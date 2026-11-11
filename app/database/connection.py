"""Engine/session management, migrations, health check, backup and restore."""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import Engine, create_engine, event, inspect, text
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session, sessionmaker

from app.database.base import Base
from app.utils.exceptions import DatabaseError
from app.utils.helpers import utcnow
from config.settings import ALEMBIC_INI, MIGRATIONS_DIR, Settings

logger = logging.getLogger(__name__)


def _enable_sqlite_pragmas(dbapi_connection: Any, _record: Any) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys = ON")
    cursor.close()


def translate_operational_error(exc: OperationalError) -> DatabaseError:
    """Convert low-level SQLite errors into clear, user-facing messages."""
    message = str(exc.orig) if exc.orig is not None else str(exc)
    if "locked" in message or "busy" in message:
        return DatabaseError(
            "The database is locked by another process (is another command still running, "
            "or the .db file open in a DB browser?). Close it and try again.")
    if "no such table" in message:
        return DatabaseError(
            "Database tables are not initialised. Run: python main.py init-db")
    if "unable to open" in message:
        return DatabaseError(f"Cannot open the database file: {message}")
    return DatabaseError(f"Database error: {message}")


@dataclass
class HealthReport:
    """Result of :meth:`Database.health_check`."""

    db_path: str
    exists: bool
    ok: bool = False
    size_bytes: int = 0
    sqlite_version: str | None = None
    current_revision: str | None = None
    head_revision: str | None = None
    integrity: str | None = None
    foreign_key_violations: int = 0
    table_counts: dict[str, int] = field(default_factory=dict)
    missing_tables: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


class Database:
    """Owns the SQLAlchemy engine for one SQLite file."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.db_path = settings.db_path
        self.engine: Engine = create_engine(
            settings.database_url,
            # check_same_thread=False: the web dashboard serves requests from several
            # threads; the connection pool hands each connection to one thread at a time.
            connect_args={"timeout": settings.db_timeout_seconds, "check_same_thread": False},
        )
        event.listen(self.engine, "connect", _enable_sqlite_pragmas)
        self._session_factory = sessionmaker(self.engine, expire_on_commit=False)

    # ------------------------------------------------------------------ sessions
    @contextmanager
    def session_scope(self) -> Iterator[Session]:
        """Transactional scope: commit on success, roll back on any error."""
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except OperationalError as exc:
            session.rollback()
            logger.error("Database operational error; transaction rolled back",
                         extra={"error": str(exc.orig)})
            raise translate_operational_error(exc) from exc
        except IntegrityError as exc:
            session.rollback()
            logger.error("Integrity error; transaction rolled back", extra={"error": str(exc.orig)})
            raise DatabaseError(f"Database integrity error: {exc.orig}") from exc
        except BaseException:
            session.rollback()
            logger.debug("Transaction rolled back because of an exception", exc_info=True)
            raise
        finally:
            session.close()

    def dispose(self) -> None:
        self.engine.dispose()

    # ------------------------------------------------------------------ migrations
    def _alembic_config(self) -> Config:
        cfg = Config(str(ALEMBIC_INI))
        cfg.set_main_option("script_location", str(MIGRATIONS_DIR).replace("%", "%%"))
        cfg.set_main_option("sqlalchemy.url", self.settings.database_url.replace("%", "%%"))
        return cfg

    def head_revision(self) -> str | None:
        return ScriptDirectory.from_config(self._alembic_config()).get_current_head()

    def current_revision(self) -> str | None:
        try:
            with self.engine.connect() as connection:
                return MigrationContext.configure(connection).get_current_revision()
        except OperationalError as exc:
            raise translate_operational_error(exc) from exc

    def upgrade(self) -> str | None:
        """Create the file if needed and apply all Alembic migrations (idempotent)."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        cfg = self._alembic_config()
        try:
            with self.engine.begin() as connection:
                cfg.attributes["connection"] = connection
                command.upgrade(cfg, "head")
        except OperationalError as exc:
            raise translate_operational_error(exc) from exc
        revision = self.current_revision()
        logger.info("Database schema at revision %s", revision, extra={"db": str(self.db_path)})
        return revision

    def ensure_ready(self) -> None:
        """Called at application start-up: initialise/upgrade the schema when needed."""
        if not self.db_path.exists() or self.current_revision() != self.head_revision():
            self.upgrade()

    # ------------------------------------------------------------------ health check
    def health_check(self) -> HealthReport:
        """Check connectivity, schema version, integrity and foreign keys (read-only)."""
        report = HealthReport(db_path=str(self.db_path), exists=self.db_path.exists())
        report.head_revision = self.head_revision()
        if not report.exists:
            report.problems.append("Database file does not exist. Run: python main.py init-db")
            return report
        report.size_bytes = self.db_path.stat().st_size
        try:
            with self.engine.connect() as connection:
                report.sqlite_version = connection.execute(
                    text("select sqlite_version()")).scalar_one()
                report.current_revision = MigrationContext.configure(
                    connection).get_current_revision()
                report.integrity = connection.execute(
                    text("PRAGMA integrity_check")).scalar_one()
                report.foreign_key_violations = len(
                    connection.execute(text("PRAGMA foreign_key_check")).fetchall())
                existing = set(inspect(connection).get_table_names())
                for table in Base.metadata.sorted_tables:
                    if table.name not in existing:
                        report.missing_tables.append(table.name)
                        continue
                    report.table_counts[table.name] = connection.execute(
                        text(f'SELECT COUNT(*) FROM "{table.name}"')).scalar_one()
        except OperationalError as exc:
            report.problems.append(str(translate_operational_error(exc)))
            return report

        if report.current_revision is None:
            report.problems.append("Schema not initialised (no migration applied). "
                                   "Run: python main.py init-db")
        elif report.current_revision != report.head_revision:
            report.problems.append(
                f"Schema revision {report.current_revision} is behind {report.head_revision}. "
                "Run: python main.py init-db")
        if report.missing_tables:
            report.problems.append(f"Missing tables: {', '.join(report.missing_tables)}")
        if report.integrity != "ok":
            report.problems.append(f"PRAGMA integrity_check returned: {report.integrity}")
        if report.foreign_key_violations:
            report.problems.append(
                f"{report.foreign_key_violations} foreign-key violation(s) found")
        report.ok = not report.problems
        return report

    # ------------------------------------------------------------------ backup / restore
    def backup(self, label: str = "manual") -> Path:
        """Create a consistent online copy using the SQLite backup API."""
        if not self.db_path.exists():
            raise DatabaseError("Nothing to back up: the database file does not exist.")
        self.settings.backups_dir.mkdir(parents=True, exist_ok=True)
        stamp = utcnow().strftime("%Y%m%dT%H%M%S%fZ")
        safe_label = "".join(ch for ch in label if ch.isalnum() or ch in "-_") or "backup"
        target = self.settings.backups_dir / f"{self.db_path.stem}_{stamp}_{safe_label}.db"
        _sqlite_copy(self.db_path, target, self.settings.db_timeout_seconds)
        _verify_sqlite_file(target)
        logger.info("Database backup created", extra={"backup": str(target)})
        return target

    def restore(self, backup_path: Path) -> Path | None:
        """Restore from *backup_path*; the current database is backed up first.

        Returns the path of the safety backup taken before restoring (or ``None``
        if no database existed).
        """
        backup_path = Path(backup_path)
        if not backup_path.is_file():
            raise DatabaseError(f"Backup file not found: {backup_path}")
        _verify_sqlite_file(backup_path, require_schema=True)
        safety_copy = self.backup(label="pre-restore") if self.db_path.exists() else None
        self.dispose()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        _sqlite_copy(backup_path, self.db_path, self.settings.db_timeout_seconds)
        self.upgrade()  # bring an older backup up to the current schema
        logger.info("Database restored", extra={"source": str(backup_path),
                                                "safety_copy": str(safety_copy)})
        return safety_copy


def _sqlite_copy(source: Path, target: Path, timeout: float) -> None:
    src = sqlite3.connect(str(source), timeout=timeout)
    dst = sqlite3.connect(str(target), timeout=timeout)
    try:
        src.backup(dst)
    except sqlite3.OperationalError as exc:
        raise DatabaseError(f"Backup/restore failed: {exc}") from exc
    finally:
        dst.close()
        src.close()


def _verify_sqlite_file(path: Path, require_schema: bool = False) -> None:
    try:
        connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
        try:
            result = connection.execute("PRAGMA integrity_check").fetchone()[0]
            tables = {row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
        finally:
            connection.close()
    except sqlite3.DatabaseError as exc:
        raise DatabaseError(f"{path} is not a valid SQLite database: {exc}") from exc
    if result != "ok":
        raise DatabaseError(f"Integrity check failed for {path}: {result}")
    if require_schema and "alembic_version" not in tables:
        raise DatabaseError(f"{path} is not a Financial Analyzer database backup.")
