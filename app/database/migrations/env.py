"""Alembic migration environment.

Supports two modes:
* programmatic (the application passes an open connection in
  ``config.attributes["connection"]``), and
* the ``alembic`` CLI (a connection is created from the configured URL).

``render_as_batch=True`` is required because SQLite cannot ALTER most constraints
in place; Alembic recreates the table instead.
"""

from __future__ import annotations

from alembic import context
from sqlalchemy import create_engine, event

from app.database import models  # noqa: F401  (registers all tables on Base.metadata)
from app.database.base import Base

config = context.config
target_metadata = Base.metadata


def _fk_on(dbapi_connection, _record) -> None:  # pragma: no cover - CLI path only
    dbapi_connection.execute("PRAGMA foreign_keys = ON")


def _run(connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata,
                      render_as_batch=True, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_offline() -> None:  # pragma: no cover - SQL script generation only
    context.configure(url=config.get_main_option("sqlalchemy.url"),
                      target_metadata=target_metadata, literal_binds=True,
                      render_as_batch=True, dialect_opts={"paramstyle": "named"})
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connection = config.attributes.get("connection")
    if connection is not None:
        _run(connection)
        return
    # CLI usage: honour FSA_DB_PATH / .env through the application settings.
    from config.settings import get_settings  # pragma: no cover

    settings = get_settings()  # pragma: no cover
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)  # pragma: no cover
    engine = create_engine(settings.database_url)  # pragma: no cover
    event.listen(engine, "connect", _fk_on)  # pragma: no cover
    with engine.begin() as cli_connection:  # pragma: no cover
        _run(cli_connection)


if context.is_offline_mode():  # pragma: no cover
    run_migrations_offline()
else:
    run_migrations_online()
