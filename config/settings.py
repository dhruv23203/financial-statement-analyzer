"""Application settings.

All filesystem locations are derived from the project directory (never from a
hard-coded user path) and can be overridden with environment variables, which
may also be placed in a ``.env`` file in the project root (see ``.env.example``).

Environment variables
---------------------
FSA_DATA_DIR            Base data folder (default: <project>/data)
FSA_DB_PATH             SQLite file (default: <data>/financial_analyzer.db)
FSA_LOGS_DIR            Log folder (default: <project>/logs)
FSA_EXPORTS_DIR         CSV export folder (default: <project>/exports)
FSA_REPORTS_DIR         Report/chart folder (default: <project>/reports)
FSA_LOG_LEVEL           Console log level (default: WARNING)
FSA_PROVIDER            Default data provider: yfinance | sample (default: yfinance)
FSA_FETCH_MAX_RETRIES   Retries for network / rate-limit errors (default: 3)
FSA_FETCH_BACKOFF_SECONDS  Initial retry back-off in seconds (default: 2)
FSA_DB_TIMEOUT_SECONDS  SQLite busy timeout (default: 30)
FSA_AVERAGE_BALANCE_FALLBACK  ending_balance | unavailable (default: ending_balance)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
CONFIG_DIR: Path = PROJECT_ROOT / "config"
SAMPLE_DATA_DIR: Path = PROJECT_ROOT / "sample_data"
ALEMBIC_INI: Path = PROJECT_ROOT / "alembic.ini"
MIGRATIONS_DIR: Path = PROJECT_ROOT / "app" / "database" / "migrations"

RED_FLAG_CONFIG_FILE: Path = CONFIG_DIR / "red_flag_thresholds.json"
SCORING_CONFIG_FILE: Path = CONFIG_DIR / "scoring_config.json"
CURRENCY_OVERRIDES_FILE: Path = CONFIG_DIR / "currency_overrides.json"

# Currency plausibility check: latest annual statement revenue divided by the
# provider's trailing-twelve-month revenue (reported in the claimed statement
# currency) must lie in this range, otherwise the currency label is flagged.
CURRENCY_CHECK_MIN_RATIO = 0.2
CURRENCY_CHECK_MAX_RATIO = 5.0

VALID_PROVIDERS = ("yfinance", "sample")
VALID_FALLBACKS = ("ending_balance", "unavailable")

# Ratio / analysis methodology version stored with every analysis run.
ANALYSIS_METHODOLOGY_VERSION = "1.0.0"

# Inventory turnover is NOT_MEANINGFUL when average inventory is below this share of
# average total assets (e.g. IT-services firms with negligible inventory would
# otherwise show turnover in the thousands).
INVENTORY_MATERIALITY_THRESHOLD = 0.005

# A fiscal period is "sparse" (excluded from analysis, with a note) when fewer than
# this many of the core metrics below are reported. Yahoo Finance often returns an
# extra, almost empty oldest year.
SPARSE_PERIOD_MIN_CORE_METRICS = 2
CORE_METRICS = ("total_revenue", "net_income", "total_assets", "stockholders_equity",
                "operating_cash_flow")

# Example peer groups offered by ``python main.py peers``.
PEER_GROUPS: dict[str, list[str]] = {
    "indian_it": ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS"],
    "indian_energy": ["RELIANCE.NS", "ONGC.NS", "BPCL.NS"],
    "us_tech": ["AAPL", "MSFT", "GOOGL"],
    "offline_demo": ["DEMOGOOD", "DEMOWEAK", "DEMOUSD"],
}

MIN_COMPARISON_COMPANIES = 2
MAX_COMPARISON_COMPANIES = 5


def _load_dotenv(path: Path) -> None:
    """Load ``KEY=VALUE`` lines from a .env file without overriding real env vars."""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got {value!r}") from exc


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a number, got {value!r}") from exc


@dataclass(frozen=True)
class Settings:
    """Resolved runtime settings. Immutable; use :meth:`with_overrides` to derive."""

    data_dir: Path
    db_path: Path
    logs_dir: Path
    exports_dir: Path
    reports_dir: Path
    log_level: str = "WARNING"
    provider: str = "yfinance"
    fetch_max_retries: int = 3
    fetch_backoff_seconds: float = 2.0
    db_timeout_seconds: float = 30.0
    average_balance_fallback: str = "ending_balance"
    red_flag_config_file: Path = field(default=RED_FLAG_CONFIG_FILE)
    scoring_config_file: Path = field(default=SCORING_CONFIG_FILE)
    currency_overrides_file: Path = field(default=CURRENCY_OVERRIDES_FILE)
    sample_data_dir: Path = field(default=SAMPLE_DATA_DIR)

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def backups_dir(self) -> Path:
        return self.data_dir / "backups"

    @property
    def database_url(self) -> str:
        return f"sqlite:///{self.db_path.as_posix()}"

    def ensure_directories(self) -> None:
        """Create every output directory (idempotent)."""
        for path in (self.data_dir, self.raw_dir, self.processed_dir, self.backups_dir,
                     self.logs_dir, self.exports_dir, self.reports_dir, self.db_path.parent):
            path.mkdir(parents=True, exist_ok=True)

    def with_overrides(self, **changes: object) -> "Settings":
        return replace(self, **changes)

    def validate(self) -> None:
        if self.provider not in VALID_PROVIDERS:
            raise ValueError(f"provider must be one of {VALID_PROVIDERS}, got {self.provider!r}")
        if self.average_balance_fallback not in VALID_FALLBACKS:
            raise ValueError(
                f"average_balance_fallback must be one of {VALID_FALLBACKS}, "
                f"got {self.average_balance_fallback!r}")
        if self.fetch_max_retries < 0:
            raise ValueError("fetch_max_retries cannot be negative")


def get_settings() -> Settings:
    """Build settings from defaults, the optional .env file and environment variables."""
    _load_dotenv(PROJECT_ROOT / ".env")
    data_dir = Path(os.environ.get("FSA_DATA_DIR") or PROJECT_ROOT / "data")
    settings = Settings(
        data_dir=data_dir,
        db_path=Path(os.environ.get("FSA_DB_PATH") or data_dir / "financial_analyzer.db"),
        logs_dir=Path(os.environ.get("FSA_LOGS_DIR") or PROJECT_ROOT / "logs"),
        exports_dir=Path(os.environ.get("FSA_EXPORTS_DIR") or PROJECT_ROOT / "exports"),
        reports_dir=Path(os.environ.get("FSA_REPORTS_DIR") or PROJECT_ROOT / "reports"),
        log_level=os.environ.get("FSA_LOG_LEVEL", "WARNING").upper(),
        provider=os.environ.get("FSA_PROVIDER", "yfinance").lower(),
        fetch_max_retries=_env_int("FSA_FETCH_MAX_RETRIES", 3),
        fetch_backoff_seconds=_env_float("FSA_FETCH_BACKOFF_SECONDS", 2.0),
        db_timeout_seconds=_env_float("FSA_DB_TIMEOUT_SECONDS", 30.0),
        average_balance_fallback=os.environ.get(
            "FSA_AVERAGE_BALANCE_FALLBACK", "ending_balance").lower(),
        scoring_config_file=Path(os.environ.get("FSA_SCORING_CONFIG") or SCORING_CONFIG_FILE),
        red_flag_config_file=Path(os.environ.get("FSA_RED_FLAG_CONFIG") or RED_FLAG_CONFIG_FILE),
    )
    settings.validate()
    return settings
