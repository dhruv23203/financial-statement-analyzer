"""Small, dependency-free helpers shared by all modules."""

from __future__ import annotations

import json
import math
import re
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from app.utils.exceptions import InvalidTickerError

# Yahoo-style tickers: letters, digits and . - & = (e.g. TCS.NS, BRK-B, M&M.NS).
_TICKER_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9.\-&=]{0,19}$")
_SAFE_FILENAME_PATTERN = re.compile(r"[^A-Za-z0-9._-]")

RATIO_DECIMAL_PLACES = Decimal("0.000001")


def normalize_ticker(raw: str | None) -> str:
    """Validate and normalise a ticker (trim + upper-case).

    Raises:
        InvalidTickerError: blank, malformed or unsupported tickers (indices such as
            ``^NSEI`` have no financial statements and are rejected).
    """
    if raw is None or not str(raw).strip():
        raise InvalidTickerError("Ticker cannot be blank. Example: TCS.NS or AAPL")
    ticker = str(raw).strip().upper()
    if ticker.startswith("^"):
        raise InvalidTickerError(
            f"Unsupported ticker {ticker!r}: market indices have no financial statements.")
    if ".." in ticker or not _TICKER_PATTERN.fullmatch(ticker):
        raise InvalidTickerError(
            f"Invalid ticker {ticker!r}. Use letters, digits and . - & = only "
            "(max 20 characters), e.g. TCS.NS, INFY.NS, AAPL.")
    return ticker


def safe_filename(text: str) -> str:
    """Turn arbitrary text into a filesystem-safe name (no path separators, no '..')."""
    cleaned = _SAFE_FILENAME_PATTERN.sub("_", text).strip("._")
    cleaned = cleaned.replace("..", "_")
    return cleaned or "unnamed"


def safe_child_path(base: Path, *parts: str) -> Path:
    """Join *parts* (sanitised) under *base* and guarantee the result stays inside *base*."""
    base_resolved = base.resolve()
    candidate = base_resolved.joinpath(*(safe_filename(p) for p in parts)).resolve()
    if base_resolved != candidate and base_resolved not in candidate.parents:
        raise ValueError(f"Refusing to write outside {base_resolved}: {candidate}")
    return candidate


def to_decimal(value: Any) -> Decimal | None:
    """Convert a provider value to ``Decimal``; NaN, None, blanks and infinities -> ``None``.

    Floats are converted through ``repr`` so the shortest round-trip representation
    is kept (``0.1`` -> ``Decimal('0.1')``, not the binary expansion).
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, Decimal):
        return value if value.is_finite() else None
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        # float(...) also unwraps numpy.float64, whose repr is "np.float64(x)" in NumPy 2.
        value = float(value)
        return None if not math.isfinite(value) else Decimal(repr(value))
    # numpy scalars and strings
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(as_float):
        return None
    text = str(value).strip()
    try:
        result = Decimal(text)
    except InvalidOperation:
        result = Decimal(repr(as_float))
    return result if result.is_finite() else None


def is_missing(value: Any) -> bool:
    """True for None, NaN, +/-inf, empty strings and textual nulls."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "nan", "none", "null", "n/a", "na", "-"}
    return to_decimal(value) is None


def quantize_ratio(value: Decimal | None) -> Decimal | None:
    """Round a calculated ratio to 6 decimal places (stored precision)."""
    if value is None:
        return None
    return value.quantize(RATIO_DECIMAL_PLACES)


def utcnow() -> datetime:
    """Timezone-aware current UTC time (the only clock used for stored timestamps)."""
    return datetime.now(timezone.utc)


def parse_period_end(value: Any) -> date | None:
    """Parse a statement column header into a ``date``; ``None`` if it is not a date."""
    import pandas as pd

    if value is None or value is pd.NaT:  # NaT subclasses datetime, so check it first
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError):
        return None
    if pd.isna(ts):
        return None
    return ts.date()


def json_default(value: Any) -> Any:
    """``json.dumps`` default that serialises Decimal, dates and paths."""
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serialisable")


def to_jsonable(value: Any) -> Any:
    """Round-trip through JSON so the result can be stored in a JSON column."""
    return json.loads(json.dumps(value, default=json_default))


def fmt_number(value: Decimal | float | None, decimals: int = 2) -> str:
    """Human-readable number with thousands separators; ``None`` -> 'N/A'."""
    if value is None:
        return "N/A"
    return f"{float(value):,.{decimals}f}"


def fmt_amount(value: Decimal | float | None, currency: str | None) -> str:
    """Large amount in millions with its currency, e.g. 'INR 2,553.24M'."""
    if value is None:
        return "N/A"
    return f"{currency or '?'} {float(value) / 1e6:,.2f}M"


def years_between(start: date, end: date) -> float:
    return (end - start).days / 365.25
