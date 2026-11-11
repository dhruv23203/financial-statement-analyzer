"""Provider calls with retry, plus conversion of statement frames into raw records.

Nothing here touches the database; :mod:`app.ingestion.service` persists results.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Any, TypeVar

import pandas as pd

from app.providers.base import FinancialDataProvider
from app.utils.exceptions import ProviderError, ProviderNetworkError, ProviderRateLimitError
from app.utils.helpers import is_missing, parse_period_end, to_decimal
from config.metric_aliases import STATEMENT_TYPES

logger = logging.getLogger(__name__)
T = TypeVar("T")
RETRYABLE = (ProviderNetworkError, ProviderRateLimitError)


def call_with_retry(func: Callable[[], T], *, retries: int, backoff_seconds: float,
                    sleep: Callable[[float], None], description: str) -> T:
    """Call *func*, retrying network/rate-limit errors with exponential back-off."""
    attempt = 0
    while True:
        try:
            return func()
        except RETRYABLE as exc:
            if attempt >= retries:
                raise
            delay = backoff_seconds * (2 ** attempt)
            if isinstance(exc, ProviderRateLimitError):
                delay *= 2
            attempt += 1
            logger.warning("%s failed (%s); retry %d/%d in %.1fs", description,
                           type(exc).__name__, attempt, retries, delay)
            sleep(delay)


@dataclass
class FetchOutcome:
    """Everything one provider round-trip produced for a ticker."""

    statements: dict[str, pd.DataFrame] = field(default_factory=dict)
    statement_errors: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    metadata_error: str | None = None
    network_failure: bool = False

    @property
    def available(self) -> list[str]:
        return [s for s in STATEMENT_TYPES if s in self.statements]

    @property
    def missing(self) -> dict[str, str]:
        return {s: self.statement_errors.get(s, "not returned")
                for s in STATEMENT_TYPES if s not in self.statements}


def fetch_all(provider: FinancialDataProvider, ticker: str, frequency: str, *,
              retries: int, backoff_seconds: float,
              sleep: Callable[[float], None]) -> FetchOutcome:
    """Fetch metadata (best effort) and all three statements; never raises ProviderError."""
    outcome = FetchOutcome()
    try:
        outcome.metadata = call_with_retry(
            lambda: provider.fetch_metadata(ticker), retries=retries,
            backoff_seconds=backoff_seconds, sleep=sleep,
            description=f"{ticker} metadata") or {}
    except ProviderError as exc:
        outcome.metadata_error = str(exc)

    network_errors = 0
    for statement_type in STATEMENT_TYPES:
        try:
            frame = call_with_retry(
                lambda st=statement_type: provider.fetch_statement(ticker, st, frequency),
                retries=retries, backoff_seconds=backoff_seconds, sleep=sleep,
                description=f"{ticker} {statement_type}")
        except RETRYABLE as exc:
            network_errors += 1
            outcome.statement_errors[statement_type] = str(exc)
            continue
        except ProviderError as exc:
            outcome.statement_errors[statement_type] = str(exc)
            continue
        if frame is None or frame.empty or frame.dropna(how="all").empty:
            outcome.statement_errors[statement_type] = "provider returned an empty statement"
            continue
        outcome.statements[statement_type] = frame
    outcome.network_failure = network_errors == len(STATEMENT_TYPES)
    return outcome


@dataclass
class RawRecord:
    metric: str
    period_end: date
    value: Decimal | None


@dataclass
class FrameConversion:
    records: list[RawRecord] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    duplicate_count: int = 0
    conflict_count: int = 0
    missing_count: int = 0


def frame_to_records(frame: pd.DataFrame, statement_type: str) -> FrameConversion:
    """Convert a provider frame to raw records.

    * Columns that are not valid dates are dropped (with a warning).
    * Missing cells (NaN/None/blank/inf) are kept as ``value=None`` so the raw
      store records that the provider had no value - they are never turned into 0.
    * Exact duplicate (metric, period) cells are stored once; conflicting
      duplicates keep the first non-missing value and are reported.
    """
    result = FrameConversion()
    columns: list[tuple[int, date]] = []
    for column_position, column in enumerate(frame.columns):
        period_end = parse_period_end(column)
        if period_end is None:
            result.warnings.append(f"{statement_type}: ignored non-date column {column!r}")
        else:
            columns.append((column_position, period_end))

    seen: dict[tuple[str, date], RawRecord] = {}
    for row_position, metric in enumerate(frame.index):
        if is_missing(metric):
            result.warnings.append(
                f"{statement_type}: ignored row {row_position} with blank name")
            continue
        metric_name = str(metric).strip()
        for column_position, period_end in columns:
            # Positional access works even with duplicate row/column labels.
            raw_value: Any = frame.iat[row_position, column_position]
            value = None if is_missing(raw_value) else to_decimal(raw_value)
            if value is None and not is_missing(raw_value):
                result.warnings.append(
                    f"{statement_type}: non-numeric value {raw_value!r} for {metric_name} "
                    f"{period_end} stored as missing")
            key = (metric_name, period_end)
            if key in seen:
                existing = seen[key]
                if existing.value == value or value is None:
                    result.duplicate_count += 1
                elif existing.value is None:
                    existing.value = value
                    result.duplicate_count += 1
                else:
                    result.conflict_count += 1
                    result.warnings.append(
                        f"{statement_type}: conflicting duplicates for {metric_name} "
                        f"{period_end}: kept {existing.value}, ignored {value}")
                continue
            record = RawRecord(metric_name, period_end, value)
            seen[key] = record
            result.records.append(record)
    result.missing_count = sum(1 for r in result.records if r.value is None)
    return result
