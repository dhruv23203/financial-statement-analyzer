"""Day 10 ingestion service: fetch -> raw CSV backup -> raw facts in the database.

Status rules for a fetch run
----------------------------
* SUCCESS          all three statements retrieved
* PARTIAL_SUCCESS  one or two statements retrieved
* FAILED           no statement retrieved (invalid ticker, network failure, ...)

Every attempt - including failures - is recorded in ``fetch_runs``. Earlier runs
are never modified or deleted, so repeated ingestion builds a history.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from app.database import repositories as repo
from app.database.models import (FREQUENCIES, RUN_FAILED, RUN_PARTIAL, RUN_RUNNING,
                                 RUN_SUCCESS, Company, FetchRun, RawFinancialFact)
from app.ingestion.currency import FAILED as CURRENCY_FAILED
from app.ingestion.currency import OVERRIDDEN as CURRENCY_OVERRIDDEN
from app.ingestion.currency import check_currency, latest_statement_revenue, \
    load_currency_overrides
from app.ingestion.fetcher import FetchOutcome, fetch_all, frame_to_records
from app.services.context import AppContext
from app.utils.exceptions import AnalyzerError
from app.utils.helpers import normalize_ticker, safe_child_path, to_jsonable, utcnow
from config.metric_aliases import STATEMENT_LABELS, STATEMENT_TYPES

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    ticker: str
    fetch_run_id: int
    status: str
    company_name: str | None = None
    currency: str | None = None
    available_statements: list[str] = field(default_factory=list)
    missing_statements: dict[str, str] = field(default_factory=dict)
    raw_fact_count: int = 0
    missing_value_count: int = 0
    periods: list[str] = field(default_factory=list)
    raw_files: dict[str, str] = field(default_factory=dict)
    metadata_error: str | None = None
    currency_check: dict[str, Any] = field(default_factory=dict)
    market_status: str | None = None
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.status in (RUN_SUCCESS, RUN_PARTIAL)


def _save_raw_csv(ctx: AppContext, ticker: str, run_id: int, statement_type: str,
                  frame: pd.DataFrame) -> str:
    """Write the untouched provider frame to data/raw/<TICKER>/ (never overwrites)."""
    folder = safe_child_path(ctx.settings.raw_dir, ticker)
    folder.mkdir(parents=True, exist_ok=True)
    stamp = utcnow().strftime("%Y%m%dT%H%M%SZ")
    path = folder / f"fetch{run_id:05d}_{stamp}_{statement_type}.csv"
    frame.to_csv(path)
    data_root = ctx.settings.data_dir.resolve()
    return path.relative_to(data_root).as_posix() if path.is_relative_to(data_root) \
        else path.as_posix()


def _status_for(outcome: FetchOutcome) -> str:
    count = len(outcome.available)
    if count == len(STATEMENT_TYPES):
        return RUN_SUCCESS
    return RUN_PARTIAL if count else RUN_FAILED


def _failure_message(ticker: str, outcome: FetchOutcome) -> str:
    if outcome.network_failure:
        return ("Could not reach the data provider (network failure, timeout or rate limit) "
                "after retries. Check the internet connection and try again later.")
    if not outcome.metadata and all("empty" in e for e in outcome.statement_errors.values()):
        return (f"No financial statements found for {ticker}. The ticker may be invalid, "
                "delisted or not an equity (Indian stocks need .NS or .BO, e.g. TCS.NS).")
    details = "; ".join(f"{STATEMENT_LABELS[k]}: {v}" for k, v in outcome.statement_errors.items())
    return f"No statements retrieved for {ticker}. {details}"


def ingest_company(ctx: AppContext, raw_ticker: str, frequency: str = "annual") -> IngestionResult:
    """Fetch and store raw statements for one ticker. Never raises for provider failures."""
    ticker = normalize_ticker(raw_ticker)
    if frequency not in FREQUENCIES:
        raise AnalyzerError(f"frequency must be one of {FREQUENCIES}")

    # 1) company + RUNNING fetch run committed first, so even a crash leaves a trace.
    with ctx.db.session_scope() as session:
        company, created = repo.get_or_create_company(session, ticker)
        run = FetchRun(company_id=company.id, provider=ctx.provider.name, frequency=frequency,
                       status=RUN_RUNNING, started_at=utcnow())
        session.add(run)
        session.flush()
        run_id, company_id = run.id, company.id
    logger.info("Fetch run started", extra={"ticker": ticker, "fetch_run_id": run_id,
                                            "provider": ctx.provider.name,
                                            "new_company": created})

    result = IngestionResult(ticker=ticker, fetch_run_id=run_id, status=RUN_RUNNING)
    try:
        outcome = fetch_all(ctx.provider, ticker, frequency,
                            retries=ctx.settings.fetch_max_retries,
                            backoff_seconds=ctx.settings.fetch_backoff_seconds, sleep=ctx.sleep)
        _persist_outcome(ctx, ticker, company_id, run_id, frequency, outcome, result)
        if result.succeeded:
            from app.ingestion.market import record_market_snapshot

            result.market_status = record_market_snapshot(ctx, ticker, company_id)
    except Exception as exc:
        # Unexpected failure: record it on the run, then re-raise for the caller to report.
        logger.exception("Ingestion failed unexpectedly", extra={"ticker": ticker,
                                                                 "fetch_run_id": run_id})
        _mark_failed(ctx, run_id, f"{type(exc).__name__}: {exc}")
        raise
    logger.info("Fetch run finished", extra={"ticker": ticker, "fetch_run_id": run_id,
                                             "status": result.status,
                                             "raw_facts": result.raw_fact_count})
    return result


def _persist_outcome(ctx: AppContext, ticker: str, company_id: int, run_id: int,
                     frequency: str, outcome: FetchOutcome, result: IngestionResult) -> None:
    status = _status_for(outcome)
    result.status = status
    result.available_statements = outcome.available
    result.missing_statements = outcome.missing
    result.metadata_error = outcome.metadata_error

    raw_files: dict[str, str] = {}
    for statement_type, frame in outcome.statements.items():
        raw_files[statement_type] = _save_raw_csv(ctx, ticker, run_id, statement_type, frame)
    result.raw_files = raw_files

    override = load_currency_overrides(ctx.settings.currency_overrides_file).get(ticker)
    with ctx.db.session_scope() as session:
        company = session.get(Company, company_id)
        if outcome.metadata:
            repo.update_company_metadata(company, outcome.metadata)
        if override is not None:
            company.currency = override["currency"].strip().upper()
        currency = company.currency
        if currency is None and outcome.statements:
            result.warnings.append("Reporting currency unknown (metadata unavailable); "
                                   "amounts stored without a currency.")
        currency_check = check_currency(
            latest_statement_revenue(outcome.statements.get("income_statement")),
            outcome.metadata, currency, override)
        result.currency_check = currency_check
        if currency_check["status"] in (CURRENCY_FAILED, CURRENCY_OVERRIDDEN):
            result.warnings.append(currency_check["message"])
        result.company_name, result.currency = company.name, currency

        periods: set[str] = set()
        retrieved_at = utcnow()
        statement_summary: dict[str, Any] = {}
        for statement_type, frame in outcome.statements.items():
            conversion = frame_to_records(frame, statement_type)
            result.warnings.extend(conversion.warnings)
            for record in conversion.records:
                session.add(RawFinancialFact(
                    fetch_run_id=run_id, company_id=company_id, statement_type=statement_type,
                    original_metric_name=record.metric[:200],
                    fiscal_period_end=record.period_end, period_type=frequency,
                    original_value=record.value, currency=currency, unit="units",
                    retrieved_at=retrieved_at))
                periods.add(record.period_end.isoformat())
            result.raw_fact_count += len(conversion.records)
            result.missing_value_count += conversion.missing_count
            statement_summary[statement_type] = {
                "rows": int(frame.shape[0]), "periods": int(frame.shape[1]),
                "records": len(conversion.records), "missing_values": conversion.missing_count,
                "duplicates_skipped": conversion.duplicate_count,
                "conflicting_duplicates": conversion.conflict_count,
            }
        result.periods = sorted(periods, reverse=True)

        run = session.get(FetchRun, run_id)
        run.status = status
        run.completed_at = utcnow()
        run.statements_retrieved = len(outcome.available)
        run.raw_file_refs = raw_files or None
        run.fetch_metadata = to_jsonable({
            "available_statements": outcome.available,
            "missing_statements": outcome.missing,
            "metadata": outcome.metadata,
            "metadata_error": outcome.metadata_error,
            "currency_check": result.currency_check,
            "statements": statement_summary,
            "periods": result.periods,
            "warnings": result.warnings[:200],
        })
        if status == RUN_FAILED:
            run.error_details = _failure_message(ticker, outcome)
            result.error = run.error_details
        elif outcome.statement_errors:
            run.error_details = "; ".join(
                f"{STATEMENT_LABELS[k]}: {v}" for k, v in outcome.statement_errors.items())


def _mark_failed(ctx: AppContext, run_id: int, message: str) -> None:
    try:
        with ctx.db.session_scope() as session:
            run = session.get(FetchRun, run_id)
            if run is not None:
                run.status = RUN_FAILED
                run.completed_at = utcnow()
                run.error_details = message[:4000]
    except AnalyzerError:
        logger.exception("Could not record failure for fetch run %s", run_id)
