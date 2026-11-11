"""Day 20 cleaning service: raw facts -> standardised, validated facts + CSV exports.

Re-cleaning the same fetch run replaces that run's standardised facts inside one
transaction (raw facts are never touched); every pass is recorded in
``cleaning_runs`` so the history of cleaning summaries is preserved.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from app.cleaning.standardizer import RawInput, StandardizationResult, standardize
from app.cleaning.validator import validate_dataset, validate_record
from app.database import repositories as repo
from app.database.models import (MAPPING_UNMAPPED, RUN_FAILED, RUN_PARTIAL, RUN_SUCCESS,
                                 VALIDATION_MISSING, VALIDATION_VALID, CleaningRun, FetchRun,
                                 StandardizedFinancialFact)
from app.services.context import AppContext
from app.utils.exceptions import DataNotFoundError
from app.utils.helpers import normalize_ticker, safe_child_path, utcnow
from config.metric_aliases import STATEMENT_TYPES

logger = logging.getLogger(__name__)


@dataclass
class CleaningSummary:
    ticker: str
    fetch_run_id: int
    cleaning_run_id: int
    status: str
    currency: str | None
    raw_record_count: int = 0
    standardized_record_count: int = 0
    mapped_record_count: int = 0
    missing_value_count: int = 0
    unmapped_metric_count: int = 0
    conflict_count: int = 0
    duplicate_count: int = 0
    validation_warning_count: int = 0
    periods: list[str] = field(default_factory=list)
    unmapped_metrics: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    exports: dict[str, str] = field(default_factory=dict)


def clean_company(ctx: AppContext, raw_ticker: str, fetch_run_id: int | None = None) -> \
        CleaningSummary:
    """Clean the latest usable fetch run of *ticker* (or a specific *fetch_run_id*)."""
    ticker = normalize_ticker(raw_ticker)
    with ctx.db.session_scope() as session:
        company = repo.get_company(session, ticker)
        if company is None:
            raise DataNotFoundError(f"{ticker} has not been ingested. Run: "
                                    f"python main.py ingest {ticker}")
        if fetch_run_id is None:
            run = repo.latest_fetch_run(session, company.id)
        else:
            run = session.get(FetchRun, fetch_run_id)
            if run is not None and run.company_id != company.id:
                run = None
        if run is None or run.status not in repo.USABLE_RUN_STATUSES:
            raise DataNotFoundError(
                f"No successful ingestion found for {ticker}. Run: python main.py ingest {ticker}")
        raw_facts = repo.raw_facts_for_run(session, run.id)
        inputs = [RawInput(f.id, f.statement_type, f.original_metric_name, f.fiscal_period_end,
                           f.original_value) for f in raw_facts]
        available = list((run.fetch_metadata or {}).get("available_statements") or
                         sorted({f.statement_type for f in raw_facts}))
        period_type, currency, run_id, company_id = (run.frequency, company.currency, run.id,
                                                     company.id)

    standardization = standardize(inputs)
    dataset_warnings = validate_dataset(standardization.records, available)
    summary = CleaningSummary(ticker=ticker, fetch_run_id=run_id, cleaning_run_id=0,
                              status=RUN_SUCCESS, currency=currency)
    summary.raw_record_count = len(inputs)

    try:
        with ctx.db.session_scope() as session:
            removed = repo.delete_standardized_for_run(session, run_id)
            cleaning_run = CleaningRun(company_id=company_id, fetch_run_id=run_id,
                                       executed_at=utcnow(), status=RUN_SUCCESS)
            session.add(cleaning_run)
            session.flush()
            rows = _build_rows(standardization, company_id, run_id, cleaning_run.id,
                               period_type, currency, summary)
            session.add_all(rows)
            summary.warnings = dataset_warnings
            summary.status = _status(summary, dataset_warnings, available)
            _fill_cleaning_run(cleaning_run, summary, standardization)
            summary.cleaning_run_id = cleaning_run.id
            if removed:
                logger.info("Replaced %d previous standardised facts", removed,
                            extra={"ticker": ticker, "fetch_run_id": run_id})
    except Exception as exc:
        _record_failed_cleaning(ctx, company_id, run_id, exc)
        raise

    summary.exports = _export_csv(ctx, ticker, run_id, standardization, period_type, currency)
    logger.info("Cleaning finished", extra={"ticker": ticker, "fetch_run_id": run_id,
                                            "status": summary.status,
                                            "standardized": summary.standardized_record_count})
    return summary


def _build_rows(standardization: StandardizationResult, company_id: int, fetch_run_id: int,
                cleaning_run_id: int, period_type: str, currency: str | None,
                summary: CleaningSummary) -> list[StandardizedFinancialFact]:
    rows = []
    periods = set()
    for record in standardization.records:
        status, messages = validate_record(record)
        if status == VALIDATION_MISSING:
            summary.missing_value_count += 1
        elif status != VALIDATION_VALID:
            summary.validation_warning_count += 1
        if record.mapping_status != MAPPING_UNMAPPED:
            summary.mapped_record_count += 1
            periods.add(record.fiscal_period_end.isoformat())
        rows.append(StandardizedFinancialFact(
            company_id=company_id, fetch_run_id=fetch_run_id, cleaning_run_id=cleaning_run_id,
            raw_fact_id=record.raw_fact_id, statement_type=record.statement_type,
            metric_code=record.metric_code, metric_name=record.metric_name,
            fiscal_period_end=record.fiscal_period_end, period_type=period_type,
            value=record.value, currency=currency, source_metric_name=record.source_metric_name,
            mapping_status=record.mapping_status, validation_status=status,
            validation_message=" ".join(messages) or None))
    summary.standardized_record_count = len(rows)
    summary.unmapped_metric_count = len(standardization.unmapped_names)
    summary.unmapped_metrics = standardization.unmapped_names
    summary.conflict_count = len(standardization.conflicts)
    summary.conflicts = standardization.conflicts
    summary.duplicate_count = standardization.duplicate_alias_count
    summary.periods = sorted(periods, reverse=True)
    if summary.unmapped_metrics:
        logger.info("Unmapped metrics preserved for review",
                    extra={"count": summary.unmapped_metric_count,
                           "metrics": summary.unmapped_metrics[:50]})
    return rows


def _status(summary: CleaningSummary, warnings: list[str], available: list[str]) -> str:
    if summary.mapped_record_count == 0:
        return RUN_FAILED
    essential_missing = any("essential field" in w for w in warnings)
    if essential_missing or len(available) < len(STATEMENT_TYPES):
        return RUN_PARTIAL
    return RUN_SUCCESS


def _fill_cleaning_run(run: CleaningRun, summary: CleaningSummary,
                       standardization: StandardizationResult) -> None:
    run.status = summary.status
    run.raw_record_count = summary.raw_record_count
    run.standardized_record_count = summary.standardized_record_count
    run.missing_value_count = summary.missing_value_count
    run.unmapped_metric_count = summary.unmapped_metric_count
    run.conflict_count = summary.conflict_count
    run.duplicate_count = summary.duplicate_count
    run.warnings = (summary.warnings + standardization.conflicts)[:500] or None
    run.unmapped_metrics = summary.unmapped_metrics[:500] or None
    if summary.status == RUN_FAILED:
        run.error_details = "No raw fact could be mapped to a standard metric."


def _record_failed_cleaning(ctx: AppContext, company_id: int, fetch_run_id: int,
                            exc: Exception) -> None:
    logger.exception("Cleaning failed", extra={"fetch_run_id": fetch_run_id})
    with ctx.db.session_scope() as session:
        session.add(CleaningRun(company_id=company_id, fetch_run_id=fetch_run_id,
                                executed_at=utcnow(), status=RUN_FAILED,
                                error_details=f"{type(exc).__name__}: {exc}"[:4000]))


def _export_csv(ctx: AppContext, ticker: str, fetch_run_id: int,
                standardization: StandardizationResult, period_type: str,
                currency: str | None) -> dict[str, str]:
    """Write long and wide CSVs of the standardised data to data/processed/<TICKER>/."""
    folder = safe_child_path(ctx.settings.processed_dir, ticker)
    folder.mkdir(parents=True, exist_ok=True)
    long_rows = [{
        "statement_type": r.statement_type, "metric_code": r.metric_code,
        "metric_name": r.metric_name, "fiscal_period_end": r.fiscal_period_end.isoformat(),
        "period_type": period_type, "value": None if r.value is None else str(r.value),
        "currency": currency, "source_metric_name": r.source_metric_name,
        "mapping_status": r.mapping_status,
    } for r in standardization.records]
    long_frame = pd.DataFrame(long_rows)
    base = f"{ticker}_fetch{fetch_run_id:05d}"
    long_path = folder / f"{base}_standardized_long.csv"
    long_frame.to_csv(long_path, index=False)
    exports = {"long": long_path.as_posix()}
    mapped = long_frame[long_frame["mapping_status"] != MAPPING_UNMAPPED] if not \
        long_frame.empty else long_frame
    if not mapped.empty:
        wide = mapped.pivot_table(index=["statement_type", "metric_code"],
                                  columns="fiscal_period_end", values="value",
                                  aggfunc="first")
        wide = wide[sorted(wide.columns, reverse=True)]
        wide_path = folder / f"{base}_standardized_wide.csv"
        wide.to_csv(wide_path)
        exports["wide"] = wide_path.as_posix()
    return exports
