"""Day 90 - end-to-end workflow built from the reusable module services.

    run:      ingest -> clean (standardise + validate) -> analyse (ratios, trends,
              red flags, scores) -> results persisted + exported
    compare:  make sure every company has an analysis (optionally running the
              pipeline for missing ones), then compare stored results
    history:  every fetch / cleaning / analysis / comparison run of a company
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from app.analysis.comparison import ComparisonReport, compare_companies, validate_tickers
from app.analysis.service import AnalysisReport, analyze_company
from app.cleaning.cleaner import CleaningSummary, clean_company
from app.database import repositories as repo
from app.ingestion.service import IngestionResult, ingest_company
from app.services.context import AppContext
from app.utils.exceptions import AnalyzerError, DataNotFoundError
from app.utils.helpers import normalize_ticker

logger = logging.getLogger(__name__)
Progress = Callable[[str], None]


def _silent(_: str) -> None:
    return None


@dataclass
class PipelineResult:
    ticker: str
    ingestion: IngestionResult | None = None
    cleaning: CleaningSummary | None = None
    analysis: AnalysisReport | None = None
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.analysis is not None


def run_pipeline(ctx: AppContext, raw_ticker: str, frequency: str = "annual",
                 progress: Progress = _silent) -> PipelineResult:
    """Fetch -> store -> clean -> standardise -> validate -> analyse -> persist."""
    ticker = normalize_ticker(raw_ticker)
    result = PipelineResult(ticker=ticker)
    progress(f"[1/3] Fetching and storing raw statements for {ticker} "
             f"(provider: {ctx.provider.name}) ...")
    result.ingestion = ingest_company(ctx, ticker, frequency)
    if not result.ingestion.succeeded:
        result.error = result.ingestion.error or "ingestion failed"
        return result
    if frequency != "annual":
        result.error = ("Quarterly data stored. Cleaning and analysis currently use annual "
                        "statements only.")
        return result
    progress("[2/3] Cleaning, standardising and validating ...")
    result.cleaning = clean_company(ctx, ticker, result.ingestion.fetch_run_id)
    progress("[3/3] Calculating ratios, trends, red flags and health scores ...")
    result.analysis = analyze_company(ctx, ticker)
    return result


def ensure_analysed(ctx: AppContext, tickers: Sequence[str], fetch_missing: bool,
                    progress: Progress = _silent) -> dict[str, str]:
    """Run the pipeline for companies that have no analysis yet; returns per-ticker errors."""
    errors: dict[str, str] = {}
    for ticker in tickers:
        with ctx.db.session_scope() as session:
            company = repo.get_company(session, ticker)
            has_analysis = company is not None and \
                repo.latest_analysis_run(session, company.id) is not None
        if has_analysis or not fetch_missing:
            continue
        progress(f"{ticker}: no stored analysis - running the full pipeline ...")
        try:
            outcome = run_pipeline(ctx, ticker)
        except AnalyzerError as exc:
            errors[ticker] = str(exc)
            continue
        if outcome.error:
            errors[ticker] = outcome.error
    return errors


def run_comparison(ctx: AppContext, raw_tickers: Sequence[str], fiscal_year: int | None = None,
                   fetch_missing: bool = True,
                   progress: Progress = _silent) -> tuple[ComparisonReport, dict[str, str]]:
    tickers, _ = validate_tickers(raw_tickers)
    errors = ensure_analysed(ctx, tickers, fetch_missing, progress)
    return compare_companies(ctx, tickers, fiscal_year), errors


def company_history(ctx: AppContext, raw_ticker: str) -> dict[str, Any]:
    """Plain-data history of all runs for one company (newest first)."""
    ticker = normalize_ticker(raw_ticker)
    with ctx.db.session_scope() as session:
        company = repo.get_company(session, ticker)
        if company is None:
            raise DataNotFoundError(f"No records for {ticker}. Run: python main.py run {ticker}")
        return {
            "company": {k: getattr(company, k) for k in (
                "ticker", "name", "exchange", "sector", "industry", "country", "currency",
                "created_at", "updated_at")},
            "fetch_runs": [{
                "id": r.id, "started_at": r.started_at, "provider": r.provider,
                "frequency": r.frequency, "status": r.status,
                "statements": r.statements_retrieved,
                "raw_facts": repo.count_raw_facts(session, r.id),
                "error": r.error_details} for r in repo.fetch_runs_for_company(session, company.id)],
            "cleaning_runs": [{
                "id": r.id, "fetch_run_id": r.fetch_run_id, "executed_at": r.executed_at,
                "status": r.status, "raw": r.raw_record_count,
                "standardized": r.standardized_record_count, "missing": r.missing_value_count,
                "unmapped": r.unmapped_metric_count, "conflicts": r.conflict_count}
                for r in repo.cleaning_runs_for_company(session, company.id)],
            "analysis_runs": [{
                "id": r.id, "fetch_run_id": r.fetch_run_id, "executed_at": r.executed_at,
                "status": r.status, "methodology": r.methodology_version,
                "latest_period": r.latest_fiscal_period,
                "overall_score": (r.summary or {}).get("overall_score"),
                "red_flags": (r.summary or {}).get("red_flags_total"),
                "error": r.error_details}
                for r in repo.analysis_runs_for_company(session, company.id)],
            "comparison_runs": [{
                "id": r.id, "executed_at": r.executed_at, "status": r.status,
                "tickers": r.requested_tickers, "currency": r.currency_compatibility,
                "periods": r.period_alignment}
                for r in repo.comparison_runs_for_company(session, company.id)],
        }
