"""Read-only views of stored results (used by the web dashboard).

Viewing never recalculates anything and never creates new runs: the latest
successful analysis run is read back from the database and converted into the same
objects the analysis service produces, so the existing table/chart builders in
``app.reporting`` can be reused unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from app.analysis.health_score import OVERALL, CategoryScore
from app.analysis.ratios import FinancialDataset, RatioOutcome
from app.analysis.red_flags import RedFlag
from app.analysis.trends import TrendOutcome
from app.database import repositories as repo
from app.database.models import FREQ_ANNUAL, FetchRun
from app.services.context import AppContext
from app.utils.exceptions import DataNotFoundError
from app.utils.helpers import normalize_ticker


@dataclass
class StoredAnalysis:
    ticker: str
    company_name: str | None
    sector: str | None
    industry: str | None
    exchange: str | None
    currency: str | None
    analysis_run_id: int
    fetch_run_id: int
    status: str
    executed_at: datetime
    methodology_version: str
    scoring_version: str | None
    dataset: FinancialDataset
    ratios: list[RatioOutcome]
    trends: list[TrendOutcome]
    flags: list[RedFlag]
    scores: dict[date, list[CategoryScore]]
    summary: dict[str, Any] = field(default_factory=dict)
    currency_check: dict[str, Any] = field(default_factory=dict)

    @property
    def latest_period(self) -> date | None:
        return max(self.scores) if self.scores else None

    def latest_scores(self) -> list[CategoryScore]:
        return self.scores.get(self.latest_period, []) if self.latest_period else []

    def overall(self) -> CategoryScore | None:
        return next((c for c in self.latest_scores() if c.category == OVERALL), None)


def list_company_summaries(ctx: AppContext) -> list[dict[str, Any]]:
    """One row per company with its latest analysis summary (if any)."""
    rows = []
    with ctx.db.session_scope() as session:
        for company in repo.list_companies(session):
            run = repo.latest_analysis_run(session, company.id)
            summary = (run.summary or {}) if run else {}
            rows.append({
                "ticker": company.ticker, "name": company.name, "sector": company.sector,
                "currency": company.currency, "analysed": run is not None,
                "latest_fy": summary.get("latest_fiscal_period"),
                "score": summary.get("overall_score"),
                "interpretation": summary.get("overall_interpretation"),
                "red_flags": summary.get("red_flags_total"),
            })
    return rows


def load_stored_analysis(ctx: AppContext, raw_ticker: str) -> StoredAnalysis:
    """Latest successful analysis of *ticker*, rebuilt from the database."""
    ticker = normalize_ticker(raw_ticker)
    with ctx.db.session_scope() as session:
        company = repo.get_company(session, ticker)
        if company is None:
            raise DataNotFoundError(f"{ticker} is not in the database yet.")
        run = repo.latest_analysis_run(session, company.id)
        if run is None:
            raise DataNotFoundError(f"{ticker} has no successful analysis yet.")
        fetch_run = session.get(FetchRun, run.fetch_run_id)
        ratios = [RatioOutcome(r.ratio_code, r.fiscal_period_end, r.status, r.value,
                               r.reason_code, r.method, r.input_values or {}, r.explanation)
                  for r in repo.ratios_for_run(session, run.id)]
        trends = [TrendOutcome(t.metric_code, t.comparison_type, t.current_period,
                               t.previous_period, t.current_value, t.previous_value,
                               t.absolute_change, t.percentage_change, t.direction, t.status,
                               t.explanation)
                  for t in repo.trends_for_run(session, run.id)]
        flags = [RedFlag(f.rule_code, f.rule_name, f.severity, f.fiscal_period_end,
                         f.previous_period_end, f.title, f.triggered_condition, f.description,
                         f.evidence or {})
                 for f in repo.red_flags_for_run(session, run.id)]
        scores: dict[date, list[CategoryScore]] = {}
        for s in repo.health_scores_for_run(session, run.id):
            scores.setdefault(s.fiscal_period_end, []).append(CategoryScore(
                s.category, s.fiscal_period_end, s.score, s.status, s.data_coverage,
                s.explanation, [], s.interpretation))
        analysed_periods = {o.period for o in ratios}
        facts = repo.standardized_facts_for_run(session, run.fetch_run_id, FREQ_ANNUAL,
                                                mapped_only=True)
        dataset = FinancialDataset.from_records(
            ticker, run.currency or company.currency,
            ((f.fiscal_period_end, f.metric_code, f.value) for f in facts
             if f.fiscal_period_end in analysed_periods))
        return StoredAnalysis(
            ticker=ticker, company_name=company.name, sector=company.sector,
            industry=company.industry, exchange=company.exchange,
            currency=run.currency or company.currency, analysis_run_id=run.id,
            fetch_run_id=run.fetch_run_id, status=run.status, executed_at=run.executed_at,
            methodology_version=run.methodology_version,
            scoring_version=run.scoring_methodology_version, dataset=dataset, ratios=ratios,
            trends=trends, flags=flags, scores=scores, summary=run.summary or {},
            currency_check=((fetch_run.fetch_metadata or {}).get("currency_check") or {})
            if fetch_run else {})


def key_financials(analysis: StoredAnalysis,
                   codes: tuple[str, ...] = ("total_revenue", "net_income",
                                             "operating_cash_flow", "total_debt")
                   ) -> dict[str, dict[date, Decimal | None]]:
    """``{metric: {period: value}}`` for charting (missing values stay ``None``)."""
    return {code: {p: analysis.dataset.get(p, code) for p in analysis.dataset.periods}
            for code in codes}
