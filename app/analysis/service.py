"""Analysis service (Days 30-70): ratios -> trends -> red flags -> health scores.

Each call creates a new ``analysis_runs`` row (history is preserved) tied to the
exact fetch run whose standardised data it used, and stores every ratio, trend,
red flag and score in one transaction. CSV exports, a Markdown report and charts
are written afterwards.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from app.analysis.config_loader import load_red_flag_config, load_scoring_config
from app.analysis.health_score import OVERALL, CategoryScore, score_all_periods
from app.analysis.insights import generate_insights, grade_for
from app.analysis.ratios import (RATIO_DEFINITIONS, FinancialDataset, RatioOutcome,
                                 calculate_all_ratios, index_outcomes)
from app.analysis.red_flags import RedFlagReport, detect_red_flags
from app.analysis.trends import TrendOutcome, analyse_trends, unavailable_trend_metrics
from app.database import repositories as repo
from app.database.models import (FREQ_ANNUAL, RUN_FAILED, RUN_PARTIAL, RUN_RUNNING,
                                 RUN_SUCCESS, AnalysisRun, HealthScore, RatioResult,
                                 RedFlagResult, TrendResult)
from app.reporting import charts, tables
from app.services.context import AppContext
from app.utils.exceptions import AnalyzerError, DataNotFoundError
from app.utils.helpers import normalize_ticker, safe_child_path, to_jsonable, utcnow
from config.settings import ANALYSIS_METHODOLOGY_VERSION

logger = logging.getLogger(__name__)


@dataclass
class AnalysisReport:
    ticker: str
    company_name: str | None
    sector: str | None
    currency: str | None
    analysis_run_id: int
    fetch_run_id: int
    status: str
    dataset: FinancialDataset
    ratios: list[RatioOutcome]
    trends: list[TrendOutcome]
    red_flags: RedFlagReport
    scores: dict[date, list[CategoryScore]]
    scoring_version: str
    red_flag_version: str
    unavailable_trends: list[str] = field(default_factory=list)
    currency_check: dict[str, Any] = field(default_factory=dict)
    pros: list[str] = field(default_factory=list)
    cons: list[str] = field(default_factory=list)
    grades: dict[str, str | None] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    exports: dict[str, str] = field(default_factory=dict)
    charts: dict[str, str] = field(default_factory=dict)

    @property
    def latest_period(self) -> date | None:
        return self.dataset.periods[-1] if self.dataset.periods else None

    def latest_scores(self) -> list[CategoryScore]:
        return self.scores.get(self.latest_period, []) if self.latest_period else []

    def overall(self) -> CategoryScore | None:
        return next((c for c in self.latest_scores() if c.category == OVERALL), None)


def load_dataset(ctx: AppContext, ticker: str) -> tuple[FinancialDataset, dict[str, Any]]:
    """Build the annual dataset from the latest cleaned fetch run of *ticker*."""
    with ctx.db.session_scope() as session:
        company = repo.get_company(session, ticker)
        if company is None:
            raise DataNotFoundError(f"{ticker} has not been ingested. Run: "
                                    f"python main.py run {ticker}")
        run = repo.latest_cleaned_fetch_run(session, company.id, FREQ_ANNUAL)
        if run is None:
            raise DataNotFoundError(f"No cleaned annual data for {ticker}. Run: "
                                    f"python main.py clean {ticker}")
        facts = repo.standardized_facts_for_run(session, run.id, FREQ_ANNUAL, mapped_only=True)
        dataset = FinancialDataset.from_records(
            ticker, company.currency, ((f.fiscal_period_end, f.metric_code, f.value)
                                       for f in facts))
        snapshot = repo.latest_market_snapshot(session, company.id)
        if snapshot is not None:
            dataset.market = {"market_cap": snapshot.market_cap, "price": snapshot.price,
                              "currency": snapshot.currency,
                              "fetched_at": snapshot.fetched_at.date().isoformat(),
                              "provider_values": snapshot.provider_values or {}}
        info = {"company_id": company.id, "fetch_run_id": run.id, "name": company.name,
                "sector": company.sector,
                "currency_check": (run.fetch_metadata or {}).get("currency_check") or {}}
    dataset.exclude_sparse_periods()
    if not dataset.periods:
        raise DataNotFoundError(f"No usable standardised annual values found for {ticker} "
                                "(all periods are empty or sparse).")
    return dataset, info


def analyze_company(ctx: AppContext, raw_ticker: str, write_outputs: bool = True) -> \
        AnalysisReport:
    ticker = normalize_ticker(raw_ticker)
    dataset, info = load_dataset(ctx, ticker)
    red_flag_config = load_red_flag_config(ctx.settings.red_flag_config_file)
    scoring_config = load_scoring_config(ctx.settings.scoring_config_file)

    with ctx.db.session_scope() as session:
        run = AnalysisRun(company_id=info["company_id"], fetch_run_id=info["fetch_run_id"],
                          executed_at=utcnow(), status=RUN_RUNNING,
                          methodology_version=ANALYSIS_METHODOLOGY_VERSION,
                          scoring_methodology_version=scoring_config["methodology_version"],
                          red_flag_config_version=str(red_flag_config["version"]),
                          currency=dataset.currency, latest_fiscal_period=dataset.periods[-1])
        session.add(run)
        session.flush()
        run_id = run.id

    try:
        ratios = calculate_all_ratios(dataset, ctx.settings.average_balance_fallback)
        ratio_index = index_outcomes(ratios)
        trends = analyse_trends(dataset)
        flags = detect_red_flags(dataset, ratio_index, red_flag_config)
        scores = score_all_periods(dataset.periods, ratio_index, scoring_config,
                                   info["sector"])
        report = AnalysisReport(
            ticker=ticker, company_name=info["name"], sector=info["sector"],
            currency=dataset.currency, analysis_run_id=run_id,
            fetch_run_id=info["fetch_run_id"], status=RUN_SUCCESS, dataset=dataset,
            ratios=ratios, trends=trends, red_flags=flags, scores=scores,
            scoring_version=scoring_config["methodology_version"],
            red_flag_version=str(red_flag_config["version"]),
            unavailable_trends=unavailable_trend_metrics(dataset),
            currency_check=info["currency_check"])
        report.notes = _notes(report)
        insights = generate_insights(ratios, trends, flags.flags, scoring_config, info["sector"],
                                     dataset.periods[-1])
        report.pros, report.cons = insights.pros, insights.cons
        report.grades = {c.category: grade_for(c.category, c.score, scoring_config)
                         for c in report.latest_scores()}
        overall = report.overall()
        report.status = RUN_SUCCESS if overall and overall.score is not None else RUN_PARTIAL
        _persist(ctx, report, info["company_id"])
    except Exception as exc:
        logger.exception("Analysis failed", extra={"ticker": ticker, "analysis_run_id": run_id})
        _mark_failed(ctx, run_id, exc)
        raise

    if write_outputs:
        report.exports = export_analysis(ctx, report)
        report.charts = draw_analysis_charts(ctx, report)
    logger.info("Analysis finished", extra={"ticker": ticker, "analysis_run_id": run_id,
                                            "status": report.status})
    return report


def _notes(report: AnalysisReport) -> list[str]:
    notes = []
    latest = report.latest_period
    if report.currency_check.get("status") in ("FAILED", "OVERRIDDEN"):
        notes.append(report.currency_check["message"] + " Ratios are unit-free and unaffected.")
    if report.dataset.excluded_periods:
        excluded = ", ".join(p.isoformat() for p in report.dataset.excluded_periods)
        notes.append(f"Excluded sparse period(s) {excluded}: the provider returned almost no "
                     "values (core metrics missing). Raw data is still stored.")
    if len(report.dataset.periods) < 2:
        notes.append("Only one fiscal year available: trends and year-over-year red-flag rules "
                     "could not be evaluated; average balances use the ending-balance fallback.")
    if report.sector and "financial" in report.sector.lower():
        notes.append("Financial-sector company: banks/insurers do not report working-capital "
                     "items, and asset turnover and operating cash flow (which includes deposit "
                     "and loan flows) do not measure their health. Categories configured as not "
                     "applicable are not scored; bank-specific metrics (e.g. NIM, NPA, capital "
                     "adequacy) are outside the Semester 1 scope.")
    fallback = [o for o in report.ratios if o.method == "ENDING_BALANCE_FALLBACK"]
    if fallback:
        years = sorted({o.period.isoformat() for o in fallback})
        notes.append(f"Ending-balance fallback used (no prior-year balance) for: {years}.")
    if latest:
        missing = [RATIO_DEFINITIONS[o.code].name for o in report.ratios
                   if o.period == latest and not o.ok]
        if missing:
            notes.append(f"Unavailable ratios in latest year ({latest}): {', '.join(missing)}.")
    if report.unavailable_trends:
        notes.append(f"Trend metrics never reported by the provider: "
                     f"{', '.join(report.unavailable_trends)}.")
    return notes


def _summary(report: AnalysisReport) -> dict[str, Any]:
    latest = report.latest_period
    overall = report.overall()
    latest_flags = [f for f in report.red_flags.flags if f.period == latest]
    return to_jsonable({
        "latest_fiscal_period": latest,
        "periods": report.dataset.periods,
        "overall_score": overall.score if overall else None,
        "overall_interpretation": overall.interpretation if overall else None,
        "overall_coverage": overall.coverage if overall else None,
        "category_scores": {c.category: c.score for c in report.latest_scores()
                            if c.category != OVERALL},
        "red_flags_latest_period": len(latest_flags),
        "red_flags_total": len(report.red_flags.flags),
        "red_flag_severity_counts": report.red_flags.counts(),
        "rules_not_evaluated": len(report.red_flags.not_evaluated()),
        "ratios_calculated": sum(1 for o in report.ratios if o.ok),
        "ratios_total": len(report.ratios),
        "notes": report.notes,
        "pros": report.pros,
        "cons": report.cons,
        "grades": report.grades,
        "market": report.dataset.market,
    })


def _persist(ctx: AppContext, report: AnalysisReport, company_id: int) -> None:
    with ctx.db.session_scope() as session:
        run = session.get(AnalysisRun, report.analysis_run_id)
        for o in report.ratios:
            definition = RATIO_DEFINITIONS[o.code]
            run.ratios.append(RatioResult(
                company_id=company_id, fiscal_period_end=o.period, category=definition.category,
                ratio_code=o.code, ratio_name=definition.name, value=o.value,
                unit=definition.unit, status=o.status, reason_code=o.reason_code,
                method=o.method, formula=definition.formula,
                input_values=to_jsonable(o.inputs), explanation=o.explanation))
        for t in report.trends:
            run.trends.append(TrendResult(
                company_id=company_id, metric_code=t.metric_code,
                comparison_type=t.comparison_type, current_period=t.current_period,
                previous_period=t.previous_period, current_value=t.current_value,
                previous_value=t.previous_value, absolute_change=t.absolute_change,
                percentage_change=t.percentage_change, direction=t.direction, status=t.status,
                explanation=t.explanation))
        for f in report.red_flags.flags:
            run.red_flags.append(RedFlagResult(
                company_id=company_id, fiscal_period_end=f.period,
                previous_period_end=f.previous_period, rule_code=f.rule_code,
                rule_name=f.rule_name, severity=f.severity, title=f.title,
                triggered_condition=f.triggered_condition, description=f.description,
                evidence=to_jsonable(f.evidence)))
        for period, categories in report.scores.items():
            for c in categories:
                run.health_scores.append(HealthScore(
                    company_id=company_id, fiscal_period_end=period, category=c.category,
                    score=c.score, status=c.status, data_coverage=c.coverage,
                    interpretation=c.interpretation, methodology_version=report.scoring_version,
                    explanation=c.explanation,
                    components=to_jsonable([vars(m) for m in c.components]) or None))
        run.status = report.status
        run.completed_at = utcnow()
        run.summary = _summary(report)


def _mark_failed(ctx: AppContext, run_id: int, exc: Exception) -> None:
    try:
        with ctx.db.session_scope() as session:
            run = session.get(AnalysisRun, run_id)
            if run is not None:
                run.status = RUN_FAILED
                run.completed_at = utcnow()
                run.error_details = f"{type(exc).__name__}: {exc}"[:4000]
    except AnalyzerError:
        logger.exception("Could not mark analysis run %s as failed", run_id)


# ---------------------------------------------------------------------- outputs
def export_analysis(ctx: AppContext, report: AnalysisReport) -> dict[str, str]:
    folder = safe_child_path(ctx.settings.exports_dir, report.ticker)
    folder.mkdir(parents=True, exist_ok=True)
    base = f"{report.ticker}_analysis{report.analysis_run_id:05d}"
    outputs: dict[str, Path] = {}
    wide, _ = tables.ratio_table(report.ratios)
    outputs["ratio_table"] = folder / f"{base}_ratio_table.csv"
    wide.to_csv(outputs["ratio_table"], index=False)
    outputs["ratios"] = folder / f"{base}_ratios_detail.csv"
    tables.ratio_long_frame(report.ratios).to_csv(outputs["ratios"], index=False)
    outputs["trends"] = folder / f"{base}_trends.csv"
    tables.trend_frame(report.trends, report.currency).to_csv(outputs["trends"], index=False)
    outputs["red_flags"] = folder / f"{base}_red_flags.csv"
    tables.red_flag_frame(report.red_flags.flags).to_csv(outputs["red_flags"], index=False)
    outputs["health_scores"] = folder / f"{base}_health_scores.csv"
    tables.score_long_frame(report.scores).to_csv(outputs["health_scores"], index=False)
    outputs["markdown_report"] = write_markdown_report(ctx, report)
    return {k: v.as_posix() for k, v in outputs.items()}


def draw_analysis_charts(ctx: AppContext, report: AnalysisReport) -> dict[str, str]:
    folder = safe_child_path(ctx.settings.reports_dir, report.ticker) / "charts"
    base = f"analysis{report.analysis_run_id:05d}"
    produced: dict[str, Path | None] = {}
    try:
        produced["financials"] = charts.financial_trend_chart(
            report.dataset, folder / f"{base}_financials.png")
        produced["profitability"] = charts.ratio_trend_chart(
            report.ratios, ["net_profit_margin", "return_on_assets", "return_on_equity"],
            f"{report.ticker}: profitability ratios (%)", folder / f"{base}_profitability.png")
        produced["liquidity_solvency"] = charts.ratio_trend_chart(
            report.ratios, ["current_ratio", "quick_ratio", "debt_to_equity", "debt_ratio"],
            f"{report.ticker}: liquidity & solvency ratios (x)",
            folder / f"{base}_liquidity_solvency.png")
        produced["health_score"] = charts.health_score_chart(
            report.ticker, report.latest_scores(), folder / f"{base}_health_score.png")
    except (OSError, ValueError, RuntimeError) as exc:
        logger.warning("Chart generation failed: %s", exc, extra={"ticker": report.ticker})
    return {k: v.as_posix() for k, v in produced.items() if v is not None}


def write_markdown_report(ctx: AppContext, report: AnalysisReport) -> Path:
    folder = safe_child_path(ctx.settings.reports_dir, report.ticker)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{report.ticker}_analysis{report.analysis_run_id:05d}.md"
    wide, notes = tables.ratio_table(report.ratios)
    overall = report.overall()
    lines = [
        f"# Financial analysis: {report.ticker}",
        "",
        f"* Company: {report.company_name or 'unknown'}",
        f"* Reporting currency: {report.currency or 'unknown'}",
        f"* Fiscal years analysed: {', '.join(p.isoformat() for p in report.dataset.periods)}",
        f"* Analysis run #{report.analysis_run_id} (fetch run #{report.fetch_run_id}), "
        f"status {report.status}",
        f"* Methodology {ANALYSIS_METHODOLOGY_VERSION}; scoring {report.scoring_version}; "
        f"red-flag rules {report.red_flag_version}",
        "",
        "## Financial health score (latest year)",
        "",
        f"**Overall: {tables.fmt_score(overall.score if overall else None)} / 100"
        f" ({overall.interpretation if overall and overall.interpretation else 'N/A'}), data "
        f"coverage {overall.coverage:.0%}**" if overall else "Overall: N/A",
        "",
        tables.to_markdown(tables.score_frame(report.scores)),
        "",
        "## Pros & cons (automatic, rule-based)",
        "",
        *[f"* PRO: {p}" for p in report.pros],
        *[f"* CON: {c}" for c in report.cons],
        "",
        "## Ratios",
        "",
        tables.to_markdown(wide),
        "",
        tables.legend(),
        "",
        *[f"* {n}" for n in notes],
        "",
        "## Trends",
        "",
        tables.to_markdown(tables.trend_summary_frame(report.trends, report.currency)),
        "",
        "## Red flags",
        "",
        tables.to_markdown(tables.red_flag_frame(report.red_flags.flags)),
        "",
        "## Notes",
        "",
        *[f"* {n}" for n in report.notes],
        "",
        "_Educational analysis of historical statements. Not a credit rating or investment "
        "advice. Source data: provider-reported figures, which may differ from audited "
        "filings._",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
