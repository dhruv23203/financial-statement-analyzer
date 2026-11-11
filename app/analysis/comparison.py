"""Day 80 - multi-company comparison (2 to 5 companies) from stored analysis results.

Safeguards
----------
* Absolute amounts are always shown with their currency and are flagged as not
  directly comparable when currencies differ or are unknown (no conversion is done).
* Ratios are unit-free and remain comparable across currencies, subject to the
  fiscal-period checks below.
* Fiscal periods: the latest analysed year of each company is used unless
  ``fiscal_year`` is given. Different fiscal year-ends (e.g. March vs December) or
  period ends more than 92 days apart produce a MISALIGNED warning.
* Companies without analysis data are excluded with a reason - nothing is invented.
* Different sectors trigger a warning: ratio norms differ by industry.
* Ranks: only calculated (OK) ratio values are ranked; direction follows each
  ratio's definition (e.g. lower debt-to-equity ranks better).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.analysis.health_score import OVERALL
from app.analysis.ratios import RATIO_DEFINITIONS
from app.database import repositories as repo
from app.database.models import (CALC_OK, RUN_FAILED, RUN_PARTIAL, RUN_SUCCESS, AnalysisRun,
                                 ComparisonCompany, ComparisonResult, ComparisonRun, FetchRun,
                                 HealthScore, RatioResult, RedFlagResult,
                                 StandardizedFinancialFact, TrendResult)
from app.reporting import charts
from app.reporting.tables import fy_label
from app.services.context import AppContext
from app.utils.exceptions import ComparisonError
from app.utils.helpers import fmt_amount, normalize_ticker, to_jsonable, utcnow
from config.settings import MAX_COMPARISON_COMPANIES, MIN_COMPARISON_COMPANIES

logger = logging.getLogger(__name__)

AMOUNT_METRICS: dict[str, str] = {
    "total_revenue": "Revenue",
    "net_income": "Net Income",
    "total_assets": "Total Assets",
    "total_debt": "Total Debt",
    "stockholders_equity": "Shareholders' Equity",
    "operating_cash_flow": "Operating Cash Flow",
}
RATIO_METRICS = ("current_ratio", "quick_ratio", "debt_to_equity", "debt_ratio",
                 "return_on_assets", "return_on_equity", "net_profit_margin", "asset_turnover",
                 "piotroski_f_score", "altman_z_score")
TREND_METRICS = {"total_revenue": "Revenue trend", "net_income": "Net income trend"}
MAX_PERIOD_GAP_DAYS = 92


@dataclass
class RatioCell:
    value: Decimal | None
    status: str
    explanation: str | None
    rank: int | None = None


@dataclass
class CompanySnapshot:
    ticker: str
    company_id: int | None = None
    name: str | None = None
    sector: str | None = None
    industry: str | None = None
    currency: str | None = None
    analysis_run_id: int | None = None
    period: date | None = None
    amounts: dict[str, Decimal | None] = field(default_factory=dict)
    ratios: dict[str, RatioCell] = field(default_factory=dict)
    overall_score: Decimal | None = None
    score_coverage: float | None = None
    score_rank: int | None = None
    red_flags_in_period: int | None = None
    red_flag_rank: int | None = None
    trends: dict[str, str] = field(default_factory=dict)
    currency_check: str | None = None
    included: bool = True
    exclusion_reason: str | None = None


@dataclass
class ComparisonReport:
    comparison_run_id: int
    status: str
    requested: list[str]
    snapshots: list[CompanySnapshot]
    currency_compatibility: str
    period_alignment: str
    selected_period: str
    warnings: list[str] = field(default_factory=list)
    exports: dict[str, str] = field(default_factory=dict)
    charts: dict[str, str] = field(default_factory=dict)

    @property
    def included(self) -> list[CompanySnapshot]:
        return [s for s in self.snapshots if s.included]


def validate_tickers(raw_tickers: Sequence[str]) -> tuple[list[str], list[str]]:
    """Normalise, de-duplicate and check the 2-5 company limit."""
    warnings, tickers = [], []
    for raw in raw_tickers:
        ticker = normalize_ticker(raw)
        if ticker in tickers:
            warnings.append(f"Duplicate ticker {ticker} ignored.")
            continue
        tickers.append(ticker)
    if len(tickers) < MIN_COMPARISON_COMPANIES:
        raise ComparisonError(f"A comparison needs at least {MIN_COMPARISON_COMPANIES} "
                              f"different companies; got {len(tickers)}.")
    if len(tickers) > MAX_COMPARISON_COMPANIES:
        raise ComparisonError(f"At most {MAX_COMPARISON_COMPANIES} companies can be compared "
                              f"at once; got {len(tickers)}.")
    return tickers, warnings


def _load_snapshot(session: Session, ticker: str, fiscal_year: int | None) -> CompanySnapshot:
    snap = CompanySnapshot(ticker=ticker)
    company = repo.get_company(session, ticker)
    if company is None:
        snap.included, snap.exclusion_reason = False, "not ingested/analysed"
        return snap
    snap.company_id, snap.name, snap.sector = company.id, company.name, company.sector
    snap.industry, snap.currency = company.industry, company.currency
    run: AnalysisRun | None = repo.latest_analysis_run(session, company.id)
    if run is None:
        snap.included, snap.exclusion_reason = False, "no successful analysis run"
        return snap
    snap.analysis_run_id, snap.currency = run.id, run.currency or company.currency
    fetch_run = session.get(FetchRun, run.fetch_run_id)
    snap.currency_check = ((fetch_run.fetch_metadata or {}).get("currency_check") or {}).get(
        "status") if fetch_run else None
    periods = sorted(set(session.scalars(select(RatioResult.fiscal_period_end).where(
        RatioResult.analysis_run_id == run.id))))
    if fiscal_year is not None:
        periods = [p for p in periods if p.year == fiscal_year]
        if not periods:
            snap.included = False
            snap.exclusion_reason = f"no analysed fiscal year ending in {fiscal_year}"
            return snap
    if not periods:
        snap.included, snap.exclusion_reason = False, "analysis has no periods"
        return snap
    snap.period = periods[-1]

    facts = session.scalars(select(StandardizedFinancialFact).where(
        StandardizedFinancialFact.fetch_run_id == run.fetch_run_id,
        StandardizedFinancialFact.fiscal_period_end == snap.period,
        StandardizedFinancialFact.metric_code.in_(list(AMOUNT_METRICS)))).all()
    values = {f.metric_code: f.value for f in facts}
    snap.amounts = {code: values.get(code) for code in AMOUNT_METRICS}
    ratio_rows = session.scalars(select(RatioResult).where(
        RatioResult.analysis_run_id == run.id, RatioResult.fiscal_period_end == snap.period,
        RatioResult.ratio_code.in_(RATIO_METRICS))).all()
    by_code = {r.ratio_code: r for r in ratio_rows}
    for code in RATIO_METRICS:
        row = by_code.get(code)
        snap.ratios[code] = RatioCell(row.value, row.status, row.explanation) if row else \
            RatioCell(None, "UNAVAILABLE", "not calculated")
    overall = session.scalar(select(HealthScore).where(
        HealthScore.analysis_run_id == run.id, HealthScore.fiscal_period_end == snap.period,
        HealthScore.category == OVERALL))
    if overall is not None:
        snap.overall_score, snap.score_coverage = overall.score, overall.data_coverage
    snap.red_flags_in_period = len(session.scalars(select(RedFlagResult.id).where(
        RedFlagResult.analysis_run_id == run.id,
        RedFlagResult.fiscal_period_end == snap.period)).all())
    trends = session.scalars(select(TrendResult).where(
        TrendResult.analysis_run_id == run.id, TrendResult.comparison_type == "MULTI_YEAR",
        TrendResult.metric_code.in_(list(TREND_METRICS)))).all()
    snap.trends = {t.metric_code: t.direction for t in trends}
    return snap


def _rank(values: dict[str, Decimal | None], higher_is_better: bool) -> dict[str, int]:
    present = [(t, v) for t, v in values.items() if v is not None]
    ordered = sorted(present, key=lambda item: item[1], reverse=higher_is_better)
    ranks: dict[str, int] = {}
    for position, (ticker, value) in enumerate(ordered, start=1):
        previous = ordered[position - 2] if position > 1 else None
        ranks[ticker] = ranks[previous[0]] if previous and previous[1] == value else position
    return ranks


def _assess(included: list[CompanySnapshot], warnings: list[str]) -> tuple[str, str]:
    currencies = {s.currency for s in included}
    suspect = [s.ticker for s in included if s.currency_check == "FAILED"]
    if suspect:
        warnings.append(f"Currency label failed the plausibility check for {', '.join(suspect)} "
                        "(see ingestion output): absolute amounts must not be compared. "
                        "Ratios are unaffected.")
        currencies.add(None)
    if None in currencies:
        currency_status = "UNKNOWN"
        warnings.append("Reporting currency unknown for at least one company: absolute "
                        "amounts are shown but must not be compared directly.")
    elif len(currencies) > 1:
        currency_status = "MIXED"
        warnings.append(f"Different reporting currencies {sorted(currencies)}: absolute "
                        "amounts are NOT comparable (no currency conversion is performed). "
                        "Ratios remain comparable.")
    else:
        currency_status = "COMPATIBLE"

    periods = [s.period for s in included if s.period]
    alignment = "ALIGNED"
    year_ends = {(p.month, p.day) for p in periods}
    if len(year_ends) > 1:
        alignment = "MISALIGNED"
        detail = ", ".join(f"{s.ticker}: {s.period}" for s in included)
        warnings.append(f"Companies have different fiscal year-ends ({detail}); figures "
                        "cover different 12-month windows.")
    if periods and (max(periods) - min(periods)).days > MAX_PERIOD_GAP_DAYS:
        alignment = "MISALIGNED"
        warnings.append(f"Selected periods are {(max(periods) - min(periods)).days} days "
                        "apart: compare with caution (use --fiscal-year to align).")
    sectors = {s.sector for s in included if s.sector}
    if len(sectors) > 1:
        warnings.append(f"Companies are in different sectors {sorted(sectors)}; typical ratio "
                        "levels differ by industry, so rankings are indicative only.")
    return currency_status, alignment


def compare_companies(ctx: AppContext, raw_tickers: Sequence[str],
                      fiscal_year: int | None = None, write_outputs: bool = True) -> \
        ComparisonReport:
    """Compare stored analysis results; persists the run and its results."""
    tickers, warnings = validate_tickers(raw_tickers)
    with ctx.db.session_scope() as session:
        snapshots = [_load_snapshot(session, t, fiscal_year) for t in tickers]
    included = [s for s in snapshots if s.included]
    for snap in snapshots:
        if not snap.included:
            warnings.append(f"{snap.ticker} excluded: {snap.exclusion_reason}.")

    selected = f"FY{fiscal_year}" if fiscal_year else "LATEST"
    configuration = {"tickers": tickers, "fiscal_year": fiscal_year,
                     "ratios": list(RATIO_METRICS), "amounts": list(AMOUNT_METRICS)}
    if len(included) < MIN_COMPARISON_COMPANIES:
        message = (f"Only {len(included)} of {len(tickers)} companies have analysis data; at "
                   f"least {MIN_COMPARISON_COMPANIES} are required. " + " ".join(warnings))
        _persist_failed(ctx, tickers, snapshots, selected, configuration, warnings, message)
        raise ComparisonError(message)

    currency_status, alignment = _assess(included, warnings)
    for code in RATIO_METRICS:
        values = {s.ticker: (s.ratios[code].value if s.ratios[code].status == CALC_OK else None)
                  for s in included}
        for ticker, rank in _rank(values, RATIO_DEFINITIONS[code].higher_is_better).items():
            next(s for s in included if s.ticker == ticker).ratios[code].rank = rank
        missing = [t for t, v in values.items() if v is None]
        if missing:
            warnings.append(f"{RATIO_DEFINITIONS[code].name} unavailable for "
                            f"{', '.join(missing)} (not ranked).")
    for ticker, rank in _rank({s.ticker: s.overall_score for s in included}, True).items():
        next(s for s in included if s.ticker == ticker).score_rank = rank
    flag_counts = {s.ticker: Decimal(s.red_flags_in_period) for s in included
                   if s.red_flags_in_period is not None}
    for ticker, rank in _rank(flag_counts, False).items():
        next(s for s in included if s.ticker == ticker).red_flag_rank = rank

    status = RUN_SUCCESS if len(included) == len(tickers) else RUN_PARTIAL
    report = ComparisonReport(0, status, tickers, snapshots, currency_status, alignment,
                              selected, warnings)
    report.comparison_run_id = _persist(ctx, report, configuration)
    if write_outputs:
        report.exports = export_comparison(ctx, report)
        report.charts = draw_comparison_charts(ctx, report)
    logger.info("Comparison finished", extra={"comparison_run_id": report.comparison_run_id,
                                              "tickers": tickers, "status": status})
    return report


def _participants(snapshots: list[CompanySnapshot]) -> list[ComparisonCompany]:
    return [ComparisonCompany(company_id=s.company_id, ticker=s.ticker,
                              analysis_run_id=s.analysis_run_id, fiscal_period_end=s.period,
                              currency=s.currency, included=s.included,
                              exclusion_reason=s.exclusion_reason) for s in snapshots]


def _persist_failed(ctx: AppContext, tickers: list[str], snapshots: list[CompanySnapshot],
                    selected: str, configuration: dict[str, Any], warnings: list[str],
                    message: str) -> None:
    with ctx.db.session_scope() as session:
        run = ComparisonRun(executed_at=utcnow(), requested_tickers=tickers,
                            selected_fiscal_period=selected, status=RUN_FAILED,
                            currency_compatibility="UNKNOWN", period_alignment="UNKNOWN",
                            configuration=configuration, warnings=warnings,
                            error_details=message[:4000])
        run.participants.extend(_participants(snapshots))
        session.add(run)


def _persist(ctx: AppContext, report: ComparisonReport, configuration: dict[str, Any]) -> int:
    with ctx.db.session_scope() as session:
        run = ComparisonRun(executed_at=utcnow(), requested_tickers=report.requested,
                            selected_fiscal_period=report.selected_period,
                            status=report.status,
                            currency_compatibility=report.currency_compatibility,
                            period_alignment=report.period_alignment,
                            configuration=to_jsonable(configuration),
                            warnings=report.warnings or None)
        run.participants.extend(_participants(report.snapshots))
        amount_note = None if report.currency_compatibility == "COMPATIBLE" else \
            "Not directly comparable across companies (currency mismatch/unknown)."
        for s in report.included:
            def add(code: str, kind: str, value: Decimal | None = None, text: str | None = None,
                    unit: str | None = None, currency: str | None = None,
                    rank: int | None = None, status: str = "OK", note: str | None = None,
                    snap: CompanySnapshot = s) -> None:
                run.results.append(ComparisonResult(
                    company_id=snap.company_id, metric_code=code, metric_kind=kind, value=value,
                    text_value=text, unit=unit, currency=currency,
                    fiscal_period_end=snap.period, rank=rank, status=status, note=note))
            for code, value in s.amounts.items():
                add(code, "AMOUNT", value, unit="amount", currency=s.currency,
                    status="OK" if value is not None else "UNAVAILABLE",
                    note=amount_note if value is not None else "Not reported by provider.")
            for code, cell in s.ratios.items():
                add(code, "RATIO", cell.value, unit=RATIO_DEFINITIONS[code].unit,
                    rank=cell.rank, status=cell.status,
                    note=None if cell.status == CALC_OK else cell.explanation)
            add("overall_health_score", "SCORE", s.overall_score, unit="points",
                rank=s.score_rank, status="OK" if s.overall_score is not None else "UNAVAILABLE")
            add("red_flags_in_period", "COUNT",
                None if s.red_flags_in_period is None else Decimal(s.red_flags_in_period),
                rank=s.red_flag_rank)
            for code in TREND_METRICS:
                direction = s.trends.get(code)
                add(f"{code}_trend", "TREND", text=direction or "UNAVAILABLE",
                    status="OK" if direction else "UNAVAILABLE")
        session.add(run)
        session.flush()
        return run.id


# ---------------------------------------------------------------------- tables & outputs
def amounts_frame(report: ComparisonReport) -> pd.DataFrame:
    rows = []
    for code, label in AMOUNT_METRICS.items():
        row: dict[str, Any] = {"Metric (amount)": label}
        for s in report.included:
            row[s.ticker] = fmt_amount(s.amounts.get(code), s.currency)
        rows.append(row)
    return pd.DataFrame(rows)


def ratios_frame(report: ComparisonReport) -> pd.DataFrame:
    rows = []
    for code in RATIO_METRICS:
        definition = RATIO_DEFINITIONS[code]
        better = "higher" if definition.higher_is_better else "lower"
        row: dict[str, Any] = {"Ratio": f"{definition.name} ({definition.unit}, {better} "
                                        "better)"}
        for s in report.included:
            cell = s.ratios[code]
            if cell.status == CALC_OK:
                row[s.ticker] = f"{float(cell.value):,.2f} (#{cell.rank})"
            else:
                row[s.ticker] = "N/M" if cell.status == "NOT_MEANINGFUL" else "N/A"
        rows.append(row)
    return pd.DataFrame(rows)


def summary_frame(report: ComparisonReport) -> pd.DataFrame:
    rows = []
    for s in report.included:
        rows.append({
            "Ticker": s.ticker, "Company": (s.name or "")[:35], "Sector": s.sector or "",
            "Currency": s.currency or "?", "Period": str(s.period),
            "Health score": "N/A" if s.overall_score is None else
            f"{float(s.overall_score):.1f} (#{s.score_rank})",
            "Coverage": "N/A" if s.score_coverage is None else f"{s.score_coverage:.0%}",
            "Red flags (year)": s.red_flags_in_period,
            "Revenue trend": s.trends.get("total_revenue", "N/A"),
            "Net income trend": s.trends.get("net_income", "N/A"),
        })
    return pd.DataFrame(rows)


def export_comparison(ctx: AppContext, report: ComparisonReport) -> dict[str, str]:
    folder = ctx.settings.exports_dir / "comparisons"
    folder.mkdir(parents=True, exist_ok=True)
    base = f"comparison{report.comparison_run_id:05d}"
    outputs: dict[str, Path] = {
        "summary": folder / f"{base}_summary.csv",
        "ratios": folder / f"{base}_ratios.csv",
        "amounts": folder / f"{base}_amounts.csv",
    }
    summary_frame(report).to_csv(outputs["summary"], index=False)
    ratios_frame(report).to_csv(outputs["ratios"], index=False)
    amounts_frame(report).to_csv(outputs["amounts"], index=False)
    return {k: v.as_posix() for k, v in outputs.items()}


def draw_comparison_charts(ctx: AppContext, report: ComparisonReport) -> dict[str, str]:
    folder = ctx.settings.reports_dir / "comparisons"
    base = f"comparison{report.comparison_run_id:05d}"
    table = {s.ticker: {c: (s.ratios[c].value if s.ratios[c].status == CALC_OK else None)
                        for c in RATIO_METRICS} for s in report.included}
    labels = {c: RATIO_DEFINITIONS[c].name for c in RATIO_METRICS}
    produced: dict[str, Path | None] = {}
    try:
        produced["percent_ratios"] = charts.comparison_ratio_chart(
            table, ["return_on_assets", "return_on_equity", "net_profit_margin"], labels,
            "Profitability ratios (%) - missing values not plotted",
            folder / f"{base}_profitability.png")
        produced["multiple_ratios"] = charts.comparison_ratio_chart(
            table, ["current_ratio", "quick_ratio", "debt_to_equity", "debt_ratio",
                    "asset_turnover"], labels,
            "Liquidity, solvency and efficiency ratios (x)", folder / f"{base}_ratios.png")
        produced["scores"] = charts.comparison_score_chart(
            {s.ticker: s.overall_score for s in report.included},
            {s.ticker: s.period for s in report.included}, folder / f"{base}_scores.png")
    except (OSError, ValueError, RuntimeError) as exc:
        logger.warning("Comparison chart generation failed: %s", exc)
    return {k: v.as_posix() for k, v in produced.items() if v is not None}


def period_labels(report: ComparisonReport) -> str:
    return ", ".join(f"{s.ticker} {fy_label(s.period)}" for s in report.included if s.period)
