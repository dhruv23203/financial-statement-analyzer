"""Stock screener and automatic peer finder over stored analyses (Screener.in-style).

Queries are conditions joined by AND, for example::

    roe > 15 and debt_to_equity < 0.5 and sector == "Technology"
    piotroski_f_score >= 8 and altman_z_score > 2.6
    score > 60 and currency == INR

They are parsed by a small whitelist parser (field names are checked against the
available columns, values must be numbers or quoted text) - user input is never
evaluated as code. Only the latest successful analysis of each company is used.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd
from sqlalchemy import select

from app.analysis.health_score import OVERALL
from app.analysis.ratios import RATIO_DEFINITIONS
from app.database import repositories as repo
from app.database.models import (RUN_SUCCESS, HealthScore, MarketSnapshot, RatioResult,
                                 StandardizedFinancialFact)
from app.services.context import AppContext
from app.utils.exceptions import AnalyzerError, DataNotFoundError
from app.utils.helpers import normalize_ticker

SHORT_NAMES = {
    "roe": "return_on_equity", "roa": "return_on_assets", "roce": "return_on_capital_employed",
    "npm": "net_profit_margin", "de": "debt_to_equity", "pe": "price_to_earnings",
    "pb": "price_to_book", "piotroski": "piotroski_f_score", "altman": "altman_z_score",
    "beneish": "beneish_m_score", "cr": "current_ratio", "qr": "quick_ratio",
}
TEXT_FIELDS = ("ticker", "name", "sector", "industry", "currency")
_CONDITION = re.compile(
    r"^\s*([a-z_]+)\s*(>=|<=|==|!=|>|<)\s*(\"[^\"]*\"|'[^']*'|-?\d+(?:\.\d+)?|[A-Za-z][\w .&-]*)\s*$")


class ScreenError(AnalyzerError):
    """Invalid screener query."""


@dataclass
class Condition:
    field: str
    operator: str
    value: float | str


def screen_frame(ctx: AppContext) -> pd.DataFrame:
    """One row per company: text fields, latest-year scores, ratios and revenue."""
    rows: dict[int, dict[str, Any]] = {}
    with ctx.db.session_scope() as session:
        latest_runs = {}
        for company in repo.list_companies(session):
            run = repo.latest_analysis_run(session, company.id)
            if run is not None and run.latest_fiscal_period is not None:
                latest_runs[run.id] = (company, run)
                rows[run.id] = {"ticker": company.ticker, "name": company.name,
                                "sector": company.sector, "industry": company.industry,
                                "currency": run.currency or company.currency,
                                "fiscal_year_end": run.latest_fiscal_period.isoformat()}
        if not latest_runs:
            return pd.DataFrame()
        run_ids = list(latest_runs)
        for chunk in (run_ids[i:i + 500] for i in range(0, len(run_ids), 500)):
            for r in session.scalars(select(RatioResult).where(
                    RatioResult.analysis_run_id.in_(chunk), RatioResult.value.is_not(None))):
                if r.fiscal_period_end == latest_runs[r.analysis_run_id][1].latest_fiscal_period:
                    rows[r.analysis_run_id][r.ratio_code] = float(r.value)
            for s in session.scalars(select(HealthScore).where(
                    HealthScore.analysis_run_id.in_(chunk), HealthScore.score.is_not(None))):
                if s.fiscal_period_end == latest_runs[s.analysis_run_id][1].latest_fiscal_period:
                    key = "score" if s.category == OVERALL else f"{s.category.lower()}_score"
                    rows[s.analysis_run_id][key] = float(s.score)
            fetch_ids = {latest_runs[i][1].fetch_run_id: i for i in chunk}
            for f in session.scalars(select(StandardizedFinancialFact).where(
                    StandardizedFinancialFact.fetch_run_id.in_(list(fetch_ids)),
                    StandardizedFinancialFact.metric_code == "total_revenue")):
                run_id = fetch_ids[f.fetch_run_id]
                if f.fiscal_period_end == latest_runs[run_id][1].latest_fiscal_period and \
                        f.value is not None:
                    rows[run_id]["revenue"] = float(f.value)
        # latest successful market snapshot per company (display / sizing only)
        run_by_company = {company.id: run_id for run_id, (company, _) in latest_runs.items()}
        for snap in session.scalars(select(MarketSnapshot).where(
                MarketSnapshot.status == RUN_SUCCESS, MarketSnapshot.market_cap.is_not(None))
                .order_by(MarketSnapshot.fetched_at, MarketSnapshot.id)):
            run_id = run_by_company.get(snap.company_id)
            if run_id is not None:
                rows[run_id]["market_cap"] = float(snap.market_cap)
    return pd.DataFrame(list(rows.values()))


def available_fields(frame: pd.DataFrame) -> list[str]:
    return sorted(frame.columns.tolist()) if not frame.empty else []


def parse_query(query: str, columns: list[str]) -> list[Condition]:
    conditions = []
    for part in re.split(r"\s+and\s+", query.strip(), flags=re.IGNORECASE):
        if not part.strip():
            continue
        match = _CONDITION.match(part)
        if not match:
            raise ScreenError(f"Cannot understand condition {part!r}. Use: field > number, "
                              "e.g. roe > 15, or sector == \"Technology\".")
        name, operator, raw = match.groups()
        name = SHORT_NAMES.get(name, name)
        if name not in columns:
            raise ScreenError(f"Unknown field {name!r}. Run `python main.py screen --fields`.")
        if name in TEXT_FIELDS:
            if operator not in ("==", "!="):
                raise ScreenError(f"Text field {name!r} supports only == and !=.")
            value: float | str = raw.strip("\"'").strip()
        else:
            try:
                value = float(raw)
            except ValueError as exc:
                raise ScreenError(f"{name!r} needs a number, got {raw!r}.") from exc
            if not math.isfinite(value):
                raise ScreenError("Numbers must be finite.")
        conditions.append(Condition(name, operator, value))
    if not conditions:
        raise ScreenError("Empty query.")
    return conditions


def latest_analysis_id(ctx: AppContext) -> int:
    """Cheap change marker: the newest analysis run id (used as a cache key)."""
    from sqlalchemy import func

    from app.database.models import AnalysisRun

    with ctx.db.session_scope() as session:
        return session.scalar(select(func.max(AnalysisRun.id))) or 0


def run_screen(ctx: AppContext, query: str, sort_by: str = "score", ascending: bool = False,
               limit: int = 25, frame: pd.DataFrame | None = None) -> tuple[pd.DataFrame, int]:
    """Filter companies; returns (matching rows, total companies screened).

    ``frame`` lets callers pass a cached :func:`screen_frame` result.
    """
    frame = screen_frame(ctx) if frame is None else frame
    if frame.empty:
        raise DataNotFoundError("No analysed companies yet. Run: python main.py analyze-universe")
    conditions = parse_query(query, available_fields(frame))
    mask = pd.Series(True, index=frame.index)
    for c in conditions:
        column = frame[c.field]
        if isinstance(c.value, str):
            hit = column.astype(str).str.lower() == c.value.lower()
            mask &= hit if c.operator == "==" else ~hit
        else:
            # Missing values never pass a numeric condition (no imputation).
            mask &= column.notna() & {
                ">": column > c.value, "<": column < c.value, ">=": column >= c.value,
                "<=": column <= c.value, "==": column == c.value, "!=": column != c.value,
            }[c.operator]
    result = frame[mask]
    sort_by = SHORT_NAMES.get(sort_by, sort_by)
    if sort_by in result.columns:
        result = result.sort_values(sort_by, ascending=ascending, na_position="last")
    shown = ["ticker", "name", "sector", "currency", "score"] + \
        [c.field for c in conditions if c.field not in ("ticker", "name", "sector", "currency",
                                                        "score")]
    shown = list(dict.fromkeys(c for c in shown if c in result.columns))
    return result[shown].head(limit).reset_index(drop=True), len(frame)


def find_peers(ctx: AppContext, raw_ticker: str, count: int = 4,
               frame: pd.DataFrame | None = None) -> list[str]:
    """Closest analysed companies: same industry, else same sector; same currency first;
    ordered by similarity of revenue size (absolute difference of log revenue)."""
    ticker = normalize_ticker(raw_ticker)
    frame = screen_frame(ctx) if frame is None else frame
    if frame.empty or ticker not in set(frame["ticker"]):
        raise DataNotFoundError(f"{ticker} has no stored analysis. Run: python main.py run {ticker}")
    me = frame[frame["ticker"] == ticker].iloc[0]
    others = frame[frame["ticker"] != ticker]
    def has(value: object) -> bool:  # NaN-safe "is a non-empty text value"
        return isinstance(value, str) and bool(value)

    same_industry = others[others["industry"] == me["industry"]] if has(me["industry"]) \
        else others[:0]
    if len(same_industry) >= count:
        pool = same_industry
    else:
        pool = others[others["sector"] == me["sector"]] if has(me["sector"]) else others[:0]
    if pool.empty:
        return []
    pool = pool.assign(same_currency=(pool["currency"] == me["currency"]).astype(int))
    if "revenue" in pool and pd.notna(me.get("revenue")) and me["revenue"] > 0:
        size = pool["revenue"].where(pool["revenue"] > 0)
        pool = pool.assign(size_gap=(size.map(math.log) - math.log(me["revenue"])).abs())
    else:
        pool = pool.assign(size_gap=0.0)
    pool = pool.sort_values(["same_currency", "size_gap"], ascending=[False, True],
                            na_position="last")
    return pool["ticker"].head(count).tolist()


def field_descriptions() -> dict[str, str]:
    """Human-readable names for the screener fields."""
    descriptions = {code: d.name for code, d in RATIO_DEFINITIONS.items()}
    descriptions.update({"score": "Overall financial health score",
                         "revenue": "Latest annual revenue (own currency)",
                         "market_cap": "Market capitalisation (trading currency, latest "
                                       "snapshot)",
                         "fiscal_year_end": "Latest fiscal year end"})
    for category in ("liquidity", "solvency", "profitability", "efficiency", "cash_flow"):
        descriptions[f"{category}_score"] = f"{category.replace('_', ' ').title()} category score"
    for short, full in SHORT_NAMES.items():
        descriptions[short] = f"alias for {full}"
    return descriptions

