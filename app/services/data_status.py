"""Data provenance, freshness and completeness for one company (shown on the Company tab).

Every number on screen should answer *where did this come from and how old is it?*

* **Source** - the provider recorded on the fetch run (Yahoo Finance via the unofficial
  yfinance library, the synthetic offline sample, or a user-restored archive).
* **Reporting date** - the latest fiscal year-end in the statements and when they were
  fetched; **freshness** compares the year-end with today (annual reports appear only
  once a year, so a statement up to 15 months old is still the current one).
* **Market price date** - the date the stored price refers to (quote time when the provider
  gave one, else the fetch date) and its age.
* **Completeness** - share of the core statement lines present in the latest year, the
  lines that are missing, and how many fiscal years are available.

Nothing is estimated: a missing value is listed as missing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

from app.database import repositories as repo
from app.database.models import FREQ_ANNUAL, RUN_SUCCESS
from app.services.context import AppContext
from app.utils.helpers import normalize_ticker

CORE_METRICS = {
    "total_revenue": "Revenue", "operating_income": "Operating income",
    "net_income": "Net income", "total_assets": "Total assets",
    "total_liabilities": "Total liabilities", "stockholders_equity": "Shareholders' equity",
    "current_assets": "Current assets", "current_liabilities": "Current liabilities",
    "total_debt": "Total debt", "cash_and_equivalents": "Cash & equivalents",
    "operating_cash_flow": "Operating cash flow", "capital_expenditure": "Capital expenditure",
}
SOURCE_LABELS = {
    "yfinance": "Yahoo Finance (via the unofficial yfinance library)",
    "sample": "Offline sample data (synthetic, for demonstration)",
    "archive": "Restored archive (user-supplied file, not re-verified with the provider)",
}
STATEMENT_CURRENT_DAYS = 456      # ~15 months: the next annual report is not due yet
STATEMENT_STALE_DAYS = 730
PRICE_FRESH_DAYS = 4              # covers weekends and a holiday
PRICE_STALE_DAYS = 10


@dataclass
class DataStatus:
    ticker: str
    source: str
    source_label: str
    currency: str | None
    statements_fetched_at: str | None
    latest_period: date | None
    fiscal_years: int
    statement_age_days: int | None
    statement_freshness: str
    price: float | None = None
    price_currency: str | None = None
    price_date: str | None = None
    price_age_days: int | None = None
    price_freshness: str = "No price"
    completeness: float = 0.0
    missing_core: list[str] = field(default_factory=list)
    currency_check: str | None = None
    notes: list[str] = field(default_factory=list)


def statement_freshness(age_days: int | None) -> str:
    if age_days is None:
        return "Unknown"
    if age_days <= STATEMENT_CURRENT_DAYS:
        return "Current"
    return "Ageing" if age_days <= STATEMENT_STALE_DAYS else "Stale"


def price_freshness(age_days: int | None) -> str:
    if age_days is None:
        return "No price"
    if age_days <= PRICE_FRESH_DAYS:
        return "Fresh"
    return "Recent" if age_days <= PRICE_STALE_DAYS else "Stale"


def data_status(ctx: AppContext, raw_ticker: str, today: date | None = None) -> DataStatus | None:
    ticker = normalize_ticker(raw_ticker)
    today = today or datetime.now(timezone.utc).date()
    with ctx.db.session_scope() as session:
        company = repo.get_company(session, ticker)
        if company is None:
            return None
        run = repo.latest_cleaned_fetch_run(session, company.id, FREQ_ANNUAL)
        if run is None:
            return None
        facts = repo.standardized_facts_for_run(session, run.id, FREQ_ANNUAL, mapped_only=True)
        periods = sorted({f.fiscal_period_end for f in facts if f.value is not None})
        latest = periods[-1] if periods else None
        present = {f.metric_code for f in facts
                   if f.fiscal_period_end == latest and f.value is not None}
        analysis = repo.latest_analysis_run(session, company.id)
        stored = ((analysis.summary or {}).get("data_completeness") if analysis else None)
        if stored:                       # measured on the full statements at analysis time
            present = set(stored.get("present") or [])
            fy = int(stored.get("fiscal_years") or len(periods))
        else:
            fy = len(periods)
        snap = repo.latest_market_snapshot(session, company.id)
        check = ((run.fetch_metadata or {}).get("currency_check") or {}).get("status")
        status = DataStatus(
            ticker=ticker, source=run.provider,
            source_label=SOURCE_LABELS.get(run.provider, run.provider),
            currency=company.currency,
            statements_fetched_at=(run.completed_at or run.started_at).date().isoformat()
            if (run.completed_at or run.started_at) else None,
            latest_period=latest, fiscal_years=fy,
            statement_age_days=(today - latest).days if latest else None,
            statement_freshness="Unknown", currency_check=check)
        status.statement_freshness = statement_freshness(status.statement_age_days)
        if snap is not None and snap.status == RUN_SUCCESS:
            status.price = float(snap.price) if snap.price is not None else None
            status.price_currency = snap.currency
            status.price_date = repo.snapshot_price_date(snap)
            if status.price_date:
                status.price_age_days = (today - date.fromisoformat(status.price_date)).days
        status.price_freshness = price_freshness(status.price_age_days)
    missing = [label for code, label in CORE_METRICS.items() if code not in present]
    status.missing_core = missing
    status.completeness = 1 - len(missing) / len(CORE_METRICS)
    if status.fiscal_years < 3:
        status.notes.append(f"Only {status.fiscal_years} fiscal year(s) available: trends, "
                            "growth and averages are limited.")
    if status.price_currency and status.currency and status.price_currency != status.currency:
        status.notes.append(f"Price in {status.price_currency}, statements in "
                            f"{status.currency}: market multiples are not computed (no FX).")
    if check == "FAILED":
        status.notes.append("The provider's currency label failed the plausibility check.")
    return status


def as_rows(s: DataStatus) -> list[dict[str, Any]]:
    """Compact table for display and export."""
    return [
        {"Item": "Source", "Value": s.source_label},
        {"Item": "Latest fiscal year-end", "Value": f"{s.latest_period} "
                                                   f"({s.statement_freshness.lower()})"},
        {"Item": "Statements fetched", "Value": s.statements_fetched_at or "unknown"},
        {"Item": "Fiscal years available", "Value": str(s.fiscal_years)},
        {"Item": "Market price date", "Value": (f"{s.price_date} ({s.price_age_days} days, "
                                                f"{s.price_freshness.lower()})")
         if s.price_date else "no stored price"},
        {"Item": "Core-line completeness", "Value": f"{s.completeness:.0%}" + (
            f" - missing: {', '.join(s.missing_core)}" if s.missing_core else "")},
        {"Item": "Reporting currency", "Value": f"{s.currency} (label check: "
                                                f"{s.currency_check or 'n/a'})"},
    ]


def universe_price_freshness(ctx: AppContext, today: date | None = None) -> dict[str, Any]:
    """Across all companies: the newest stored price date, how many companies have a
    stored price, and how many of those prices are older than PRICE_STALE_DAYS."""
    from sqlalchemy import select

    from app.database.models import MarketSnapshot

    today = today or datetime.now(timezone.utc).date()
    latest: dict[int, Any] = {}
    with ctx.db.session_scope() as session:
        for snap in session.scalars(select(MarketSnapshot).where(
                MarketSnapshot.status == RUN_SUCCESS, MarketSnapshot.price.is_not(None))
                .order_by(MarketSnapshot.fetched_at, MarketSnapshot.id)):
            latest[snap.company_id] = repo.snapshot_price_date(snap)
    dates = [d for d in latest.values() if d]
    stale = sum(1 for d in dates if (today - date.fromisoformat(d)).days > PRICE_STALE_DAYS)
    return {"priced": len(dates), "latest_price_date": max(dates) if dates else None,
            "stale_prices": stale if dates else None, "stale_days": PRICE_STALE_DAYS}
