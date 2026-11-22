"""Data-consistency validation of the database (``python main.py db-validate`` and the
Database tab).

``Database.health_check`` verifies the *file* (SQLite integrity, foreign keys, schema
revision). This module verifies the *content*: duplicates, interrupted runs, impossible
values, stale data and currency consistency. Each check reports a status (OK / WARN /
FAIL), a count and up to five example tickers - it never changes data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

from sqlalchemy import func, select, text

from app.database import repositories as repo
from app.database.models import (RUN_RUNNING, AnalysisRun, Company, FetchRun,
                                 StandardizedFinancialFact)
from app.services.context import AppContext

OK, WARN, FAIL = "OK", "WARN", "FAIL"
STALE_PRICE_DAYS = 30
STALE_STATEMENT_DAYS = 730
STUCK_RUN = timedelta(hours=1)


@dataclass
class Check:
    name: str
    status: str
    count: int
    detail: str
    examples: list[str] = field(default_factory=list)


def _status(count: int, fail: bool = False) -> str:
    return OK if count == 0 else (FAIL if fail else WARN)


def validate_database(ctx: AppContext, now: datetime | None = None) -> list[Check]:
    now = now or datetime.now(timezone.utc)
    today = now.date()
    checks: list[Check] = []
    with ctx.db.session_scope() as session:
        tickers = {c.id: c.ticker for c in repo.list_companies(session)}
        runs = repo.latest_analysis_runs(session)
        snaps = repo.latest_market_snapshots(session)

        missing = [t for cid, t in tickers.items() if cid not in runs]
        checks.append(Check("Companies without a usable analysis", _status(len(missing)),
                            len(missing), "Ingested but never analysed successfully (run the "
                            "pipeline again).", missing[:5]))

        dup = session.execute(text(
            "SELECT company_id, COUNT(*) FROM standardized_financial_facts "
            "GROUP BY company_id, fetch_run_id, statement_type, metric_code, "
            "fiscal_period_end, period_type HAVING COUNT(*) > 1")).all()
        checks.append(Check("Duplicate standardised facts", _status(len(dup), fail=True),
                            len(dup), "Same company, run, statement, metric and period stored "
                            "more than once.", [tickers.get(r[0], "?") for r in dup[:5]]))

        stuck = session.scalars(select(FetchRun).where(
            FetchRun.status == RUN_RUNNING,
            FetchRun.started_at < (now - STUCK_RUN).replace(tzinfo=None))).all()
        stuck_a = session.scalars(select(AnalysisRun).where(
            AnalysisRun.status == RUN_RUNNING,
            AnalysisRun.executed_at < (now - STUCK_RUN).replace(tzinfo=None))).all()
        n = len(stuck) + len(stuck_a)
        checks.append(Check("Interrupted runs", _status(n), n,
                            "Fetch or analysis runs still 'RUNNING' after an hour (the process "
                            "stopped); they are ignored by every reader.",
                            [tickers.get(r.company_id, "?") for r in (stuck + stuck_a)[:5]]))

        bad_snaps = [s for s in snaps.values() if (s.price is not None and s.price <= 0)
                     or (s.fetched_at and s.fetched_at.date() > today)]
        checks.append(Check("Impossible market snapshots", _status(len(bad_snaps), fail=True),
                            len(bad_snaps), "Price not positive or dated in the future.",
                            [tickers.get(s.company_id, "?") for s in bad_snaps[:5]]))

        stale = [s for s in snaps.values()
                 if (today - (repo_date(s) or today)).days > STALE_PRICE_DAYS]
        checks.append(Check(f"Prices older than {STALE_PRICE_DAYS} days", _status(len(stale)),
                            len(stale), "Market multiples and price-vs-value use these dated "
                            "prices; refresh with `python main.py market ...`.",
                            [tickers.get(s.company_id, "?") for s in stale[:5]]))
        no_price = [t for cid, t in tickers.items() if cid in runs and cid not in snaps]
        checks.append(Check("Analysed companies without a price", _status(len(no_price)),
                            len(no_price), "No successful market snapshot: multiples and DCF "
                            "comparisons are N/A for them.", no_price[:5]))

        old = [tickers.get(cid, "?") for cid, r in runs.items() if r.latest_fiscal_period and
               (today - r.latest_fiscal_period).days > STALE_STATEMENT_DAYS]
        checks.append(Check("Statements older than two years", _status(len(old)), len(old),
                            "Latest fiscal year ended more than 730 days ago.", old[:5]))

        mismatch = session.execute(
            select(Company.ticker, func.count()).join(
                StandardizedFinancialFact, StandardizedFinancialFact.company_id == Company.id)
            .where(StandardizedFinancialFact.currency.is_not(None), Company.currency.is_not(None),
                   StandardizedFinancialFact.currency != Company.currency)
            .group_by(Company.ticker)).all()
        checks.append(Check("Fact currency differs from company currency",
                            _status(len(mismatch)), len(mismatch),
                            "Facts stored under a different currency label than the company "
                            "(e.g. after a currency-label override): re-clean to align.",
                            [r[0] for r in mismatch[:5]]))

    revision, head = ctx.db.current_revision(), ctx.db.head_revision()
    checks.append(Check("Schema revision", OK if revision == head else FAIL,
                        0 if revision == head else 1,
                        f"database at {revision}, code expects {head}"))
    return checks


def repo_date(snapshot) -> date | None:
    value = repo.snapshot_price_date(snapshot)
    return date.fromisoformat(value) if value else None


def prune_superseded_analyses(ctx: AppContext, keep: int = 3) -> dict[str, int]:
    """Delete analysis runs older than the newest ``keep`` per company (their ratios,
    trends, flags and scores cascade). Fetch runs, statements and market data are kept,
    so any company can be re-analysed. A verified backup is taken first; nothing runs
    automatically - this is an explicit maintenance command."""
    if keep < 1:
        raise ValueError("keep must be at least 1")
    backup = ctx.db.backup(label="before_prune")
    deleted = 0
    with ctx.db.session_scope() as session:
        ids_by_company: dict[int, list[int]] = {}
        for company_id, run_id in session.execute(
                select(AnalysisRun.company_id, AnalysisRun.id)
                .order_by(AnalysisRun.company_id, AnalysisRun.executed_at.desc(),
                          AnalysisRun.id.desc())):
            ids_by_company.setdefault(company_id, []).append(run_id)
        doomed = [i for ids in ids_by_company.values() for i in ids[keep:]]
        for chunk in (doomed[i:i + 500] for i in range(0, len(doomed), 500)):
            for run in session.scalars(select(AnalysisRun).where(AnalysisRun.id.in_(chunk))):
                session.delete(run)
                deleted += 1
    with ctx.db.engine.connect() as conn:
        conn.exec_driver_sql("VACUUM")
    return {"deleted_analysis_runs": deleted, "backup": str(backup)}
