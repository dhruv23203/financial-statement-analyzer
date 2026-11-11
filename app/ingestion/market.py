"""Market snapshots (price, market capitalisation) - stored separately from statements.

Fetched best-effort during ingestion and on demand (``python main.py market``). A failed
snapshot never affects statement ingestion; it is recorded with status FAILED.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from app.database import repositories as repo
from app.database.models import RUN_FAILED, RUN_SUCCESS, MarketSnapshot
from app.ingestion.fetcher import call_with_retry
from app.services.context import AppContext
from app.utils.exceptions import AnalyzerError, ProviderError
from app.utils.helpers import normalize_ticker, to_decimal, to_jsonable, utcnow

logger = logging.getLogger(__name__)


def record_market_snapshot(ctx: AppContext, ticker: str, company_id: int) -> str:
    """Fetch and store one snapshot; returns SUCCESS or FAILED (never raises provider errors)."""
    snapshot: dict[str, Any] = {}
    error = None
    try:
        snapshot = call_with_retry(lambda: ctx.provider.fetch_market_snapshot(ticker),
                                   retries=ctx.settings.fetch_max_retries,
                                   backoff_seconds=ctx.settings.fetch_backoff_seconds,
                                   sleep=ctx.sleep, description=f"{ticker} market data") or {}
    except ProviderError as exc:
        error = str(exc)
    market_cap = to_decimal(snapshot.get("market_cap"))
    if error is None and (market_cap is None or market_cap <= 0):
        error = "provider returned no market capitalisation"
    status = RUN_FAILED if error else RUN_SUCCESS
    with ctx.db.session_scope() as session:
        session.add(MarketSnapshot(
            company_id=company_id, fetched_at=utcnow(), provider=ctx.provider.name,
            status=status, price=to_decimal(snapshot.get("price")) if not error else None,
            market_cap=market_cap if not error else None,
            shares_outstanding=to_decimal(snapshot.get("shares_outstanding"))
            if not error else None,
            currency=(snapshot.get("currency") or None) if not error else None,
            provider_values=to_jsonable(snapshot.get("provider_values")) or None,
            error_details=error))
    if error:
        logger.info("Market snapshot unavailable", extra={"ticker": ticker, "error": error})
    return status


def refresh_market_snapshots(ctx: AppContext, raw_tickers: list[str],
                             progress: Callable[[str], None] = lambda _m: None,
                             pause_seconds: float = 0.0) -> dict[str, str]:
    """Fetch snapshots for companies already in the database."""
    results: dict[str, str] = {}
    for position, raw in enumerate(raw_tickers, start=1):
        ticker = normalize_ticker(raw)
        with ctx.db.session_scope() as session:
            company = repo.get_company(session, ticker)
            company_id = company.id if company else None
        if company_id is None:
            results[ticker] = "NOT_IN_DATABASE"
            continue
        if pause_seconds and position > 1:
            ctx.sleep(pause_seconds)
        progress(f"[{position}/{len(raw_tickers)}] market data {ticker}")
        try:
            results[ticker] = record_market_snapshot(ctx, ticker, company_id)
        except AnalyzerError as exc:
            results[ticker] = f"FAILED: {exc}"
    return results
