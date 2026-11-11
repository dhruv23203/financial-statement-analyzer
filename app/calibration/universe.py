"""Reference universe: load the ticker list and make sure each company is ingested and cleaned.

Only ingestion and cleaning are needed for calibration - ratios are recomputed in
memory from the standardised data, so no analysis runs or charts are produced.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from app.cleaning.cleaner import clean_company
from app.database import repositories as repo
from app.ingestion.service import ingest_company
from app.services.context import AppContext
from app.utils.exceptions import AnalyzerError, ConfigError
from app.utils.helpers import normalize_ticker
from config.settings import CONFIG_DIR

UNIVERSE_FILE = CONFIG_DIR / "reference_universe.json"


def load_universe(path: Path = UNIVERSE_FILE) -> list[str]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"Cannot read reference universe {path}: {exc}") from exc
    groups = data.get("tickers")
    if not isinstance(groups, dict):
        raise ConfigError(f"{path}: 'tickers' must map group names to ticker lists")
    tickers: list[str] = []
    for group in groups.values():
        for raw in group:
            ticker = normalize_ticker(raw)
            if ticker not in tickers:
                tickers.append(ticker)
    return tickers


@dataclass
class PreparationReport:
    ready: list[str] = field(default_factory=list)
    fetched: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)


def prepare_universe(ctx: AppContext, tickers: list[str], refresh: bool = False,
                     progress: Callable[[str], None] = lambda _m: None,
                     pause_seconds: float = 0.0) -> PreparationReport:
    """Ingest + clean every ticker that has no cleaned annual data yet (or all if refresh).

    Resumable: companies already cleaned are skipped, so an interrupted run can simply be
    started again. ``pause_seconds`` spaces out downloads to respect provider rate limits.
    """
    report = PreparationReport()
    for position, ticker in enumerate(tickers, start=1):
        if pause_seconds and report.fetched:
            ctx.sleep(pause_seconds)
        if not refresh:
            with ctx.db.session_scope() as session:
                company = repo.get_company(session, ticker)
                cleaned = company is not None and \
                    repo.latest_cleaned_fetch_run(session, company.id) is not None
            if cleaned:
                report.ready.append(ticker)
                continue
        progress(f"[{position}/{len(tickers)}] fetching {ticker}")
        try:
            result = ingest_company(ctx, ticker)
            if not result.succeeded:
                report.failed[ticker] = result.error or result.status
                continue
            clean_company(ctx, ticker, result.fetch_run_id)
        except AnalyzerError as exc:
            report.failed[ticker] = str(exc)
            continue
        report.fetched.append(ticker)
        report.ready.append(ticker)
    return report
