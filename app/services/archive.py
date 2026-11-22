"""Portable company archives: keep companies you added when the hosted app restarts.

Streamlit Community Cloud gives the app a temporary disk: the preloaded database (shipped
in the repository) comes back after a restart, but companies fetched through the sidebar
are lost. No free persistent storage is available without an external account, so the
data can travel with the user instead:

* :func:`export_archive` writes the **original provider statements** (raw facts, exactly
  as received), the company metadata and the latest dated market snapshot of each company
  into a zip, in the offline-provider layout (``<TICKER>/income_statement.csv`` ...)
  plus a manifest with SHA-256 checksums.
* :func:`import_archive` validates the zip (format version, size, safe paths, checksums)
  and re-runs the normal pipeline offline from it (ingest -> clean -> analyse) with an
  ``archive`` provider, so every validation rule applies again and the data source is
  labelled *Restored archive (user-supplied, not re-verified)*. Nothing is merged
  silently, and existing records are never overwritten: a restore adds a new fetch run.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd

from app.database import repositories as repo
from app.database.models import FREQ_ANNUAL
from app.providers.sample_provider import SampleDataProvider
from app.services.context import AppContext, build_context
from app.utils.exceptions import AnalyzerError
from app.utils.helpers import normalize_ticker
from config.metric_aliases import STATEMENT_TYPES

logger = logging.getLogger(__name__)
FORMAT_VERSION = 1
MAX_ARCHIVE_BYTES = 20 * 1024 * 1024
MAX_FILES = 400


class ArchiveError(AnalyzerError):
    """The archive cannot be read or fails validation."""


class ArchiveProvider(SampleDataProvider):
    """Serves statements from an extracted archive (same layout as the sample data)."""

    name = "archive"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def export_archive(ctx: AppContext, tickers: list[str]) -> tuple[bytes, dict[str, str]]:
    """Zip of the original statements of *tickers*; returns (bytes, per-ticker status)."""
    buffer, status, files = io.BytesIO(), {}, {}
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf, \
            ctx.db.session_scope() as session:
        for raw in tickers:
            ticker = normalize_ticker(raw)
            company = repo.get_company(session, ticker)
            run = repo.latest_fetch_run(session, company.id, FREQ_ANNUAL) if company else None
            facts = repo.raw_facts_for_run(session, run.id) if run else []
            if not facts:
                status[ticker] = ("skipped: no original statements stored (preloaded "
                                  "companies ship with the app and need no archive)")
                continue
            for statement in STATEMENT_TYPES:
                rows = [f for f in facts if f.statement_type == statement]
                if not rows:
                    continue
                frame = pd.DataFrame([{"metric": f.original_metric_name,
                                       "period": f.fiscal_period_end.isoformat(),
                                       "value": None if f.original_value is None
                                       else float(f.original_value)} for f in rows])
                wide = frame.pivot_table(index="metric", columns="period", values="value",
                                         aggfunc="first", dropna=False)
                # keep every metric row, even when all its values are missing
                wide = wide.reindex(sorted(frame["metric"].unique()))
                data = wide.to_csv().encode("utf-8")
                files[f"{ticker}/{statement}.csv"] = data
                zf.writestr(f"{ticker}/{statement}.csv", data)
            snap = repo.latest_market_snapshot(session, company.id)
            meta = {"name": company.name, "exchange": company.exchange,
                    "sector": company.sector, "industry": company.industry,
                    "country": company.country, "currency": company.currency,
                    "trading_currency": (run.fetch_metadata or {}).get("trading_currency"),
                    "original_provider": run.provider,
                    "statements_fetched_at": run.started_at.isoformat(),
                    "market": {"price": float(snap.price) if snap and snap.price else None,
                               "market_cap": float(snap.market_cap)
                               if snap and snap.market_cap else None,
                               "shares_outstanding": float(snap.shares_outstanding)
                               if snap and snap.shares_outstanding else None,
                               "currency": snap.currency if snap else None,
                               "provider_values": (snap.provider_values or {}) if snap else {}}
                    if snap is not None else {}}
            data = json.dumps(meta, indent=2, default=str).encode("utf-8")
            files[f"{ticker}/metadata.json"] = data
            zf.writestr(f"{ticker}/metadata.json", data)
            status[ticker] = f"exported ({len(facts)} statement values)"
        manifest = {"format": "fsa-company-archive", "version": FORMAT_VERSION,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "tickers": sorted(t for t, s in status.items() if s.startswith("exported")),
                    "sha256": {name: _sha(data) for name, data in files.items()}}
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
    logger.info("Archive exported", extra={"tickers": manifest["tickers"]})
    return buffer.getvalue(), status


def validate_archive(data: bytes) -> dict[str, Any]:
    """Manifest of a valid archive; raises :class:`ArchiveError` with the reason."""
    if len(data) > MAX_ARCHIVE_BYTES:
        raise ArchiveError(f"Archive is larger than {MAX_ARCHIVE_BYTES // 1_048_576} MB.")
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise ArchiveError("Not a zip archive.") from exc
    names = zf.namelist()
    if len(names) > MAX_FILES:
        raise ArchiveError("Archive has too many files.")
    for name in names:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name:
            raise ArchiveError(f"Unsafe path in archive: {name!r}")
    if "manifest.json" not in names:
        raise ArchiveError("manifest.json missing - not an archive exported by this app.")
    manifest = json.loads(zf.read("manifest.json"))
    if manifest.get("format") != "fsa-company-archive" or \
            manifest.get("version") != FORMAT_VERSION:
        raise ArchiveError("Unsupported archive format or version.")
    for name, digest in (manifest.get("sha256") or {}).items():
        if name not in names or _sha(zf.read(name)) != digest:
            raise ArchiveError(f"Checksum mismatch for {name}: the archive is damaged or "
                               "was edited.")
    for ticker in manifest.get("tickers") or []:
        normalize_ticker(ticker)                # raises for malformed tickers
    return manifest


@dataclass
class ImportResult:
    restored: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)


def import_archive(ctx: AppContext, data: bytes) -> ImportResult:
    """Validate, extract and re-run the pipeline offline for every archived company."""
    from app.services.pipeline import run_pipeline

    manifest = validate_archive(data)
    result = ImportResult()
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for name in manifest["sha256"]:
                target = Path(tmp) / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zf.read(name))
        archive_ctx = build_context(ctx.settings, provider_instance=ArchiveProvider(Path(tmp)),
                                    sleep=lambda _s: None)
        try:
            for ticker in manifest["tickers"]:
                try:
                    outcome = run_pipeline(archive_ctx, ticker)
                except AnalyzerError as exc:
                    result.failed[ticker] = str(exc)
                    continue
                if outcome.succeeded:
                    result.restored.append(ticker)
                else:
                    result.failed[ticker] = outcome.error or "pipeline stopped"
        finally:
            archive_ctx.close()
    logger.info("Archive restored", extra={"restored": result.restored,
                                           "failed": list(result.failed)})
    return result
