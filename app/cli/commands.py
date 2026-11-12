"""argparse CLI. Every command is a thin wrapper around a service function.

Exit codes: 0 success, 1 handled error (bad input, missing data, provider or
database problem), 2 invalid command-line usage (argparse).
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from app.analysis.comparison import (ComparisonReport, amounts_frame, period_labels,
                                     ratios_frame, summary_frame)
from app.analysis.service import AnalysisReport, analyze_company
from app.cleaning.cleaner import CleaningSummary, clean_company
from app.database.connection import Database
from app.ingestion.service import IngestionResult, ingest_company
from app.reporting import tables
from app.services.context import AppContext, build_context
from app.services.pipeline import company_history, run_comparison, run_pipeline
from app.utils.exceptions import AnalyzerError
from app.utils.logging_config import configure_logging
from config.metric_aliases import STATEMENT_LABELS
from config.settings import PEER_GROUPS, PROJECT_ROOT, VALID_PROVIDERS, get_settings

logger = logging.getLogger(__name__)
LINE = "=" * 78
DISCLAIMER = ("Educational analysis of historical financial statements - not a credit rating "
              "or investment advice.")


def out(text: str = "") -> None:
    print(text, flush=True)


def heading(title: str) -> None:
    out()
    out(LINE)
    out(f"  {title}")
    out(LINE)


# ---------------------------------------------------------------------- printers
def print_ingestion(result: IngestionResult) -> None:
    heading(f"DAY 10 - DATA ACQUISITION: {result.ticker}  (fetch run #{result.fetch_run_id})")
    out(f"Status            : {result.status}")
    out(f"Company           : {result.company_name or 'unknown (metadata unavailable)'}")
    out(f"Reporting currency: {result.currency or 'unknown'}")
    if result.currency_check:
        out(f"Currency check    : {result.currency_check.get('status')}")
    for statement, label in STATEMENT_LABELS.items():
        if statement in result.available_statements:
            out(f"  [OK]      {label}")
        else:
            out(f"  [MISSING] {label}: {result.missing_statements.get(statement, '')}")
    if result.succeeded:
        out(f"Raw facts stored  : {result.raw_fact_count} "
            f"({result.missing_value_count} provider cells were empty)")
        out(f"Fiscal periods    : {', '.join(result.periods)}")
        for statement, path in result.raw_files.items():
            out(f"Raw CSV backup    : data/{path}")
    if result.metadata_error:
        out(f"Note: company metadata unavailable ({result.metadata_error}); "
            "statements were still stored.")
    for warning in result.warnings[:10]:
        out(f"Warning: {warning}")
    if result.error:
        out(f"ERROR: {result.error}")


def print_cleaning(summary: CleaningSummary) -> None:
    heading(f"DAY 20 - CLEANING & STANDARDISATION: {summary.ticker}  "
            f"(cleaning run #{summary.cleaning_run_id}, fetch run #{summary.fetch_run_id})")
    rows = [("Status", summary.status), ("Raw records", summary.raw_record_count),
            ("Standardised records", summary.standardized_record_count),
            ("  of which mapped", summary.mapped_record_count),
            ("Missing values (kept as missing, not 0)", summary.missing_value_count),
            ("Unmapped metrics (preserved)", summary.unmapped_metric_count),
            ("Mapping conflicts (precedence applied)", summary.conflict_count),
            ("Duplicate aliases merged", summary.duplicate_count),
            ("Values with validation warnings", summary.validation_warning_count),
            ("Fiscal periods", ", ".join(summary.periods))]
    for label, value in rows:
        out(f"{label:<42}: {value}")
    if summary.unmapped_metrics:
        shown = summary.unmapped_metrics[:8]
        more = len(summary.unmapped_metrics) - len(shown)
        out("Unmapped (review): " + "; ".join(shown) + (f" ... (+{more} more)" if more else ""))
    for conflict in summary.conflicts[:5]:
        out(f"Conflict: {conflict}")
    for warning in summary.warnings[:10]:
        out(f"Validation warning: {warning}")
    if summary.exports:
        out(f"Processed CSV     : {summary.exports.get('wide') or summary.exports.get('long')}")


def print_analysis(report: AnalysisReport) -> None:
    currency = report.currency or "unknown currency"
    heading(f"ANALYSIS: {report.ticker} - {report.company_name or ''}  "
            f"(analysis run #{report.analysis_run_id}, {currency})")
    wide, notes = tables.ratio_table(report.ratios)
    out("\nDAY 30/40 - FINANCIAL RATIOS (fiscal years, newest first)")
    out(tables.to_text(wide))
    out(tables.legend())
    for note in notes[:12]:
        out(f"  - {note}")
    if len(notes) > 12:
        out(f"  ... {len(notes) - 12} more explanations in the exported ratio detail CSV")

    out(f"\nDAY 50 - TREND SUMMARY (amounts in {currency})")
    out(tables.to_text(tables.trend_summary_frame(report.trends, report.currency)))
    special = [t for t in report.trends if t.comparison_type == "YOY" and t.status != "OK"
               and t.explanation]
    for trend in special[:6]:
        out(f"  - {trend.label} {trend.previous_period}->{trend.current_period}: "
            f"{trend.explanation}")

    out("\nDAY 60 - RED FLAGS (potential concerns, not proof of fraud or distress)")
    if report.red_flags.flags:
        out(tables.to_text(tables.red_flag_frame(report.red_flags.flags)))
    else:
        out("No red-flag rule was triggered.")
    skipped = report.red_flags.not_evaluated()
    if skipped:
        out(f"  ({len(skipped)} rule checks could not be evaluated because inputs were missing)")

    out("\nDAY 70 - FINANCIAL HEALTH SCORE (0-100, by fiscal year)")
    out(tables.to_text(tables.score_frame(report.scores)))
    overall = report.overall()
    if overall is not None:
        out(f"Latest overall score: {tables.fmt_score(overall.score)} / 100  "
            f"[{overall.interpretation or 'N/A'}], data coverage {overall.coverage:.0%}, "
            f"scoring methodology v{report.scoring_version}")
    grades = {k: v for k, v in report.grades.items() if v}
    if grades:
        out("Grades (A = top 10% ... F = bottom 10% of the reference universe): " + ", ".join(
            f"{k.title().replace('_', ' ')} {v}" for k, v in grades.items()))
    market = report.dataset.market
    if market:
        valuation = {o.code: o for o in report.ratios if o.definition.category == "VALUATION"}
        parts = [f"market cap {market.get('currency', '')} "
                 f"{float(market['market_cap']) / 1e9:,.1f}bn (as of {market.get('fetched_at')})"]
        for code in ("price_to_earnings", "price_to_book", "ev_to_ebitda", "dividend_yield"):
            o = valuation.get(code)
            if o is not None:
                parts.append(f"{o.definition.name} {tables.format_ratio_value(o)}")
        out("\nMARKET SNAPSHOT (descriptive, not a valuation): " + "; ".join(parts))
    if report.pros or report.cons:
        out("\nPROS & CONS (automatic, rule-based)")
        for item in report.pros:
            out(f"  + {item}")
        for item in report.cons:
            out(f"  - {item}")
    for note in report.notes:
        out(f"Note: {note}")
    if report.exports:
        out(f"\nReport  : {report.exports.get('markdown_report')}")
        out(f"Exports : {Path(report.exports['ratio_table']).parent.as_posix()}/")
    if report.charts:
        out(f"Charts  : {Path(next(iter(report.charts.values()))).parent.as_posix()}/")
    out(DISCLAIMER)


def print_comparison(report: ComparisonReport) -> None:
    heading(f"DAY 80 - COMPARISON #{report.comparison_run_id}: {', '.join(report.requested)}")
    out(f"Status: {report.status} | currency: {report.currency_compatibility} | "
        f"periods: {report.period_alignment} ({period_labels(report)})")
    out("\nSummary")
    out(tables.to_text(summary_frame(report)))
    out("\nRatios (rank in brackets; unit-free, comparable across currencies)")
    out(tables.to_text(ratios_frame(report)))
    out("\nAbsolute amounts (each in its own reporting currency, millions)")
    out(tables.to_text(amounts_frame(report)))
    if report.warnings:
        out("\nWarnings")
        for warning in report.warnings:
            out(f"  - {warning}")
    if report.exports:
        out(f"\nExports: {Path(report.exports['summary']).parent.as_posix()}/")
    if report.charts:
        out(f"Charts : {Path(next(iter(report.charts.values()))).parent.as_posix()}/")
    out(DISCLAIMER)


def print_history(history: dict[str, Any]) -> None:
    company = history["company"]
    heading(f"HISTORY: {company['ticker']} - {company['name'] or ''}")
    out(f"Exchange {company['exchange']} | sector {company['sector']} | industry "
        f"{company['industry']} | currency {company['currency']}")
    sections = [("Fetch runs", "fetch_runs"), ("Cleaning runs", "cleaning_runs"),
                ("Analysis runs", "analysis_runs"), ("Comparison runs", "comparison_runs")]
    import pandas as pd

    for title, key in sections:
        out(f"\n{title}")
        frame = pd.DataFrame(history[key])
        if not frame.empty:
            for column in frame.columns:
                if column.endswith("_at"):
                    frame[column] = frame[column].map(
                        lambda v: v.strftime("%Y-%m-%d %H:%M UTC") if v else "")
            if "error" in frame:
                frame["error"] = frame["error"].fillna("").map(lambda v: str(v)[:60])
        out(tables.to_text(frame))


def print_db_report(report: Any) -> None:
    heading("DATABASE HEALTH CHECK")
    out(f"File            : {report.db_path}")
    out(f"Exists          : {report.exists}")
    if report.exists:
        out(f"Size            : {report.size_bytes:,} bytes")
        out(f"SQLite version  : {report.sqlite_version}")
        out(f"Schema revision : {report.current_revision} (head {report.head_revision})")
        out(f"Integrity check : {report.integrity}")
        out(f"FK violations   : {report.foreign_key_violations}")
        for table, count in report.table_counts.items():
            out(f"  {table:<32}{count:>10,}")
    out(f"Result          : {'OK' if report.ok else 'PROBLEMS FOUND'}")
    for problem in report.problems:
        out(f"  - {problem}")


# ---------------------------------------------------------------------- commands
def cmd_init_db(args: argparse.Namespace) -> int:
    settings = _settings(args)
    settings.ensure_directories()
    configure_logging(settings.logs_dir, settings.log_level)
    database = Database(settings)
    revision = database.upgrade()
    out(f"Database ready at {settings.db_path} (schema revision {revision}).")
    database.dispose()
    return 0


def cmd_db_check(args: argparse.Namespace) -> int:
    settings = _settings(args)
    configure_logging(settings.logs_dir, settings.log_level)
    database = Database(settings)
    report = database.health_check()
    print_db_report(report)
    database.dispose()
    return 0 if report.ok else 1


def cmd_backup_db(args: argparse.Namespace) -> int:
    ctx = _context(args)
    path = ctx.db.backup(label=args.label)
    out(f"Backup created: {path}")
    return 0


def cmd_restore_db(args: argparse.Namespace) -> int:
    if not args.yes:
        out("Restoring replaces the current database. A safety backup is taken first.")
        out("Re-run with --yes to confirm.")
        return 1
    settings = _settings(args)
    settings.ensure_directories()
    configure_logging(settings.logs_dir, settings.log_level)
    database = Database(settings)
    safety = database.restore(Path(args.backup_file))
    out(f"Database restored from {args.backup_file}.")
    if safety:
        out(f"The previous database was saved to {safety}.")
    database.dispose()
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    ctx = _context(args)
    failures = 0
    for ticker in args.tickers:
        result = ingest_company(ctx, ticker, args.frequency)
        print_ingestion(result)
        failures += not result.succeeded
    return 1 if failures else 0


def cmd_clean(args: argparse.Namespace) -> int:
    ctx = _context(args)
    for ticker in args.tickers:
        print_cleaning(clean_company(ctx, ticker))
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    ctx = _context(args)
    for ticker in args.tickers:
        print_analysis(analyze_company(ctx, ticker))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    ctx = _context(args)
    failures = 0
    for ticker in args.tickers:
        result = run_pipeline(ctx, ticker, args.frequency, progress=out)
        if result.ingestion:
            print_ingestion(result.ingestion)
        if result.cleaning:
            print_cleaning(result.cleaning)
        if result.analysis:
            print_analysis(result.analysis)
        if result.error:
            failures += 1
            out(f"\n{ticker}: pipeline stopped - {result.error}")
    return 1 if failures else 0


def cmd_compare(args: argparse.Namespace) -> int:
    tickers = list(args.tickers or [])
    if args.peer_group:
        if args.peer_group not in PEER_GROUPS:
            raise AnalyzerError(f"Unknown peer group {args.peer_group!r}. "
                                f"Options: {', '.join(PEER_GROUPS)}")
        tickers = PEER_GROUPS[args.peer_group] + tickers
    ctx = _context(args)
    report, errors = run_comparison(ctx, tickers, args.fiscal_year,
                                    fetch_missing=not args.no_fetch, progress=out)
    for ticker, error in errors.items():
        out(f"{ticker}: could not be prepared - {error}")
    print_comparison(report)
    return 0


def cmd_history(args: argparse.Namespace) -> int:
    ctx = _context(args)
    print_history(company_history(ctx, args.ticker))
    return 0


def cmd_companies(args: argparse.Namespace) -> int:
    from app.database import repositories as repo

    ctx = _context(args)
    with ctx.db.session_scope() as session:
        companies = repo.list_companies(session)
        rows = []
        for c in companies:
            latest = repo.latest_analysis_run(session, c.id)
            summary = (latest.summary or {}) if latest else {}
            rows.append({"Ticker": c.ticker, "Name": (c.name or "")[:40],
                         "Currency": c.currency or "?", "Sector": c.sector or "",
                         "Latest FY": summary.get("latest_fiscal_period", ""),
                         "Score": summary.get("overall_score") or "N/A",
                         "Red flags": summary.get("red_flags_total", "")})
    import pandas as pd

    heading("COMPANIES IN DATABASE")
    out(tables.to_text(pd.DataFrame(rows)))
    return 0


def cmd_peers(args: argparse.Namespace) -> int:
    if args.ticker:
        from app.services.screener import find_peers

        ctx = _context(args)
        peers = find_peers(ctx, args.ticker, args.count)
        heading(f"PEERS OF {args.ticker.upper()} (same industry/sector, closest revenue size)")
        out("  " + (" ".join(peers) if peers else "No analysed peers found."))
        if peers:
            out(f"Compare with: python main.py compare {args.ticker.upper()} {' '.join(peers)}")
        return 0
    heading("EXAMPLE PEER GROUPS (use: python main.py compare --peer-group NAME)")
    for name, tickers in PEER_GROUPS.items():
        out(f"  {name:<15} {' '.join(tickers)}")
    return 0


def cmd_screen(args: argparse.Namespace) -> int:
    from app.services.screener import field_descriptions, run_screen, screen_frame

    ctx = _context(args)
    if args.fields or not args.query:
        frame = screen_frame(ctx)
        heading(f"SCREENER FIELDS ({len(frame)} analysed companies)")
        descriptions = field_descriptions()
        for name in sorted(set(frame.columns) | {k for k in descriptions if len(k) <= 9}):
            out(f"  {name:<30} {descriptions.get(name, '')}")
        out('Example: python main.py screen "roe > 15 and de < 0.5 and piotroski >= 7"')
        return 0
    result, total = run_screen(ctx, args.query, args.sort, args.ascending, args.limit)
    heading(f"SCREEN: {args.query}  ->  {len(result)} shown of {total} companies screened")
    out(tables.to_text(result.round(2)))
    if args.export:
        path = ctx.settings.exports_dir / "screens"
        path.mkdir(parents=True, exist_ok=True)
        target = path / "screen_results.csv"
        result.to_csv(target, index=False)
        out(f"Exported: {target}")
    return 0


def cmd_market(args: argparse.Namespace) -> int:
    from app.ingestion.market import refresh_market_snapshots

    ctx = _context(args)
    for ticker, status in refresh_market_snapshots(ctx, args.tickers).items():
        out(f"{ticker:<14} {status}")
    return 0


def cmd_analyze_universe(args: argparse.Namespace) -> int:
    """Store an analysis for every universe company (needed by the screener)."""
    from app.calibration.universe import load_universe
    from app.database import repositories as repo

    ctx = _context(args)
    tickers = load_universe()
    done, failed = 0, 0
    for position, ticker in enumerate(tickers, start=1):
        with ctx.db.session_scope() as session:
            company = repo.get_company(session, ticker)
            ready = company is not None and \
                repo.latest_cleaned_fetch_run(session, company.id) is not None
        if not ready:
            continue
        try:
            analyze_company(ctx, ticker, write_outputs=False)
            done += 1
        except AnalyzerError as exc:
            failed += 1
            logger.warning("analyze-universe: %s failed: %s", ticker, exc)
        if position % 100 == 0:
            out(f"[{position}/{len(tickers)}] analysed {done}")
    out(f"Analysed {done} companies ({failed} failed). Use: python main.py screen --fields")
    return 0


def cmd_sensitivity(args: argparse.Namespace) -> int:
    """How much the overall score depends on the category weights (stored scores only)."""
    import json

    from app.calibration.sensitivity import compare_weightings, default_scenarios
    from app.services.screener import screen_frame

    ctx = _context(args)
    config = json.loads(ctx.settings.scoring_config_file.read_text(encoding="utf-8"))
    results = PROJECT_ROOT / "docs" / "calibration" / "calibration_results.json"
    critic = None
    if results.is_file():
        critic = json.loads(results.read_text(encoding="utf-8"))["category_weights"][
            "candidates"].get("CRITIC")
    frame = screen_frame(ctx)
    if frame.empty:
        out("No analysed companies. Run: python main.py analyze-universe")
        return 1
    frame = frame[frame["sector"] != "Financial Services"]
    table = compare_weightings(frame, default_scenarios(config, critic),
                               "shipped (logistic, liquidity 0)",
                               config["interpretation_bands"])
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(target, index=False)
    out(table.to_string(index=False))
    out(f"Saved {target}")
    return 0


def cmd_validate_data(args: argparse.Namespace) -> int:
    """Compare stored figures with official filings (SEC EDGAR and/or a reference CSV)."""
    from app.validation import reference_check as rc

    ctx = _context(args)
    references = []
    if args.csv:
        references.extend(rc.load_reference_csv(Path(args.csv)))
    for ticker in args.tickers:
        try:
            values, _ = rc.stored_values(ctx, ticker)
        except AnalyzerError as exc:
            out(f"{ticker}: {exc}")
            continue
        periods = sorted(values)[-args.years:]
        try:
            found = rc.sec_reference_values(ticker, periods)
        except OSError as exc:  # network errors, HTTP errors
            out(f"{ticker}: SEC EDGAR request failed ({exc}); skipped.")
            continue
        if not found:
            out(f"{ticker}: not found in SEC EDGAR (only US filers are covered).")
        references.extend(found)
    if not references:
        out("Nothing to validate: give US tickers and/or --csv with reference values.")
        return 1
    rows = rc.validate(ctx, references)
    csv_path, md_path = rc.write_report(rows, Path(args.output))
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.status] = counts.get(row.status, 0) + 1
    out(f"Validated {len(rows)} values: " + ", ".join(f"{k} {v}" for k, v in
                                                      sorted(counts.items())))
    out(f"Report: {md_path}")
    out(f"Data:   {csv_path}")
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Derive scoring breakpoints, weights and red-flag thresholds from a reference universe."""
    import json

    from app.calibration.calibrate import calibrate, write_outputs
    from app.calibration.universe import load_universe, prepare_universe

    ctx = _context(args)
    tickers = load_universe(Path(args.universe)) if args.universe else load_universe()
    out(f"Reference universe: {len(tickers)} tickers")
    if args.fetch or args.refresh:
        report = prepare_universe(ctx, tickers, refresh=args.refresh, progress=out)
        out(f"Ready: {len(report.ready)} | newly fetched: {len(report.fetched)} | "
            f"failed: {len(report.failed)}")
    scoring_path, red_flag_path = ctx.settings.scoring_config_file, \
        ctx.settings.red_flag_config_file
    result = calibrate(ctx, tickers, json.loads(scoring_path.read_text(encoding="utf-8")),
                       json.loads(red_flag_path.read_text(encoding="utf-8")))
    outputs = write_outputs(result, Path(args.output), args.apply, scoring_path, red_flag_path)
    heading("CALIBRATION")
    out(f"Companies used: {len(result.companies)} | company-years: {result.observations} | "
        f"excluded: {len(result.excluded)}")
    out(f"Category weights ({result.category_weights['chosen']}): " + ", ".join(
        f"{k} {v['weight']:.3f}" for k, v in result.scoring_config["categories"].items()))
    for label, sup in result.supervised.items():
        if "skipped" in sup:
            out(f"Supervised model [{label}]: skipped ({sup['skipped']})")
        else:
            out(f"Supervised model [{label}]: AUC {sup['oof_auc']:.3f} (95% CI "
                f"{sup['oof_auc_ci'][0]:.3f}-{sup['oof_auc_ci'][1]:.3f}), "
                f"accepted={sup['accepted']}")
    validation = result.diagnostics.get("validation", {})
    for code in ("piotroski_f_score", "altman_z_score", "sws_health_checks", "beneish_m_score"):
        if code in validation:
            out(f"Validation vs {code}: Spearman rho {validation[code]['spearman_rho']} "
                f"(n={validation[code]['n']})")
    out("Bands: " + ", ".join(f"{b['label']} >= {b['min']}" for b in result.bands))
    out(f"Report: {outputs['report']}")
    if args.apply:
        out("Applied: config files updated (previous versions in config/archive/). Re-run "
            "`python main.py analyze TICKER` to score companies with the new methodology.")
    else:
        out("Proposed configs written; add --apply to make them the active methodology.")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Offline end-to-end demonstration with the synthetic sample companies."""
    args.provider = "sample"
    ctx = _context(args)
    out("OFFLINE DEMO - synthetic companies from sample_data/ (no internet needed).")
    for ticker in ("DEMOGOOD", "DEMOWEAK", "DEMOUSD", "DEMOPARTIAL"):
        result = run_pipeline(ctx, ticker, progress=out)
        if result.analysis:
            report = result.analysis
            overall = report.overall()
            out(f"  {ticker:<12} score {tables.fmt_score(overall.score if overall else None):>6}"
                f"  flags {len(report.red_flags.flags):>2}  status {report.status}")
    report, _ = run_comparison(ctx, PEER_GROUPS["offline_demo"], fetch_missing=False)
    print_comparison(report)
    if args.verbose:
        print_analysis(analyze_company(ctx, "DEMOWEAK"))
    return 0


# ---------------------------------------------------------------------- parser
def _settings(args: argparse.Namespace):
    settings = get_settings()
    overrides: dict[str, Any] = {}
    if getattr(args, "provider", None):
        overrides["provider"] = args.provider
    if getattr(args, "log_level", None):
        overrides["log_level"] = args.log_level.upper()
    return settings.with_overrides(**overrides) if overrides else settings


def _context(args: argparse.Namespace) -> AppContext:
    """Build the application context; ``main`` closes it when the command finishes."""
    ctx = build_context(_settings(args))
    args.app_context = ctx
    return ctx


def _positive_year(text: str) -> int:
    value = int(text)
    if not 1900 <= value <= 2100:
        raise argparse.ArgumentTypeError("fiscal year must be between 1900 and 2100")
    return value


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--provider", choices=VALID_PROVIDERS, default=argparse.SUPPRESS,
                        help="data source (default: yfinance, or FSA_PROVIDER)")
    common.add_argument("--log-level", default=argparse.SUPPRESS,
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="console log level (file log is always DEBUG)")

    parser = argparse.ArgumentParser(
        prog="python main.py", parents=[common],
        description="Automated Financial Statement Analyzer - Semester 1 backend",
        epilog="Example: python main.py run TCS.NS | python main.py compare TCS.NS INFY.NS")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    def add(name: str, func: Any, help_text: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, parents=[common], help=help_text, description=help_text)
        p.set_defaults(func=func)
        return p

    add("init-db", cmd_init_db, "create/upgrade the database schema (Alembic migrations)")
    add("db-check", cmd_db_check, "check database connectivity, schema and integrity")
    p = add("backup-db", cmd_backup_db, "create a timestamped backup in data/backups/")
    p.add_argument("--label", default="manual", help="text added to the backup file name")
    p = add("restore-db", cmd_restore_db, "restore the database from a backup file")
    p.add_argument("backup_file", help="path to a .db backup")
    p.add_argument("--yes", action="store_true", help="confirm replacing the current database")
    for name, func, text in (
            ("ingest", cmd_ingest, "Day 10: fetch statements and store raw data"),
            ("run", cmd_run, "Days 10-70: full pipeline (ingest, clean, analyse)")):
        p = add(name, func, text)
        p.add_argument("tickers", nargs="+", help="e.g. TCS.NS INFY.NS AAPL")
        p.add_argument("--frequency", choices=["annual", "quarterly"], default="annual")
    p = add("clean", cmd_clean, "Day 20: clean and standardise stored raw data")
    p.add_argument("tickers", nargs="+")
    p = add("analyze", cmd_analyze, "Days 30-70: ratios, trends, red flags, health score")
    p.add_argument("tickers", nargs="+")
    p = add("compare", cmd_compare, "Day 80: compare 2-5 companies")
    p.add_argument("tickers", nargs="*")
    p.add_argument("--peer-group", help=f"one of: {', '.join(PEER_GROUPS)}")
    p.add_argument("--fiscal-year", type=_positive_year,
                   help="compare the fiscal year ending in this calendar year")
    p.add_argument("--no-fetch", action="store_true",
                   help="use stored analyses only; never download missing companies")
    p = add("history", cmd_history, "show ingestion/cleaning/analysis/comparison history")
    p.add_argument("ticker")
    add("companies", cmd_companies, "list companies stored in the database")
    p = add("peers", cmd_peers, "list example peer groups, or find peers of a ticker")
    p.add_argument("ticker", nargs="?", help="find the closest analysed peers of this ticker")
    p.add_argument("--count", type=int, default=4)
    p = add("screen", cmd_screen, 'stock screener, e.g. screen "roe > 15 and de < 0.5"')
    p.add_argument("query", nargs="?", help="conditions joined by AND")
    p.add_argument("--fields", action="store_true", help="list fields you can screen on")
    p.add_argument("--sort", default="score", help="field to sort by (default: score)")
    p.add_argument("--ascending", action="store_true")
    p.add_argument("--limit", type=int, default=25)
    p.add_argument("--export", action="store_true", help="save results to exports/screens/")
    p = add("market", cmd_market, "fetch current market data (price, market cap)")
    p.add_argument("tickers", nargs="+")
    add("analyze-universe", cmd_analyze_universe,
        "analyse every reference-universe company (for the screener)")
    p = add("calibrate", cmd_calibrate,
            "derive score breakpoints, weights and red-flag thresholds from a reference universe")
    p.add_argument("--fetch", action="store_true", help="download universe companies not yet stored")
    p.add_argument("--refresh", action="store_true", help="re-download every universe company")
    p.add_argument("--apply", action="store_true",
                   help="replace the active config files (old versions archived)")
    p.add_argument("--universe", help="path to a universe JSON (default config/reference_universe.json)")
    p.add_argument("--output", default=str(PROJECT_ROOT / "docs" / "calibration"),
                   help="folder for the report and proposed configs")
    p = add("sensitivity", cmd_sensitivity,
            "how much the overall score depends on the category weights")
    p.add_argument("--output", default=str(PROJECT_ROOT / "docs" / "calibration" /
                                           "weight_sensitivity.csv"))
    p = add("validate-data", cmd_validate_data,
            "compare stored figures with official filings (SEC EDGAR / reference CSV)")
    p.add_argument("tickers", nargs="*", help="US tickers to check against SEC EDGAR")
    p.add_argument("--csv", help="reference CSV (e.g. values from Indian annual reports)")
    p.add_argument("--years", type=int, default=2, help="latest N fiscal years per ticker")
    p.add_argument("--output", default=str(PROJECT_ROOT / "docs" / "validation"))
    p = add("demo", cmd_demo, "offline demonstration with synthetic sample companies")
    p.add_argument("--verbose", action="store_true", help="also print one full analysis")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    # Never crash on characters the Windows console code page cannot encode.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except AnalyzerError as exc:
        logger.error("%s", exc, extra={"command": args.command})
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    finally:
        ctx = getattr(args, "app_context", None)
        if ctx is not None:
            ctx.close()


