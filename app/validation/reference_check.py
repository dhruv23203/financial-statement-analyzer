"""Data validation: compare stored (Yahoo Finance) figures with official filings.

Yahoo Finance is a convenient secondary source, not an authoritative one. This module
compares the standardised values our pipeline stored with reference values taken from
official filings and records every difference transparently:

* **US companies** - SEC EDGAR XBRL "company facts" (the machine-readable data of the
  10-K filings), fetched automatically: https://www.sec.gov/edgar/sec-api-documentation
* **Any company** (e.g. Indian annual reports) - a reference CSV that a person fills in
  from the annual report, with the source and access date of every value
  (``data/validation/reference_values.csv``).

Nothing here changes stored data or any calculation: it only reports. Status per value:
MATCH (|diff| <= 0.5 %), CLOSE (<= 2 %), DIFFERENT (> 2 %), MISSING_OURS,
MISSING_REFERENCE. Definitions can legitimately differ (e.g. Yahoo's "Total Debt"
includes lease liabilities); such metrics are marked ``definition_note``.

    python main.py validate-data AAPL MSFT          # SEC EDGAR, US tickers
    python main.py validate-data --csv data/validation/reference_values.csv
"""

from __future__ import annotations

import csv
import json
import os
import urllib.request
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Iterable

from app.database import repositories as repo
from app.database.models import StandardizedFinancialFact
from app.services.context import AppContext
from app.utils.exceptions import DataNotFoundError
from app.utils.helpers import normalize_ticker

METRICS = ("total_revenue", "net_income", "total_assets", "total_liabilities", "total_debt",
           "operating_cash_flow", "capital_expenditure")
MATCH_PCT, CLOSE_PCT = 0.5, 2.0
MAX_END_GAP_DAYS = 7
CSV_COLUMNS = ("ticker", "metric", "period_end", "currency", "reference_value", "source",
               "accessed_on", "notes")

# us-gaap concepts per metric, in order of preference. Sums are used when a filer does
# not report a single total (recorded in the "method" column).
SEC_CONCEPTS: dict[str, list[tuple[str, ...]]] = {
    "total_revenue": [("RevenueFromContractWithCustomerExcludingAssessedTax",),
                      ("Revenues",), ("SalesRevenueNet",)],
    "net_income": [("NetIncomeLoss",)],
    "total_assets": [("Assets",)],
    "total_liabilities": [("Liabilities",)],
    "operating_cash_flow": [("NetCashProvidedByUsedInOperatingActivities",)],
    "capital_expenditure": [("PaymentsToAcquirePropertyPlantAndEquipment",)],
}
# Total debt is NOT compared automatically: filers tag borrowings with many different
# concepts (and Yahoo's figure includes lease liabilities), so a reconstruction from
# XBRL tags was found to be unreliable (differences of 20 % to >1000 % in a trial run).
# Validate it with values read from the balance sheet / debt note via the reference CSV.
INSTANT = {"total_assets", "total_liabilities", "total_debt"}
DEFINITION_NOTES = {
    "total_debt": "Yahoo 'Total Debt' = current + long-term debt + capital/operating lease "
                  "liabilities; compare with the same components from the annual report.",
    "total_revenue": "Revenue definitions differ between sources (e.g. Yahoo may include or "
                     "exclude other operating income, excise taxes or membership fees); "
                     "differences of 1-3 % are usually definitional.",
    "capital_expenditure": "Yahoo reports capital expenditure as a negative cash outflow; "
                           "absolute values are compared.",
    "total_liabilities": "Yahoo 'Total Liabilities Net Minority Interest'.",
}
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
# SEC asks automated clients to identify themselves. Set FSA_SEC_USER_AGENT to
# "<your name or organisation> <contact email>" (SEC fair-access policy); the default
# identifies the tool only.
DEFAULT_USER_AGENT = "FinancialStatementAnalyzer/1.0 (educational project)"


@dataclass
class ReferenceValue:
    ticker: str
    metric: str
    period_end: date
    currency: str
    value: Decimal
    source: str
    accessed_on: str
    notes: str = ""


@dataclass
class ValidationRow:
    company: str
    metric: str
    period_end: str
    currency: str
    our_value: float | None
    reference_value: float | None
    difference: float | None
    difference_pct: float | None
    status: str
    source: str
    accessed_on: str
    notes: str = ""


# ------------------------------------------------------------------------- our values
def stored_values(ctx: AppContext, ticker: str) -> tuple[dict[date, dict[str, Decimal]],
                                                         str | None]:
    """Standardised annual values of the latest cleaned fetch run, and the currency."""
    from sqlalchemy import select

    ticker = normalize_ticker(ticker)
    with ctx.db.session_scope() as session:
        company = repo.get_company(session, ticker)
        if company is None:
            raise DataNotFoundError(f"{ticker} is not in the database.")
        run = repo.latest_cleaned_fetch_run(session, company.id)
        if run is None:
            raise DataNotFoundError(f"{ticker} has no cleaned data.")
        values: dict[date, dict[str, Decimal]] = {}
        for fact in session.scalars(select(StandardizedFinancialFact).where(
                StandardizedFinancialFact.fetch_run_id == run.id,
                StandardizedFinancialFact.metric_code.in_(METRICS))):
            if fact.value is not None:  # latest_cleaned_fetch_run is the annual run
                values.setdefault(fact.fiscal_period_end, {})[fact.metric_code] = fact.value
        return values, company.currency


# ------------------------------------------------------------------------- comparison
def compare_value(ours: Decimal | None, reference: Decimal | None,
                  metric: str) -> tuple[float | None, float | None, str]:
    if ours is None:
        return None, None, "MISSING_OURS"
    if reference is None:
        return None, None, "MISSING_REFERENCE"
    a, b = float(ours), float(reference)
    if metric == "capital_expenditure":
        a, b = abs(a), abs(b)
    diff = a - b
    if b == 0:
        return diff, None, "MATCH" if a == 0 else "DIFFERENT"
    pct = diff / abs(b) * 100
    status = "MATCH" if abs(pct) <= MATCH_PCT else "CLOSE" if abs(pct) <= CLOSE_PCT \
        else "DIFFERENT"
    return diff, pct, status


def validate(ctx: AppContext, references: Iterable[ReferenceValue]) -> list[ValidationRow]:
    rows: list[ValidationRow] = []
    cache: dict[str, tuple[dict[date, dict[str, Decimal]], str | None]] = {}
    for ref in references:
        if ref.ticker not in cache:
            try:
                cache[ref.ticker] = stored_values(ctx, ref.ticker)
            except DataNotFoundError:
                cache[ref.ticker] = ({}, None)
        values, our_currency = cache[ref.ticker]
        ours = values.get(ref.period_end, {}).get(ref.metric)
        diff, pct, status = compare_value(ours, ref.value, ref.metric)
        notes = [n for n in (ref.notes, DEFINITION_NOTES.get(ref.metric)) if n]
        if our_currency and ref.currency and our_currency != ref.currency:
            notes.append(f"currency differs (ours {our_currency})")
            status = "DIFFERENT" if status in ("MATCH", "CLOSE") else status
        rows.append(ValidationRow(
            ref.ticker, ref.metric, ref.period_end.isoformat(), ref.currency,
            None if ours is None else float(ours), float(ref.value),
            None if diff is None else round(diff, 2),
            None if pct is None else round(pct, 3), status, ref.source, ref.accessed_on,
            " ".join(notes)))
    return rows


# ------------------------------------------------------------------------- CSV source
def load_reference_csv(path: Path) -> list[ReferenceValue]:
    """Rows with columns ticker, metric, period_end (YYYY-MM-DD), currency,
    reference_value (full units, not crore/millions), source, accessed_on, notes."""
    refs = []
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for line, row in enumerate(csv.DictReader(handle), start=2):
            if not (row.get("ticker") or "").strip():
                continue
            metric = row["metric"].strip()
            if metric not in METRICS:
                raise ValueError(f"{path}:{line}: unknown metric {metric!r} "
                                 f"(use one of {', '.join(METRICS)})")
            if not (row.get("source") or "").strip():
                raise ValueError(f"{path}:{line}: every reference value needs a source")
            refs.append(ReferenceValue(
                normalize_ticker(row["ticker"]), metric,
                date.fromisoformat(row["period_end"].strip()), row["currency"].strip(),
                Decimal(row["reference_value"].replace(",", "").strip()),
                row["source"].strip(), (row.get("accessed_on") or "").strip(),
                (row.get("notes") or "").strip()))
    return refs


# ------------------------------------------------------------------------- SEC EDGAR
def _get_json(url: str, user_agent: str) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": user_agent,
                                                   "Accept-Encoding": "identity"})
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 (https)
        return json.loads(response.read().decode("utf-8"))


# Public SEC Central Index Keys of frequently validated companies. The full ticker map
# (www.sec.gov/files/company_tickers.json) requires a User-Agent with a contact address,
# so it is only used for other tickers when FSA_SEC_USER_AGENT is set.
KNOWN_CIKS = {"AAPL": 320193, "MSFT": 789019, "NVDA": 1045810, "GOOGL": 1652044,
              "GOOG": 1652044, "AMZN": 1018724, "META": 1326801, "JNJ": 200406,
              "XOM": 34088, "WMT": 104169, "KO": 21344, "PG": 80424, "JPM": 19617,
              "V": 1403161, "UNH": 731766, "HD": 354950, "PEP": 77476, "CSCO": 858877,
              "INTC": 50863, "ORCL": 1341439}


def sec_cik(ticker: str, fetch: Callable[[str], Any]) -> int | None:
    symbol = ticker.upper().replace(".", "-")
    if symbol in KNOWN_CIKS:
        return KNOWN_CIKS[symbol]
    for entry in fetch(SEC_TICKERS_URL).values():
        if entry.get("ticker", "").upper() == symbol:
            return int(entry["cik_str"])
    return None


def _annual_fact(units: list[dict[str, Any]], period_end: date,
                 instant: bool) -> dict[str, Any] | None:
    """The 10-K value for the fiscal year ending on *period_end* (latest filing wins,
    so restatements are reflected)."""
    candidates = []
    for item in units:
        if item.get("form") not in ("10-K", "10-K/A") or not item.get("end"):
            continue
        end = date.fromisoformat(item["end"])
        # Yahoo stores month-end dates; many filers end on a weekday rule (e.g. Apple:
        # last Saturday of September), so allow up to a week's difference.
        if abs((end - period_end).days) > MAX_END_GAP_DAYS:
            continue
        if not instant:
            start = item.get("start")
            if not start or not 350 <= (end - date.fromisoformat(start)).days <= 380:
                continue
        candidates.append(item)
    if not candidates:
        return None
    # closest end date; among equals the latest filing (restatements are reflected).
    # min() keeps the first of equal keys, so sort by filing date (newest first) before.
    candidates.sort(key=lambda i: i.get("filed", ""), reverse=True)
    return min(candidates, key=lambda i: abs((date.fromisoformat(i["end"]) - period_end).days))


def sec_reference_values(ticker: str, periods: Iterable[date],
                         user_agent: str | None = None,
                         fetch: Callable[[str], Any] | None = None) -> list[ReferenceValue]:
    """Reference values from SEC EDGAR company facts for the given fiscal year ends."""
    agent = user_agent or os.environ.get("FSA_SEC_USER_AGENT") or DEFAULT_USER_AGENT
    get = fetch or (lambda url: _get_json(url, agent))
    cik = sec_cik(ticker, get)
    if cik is None:
        return []
    facts = get(SEC_FACTS_URL.format(cik=cik)).get("facts", {}).get("us-gaap", {})
    accessed = datetime.now(timezone.utc).date().isoformat()
    refs = []
    for period in periods:
        for metric, options in SEC_CONCEPTS.items():
            for concepts in options:
                found = [_annual_fact(facts.get(c, {}).get("units", {}).get("USD", []),
                                      period, metric in INSTANT) for c in concepts]
                present = [f for f in found if f is not None]
                # single concept must exist; for sums use what the filer reports
                if not present or (len(concepts) == 1 and found[0] is None):
                    continue
                total = sum(Decimal(str(f["val"])) for f in present)
                used = [c for c, f in zip(concepts, found) if f is not None]
                accession = present[0].get("accn", "")
                filed_end = present[0].get("end")
                note = "" if filed_end == period.isoformat() else \
                    f"filing period ends {filed_end} (Yahoo date {period})."
                refs.append(ReferenceValue(
                    normalize_ticker(ticker), metric, period, "USD", total,
                    f"SEC EDGAR XBRL 10-K, CIK {cik}, accession {accession}, us-gaap "
                    f"{' + '.join(used)}", accessed, note))
                break
    return refs


# ------------------------------------------------------------------------- reporting
def write_report(rows: list[ValidationRow], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "validation_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ValidationRow.__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.status] = counts.get(row.status, 0) + 1
    lines = ["# Data validation results", "",
             f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC by "
             "`python main.py validate-data`. Our value = standardised value stored from "
             "Yahoo Finance; reference = official filing. Thresholds: MATCH <= "
             f"{MATCH_PCT}%, CLOSE <= {CLOSE_PCT}%, otherwise DIFFERENT.", "",
             "Summary: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())), "",
             "| Company | Metric | Period end | Currency | Our value | Reference | Diff % | "
             "Status | Source (accessed) |",
             "|---|---|---|---|---:|---:|---:|---|---|"]
    for r in rows:
        fmt = (lambda v: "-" if v is None else f"{v:,.0f}")
        lines.append(f"| {r.company} | {r.metric} | {r.period_end} | {r.currency} | "
                     f"{fmt(r.our_value)} | {fmt(r.reference_value)} | "
                     f"{'-' if r.difference_pct is None else f'{r.difference_pct:+.2f}%'} | "
                     f"{r.status} | {r.source} ({r.accessed_on}) |")
    notes = sorted({f"* **{r.metric}**: {DEFINITION_NOTES[r.metric]}" for r in rows
                    if r.metric in DEFINITION_NOTES})
    if notes:
        lines += ["", "Definition notes:", *notes]
    md_path = output_dir / "VALIDATION_RESULTS.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path
