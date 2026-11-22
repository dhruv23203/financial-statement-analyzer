"""Portfolio Doctor: review a broker holdings export against the screening results.

Input: the holdings CSV/Excel a broker exports (Zerodha, Upstox, Groww, ICICI Direct, a
US broker ...) with at least a symbol, quantity and average-cost column; column names
are matched loosely and every rejected row is reported with its reason.

Output: a descriptive observation per holding (financial strength, valuation, position
size), portfolio-level metrics and prioritised alerts. Pure functions over the screen
frame; nothing is stored. Observations are not buy/sell instructions - educational
project, not investment advice.

Correctness rules (see docs/PORTFOLIO_METHODOLOGY.md):

* a market price is used only when it is fresh (``price_max_age_days``); average cost is
  never substituted for a missing price - value, P&L and weight are then N/A;
* every holding has a currency; amounts in different currencies are never added
  (no FX conversion is implemented), so the portfolio is analysed one currency at a time;
* the portfolio health score covers only analysed, priced holdings and its coverage is
  always reported; the composite score is withheld below ``min_analysed_coverage``;
* thresholds come from ``config/portfolio_config.json`` and are illustrative.
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

from config.settings import CONFIG_DIR

SYMBOL_COLS = ["symbol", "instrument", "ticker", "trading symbol", "stock symbol", "scrip",
               "stock", "company"]
QTY_COLS = ["quantity", "qty", "qty.", "shares", "units", "holding", "quantity available"]
COST_COLS = ["avg_price", "avg. cost", "avg cost", "avg price", "average price", "average cost",
             "average buy price", "buy price", "buy avg", "cost price", "purchase price",
             "avg_cost", "cost basis"]
LTP_COLS = ["ltp", "last price", "current price", "closing price", "close price", "cmp",
            "market price"]

# Illustrative, project-defined thresholds (config/portfolio_config.json) - not rules.
CONFIG = json.loads((CONFIG_DIR / "portfolio_config.json").read_text(encoding="utf-8"))
MAX_STOCK_PCT = float(CONFIG["max_stock_pct"])
MAX_SECTOR_PCT = float(CONFIG["max_sector_pct"])
MIN_HOLDINGS = int(CONFIG["min_holdings"])
SPREAD_TARGET = int(CONFIG["position_spread_target"])
PRICE_MAX_AGE_DAYS = int(CONFIG["price_max_age_days"])
DISAGREEMENT_PCT = float(CONFIG["price_disagreement_pct"])
MIN_COVERAGE = float(CONFIG["min_analysed_coverage"])
WEAK_KEYS = ("AVOID", "TRAP")
EXPENSIVE_KEYS = ("WAIT", "NOADD")
BUY_KEYS = ("BUY", "BUY_CHECK")


# ----------------------------------------------------------------------------- parsing
def _norm(c: str) -> str:
    return re.sub(r"[^a-z0-9 ._]", "", str(c).lower()).strip()


def _find(columns, candidates):
    normed = {_norm(c): c for c in columns}
    for cand in candidates:
        if cand in normed:
            return normed[cand]
    for cand in candidates:
        for n, orig in normed.items():
            if cand in n:
                return orig
    return None


def clean_symbol(raw, default_suffix: str = ".NS") -> str:
    s = str(raw).strip().upper().replace(" ", "")
    s = re.sub(r"-(BE|EQ|BL|BZ|SM)$", "", s)       # NSE series suffixes in broker files
    if not s or s == "NAN":
        return ""
    if "." in s or s.startswith("^"):
        return s
    return s + default_suffix


def validate_holdings(df: pd.DataFrame, default_suffix: str = ".NS"
                      ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Normalise a broker export and report every rejected row with its reason.

    Returns (holdings with columns symbol, quantity, avg_price, ltp; rejected rows with
    columns row, symbol, reason; counts). Rows are never dropped silently."""
    if df is None or len(df.columns) == 0 or df.dropna(how="all").empty:
        raise ValueError("The file has no rows. Export your holdings again (CSV or Excel).")
    sym, qty = _find(df.columns, SYMBOL_COLS), _find(df.columns, QTY_COLS)
    cost, ltp = _find(df.columns, COST_COLS), _find(df.columns, LTP_COLS)
    missing = [n for n, c in (("symbol/instrument", sym), ("quantity", qty),
                              ("average price", cost)) if c is None]
    if missing:
        raise ValueError("Could not find column(s): " + ", ".join(missing) +
                         f". Columns in your file: {list(df.columns)}. Rename them, e.g. "
                         "'Symbol', 'Quantity', 'Avg cost'.")
    df = df.dropna(how="all")
    out = pd.DataFrame({
        "symbol": df[sym].map(lambda x: clean_symbol(x, default_suffix)).values,
        "quantity": pd.to_numeric(df[qty], errors="coerce").values,
        "avg_price": pd.to_numeric(df[cost], errors="coerce").values,
        "ltp": pd.to_numeric(df[ltp], errors="coerce").values if ltp else np.nan,
    })
    out["row"] = np.arange(2, len(out) + 2)  # spreadsheet row numbers (header = row 1)
    reason = np.select(
        [out["symbol"] == "", out["quantity"].isna(), out["quantity"] <= 0,
         out["avg_price"].isna(), out["avg_price"] <= 0],
        ["symbol missing", "quantity is not a number",
         "quantity must be positive (short or closed positions are not supported)",
         "average cost missing or not a number", "average cost must be positive"],
        default="")
    rejected = out[reason != ""].assign(reason=reason[reason != ""])
    ok = out[reason == ""].copy()
    ok["ltp"] = ok["ltp"].where(ok["ltp"] > 0)
    ok["cost_total"] = ok["quantity"] * ok["avg_price"]
    g = ok.groupby("symbol", as_index=False).agg(quantity=("quantity", "sum"),
                                                 cost_total=("cost_total", "sum"),
                                                 ltp=("ltp", "last"), rows=("row", "count"))
    g["avg_price"] = g["cost_total"] / g["quantity"]
    counts = {"rows": int(len(out)), "accepted": int(len(ok)), "rejected": int(len(rejected)),
              "holdings": int(len(g)), "duplicates_merged": int((g["rows"] > 1).sum())}
    return (g[["symbol", "quantity", "avg_price", "ltp"]].reset_index(drop=True),
            rejected[["row", "symbol", "reason"]].reset_index(drop=True), counts)


def parse_holdings(df: pd.DataFrame, default_suffix: str = ".NS") -> pd.DataFrame:
    """Normalise a broker export to columns: symbol, quantity, avg_price, ltp
    (see :func:`validate_holdings` for the rejected rows and counts)."""
    return validate_holdings(df, default_suffix)[0]


# ----------------------------------------------------------------------------- currency
def holding_currency(symbol: str, row: Any = None) -> tuple[str, str]:
    """(currency, basis). Stored market data wins; otherwise the exchange suffix."""
    if row is not None and isinstance(row.get("price_currency"), str):
        return row["price_currency"], "stored market data"
    if symbol.endswith((".NS", ".BO")):
        return "INR", "NSE/BSE suffix"
    if "." not in symbol:
        return "USD", "assumed: no exchange suffix (US listing)"
    return "Unknown", "unrecognised exchange suffix"


# ----------------------------------------------------------------------------- prices
def _age_days(as_of, today: date) -> int | None:
    if as_of is None or (isinstance(as_of, float) and np.isnan(as_of)):
        return None
    if isinstance(as_of, datetime):
        as_of = as_of.date()
    if isinstance(as_of, str):
        as_of = date.fromisoformat(as_of[:10])
    return (today - as_of).days


def select_price(broker_ltp: float | None, broker_as_of: date | None,
                 live: tuple[float, date] | None, snapshot: tuple[float, Any] | None,
                 today: date, max_age_days: int = PRICE_MAX_AGE_DAYS,
                 disagreement_pct: float = DISAGREEMENT_PCT) -> dict[str, Any]:
    """Pick the current price by a documented priority, never using a stale price:

    1. broker LTP from the uploaded file, if its as-of date is given and fresh;
    2. latest Yahoo Finance close fetched now, if fresh;
    3. stored database snapshot, if its date is fresh;
    4. otherwise no price (N/A) - average cost is never used as a market price.

    A warning is returned when two available sources disagree by more than
    ``disagreement_pct`` percent."""
    candidates = []
    if broker_ltp is not None and not pd.isna(broker_ltp):
        candidates.append(("Broker file LTP", float(broker_ltp), broker_as_of))
    if live is not None and live[0] is not None:
        candidates.append(("Yahoo Finance close (fetched now)", float(live[0]), live[1]))
    if snapshot is not None and snapshot[0] is not None and not pd.isna(snapshot[0]):
        candidates.append(("Stored snapshot", float(snapshot[0]), snapshot[1]))
    chosen, stale = None, []
    for source, price, as_of in candidates:
        age = _age_days(as_of, today)
        if age is not None and 0 <= age <= max_age_days:
            chosen = (source, price, as_of, age)
            break
        stale.append(f"{source} {price:,.2f} ("
                     f"{'date unknown' if age is None else f'{age} days old'})")
    if chosen is None:
        return {"price": None, "source": "None", "as_of": None, "status": "No reliable price",
                "note": ("Not used (stale or undated): " + "; ".join(stale)) if stale
                else "No price available", "warning": None}
    warning = None
    for source, price, _ in candidates:
        if source != chosen[0] and abs(price / chosen[1] - 1) * 100 > disagreement_pct:
            warning = (f"{chosen[0]} {chosen[1]:,.2f} and {source} {price:,.2f} differ by "
                       f"{abs(price / chosen[1] - 1) * 100:.0f}%")
            break
    return {"price": chosen[1], "source": chosen[0], "as_of": str(chosen[2])[:10],
            "status": "Fresh" if chosen[3] <= 1 else f"Fresh ({chosen[3]} days old)",
            "note": "", "warning": warning}


# ----------------------------------------------------------------------------- observations
def _action(key: str | None, weight: float | None) -> tuple[str, str]:
    if key is None:
        return "NO DATA", "Not in the database - analyse it from the sidebar first."
    if key in WEAK_KEYS:
        return "REVIEW: WEAK FINANCIALS", "Financial health is in the bottom of the universe."
    if weight is None or pd.isna(weight):
        return "NO PRICE", "No reliable current price, so the position size is unknown."
    if weight > MAX_STOCK_PCT:
        return "REVIEW: LARGE POSITION", (f"Position is {weight:.0f}% of the portfolio "
                                           f"(illustrative limit {MAX_STOCK_PCT:.0f}%).")
    if key in EXPENSIVE_KEYS:
        if weight > 10:
            return "REVIEW: LARGE & HIGH VALUATION", ("Valued above its model value and "
                                                     "already a large position.")
        return "HIGH VALUATION", "Price is above the model value for its quality."
    if key == "BUY":
        if weight < 10:
            return "STRONG · SMALL POSITION", ("Strong financials, price at or below the "
                                              "margin-of-safety price, small position.")
        return "STRONG · WELL SIZED", ("Strong financials, price at or below the "
                                       "margin-of-safety price, well-sized position.")
    if key == "BUY_CHECK":
        # valuation not conclusive: no DCF, no dated price, or price between the
        # margin-of-safety price and the model value - never "below model value"
        return "STRONG · VALUATION UNCLEAR", ("Strong financials; the valuation is not "
                                             "conclusive (no DCF or dated price, or price "
                                             "close to the model value).")
    if key == "WATCH":
        return "BELOW-AVERAGE FINANCIALS", "Financial health is below the universe median."
    if key == "NODATA":
        return "NOT SCORED", "No overall health score (e.g. banks are not scored overall)."
    return "AVERAGE", "Average financial health and valuation."


def position_spread(weights_pct: pd.Series, target: int = SPREAD_TARGET) -> tuple[float, float]:
    """(score 0-100, effective number of positions). Effective number = 1 / sum(w^2)
    (inverse Herfindahl-Hirschman index); score = effective / target x 100, capped."""
    w = weights_pct.dropna() / 100
    if w.empty or w.sum() <= 0:
        return float("nan"), float("nan")
    w = w / w.sum()
    effective = float(1 / (w ** 2).sum())
    return float(min(100.0, effective / target * 100)), effective


# ----------------------------------------------------------------------------- analysis
def analyse_portfolio(holdings: pd.DataFrame, advisor: pd.DataFrame,
                      reporting_currency: str | None = None,
                      live_prices: dict[str, tuple[float, date]] | None = None,
                      broker_as_of: date | None = None,
                      today: date | None = None) -> dict[str, Any]:
    """``holdings`` from :func:`parse_holdings`; ``advisor`` from ``advisor_frame``.

    Only holdings in ``reporting_currency`` are aggregated (default: the currency with
    the largest cost basis); the others are listed as excluded - INR and USD are never
    added together."""
    today = today or date.today()
    live_prices = live_prices or {}
    by_ticker = {r["ticker"]: r for _, r in advisor.iterrows()} if len(advisor) else {}
    rows = []
    for _, h in holdings.iterrows():
        r = by_ticker.get(h["symbol"])
        currency, currency_basis = holding_currency(h["symbol"], r)

        def g(col, default=np.nan):
            return r[col] if (r is not None and col in r and not pd.isna(r[col])) else default

        sel = select_price(h.get("ltp"), broker_as_of, live_prices.get(h["symbol"]),
                           (g("price", None), g("price_date", None)), today)
        invested = h["quantity"] * h["avg_price"]
        value = h["quantity"] * sel["price"] if sel["price"] is not None else np.nan
        rows.append({
            "Symbol": h["symbol"], "Company": g("name", h["symbol"]),
            "Sector": g("sector", "Unknown"), "Currency": currency,
            "Currency basis": currency_basis, "Qty": h["quantity"], "Avg cost": h["avg_price"],
            "Price": sel["price"] if sel["price"] is not None else np.nan,
            "Price source": sel["source"], "Price date": sel["as_of"],
            "Price status": sel["status"], "Price note": sel["note"] or (sel["warning"] or ""),
            "Price warning": sel["warning"],
            "Invested": invested, "Value": value,
            "P&L": value - invested if not pd.isna(value) else np.nan,
            "P&L %": (value / invested - 1) * 100 if not pd.isna(value) else np.nan,
            "Health": g("score"), "Valuation": g("val_label", None),
            "Verdict": g("headline", "Not analysed"),
            "_key": r["decision"] if r is not None else None,
        })
    allt = pd.DataFrame(rows)
    currencies = allt.groupby("Currency")["Invested"].sum().sort_values(ascending=False) \
        if len(allt) else pd.Series(dtype=float)
    if reporting_currency is None and len(currencies):
        reporting_currency = currencies.index[0]
    t = allt[allt["Currency"] == reporting_currency].copy()
    excluded = allt[allt["Currency"] != reporting_currency]

    priced = t["Value"].notna()
    total = float(t.loc[priced, "Value"].sum())
    t["Weight %"] = np.where(priced, t["Value"] / total * 100 if total else np.nan, np.nan)
    acts = [_action(k, w) for k, w in zip(t["_key"], t["Weight %"])]
    t["Action"], t["Why"] = [a[0] for a in acts], [a[1] for a in acts]

    analysed = t[priced & t["Health"].notna()]
    analysed_value = float(analysed["Value"].sum())
    coverage = analysed_value / total if total else 0.0
    w_health = (float(np.average(analysed["Health"], weights=analysed["Value"]))
                if len(analysed) else None)
    spread, effective = position_spread(t["Weight %"])
    sector_w = (t[priced].groupby("Sector")["Weight %"].sum().sort_values(ascending=False)
                if priced.any() else pd.Series(dtype=float))
    _, sector_effective = position_spread(sector_w)
    cw = CONFIG["composite_weights"]
    composite = None
    if w_health is not None and not np.isnan(spread) and coverage >= MIN_COVERAGE:
        composite = round(cw["health"] * w_health + cw["position_spread"] * spread, 1)
    invested_priced = float(t.loc[priced, "Invested"].sum())
    summary = {
        "currency": reporting_currency, "currencies": currencies.to_dict(),
        "excluded_other_currency": int(len(excluded)),
        "n": int(len(t)), "n_priced": int(priced.sum()), "n_unpriced": int((~priced).sum()),
        "n_analysed": int(len(analysed)),
        "invested": float(t["Invested"].sum()), "invested_priced": invested_priced,
        # share of the cost basis that has a current price (the weight denominator)
        "priced_cost_share": (invested_priced / float(t["Invested"].sum())
                              if float(t["Invested"].sum()) else None),
        "weight_basis": "priced holdings only" if (~priced).any() else "all holdings",
        "cost_unpriced": float(t.loc[~priced, "Invested"].sum()),
        "value": total if priced.any() else None,
        "pnl": (total - invested_priced) if priced.any() else None,
        "pnl_pct": ((total / invested_priced - 1) * 100) if invested_priced else None,
        "analysed_coverage": coverage, "unanalysed_coverage": 1 - coverage if total else None,
        "weighted_health": w_health, "position_spread": None if np.isnan(spread) else
        round(spread, 1), "effective_positions": None if np.isnan(effective) else
        round(effective, 1), "largest_position": float(t["Weight %"].max())
        if priced.any() else None, "largest_position_name": (
            t.loc[t["Weight %"].idxmax(), "Company"] if priced.any() else None),
        "n_sectors": int(len(sector_w)), "largest_sector": (sector_w.index[0]
                                                            if len(sector_w) else None),
        "largest_sector_pct": float(sector_w.iloc[0]) if len(sector_w) else None,
        "effective_sectors": None if np.isnan(sector_effective) else round(sector_effective, 1),
        "health": composite, "composite_withheld": composite is None and w_health is not None,
        "weak_pct": float(t.loc[t["_key"].isin(WEAK_KEYS), "Weight %"].sum()),
        "expensive_pct": float(t.loc[t["Valuation"] == "Overvalued", "Weight %"].sum()),
        "sector_weights": sector_w,
        # backwards-compatible name used by older code paths
        "diversification": None if np.isnan(spread) else round(spread, 1),
    }
    return {"table": t.drop(columns=["_key"]), "excluded": excluded.drop(columns=["_key"]),
            "summary": summary, "alerts": build_alerts(t, sector_w, summary)}


def build_alerts(t: pd.DataFrame, sector_w: pd.Series,
                 summary: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    alerts: list[dict[str, Any]] = []
    summary = summary or {}

    def add(sev, title, detail, tickers=()):
        alerts.append({"severity": sev, "title": title, "detail": detail,
                       "tickers": list(tickers)})

    if summary.get("excluded_other_currency"):
        add("High", "Holdings in another currency excluded",
            f"{summary['excluded_other_currency']} holding(s) are not in "
            f"{summary.get('currency')}. Amounts in different currencies are never added "
            "(no FX conversion); switch the reporting currency to review them.")
    unpriced = t[t["Value"].isna()]
    if len(unpriced):
        add("High" if len(unpriced) * 2 >= len(t) else "Medium",
            "No reliable current price",
            f"{len(unpriced)} holding(s) have no fresh price, so their value, P&L and weight "
            "are N/A (cost basis is shown separately, never as market value). Fetch live "
            "prices or upload a dated broker file.", unpriced["Symbol"])
    if summary.get("analysed_coverage") is not None and summary.get("value") and \
            summary["analysed_coverage"] < MIN_COVERAGE:
        add("Medium", "Partial assessment",
            f"Only {summary['analysed_coverage']:.0%} of the priced value has a financial-"
            f"health score (minimum {MIN_COVERAGE:.0%} for the composite score).")
    for _, r in t[t["_key"].isna()].iterrows():
        add("Low", f"{r['Symbol']} is not analysed yet",
            "Enter the ticker in the sidebar ('Fetch & analyse'). NSE stocks need .NS.",
            [r["Symbol"]])
    for _, r in t[t["_key"].isin(WEAK_KEYS)].iterrows():
        weight = r["Weight %"]
        sev = "High" if (not pd.isna(weight) and weight >= 5) else "Medium"
        extra = ""
        if pd.notna(r["P&L %"]) and r["P&L %"] < -25:
            extra = f" It is also down {abs(r['P&L %']):.0f}% from the average cost."
        hs = f"health score {r['Health']:.0f}/100" if pd.notna(r["Health"]) else "weak fundamentals"
        size = f"{weight:.1f}% of your portfolio" if not pd.isna(weight) else "position size N/A"
        add(sev, f"Weak financial health: {r['Company']}", f"{hs}, {size}.{extra}",
            [r["Symbol"]])
    for _, r in t[(t["Valuation"] == "Overvalued") & (t["Weight %"] > 10)
                  & (~t["_key"].isin(WEAK_KEYS))].iterrows():
        add("Medium", f"Large holding with a high valuation: {r['Company']}",
            f"{r['Weight %']:.0f}% of the portfolio and priced above what its fundamentals "
            "support (above its model value).", [r["Symbol"]])
    for _, r in t[(t["P&L %"] > 100) & (t["Valuation"] == "Overvalued")].iterrows():
        add("Low", f"Large gain, now above model value: {r['Company']}",
            f"Up {r['P&L %']:.0f}% and now priced above its model value.", [r["Symbol"]])
    for _, r in t[t["Weight %"] > MAX_STOCK_PCT].iterrows():
        add("High", f"Over-concentrated: {r['Company']}",
            f"{r['Weight %']:.0f}% of the portfolio is in one stock (illustrative limit "
            f"{MAX_STOCK_PCT:.0f}%).", [r["Symbol"]])
    for sec, w in sector_w.items():
        if sec != "Unknown" and w > MAX_SECTOR_PCT:
            add("High", f"Sector concentration: {sec}",
                f"{w:.0f}% of the portfolio is in {sec} (illustrative limit "
                f"{MAX_SECTOR_PCT:.0f}%).", t.loc[t["Sector"] == sec, "Symbol"])
    for _, r in t[t["Price warning"].notna()].iterrows():
        add("Low", f"Price sources disagree: {r['Symbol']}", r["Price warning"], [r["Symbol"]])
    if len(t) < MIN_HOLDINGS:
        add("Medium", "Few holdings",
            f"Only {len(t)} stocks (illustrative minimum {MIN_HOLDINGS}): one company's "
            "outcome dominates the portfolio.")
    order = {"High": 0, "Medium": 1, "Low": 2}
    return sorted(alerts, key=lambda a: order[a["severity"]])


def sample_holdings(advisor: pd.DataFrame) -> pd.DataFrame:
    """DEMONSTRATION DATA: a six-stock Indian portfolio. Quantities are invented and the
    average costs are synthetic (stored price x a fixed factor), so the P&L is illustrative
    only. The dashboard labels it as demo data."""
    picks = [("TCS", 40, 1.15), ("INFY", 60, 1.05), ("HDFCBANK", 90, 0.85),
             ("RELIANCE", 30, 0.95), ("TATASTEEL", 400, 0.7), ("ITC", 300, 1.1)]
    prices = dict(zip(advisor["ticker"], advisor["price"])) if len(advisor) else {}
    rows = []
    for sym, qty, factor in picks:
        p = prices.get(sym + ".NS")
        if not p or pd.isna(p):
            continue   # no stored price: the demo row is skipped, never given a made-up cost
        rows.append({"Instrument": sym, "Qty.": qty, "Avg. cost": round(float(p) * factor, 2)})
    return pd.DataFrame(rows, columns=["Instrument", "Qty.", "Avg. cost"])
