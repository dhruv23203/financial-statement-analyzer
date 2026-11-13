"""Portfolio Doctor: review a broker holdings export against the advisor verdicts.

Input: the holdings CSV/Excel a broker exports (Zerodha, Upstox, Groww, ICICI Direct, a
US broker ...) with at least a symbol, quantity and average-cost column; column names
are matched loosely. Output: an action per holding (ADD / HOLD / TRIM / REVIEW),
portfolio-level health and prioritised alerts. Pure functions over the advisor frame;
nothing is stored. Educational project - not investment advice.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

SYMBOL_COLS = ["symbol", "instrument", "ticker", "trading symbol", "stock symbol", "scrip",
               "stock", "company"]
QTY_COLS = ["quantity", "qty", "qty.", "shares", "units", "holding", "quantity available"]
COST_COLS = ["avg_price", "avg. cost", "avg cost", "avg price", "average price", "average cost",
             "average buy price", "buy price", "buy avg", "cost price", "purchase price",
             "avg_cost", "cost basis"]
LTP_COLS = ["ltp", "last price", "current price", "closing price", "close price", "cmp",
            "market price"]

MAX_STOCK_PCT = 25.0
MAX_SECTOR_PCT = 40.0
MIN_HOLDINGS = 5
WEAK_KEYS = ("AVOID", "TRAP")
EXPENSIVE_KEYS = ("WAIT", "NOADD")
BUY_KEYS = ("BUY", "BUY_CHECK")


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


def parse_holdings(df: pd.DataFrame, default_suffix: str = ".NS") -> pd.DataFrame:
    """Normalise a broker export to columns: symbol, quantity, avg_price, ltp."""
    sym, qty = _find(df.columns, SYMBOL_COLS), _find(df.columns, QTY_COLS)
    cost, ltp = _find(df.columns, COST_COLS), _find(df.columns, LTP_COLS)
    missing = [n for n, c in (("symbol/instrument", sym), ("quantity", qty),
                              ("average price", cost)) if c is None]
    if missing:
        raise ValueError("Could not find column(s): " + ", ".join(missing) +
                         f". Columns in your file: {list(df.columns)}")
    out = pd.DataFrame({
        "symbol": df[sym].map(lambda x: clean_symbol(x, default_suffix)),
        "quantity": pd.to_numeric(df[qty], errors="coerce"),
        "avg_price": pd.to_numeric(df[cost], errors="coerce"),
    })
    out["ltp"] = pd.to_numeric(df[ltp], errors="coerce").where(lambda x: x > 0) if ltp else np.nan
    out = out[(out["symbol"] != "") & (out["quantity"] > 0) & (out["avg_price"] > 0)]
    out["cost_total"] = out["quantity"] * out["avg_price"]
    g = out.groupby("symbol", as_index=False).agg(quantity=("quantity", "sum"),
                                                  cost_total=("cost_total", "sum"),
                                                  ltp=("ltp", "last"))
    g["avg_price"] = g["cost_total"] / g["quantity"]
    return g[["symbol", "quantity", "avg_price", "ltp"]].reset_index(drop=True)


def _action(key: str | None, weight: float) -> tuple[str, str]:
    if key is None:
        return "NO DATA", "Not in the database - analyse it from the sidebar first."
    if key in WEAK_KEYS:
        return "REVIEW / EXIT", "Weak fundamentals - check whether the reason you bought still holds."
    if weight > MAX_STOCK_PCT:
        return "TRIM", f"Position is {weight:.0f}% of the portfolio (limit {MAX_STOCK_PCT:.0f}%)."
    if key in EXPENSIVE_KEYS:
        if weight > 10:
            return "TRIM", "Expensive for its quality and already a large position."
        return "HOLD - DON'T ADD", "Price looks high; don't add more now."
    if key in BUY_KEYS:
        if weight < 10:
            return "ADD", "Strong fundamentals at a reasonable price and a small position."
        return "HOLD", "Strong holding and already well sized."
    if key == "WATCH":
        return "HOLD - WATCH", "Below-average financial health; don't add until it improves."
    if key == "NODATA":
        return "HOLD", "No overall health score (e.g. banks are not scored overall); judge on valuation."
    return "HOLD", "Average fundamentals; keep watching."


def analyse_portfolio(holdings: pd.DataFrame, advisor: pd.DataFrame) -> dict[str, Any]:
    """``holdings`` from :func:`parse_holdings`; ``advisor`` from ``advisor_frame``."""
    by_ticker = {r["ticker"]: r for _, r in advisor.iterrows()} if len(advisor) else {}
    rows = []
    for _, h in holdings.iterrows():
        r = by_ticker.get(h["symbol"])
        price = None
        if r is not None and r.get("price") and not pd.isna(r.get("price")):
            price = float(r["price"])
        elif pd.notna(h.get("ltp")):
            price = float(h["ltp"])
        known = price is not None
        price = price if known else float(h["avg_price"])
        invested, value = h["quantity"] * h["avg_price"], h["quantity"] * price

        def g(col, default=np.nan):
            return r[col] if (r is not None and col in r and not pd.isna(r[col])) else default

        rows.append({
            "Symbol": h["symbol"], "Company": g("name", h["symbol"]), "Sector": g("sector", "Unknown"),
            "Qty": h["quantity"], "Avg cost": h["avg_price"], "Price": price,
            "Invested": invested, "Value": value, "P&L": value - invested,
            "P&L %": (value / invested - 1) * 100 if known else np.nan,
            "Health": g("score"), "Valuation": g("val_label", None),
            "Verdict": g("headline", "Not analysed"),
            "_key": r["decision"] if r is not None else None,
        })
    t = pd.DataFrame(rows)
    total = t["Value"].sum()
    t["Weight %"] = t["Value"] / total * 100 if total else 0.0
    acts = [_action(k, w) for k, w in zip(t["_key"], t["Weight %"])]
    t["Action"], t["Why"] = [a[0] for a in acts], [a[1] for a in acts]

    known = t[t["Health"].notna()]
    w_health = float(np.average(known["Health"], weights=known["Value"])) if len(known) else None
    shares = (t["Weight %"] / 100).values
    hhi = float((shares ** 2).sum()) if len(t) else 1.0
    diversification = float(np.clip((1 / hhi) / 8 * 100, 0, 100))     # 8 equal positions = 100
    sector_w = t.groupby("Sector")["Weight %"].sum().sort_values(ascending=False)
    health = round(0.7 * w_health + 0.3 * diversification, 1) if w_health is not None else None
    summary = {
        "invested": float(t["Invested"].sum()), "value": float(total),
        "pnl": float(total - t["Invested"].sum()),
        "pnl_pct": float((total / t["Invested"].sum() - 1) * 100) if t["Invested"].sum() else 0.0,
        "n": len(t), "health": health, "weighted_health": w_health,
        "diversification": round(diversification, 1),
        "weak_pct": float(t.loc[t["_key"].isin(WEAK_KEYS), "Weight %"].sum()),
        "expensive_pct": float(t.loc[t["Valuation"] == "Overvalued", "Weight %"].sum()),
        "unknown_pct": float(t.loc[t["_key"].isna(), "Weight %"].sum()),
        "sector_weights": sector_w,
    }
    return {"table": t.drop(columns=["_key"]), "summary": summary,
            "alerts": build_alerts(t, sector_w)}


def build_alerts(t: pd.DataFrame, sector_w: pd.Series) -> list[dict[str, Any]]:
    alerts: list[dict[str, Any]] = []

    def add(sev, title, detail, tickers=()):
        alerts.append({"severity": sev, "title": title, "detail": detail,
                       "tickers": list(tickers)})

    for _, r in t[t["_key"].isna()].iterrows():
        add("Low", f"{r['Symbol']} is not analysed yet",
            "Enter the ticker in the sidebar ('Fetch & analyse'). NSE stocks need .NS.",
            [r["Symbol"]])
    for _, r in t[t["_key"].isin(WEAK_KEYS)].iterrows():
        sev = "High" if r["Weight %"] >= 5 else "Medium"
        extra = ""
        if pd.notna(r["P&L %"]) and r["P&L %"] < -25:
            extra = (f" It is also down {abs(r['P&L %']):.0f}% - check whether the business, "
                     "not just the price, has weakened.")
        hs = f"health score {r['Health']:.0f}/100" if pd.notna(r["Health"]) else "weak fundamentals"
        add(sev, f"Weak fundamentals: {r['Company']}",
            f"{hs}, {r['Weight %']:.1f}% of your portfolio.{extra}", [r["Symbol"]])
    for _, r in t[(t["Valuation"] == "Overvalued") & (t["Weight %"] > 10)
                  & (~t["_key"].isin(WEAK_KEYS))].iterrows():
        add("Medium", f"Expensive large holding: {r['Company']}",
            f"{r['Weight %']:.0f}% of the portfolio and priced above what its fundamentals "
            "support. Avoid adding; consider trimming.", [r["Symbol"]])
    for _, r in t[(t["P&L %"] > 100) & (t["Valuation"] == "Overvalued")].iterrows():
        add("Low", f"Big gain on an expensive stock: {r['Company']}",
            f"Up {r['P&L %']:.0f}% and now overvalued - you may want to book part of the profit.",
            [r["Symbol"]])
    for _, r in t[t["Weight %"] > MAX_STOCK_PCT].iterrows():
        add("High", f"Over-concentrated: {r['Company']}",
            f"{r['Weight %']:.0f}% of the portfolio is in one stock (limit {MAX_STOCK_PCT:.0f}%).",
            [r["Symbol"]])
    for sec, w in sector_w.items():
        if sec != "Unknown" and w > MAX_SECTOR_PCT:
            add("High", f"Sector concentration: {sec}",
                f"{w:.0f}% of the portfolio is in {sec} (limit {MAX_SECTOR_PCT:.0f}%).",
                t.loc[t["Sector"] == sec, "Symbol"])
    if len(t) < MIN_HOLDINGS:
        add("Medium", "Too few holdings",
            f"Only {len(t)} stocks. Fewer than {MIN_HOLDINGS} leaves you exposed to one bad outcome.")
    order = {"High": 0, "Medium": 1, "Low": 2}
    return sorted(alerts, key=lambda a: order[a["severity"]])


def sample_holdings(advisor: pd.DataFrame) -> pd.DataFrame:
    """A plausible six-stock Indian portfolio for the demo: average cost is derived from the
    stored price so the P&L looks realistic whatever the snapshot date."""
    picks = [("TCS", 40, 1.15), ("INFY", 60, 1.05), ("HDFCBANK", 90, 0.85),
             ("RELIANCE", 30, 0.95), ("TATASTEEL", 400, 0.7), ("ITC", 300, 1.1)]
    prices = dict(zip(advisor["ticker"], advisor["price"])) if len(advisor) else {}
    rows = []
    for sym, qty, factor in picks:
        p = prices.get(sym + ".NS")
        rows.append({"Instrument": sym, "Qty.": qty,
                     "Avg. cost": round(float(p) * factor, 2) if p and not pd.isna(p) else 100.0})
    return pd.DataFrame(rows)
