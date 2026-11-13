"""Investment Advisor layer: turns stored analyses into an investor's decision.

The existing pipeline answers "is this a healthy business?" (health score, grades, red
flags). This module adds the second half of an investment decision, "is the price fair?",
and combines the two into one verdict with reasons:

    quality tier  (health-score interpretation band, adjusted for HIGH red flags)
  + valuation     (P/E and P/B against sector medians, PEG, free-cash-flow yield and a
                   10-year DCF fair value from stored statements and the market snapshot)
  = decision      (BUY / WAIT / RESEARCH / HOLD / DON'T ADD / AVOID ...)

It only READS what the pipeline already stored (``screen_frame``, analysis-run summaries,
multi-year trends and market snapshots); nothing here writes to the database or changes
the existing scoring. Educational project - not investment advice.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import select

from app.database import repositories as repo
from app.database.models import RUN_SUCCESS, MarketSnapshot, TrendResult
from app.services.context import AppContext
from app.services.screener import screen_frame

# ----------------------------------------------------------------------------- assumptions
# Discount rate by investor profile and long-run (terminal) growth, per market.
DCF_ASSUMPTIONS = {
    "INR": {"rates": {"conservative": 0.13, "moderate": 0.12, "aggressive": 0.11}, "terminal": 0.05},
    "other": {"rates": {"conservative": 0.10, "moderate": 0.09, "aggressive": 0.08}, "terminal": 0.03},
}
GROWTH_HAIRCUT = {"conservative": 0.6, "moderate": 0.8, "aggressive": 1.0}
MAX_GROWTH = 0.15          # never project more than 15 % a year
DCF_YEARS = 10
MARGIN_OF_SAFETY = 0.20    # "buy below" = fair value less 20 %
MIN_SECTOR_PEERS = 5

# Piecewise-linear curves: x = measured value, y = 0..1 (1 = cheapest).
CURVES = {
    "dcf_mos": ([-50, -20, 0, 20, 40], [0, .15, .4, .75, 1]),          # % below fair value
    "pe_vs_sector": ([.5, .8, 1, 1.3, 1.8], [1, .85, .5, .25, 0]),     # ratio to sector median
    "pb_vs_sector": ([.5, .8, 1, 1.3, 1.8], [1, .85, .5, .25, 0]),
    "peg": ([.5, 1, 1.5, 2.5, 3.5], [1, .8, .5, .15, 0]),
    "fcf_yield": ([0, 2, 4, 6, 8], [0, .25, .55, .85, 1]),              # %
}
UNDERVALUED, FAIR, OVERVALUED, NO_DATA = "Undervalued", "Fairly valued", "Overvalued", "Not enough data"
FINANCIAL_SECTORS = ("Financial Services",)

# decision key -> (headline, colour)   (colours follow app/reporting/ui.py)
ACTIONS = {
    "BUY": ("BUY / ACCUMULATE", "#26a69a"),
    "BUY_CHECK": ("BUY - CHECK THE PRICE", "#9ccc65"),
    "WAIT": ("STRONG BUSINESS - WAIT FOR A BETTER PRICE", "#f9a825"),
    "RESEARCH": ("CHEAP BUT AVERAGE QUALITY - RESEARCH", "#f9a825"),
    "HOLD": ("HOLD / WATCHLIST", "#8b97a8"),
    "NOADD": ("HOLD - DON'T ADD AT THIS PRICE", "#fb8c00"),
    "WATCH": ("BELOW-AVERAGE QUALITY - WATCH ONLY", "#fb8c00"),
    "TRAP": ("AVOID - POSSIBLE VALUE TRAP", "#ef5350"),
    "AVOID": ("AVOID", "#ef5350"),
    "NODATA": ("INSUFFICIENT DATA", "#8b97a8"),
}
TIER_ORDER = ("high", "mid", "weak", "low")


# ----------------------------------------------------------------------------- helpers
def _curve(name: str, x: float) -> float:
    xs, ys = CURVES[name]
    return float(np.interp(x, xs, ys)) * 100


def dcf_fair_value(base_fcf: float, growth: float, rate: float, terminal: float,
                   shares: float) -> float:
    """Value per share: 5 years at ``growth``, 5 years fading to ``terminal``, then a
    terminal value (Gordon growth)."""
    pv, fcf = 0.0, base_fcf
    for t in range(1, DCF_YEARS + 1):
        g = growth if t <= 5 else growth + (terminal - growth) * (t - 5) / 5
        fcf *= 1 + g
        pv += fcf / (1 + rate) ** t
    pv += fcf * (1 + terminal) / (rate - terminal) / (1 + rate) ** DCF_YEARS
    return pv / shares


def _num(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(f) else f


def currency_symbol(code: str | None) -> str:
    return {"INR": "₹", "USD": "$", "EUR": "€", "GBP": "£"}.get(code or "", f"{code or ''} ")


# ----------------------------------------------------------------------------- data
def _extras(ctx: AppContext, tickers: list[str]) -> dict[str, dict[str, Any]]:
    """Per ticker: interpretation, HIGH red-flag count, revenue/net-income CAGR (%),
    latest price and share count from the stored market snapshot."""
    out: dict[str, dict[str, Any]] = {}
    with ctx.db.session_scope() as session:
        runs: dict[int, str] = {}
        for company in repo.list_companies(session):
            if company.ticker not in tickers:
                continue
            run = repo.latest_analysis_run(session, company.id)
            if run is None:
                continue
            s = run.summary or {}
            out[company.ticker] = {
                "interpretation": s.get("overall_interpretation"),
                "high_flags": int((s.get("red_flag_severity_counts") or {}).get("HIGH", 0)),
                "pros": s.get("pros") or [], "cons": s.get("cons") or [],
                "revenue_cagr": None, "net_income_cagr": None, "price": None, "shares": None,
                "price_currency": None,
            }
            runs[run.id] = company.ticker
            snap = repo.latest_market_snapshot(session, company.id)
            if snap is not None and snap.status == RUN_SUCCESS:
                out[company.ticker]["price"] = _num(snap.price)
                out[company.ticker]["shares"] = _num(snap.shares_outstanding)
                out[company.ticker]["price_currency"] = snap.currency
        ids = list(runs)
        for chunk in (ids[i:i + 500] for i in range(0, len(ids), 500)):
            for t in session.scalars(select(TrendResult).where(
                    TrendResult.analysis_run_id.in_(chunk),
                    TrendResult.comparison_type == "MULTI_YEAR",
                    TrendResult.metric_code.in_(["total_revenue", "net_income"]))):
                key = "revenue_cagr" if t.metric_code == "total_revenue" else "net_income_cagr"
                out[runs[t.analysis_run_id]][key] = _num(t.percentage_change)
    return out


# ----------------------------------------------------------------------------- valuation
def _value_row(r: pd.Series, sector_pe: float | None, sector_pb: float | None,
               risk: str) -> dict[str, Any]:
    is_fin = r.get("sector") in FINANCIAL_SECTORS
    pe, pb = _num(r.get("price_to_earnings")), _num(r.get("price_to_book"))
    mcap, price, shares = _num(r.get("market_cap")), _num(r.get("price")), _num(r.get("shares"))
    rev, fcf_m = _num(r.get("revenue")), _num(r.get("free_cash_flow_margin"))
    rev_g, ni_g = _num(r.get("revenue_cagr")), _num(r.get("net_income_cagr"))
    fcf = rev * fcf_m / 100 if (rev is not None and fcf_m is not None) else None
    comps: dict[str, dict[str, Any]] = {}
    notes: list[str] = []
    # Statements and market data must be in the same currency before they are mixed
    # (e.g. Infosys reports in USD while its NSE price is in INR).
    pc = r.get("price_currency")
    if isinstance(pc, str) and r.get("currency") and pc != r.get("currency"):
        notes.append(f"Statements are in {r.get('currency')} but the price is in {pc}; "
                     "DCF and cash-flow yield are skipped.")
        fcf = None

    def add(key, value, text):
        comps[key] = {"value": round(value, 2), "score": round(_curve(key, value)), "text": text}

    dcf = None
    if not is_fin and fcf and fcf > 0 and shares and price and rev_g is not None:
        market = "INR" if r.get("currency") == "INR" else "other"
        a = DCF_ASSUMPTIONS[market]
        rate, term = a["rates"].get(risk, a["rates"]["moderate"]), a["terminal"]
        growth = float(np.clip(rev_g / 100, 0, MAX_GROWTH)) * GROWTH_HAIRCUT.get(risk, 0.8)
        fair = dcf_fair_value(fcf, growth, rate, term, shares)
        if fair > 0:
            mos = (fair - price) / fair * 100
            dcf = {"fair_value": fair, "buy_below": fair * (1 - MARGIN_OF_SAFETY),
                   "upside": (fair / price - 1) * 100, "margin_of_safety": mos,
                   "growth": growth, "discount_rate": rate, "terminal_growth": term,
                   "base_fcf": fcf}
            add("dcf_mos", mos, f"Price is {abs(mos):.0f}% {'below' if mos >= 0 else 'above'} "
                                f"DCF fair value (upside {dcf['upside']:+.0f}%)")
    elif is_fin:
        notes.append("DCF is not used for financial companies (cash flow is not meaningful).")
    elif fcf is not None and fcf <= 0:
        notes.append("No DCF: free cash flow is negative.")

    if pe and sector_pe:
        add("pe_vs_sector", pe / sector_pe, f"P/E {pe:.1f} vs sector median {sector_pe:.1f}")
    elif pe is None:
        notes.append("P/E not available (loss-making or no market snapshot).")
    if is_fin and pb and sector_pb:
        add("pb_vs_sector", pb / sector_pb, f"P/B {pb:.2f} vs sector median {sector_pb:.2f}")
    peg = pe / ni_g if (pe and ni_g and ni_g > 0) else None
    if peg:
        add("peg", peg, f"PEG {peg:.2f} (P/E {pe:.1f} / {ni_g:.0f}% profit growth)")
    fcf_yield = fcf / mcap * 100 if (fcf is not None and mcap and not is_fin) else None
    if fcf_yield is not None:
        add("fcf_yield", fcf_yield, f"Free-cash-flow yield {fcf_yield:.1f}%")

    if len(comps) >= 2:
        score = round(float(np.mean([c["score"] for c in comps.values()])), 1)
        label = UNDERVALUED if score >= 65 else FAIR if score >= 40 else OVERVALUED
    else:
        score, label = None, NO_DATA
        notes.append("Fewer than two valuation checks were possible.")
    return {"val_score": score, "val_label": label, "components": comps, "dcf": dcf,
            "peg": peg, "fcf_yield": fcf_yield, "val_notes": notes,
            "fair_value": dcf["fair_value"] if dcf else None,
            "buy_below": dcf["buy_below"] if dcf else None,
            "upside": dcf["upside"] if dcf else None}


# ----------------------------------------------------------------------------- decision
def quality_tier(interpretation: str | None, high_flags: int) -> tuple[str | None, list[str]]:
    """'high' / 'mid' / 'weak' / 'low' from the health-score band (top quartile, above
    median, below median, bottom quartile); HIGH red flags push the tier down one step."""
    if not isinstance(interpretation, str) or not interpretation:
        return None, []
    tier = {"TOP QUARTILE": "high", "ABOVE MEDIAN": "mid",
            "BELOW MEDIAN": "weak"}.get(interpretation, "low")
    why = [f"Financial health is {interpretation.lower()} of the analysed universe."]
    if high_flags:
        idx = min(TIER_ORDER.index(tier) + 1, len(TIER_ORDER) - 1)
        if TIER_ORDER[idx] != tier:
            why.append(f"{high_flags} HIGH-severity red flag(s) lower the quality tier.")
        tier = TIER_ORDER[idx]
    return tier, why


def decide(tier: str | None, val: dict[str, Any], currency: str) -> dict[str, Any]:
    label = val["val_label"]
    if tier is None:
        key = "NODATA"
    elif tier == "high":
        key = {UNDERVALUED: "BUY", FAIR: "BUY", OVERVALUED: "WAIT"}.get(label, "BUY_CHECK")
    elif tier == "mid":
        key = {UNDERVALUED: "RESEARCH", OVERVALUED: "NOADD"}.get(label, "HOLD")
    elif tier == "weak":
        key = "AVOID" if label == OVERVALUED else "WATCH"
    else:
        key = "TRAP" if label == UNDERVALUED else "AVOID"
    why: list[str] = []
    d = val.get("dcf")
    if d and key == "BUY" and d["upside"] < -25:
        key = "BUY_CHECK"
        why.append(f"Multiples look fair, but the DCF puts the price {abs(d['upside']):.0f}% "
                   "above fair value - confirm the growth assumptions before buying.")
    if label != NO_DATA:
        why.append(f"Price looks {label.lower()} (valuation score {val['val_score']:.0f}/100).")
    else:
        why.append("Valuation could not be assessed, so the call rests on quality alone.")
    sym = currency_symbol(currency)
    if d:
        why.append(f"DCF fair value {sym}{d['fair_value']:,.0f} per share (assumes "
                   f"{d['growth'] * 100:.1f}% growth, {d['discount_rate'] * 100:.0f}% discount rate).")
        if key in ("BUY", "BUY_CHECK", "WAIT", "HOLD", "NOADD"):
            why.append(f"With a {MARGIN_OF_SAFETY * 100:.0f}% margin of safety, buy below "
                       f"{sym}{d['buy_below']:,.0f}.")
    headline, color = ACTIONS[key]
    return {"decision": key, "headline": headline, "color": color, "decision_why": why}


# ----------------------------------------------------------------------------- public
def advisor_frame(ctx: AppContext, risk: str = "moderate") -> pd.DataFrame:
    """One row per analysed company with valuation, quality tier and decision.

    Columns include: ticker, name, sector, currency, score, interpretation, high_flags,
    tier, price_to_earnings, price_to_book, peg, fcf_yield, price, fair_value, buy_below,
    upside, val_score, val_label, decision, headline, color, why (list of reasons),
    components (dict), dcf (dict or None)."""
    base = screen_frame(ctx)
    if base.empty:
        return base
    extras = _extras(ctx, list(base["ticker"]))
    frame = base.merge(pd.DataFrame.from_dict(extras, orient="index").rename_axis("ticker")
                       .reset_index(), on="ticker", how="left")
    # sector medians (fall back to the whole universe for small sectors)
    med_pe_all = frame["price_to_earnings"].median() if "price_to_earnings" in frame else None
    med_pb_all = frame["price_to_book"].median() if "price_to_book" in frame else None
    frame["_mkt"] = np.where(frame["currency"] == "INR", "INR", "other")
    sec_pe = frame.groupby(["_mkt", "sector"])["price_to_earnings"].agg(["median", "count"]) \
        if "price_to_earnings" in frame else pd.DataFrame()
    sec_pb = frame.groupby(["_mkt", "sector"])["price_to_book"].agg(["median", "count"]) \
        if "price_to_book" in frame else pd.DataFrame()

    def sector_median(table, sector, fallback):
        if sector in table.index and table.loc[sector, "count"] >= MIN_SECTOR_PEERS:
            return _num(table.loc[sector, "median"])
        return _num(fallback)

    rows = []
    for _, r in frame.iterrows():
        key = (r["_mkt"], r.get("sector"))
        val = _value_row(r, sector_median(sec_pe, key, med_pe_all),
                         sector_median(sec_pb, key, med_pb_all), risk)
        hf = _num(r.get("high_flags")) or 0
        tier, q_why = quality_tier(r.get("interpretation"), int(hf))
        dec = decide(tier, val, r.get("currency"))
        rows.append({**val, "tier": tier, **dec, "why": q_why + dec["decision_why"]})
    out = pd.concat([frame.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    return out.drop(columns=["decision_why", "_mkt"])


def ideas(frame: pd.DataFrame, keys: tuple[str, ...] = ("BUY", "BUY_CHECK"),
          limit: int = 25) -> pd.DataFrame:
    """Best-scored companies with a buy-type decision, for the ideas list."""
    f = frame[frame["decision"].isin(keys)]
    return f.sort_values(["score", "val_score"], ascending=False).head(limit)


def company_row(frame: pd.DataFrame, ticker: str) -> pd.Series | None:
    m = frame[frame["ticker"] == ticker]
    return m.iloc[0] if len(m) else None
