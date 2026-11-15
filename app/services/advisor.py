"""Quality & Value screen: combines stored analyses with a valuation check.

The existing pipeline answers "is this a healthy business?" (health score, grades, red
flags). This module adds the second half of an investment decision, "is the price fair?",
and combines the two into one descriptive screening result with reasons:

    quality tier  (health-score interpretation band, adjusted for HIGH red flags)
  + valuation     (P/E and P/B against sector medians, PEG, free-cash-flow yield and a
                   10-year DCF fair value from stored statements and the market snapshot)
  = result        (e.g. "Strong financials - price below model value"; internal keys
                   BUY / WAIT / ... are kept for the logic but never shown to users)

It only READS what the pipeline already stored (``screen_frame``, analysis-run summaries,
multi-year trends and market snapshots); nothing here writes to the database or changes
the existing scoring. The results are descriptive screening labels, deliberately not
buy/sell recommendations: this is an educational project, not investment advice, and
giving personalised buy/sell advice to the public requires regulatory registration
(e.g. SEBI Investment Adviser / Research Analyst in India).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import select

from app.database import repositories as repo
from app.database.models import RUN_SUCCESS, TrendResult
from app.services.context import AppContext
from app.services.screener import screen_frame
from app.valuation.capm import load_assumptions
from app.valuation.company import ValuationInputs, load_inputs, screening_valuation

# ----------------------------------------------------------------------------- assumptions
# All valuation assumptions (screening discount rates, growth bounds, margin of safety,
# valuation-score weights) live in config/valuation_assumptions.json with their sources.
_CFG = load_assumptions()
MARGIN_OF_SAFETY = float(_CFG["screening"]["margin_of_safety"])
SCORE_CFG = _CFG["valuation_score"]
MIN_SECTOR_PEERS = int(SCORE_CFG["min_sector_peers"])

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

# internal decision key -> (descriptive label shown to users, colour)
# Labels describe what the numbers show; they are not buy/sell instructions.
ACTIONS = {
    "BUY": ("Strong financials · price below model value", "#26a69a"),
    "BUY_CHECK": ("Strong financials · valuation not conclusive", "#9ccc65"),
    "WAIT": ("Strong financials · price above model value", "#f9a825"),
    "RESEARCH": ("Average financials · low valuation", "#f9a825"),
    "HOLD": ("Average financials · fair valuation", "#8b97a8"),
    "NOADD": ("Average financials · high valuation", "#fb8c00"),
    "WATCH": ("Below-average financials", "#fb8c00"),
    "TRAP": ("Weak financials · low valuation (value-trap profile)", "#ef5350"),
    "AVOID": ("Weak financials", "#ef5350"),
    "NODATA": ("Insufficient data", "#8b97a8"),
}
TIER_ORDER = ("high", "mid", "weak", "low")


# ----------------------------------------------------------------------------- helpers
def _curve(name: str, x: float) -> float:
    xs, ys = CURVES[name]
    return float(np.interp(x, xs, ys)) * 100


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
                "price_currency": None, "price_date": None,
            }
            runs[run.id] = company.ticker
            snap = repo.latest_market_snapshot(session, company.id)
            if snap is not None and snap.status == RUN_SUCCESS:
                out[company.ticker]["price"] = _num(snap.price)
                out[company.ticker]["shares"] = _num(snap.shares_outstanding)
                out[company.ticker]["price_currency"] = snap.currency
                out[company.ticker]["price_date"] = (snap.fetched_at.date().isoformat()
                                                     if snap.fetched_at else None)
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
def _value_row(r: pd.Series, peer_pe: tuple[float | None, str | None],
               peer_pb: tuple[float | None, str | None], risk: str,
               inputs: ValuationInputs | None) -> dict[str, Any]:
    """Valuation checks for one company. ``peer_pe`` / ``peer_pb`` are (median, basis)
    where basis says which peer group was used (or why none was available)."""
    is_fin = r.get("sector") in FINANCIAL_SECTORS
    pe, pb = _num(r.get("price_to_earnings")), _num(r.get("price_to_book"))
    mcap = _num(r.get("market_cap"))
    rev, fcf_m = _num(r.get("revenue")), _num(r.get("free_cash_flow_margin"))
    ni_g = _num(r.get("net_income_cagr"))
    fcf = rev * fcf_m / 100 if (rev is not None and fcf_m is not None) else None
    comps: dict[str, dict[str, Any]] = {}
    notes: list[str] = []
    pc = r.get("price_currency")
    if isinstance(pc, str) and r.get("currency") and pc != r.get("currency"):
        notes.append(f"Statements are in {r.get('currency')} but the price is in {pc}; "
                     "cash-flow yield and the DCF market comparison are skipped (no FX).")
        fcf = None

    def add(key, value, text):
        comps[key] = {"value": round(value, 2), "score": round(_curve(key, value)), "text": text,
                      "weight": float(SCORE_CFG["weights"].get(key, 0.0))}

    # ---- DCF: FCFF at a disclosed screening discount rate, then the equity bridge
    dcf = None
    valuation = screening_valuation(inputs, risk) if inputs is not None else None
    if valuation is not None and valuation.ok:
        fair = valuation.value_per_share
        dcf = {"fair_value": fair,
               "buy_below": fair * (1 - MARGIN_OF_SAFETY) if fair and fair > 0 else None,
               "price": valuation.price,
               "upside": valuation.upside * 100 if valuation.upside is not None else None,
               "margin_of_safety": (valuation.margin_of_safety * 100
                                    if valuation.margin_of_safety is not None else None),
               "growth": valuation.growth, "growth_note": valuation.growth_source,
               "discount_rate": valuation.discount_rate,
               "discount_note": valuation.discount_note,
               "terminal_growth": valuation.terminal_growth, "base_fcf": valuation.base_fcff,
               "fcff_components": valuation.fcff_components,
               "enterprise_value": valuation.dcf.enterprise_value,
               "equity_value": valuation.dcf.equity_value, "bridge": valuation.dcf.bridge,
               "shares": valuation.dcf.shares, "shares_source": valuation.shares_source,
               "terminal_share": valuation.dcf.terminal_share,
               "warnings": valuation.warnings}
        if dcf["margin_of_safety"] is not None and fair > 0:
            mos = dcf["margin_of_safety"]
            add("dcf_mos", mos, f"Price is {abs(mos):.0f}% {'below' if mos >= 0 else 'above'} "
                                f"the DCF model value")
        elif dcf["price"] is None:
            notes.append("DCF model value calculated, but no current price in the statement "
                         "currency to compare it with.")
    elif valuation is not None:
        notes.append(f"DCF: {valuation.reason}")

    pe_med, pe_basis = peer_pe
    if pe and pe_med:
        add("pe_vs_sector", pe / pe_med, f"P/E {pe:.1f} vs {pe_basis} median {pe_med:.1f}")
        if pe_basis and pe_basis.startswith("market"):
            notes.append(f"Fewer than {MIN_SECTOR_PEERS} same-market sector peers: P/E compared "
                         f"with the {pe_basis} median instead (fallback).")
    elif pe is None:
        notes.append("P/E not available (loss-making or no market snapshot).")
    elif pe_med is None:
        notes.append("Not enough comparable companies in the same market for a P/E comparison.")
    pb_med, pb_basis = peer_pb
    if is_fin and pb and pb_med:
        add("pb_vs_sector", pb / pb_med, f"P/B {pb:.2f} vs {pb_basis} median {pb_med:.2f}")
    peg = pe / ni_g if (pe and ni_g and ni_g > 0) else None
    if peg:
        add("peg", peg, f"PEG {peg:.2f} (P/E {pe:.1f} / {ni_g:.0f}% profit CAGR)")
    fcf_yield = fcf / mcap * 100 if (fcf is not None and mcap and not is_fin) else None
    if fcf_yield is not None:
        add("fcf_yield", fcf_yield, f"Free-cash-flow yield {fcf_yield:.1f}%")

    # ---- composite: documented (illustrative) weights over the applicable checks
    applicable = ["pe_vs_sector", "peg"] + (["pb_vs_sector"] if is_fin
                                            else ["dcf_mos", "fcf_yield"])
    weights = SCORE_CFG["weights"]
    w_applicable = sum(float(weights.get(k, 0)) for k in applicable)
    w_used = sum(c["weight"] for c in comps.values())
    coverage = w_used / w_applicable if w_applicable else 0.0
    if len(comps) >= int(SCORE_CFG["min_components"]) and w_used > 0:
        score = round(sum(c["score"] * c["weight"] for c in comps.values()) / w_used, 1)
        label = UNDERVALUED if score >= 65 else FAIR if score >= 40 else OVERVALUED
        if coverage < 0.75:
            notes.append(f"Only {len(comps)} of {len(applicable)} valuation checks were possible "
                         f"({coverage:.0%} of the weight): compare this score with care.")
    else:
        score, label = None, NO_DATA
        notes.append(f"Fewer than {SCORE_CFG['min_components']} valuation checks were possible.")
    return {"val_score": score, "val_label": label, "components": comps, "dcf": dcf,
            "peg": peg, "fcf_yield": fcf_yield, "val_notes": notes,
            "val_components_used": len(comps), "val_components_applicable": len(applicable),
            "val_coverage": round(coverage, 3),
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
    sym = currency_symbol(currency)
    # The margin-of-safety price shown to the user gates the most positive results, so the
    # screen can never call a stock "below model value" at a price its own DCF calls too high:
    #   price <= buy below (fair value - margin of safety)  -> BUY
    #   buy below < price <= fair value                      -> BUY - CHECK THE PRICE
    #   price > fair value                                   -> WAIT FOR A BETTER PRICE
    if d and key in ("BUY", "BUY_CHECK") and d.get("price") is None:
        key = "BUY_CHECK"
        why.append("No current price in the statement currency, so the price cannot be "
                   "compared with the DCF model value.")
    elif d and key in ("BUY", "BUY_CHECK") and (d["fair_value"] or 0) <= 0:
        key = "BUY_CHECK"
        why.append("The DCF equity value is not positive at these inputs.")
    elif d and key in ("BUY", "BUY_CHECK"):
        price = d["price"]
        if price > d["fair_value"]:
            key = "WAIT"
            premium = (price / d["fair_value"] - 1) * 100
            why.append(f"The price {sym}{price:,.0f} is {premium:.0f}% above the DCF model "
                       f"value {sym}{d['fair_value']:,.0f}, although the multiples alone look "
                       f"{label.lower() if label != NO_DATA else 'acceptable'}.")
        elif price > d["buy_below"]:
            key = "BUY_CHECK"
            why.append(f"The price {sym}{price:,.0f} is below the DCF model value but above "
                       f"the margin-of-safety price {sym}{d['buy_below']:,.0f} (model value less "
                       f"{MARGIN_OF_SAFETY * 100:.0f}%).")
    if label != NO_DATA:
        why.append(f"Price looks {label.lower()} (valuation score {val['val_score']:.0f}/100).")
    else:
        why.append("Valuation could not be assessed, so the result rests on quality alone.")
    if d:
        why.append(f"DCF model value {sym}{d['fair_value']:,.0f} per share (FCFF, "
                   f"{d['growth'] * 100:+.1f}% growth, {d['discount_rate'] * 100:.0f}% "
                   "screening discount rate, equity bridge for cash and debt).")
        if key in ("BUY", "BUY_CHECK", "WAIT", "HOLD", "NOADD") and d.get("buy_below"):
            why.append(f"Margin-of-safety price ({MARGIN_OF_SAFETY * 100:.0f}% below model "
                       f"value): {sym}{d['buy_below']:,.0f}.")
    headline, color = ACTIONS[key]
    return {"decision": key, "headline": headline, "color": color, "decision_why": why}


# ----------------------------------------------------------------------------- public
def peer_median(frame: pd.DataFrame, column: str, market: str, sector: str | None
                ) -> tuple[float | None, str | None]:
    """Peer median with an explicit hierarchy - never mixing markets:
    1. same market and same sector (at least MIN_SECTOR_PEERS values);
    2. same market, all sectors (fallback, flagged);
    3. None when the market itself has too few values."""
    if column not in frame:
        return None, None
    same_market = frame[frame["_mkt"] == market]
    sector_values = same_market.loc[same_market["sector"] == sector, column].dropna()
    if sector is not None and len(sector_values) >= MIN_SECTOR_PEERS:
        return _num(sector_values.median()), f"{market} {sector} sector"
    market_values = same_market[column].dropna()
    if len(market_values) >= MIN_SECTOR_PEERS:
        return _num(market_values.median()), f"market-wide {market}"
    return None, None


def advisor_frame(ctx: AppContext, risk: str = "moderate",
                  inputs: dict[str, ValuationInputs] | None = None) -> pd.DataFrame:
    """One row per analysed company with valuation, quality tier and screening result.

    Columns include: ticker, name, sector, currency, score, interpretation, high_flags,
    tier, price_to_earnings, price_to_book, peg, fcf_yield, price, price_date, fair_value,
    buy_below, upside, val_score, val_label, val_coverage, decision, headline, color, why
    (list of reasons), components (dict), dcf (dict or None)."""
    base = screen_frame(ctx)
    if base.empty:
        return base
    extras = _extras(ctx, list(base["ticker"]))
    frame = base.merge(pd.DataFrame.from_dict(extras, orient="index").rename_axis("ticker")
                       .reset_index(), on="ticker", how="left")
    inputs = inputs if inputs is not None else load_inputs(ctx, list(frame["ticker"]))
    frame["_mkt"] = np.where(frame["currency"] == "INR", "INR", "USD")
    rows = []
    for _, r in frame.iterrows():
        val = _value_row(r, peer_median(frame, "price_to_earnings", r["_mkt"], r.get("sector")),
                         peer_median(frame, "price_to_book", r["_mkt"], r.get("sector")), risk,
                         inputs.get(r["ticker"]))
        hf = _num(r.get("high_flags")) or 0
        tier, q_why = quality_tier(r.get("interpretation"), int(hf))
        dec = decide(tier, val, r.get("currency"))
        rows.append({**val, "tier": tier, **dec, "why": q_why + dec["decision_why"]})
    out = pd.concat([frame.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    return out.drop(columns=["decision_why", "_mkt"])


def ideas(frame: pd.DataFrame, keys: tuple[str, ...] = ("BUY", "BUY_CHECK"),
          limit: int = 25) -> pd.DataFrame:
    """Best-scored companies whose result is 'strong financials' with a price at or near
    model value (internal keys BUY / BUY_CHECK), for the screen list."""
    f = frame[frame["decision"].isin(keys)]
    return f.sort_values(["score", "val_score"], ascending=False).head(limit)


def company_row(frame: pd.DataFrame, ticker: str) -> pd.Series | None:
    m = frame[frame["ticker"] == ticker]
    return m.iloc[0] if len(m) else None
