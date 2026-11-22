"""Valuation research on top of one DCF: sensitivity, scenarios and reverse DCF.

All three re-run the *same* engine (``project_cash_flows`` -> ``value_firm``) with one or
more inputs changed; the base FCFF, forecast shape, equity-bridge items and share count
are those of the valuation being analysed, so every number is reproducible from it.

* Sensitivity: value per share over a grid of discount rate x terminal growth. Cells where
  the rate is not above terminal growth are N/A (Gordon model undefined), never a number.
* Scenarios: bear / base / bull shift growth, discount rate and terminal growth by the
  amounts in ``config/valuation_assumptions.json`` ("scenarios"). They are assumption
  sets, not forecasts or probabilities.
* Reverse DCF: the explicit-period growth rate at which the model value equals the
  current market price - what the price "implies" - found by bisection. It describes
  market expectations under this model; it is not a prediction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pandas as pd

from app.valuation import capm
from app.valuation.company import CompanyValuation
from app.valuation.dcf import ValuationError, project_cash_flows, value_firm

DEFAULT_SCENARIOS = {
    "Bear": {"growth_delta": -0.03, "rate_delta": 0.01, "terminal_delta": -0.01},
    "Base": {"growth_delta": 0.0, "rate_delta": 0.0, "terminal_delta": 0.0},
    "Bull": {"growth_delta": 0.03, "rate_delta": -0.01, "terminal_delta": 0.005},
}
REVERSE_DCF_BRACKET = (-0.30, 0.50)   # search range for the implied growth rate


def _bridge_args(v: CompanyValuation) -> dict[str, Any]:
    b = v.dcf.bridge
    return {"cash": b.get("plus_cash"), "debt": b.get("minus_debt"),
            "minority_interest": b.get("minus_minority_interest"),
            "other_senior_claims": b.get("minus_other_senior_claims"),
            "non_operating_assets": b.get("plus_non_operating_assets"),
            "shares": v.dcf.shares}


def revalue(v: CompanyValuation, growth: float | None = None, rate: float | None = None,
            terminal: float | None = None) -> float | None:
    """Value per share with some inputs replaced; None when the model is undefined
    (e.g. rate <= terminal growth) - never a substitute number."""
    if not v.ok or v.dcf is None:
        return None
    cfg = capm.load_assumptions()["forecast"]
    g = v.growth if growth is None else growth
    r = v.discount_rate if rate is None else rate
    tg = v.terminal_growth if terminal is None else terminal
    if v.risk_free is not None and tg >= v.risk_free:
        return None            # perpetual growth at or above the risk-free rate: not reasonable
    try:
        flows = project_cash_flows(v.base_fcff, g, tg, cfg["explicit_years"], cfg["fade_years"])
        out = value_firm(flows, r, tg, **_bridge_args(v)).value_per_share
    except ValuationError:
        return None
    if out is None or not math.isfinite(out):
        return None
    # same date as the base valuation: rolled to the price date at this scenario's rate
    return out * (1 + r) ** v.roll_years if v.roll_years else out


def sensitivity_grid(v: CompanyValuation, rate_steps=(-0.02, -0.01, 0.0, 0.01, 0.02),
                     terminal_steps=(-0.01, -0.005, 0.0, 0.005, 0.01)) -> pd.DataFrame:
    """Value per share: rows = discount rate, columns = terminal growth (both absolute)."""
    if not v.ok:
        return pd.DataFrame()
    rates = [round(v.discount_rate + d, 4) for d in rate_steps]
    terms = [round(v.terminal_growth + d, 4) for d in terminal_steps]
    data = {tg: [revalue(v, rate=r, terminal=tg) for r in rates] for tg in terms}
    frame = pd.DataFrame(data, index=rates)
    frame.index.name, frame.columns.name = "discount_rate", "terminal_growth"
    return frame


@dataclass
class Scenario:
    name: str
    growth: float
    rate: float
    terminal: float
    value_per_share: float | None
    vs_price: float | None


def scenarios(v: CompanyValuation, definitions: dict[str, dict[str, float]] | None = None
              ) -> list[Scenario]:
    """Bear / base / bull with growth kept inside the configured forecast bounds."""
    if not v.ok:
        return []
    a = capm.load_assumptions()
    defs = definitions or a.get("scenarios", {}).get("cases") or DEFAULT_SCENARIOS
    lo, hi = a["forecast"]["min_growth"], a["forecast"]["max_growth"]
    out = []
    for name, d in defs.items():
        g = min(max(v.growth + d["growth_delta"], lo), hi)
        r, tg = v.discount_rate + d["rate_delta"], v.terminal_growth + d["terminal_delta"]
        if v.risk_free is not None and tg >= v.risk_free:
            tg = v.risk_free - 0.0025     # keep perpetual growth below the risk-free rate
        value = revalue(v, growth=g, rate=r, terminal=tg)
        vs = (value / v.price - 1) if (value is not None and v.price) else None
        out.append(Scenario(name, g, r, tg, value, vs))
    return out


@dataclass
class ReverseDCF:
    implied_growth: float | None
    model_growth: float
    price: float | None
    note: str


def reverse_dcf(v: CompanyValuation, price: float | None = None, tol: float = 1e-4
                ) -> ReverseDCF:
    """Explicit-period growth that makes the model value equal the price (bisection;
    value per share rises with growth when the base FCFF is positive)."""
    price = v.price if price is None else price
    if not v.ok:
        return ReverseDCF(None, math.nan, price, "No valid DCF to invert.")
    if price is None or not math.isfinite(price) or price <= 0:
        return ReverseDCF(None, v.growth, price, "No current price in the statement "
                          "currency: the implied growth cannot be computed.")
    lo, hi = REVERSE_DCF_BRACKET
    f_lo, f_hi = revalue(v, growth=lo), revalue(v, growth=hi)
    if f_lo is None or f_hi is None:
        return ReverseDCF(None, v.growth, price, "Model undefined at the search bounds.")
    if not f_lo <= price <= f_hi:
        side = "below" if price < f_lo else "above"
        bound = lo if price < f_lo else hi
        return ReverseDCF(None, v.growth, price,
                          f"The price is {side} the model value even at {bound:+.0%} annual "
                          "growth: the price is not explained by growth alone under these "
                          "discount-rate and margin assumptions.")
    for _ in range(100):
        mid = (lo + hi) / 2
        if revalue(v, growth=mid) < price:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    g = (lo + hi) / 2
    return ReverseDCF(g, v.growth, price,
                      f"The price implies {g:+.1%} a year for the first five years (fading to "
                      f"{v.terminal_growth:.1%}); the model uses {v.growth:+.1%}.")
