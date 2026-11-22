"""Cost of capital: CAPM cost of equity, cost of debt and WACC (pure functions).

    Cost of equity (CAPM)  Ke = Rf + beta x ERP
    After-tax cost of debt Kd_at = Kd x (1 - t)
    WACC                   = E/(D+E) x Ke + D/(D+E) x Kd_at

Rf, ERP and the tax rate come from ``config/valuation_assumptions.json`` (each with a
source and as-of date) and must belong to the same market and currency as the company's
cash flows. Beta is the OLS slope of the stock's periodic returns on its market
benchmark's returns (estimation period and benchmark recorded with the result).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.valuation.dcf import ValuationError
from config.settings import CONFIG_DIR

ASSUMPTIONS_FILE = CONFIG_DIR / "valuation_assumptions.json"


@lru_cache(maxsize=4)
def load_assumptions(path: str | None = None) -> dict[str, Any]:
    return json.loads(Path(path or ASSUMPTIONS_FILE).read_text(encoding="utf-8"))


def market_assumptions(currency: str | None, path: str | None = None) -> dict[str, Any]:
    """Market parameters for the cash-flow currency; raises if the market is not covered
    (a USD risk-free rate is never used for INR cash flows or vice versa)."""
    markets = load_assumptions(path)["markets"]
    if currency not in markets:
        raise ValuationError(f"No market assumptions for currency {currency!r}; CAPM inputs "
                             f"are configured for {', '.join(markets)} only.")
    return markets[currency]


@dataclass
class BetaEstimate:
    beta: float
    blume_adjusted: float
    r_squared: float
    observations: int
    benchmark: str
    period: str
    interval: str
    price_basis: str = "close prices (basis not recorded)"


def estimate_beta(stock_prices: pd.Series, market_prices: pd.Series, benchmark: str,
                  period: str = "5y", interval: str = "1mo",
                  min_observations: int = 36,
                  price_basis: str = "close prices (basis not recorded)") -> BetaEstimate:
    """OLS beta of the stock's simple returns on the benchmark's, over common dates only
    (periods missing in either series are dropped, not filled)."""
    joined = pd.concat([stock_prices.rename("s"), market_prices.rename("m")], axis=1,
                       join="inner", sort=True).dropna()
    returns = joined.pct_change().dropna()
    n = len(returns)
    if n < min_observations:
        raise ValuationError(f"Only {n} common return observations against {benchmark} "
                             f"(need {min_observations}); beta not estimated.")
    if not np.isfinite(returns.to_numpy()).all():
        raise ValuationError("Non-finite returns in the price series; beta not estimated.")
    var_m = float(returns["m"].var(ddof=1))
    if var_m == 0:
        raise ValuationError("Benchmark returns have zero variance; beta undefined.")
    beta = float(returns["s"].cov(returns["m"]) / var_m)
    r2 = float(np.corrcoef(returns["s"], returns["m"])[0, 1] ** 2)
    return BetaEstimate(beta, 0.67 * beta + 0.33, r2, n, benchmark, period, interval,
                        price_basis)


def cost_of_equity(risk_free: float, beta: float, equity_risk_premium: float) -> float:
    """CAPM: Ke = Rf + beta x ERP."""
    return risk_free + beta * equity_risk_premium


@dataclass
class CostOfDebt:
    pre_tax: float
    method: str
    warnings: list[str] = field(default_factory=list)


def cost_of_debt(interest_expense: float | None, debt_now: float | None,
                 debt_previous: float | None, risk_free: float, max_spread: float = 0.10,
                 fallback_spread: float = 0.02) -> CostOfDebt:
    """Pre-tax cost of debt = interest expense / average total debt, accepted when it lies
    between the risk-free rate and risk-free + ``max_spread``; otherwise risk-free +
    ``fallback_spread`` with a warning (the substitution is always disclosed)."""
    debts = [d for d in (debt_now, debt_previous) if d is not None and d > 0]
    if interest_expense is not None and debts:
        implied = abs(interest_expense) / (sum(debts) / len(debts))
        if risk_free <= implied <= risk_free + max_spread:
            return CostOfDebt(implied, "interest expense / average total debt")
        fallback = risk_free + fallback_spread
        return CostOfDebt(fallback, f"risk-free + {fallback_spread:.0%} (fallback)",
                          [f"Implied cost of debt {implied:.2%} is outside the plausible range "
                           f"[{risk_free:.2%}, {risk_free + max_spread:.2%}]; using "
                           f"{fallback:.2%} instead."])
    fallback = risk_free + fallback_spread
    return CostOfDebt(fallback, f"risk-free + {fallback_spread:.0%} (fallback)",
                      ["Interest expense or debt not reported; cost of debt set to "
                       f"risk-free + {fallback_spread:.0%} = {fallback:.2%}."])


@dataclass
class WACCResult:
    wacc: float
    cost_of_equity: float
    cost_of_debt_pre_tax: float
    cost_of_debt_after_tax: float
    tax_rate: float
    equity_weight: float
    debt_weight: float
    equity_value_basis: str
    warnings: list[str] = field(default_factory=list)


def wacc(cost_of_equity_rate: float, pre_tax_cost_of_debt: float, tax_rate: float,
         equity_value: float | None, debt_value: float | None,
         equity_value_basis: str = "market capitalisation") -> WACCResult:
    """Weighted average cost of capital with capital-structure weights E/(D+E), D/(D+E)."""
    if equity_value is None or equity_value <= 0:
        raise ValuationError("Equity value for the WACC weights is missing or not positive.")
    if debt_value is None:
        raise ValuationError("Total debt is missing: WACC weights cannot be calculated.")
    if not 0 <= tax_rate < 1:
        raise ValuationError(f"Tax rate {tax_rate} must be between 0 and 1.")
    total = equity_value + max(debt_value, 0.0)
    we, wd = equity_value / total, max(debt_value, 0.0) / total
    kd_at = pre_tax_cost_of_debt * (1 - tax_rate)
    return WACCResult(we * cost_of_equity_rate + wd * kd_at, cost_of_equity_rate,
                      pre_tax_cost_of_debt, kd_at, tax_rate, we, wd, equity_value_basis)
