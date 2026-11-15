"""Discounted-cash-flow mathematics (pure functions, no database or network access).

Two valuation bases are supported and their discount rates are never mixed:

* **FCFF** (free cash flow to the firm) is discounted at **WACC** and gives the
  **enterprise value**; equity value is then obtained with the bridge
  ``EV + cash and non-operating assets - debt - minority interest - other senior claims``.
* **FCFE** (free cash flow to equity) is discounted at the **cost of equity** and gives the
  **equity value** directly (no bridge).

Conventions (stated so a manual calculation can reproduce every number):

* cash flows arrive at the **end** of each forecast year (year-end discounting):
  ``PV_t = CF_t / (1 + r)^t``;
* the terminal value is a Gordon growth perpetuity at the end of the last forecast year,
  ``TV_N = CF_N x (1 + g) / (r - g)``, discounted with the same factor as year N;
* the Gordon model is only valid when ``r > g``; otherwise a :class:`ValuationError`
  is raised instead of returning a meaningless number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

FCFF, FCFE = "FCFF", "FCFE"
WACC, COST_OF_EQUITY = "WACC", "COST_OF_EQUITY"
# The only valid (cash-flow basis -> discount rate) pairings.
VALID_PAIRS = {FCFF: WACC, FCFE: COST_OF_EQUITY}
MAX_TERMINAL_GROWTH = 0.08  # above this a perpetual growth rate is not economically plausible


class ValuationError(ValueError):
    """The inputs cannot produce a meaningful valuation (the message says why)."""


@dataclass
class DCFResult:
    basis: str
    rate_kind: str
    discount_rate: float
    terminal_growth: float
    cash_flows: list[float]
    discount_factors: list[float]
    pv_cash_flows: list[float]
    pv_forecast: float
    terminal_value: float
    pv_terminal_value: float
    enterprise_value: float | None        # FCFF only
    equity_value: float
    bridge: dict[str, float] = field(default_factory=dict)
    shares: float | None = None
    value_per_share: float | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def terminal_share(self) -> float:
        """Share of the total present value that comes from the terminal value."""
        total = self.pv_forecast + self.pv_terminal_value
        return self.pv_terminal_value / total if total else float("nan")


def check_pairing(basis: str, rate_kind: str) -> None:
    """FCFF must be discounted at WACC and FCFE at the cost of equity - nothing else."""
    if VALID_PAIRS.get(basis) != rate_kind:
        raise ValuationError(
            f"{basis} cannot be discounted at {rate_kind}: FCFF pairs with WACC (gives "
            "enterprise value) and FCFE pairs with the cost of equity (gives equity value).")


def project_cash_flows(base: float, growth: float, terminal_growth: float,
                       explicit_years: int = 5, fade_years: int = 5) -> list[float]:
    """Cash flows for years 1..N: ``explicit_years`` at ``growth``, then ``fade_years`` with
    the growth rate moving linearly from ``growth`` to ``terminal_growth``.
    Negative growth is allowed (a shrinking business is valued as shrinking)."""
    flows, cf = [], float(base)
    for t in range(1, explicit_years + fade_years + 1):
        if t <= explicit_years:
            g = growth
        else:
            step = t - explicit_years
            g = growth + (terminal_growth - growth) * step / fade_years
        cf *= 1 + g
        flows.append(cf)
    return flows


def terminal_value(last_cash_flow: float, rate: float, growth: float) -> float:
    """Gordon growth value at the end of the last forecast year."""
    if rate <= growth:
        raise ValuationError(
            f"Discount rate {rate:.2%} must be greater than terminal growth {growth:.2%}; "
            "the Gordon growth model is undefined (division by zero or a negative "
            "denominator) when it is not.")
    return last_cash_flow * (1 + growth) / (rate - growth)


def discount_cash_flows(cash_flows: Sequence[float], rate: float, terminal_growth: float,
                        basis: str, rate_kind: str) -> DCFResult:
    """Present value of the forecast and the terminal value (year-end convention)."""
    check_pairing(basis, rate_kind)
    if not cash_flows:
        raise ValuationError("No forecast cash flows.")
    if rate <= -1:
        raise ValuationError("Discount rate must be greater than -100%.")
    warnings = []
    if terminal_growth > MAX_TERMINAL_GROWTH:
        warnings.append(f"Terminal growth {terminal_growth:.1%} is above "
                        f"{MAX_TERMINAL_GROWTH:.0%}, which is not plausible in perpetuity.")
    if rate - terminal_growth < 0.01 and rate > terminal_growth:
        warnings.append("Discount rate is within one percentage point of terminal growth: "
                        "the terminal value is extremely sensitive to both inputs.")
    tv = terminal_value(cash_flows[-1], rate, terminal_growth)
    factors = [1 / (1 + rate) ** t for t in range(1, len(cash_flows) + 1)]
    pv_flows = [cf * f for cf, f in zip(cash_flows, factors)]
    pv_forecast = sum(pv_flows)
    pv_tv = tv * factors[-1]
    if cash_flows[-1] <= 0:
        warnings.append("The final forecast cash flow is not positive, so the terminal value "
                        "is not positive: the business does not create value at these inputs.")
    total = pv_forecast + pv_tv
    return DCFResult(basis, rate_kind, rate, terminal_growth, list(cash_flows), factors,
                     pv_flows, pv_forecast, tv, pv_tv,
                     enterprise_value=total if basis == FCFF else None,
                     equity_value=total, warnings=warnings)


def equity_bridge(enterprise_value: float, cash: float | None, debt: float | None,
                  minority_interest: float | None = 0.0, other_senior_claims: float | None = 0.0,
                  non_operating_assets: float | None = 0.0) -> tuple[float, dict[str, float],
                                                                      list[str]]:
    """Enterprise value -> equity value. Cash and debt are required: missing values are
    never assumed to be zero (a missing bridge item makes the equity value N/A)."""
    if cash is None:
        raise ValuationError("Cash and cash equivalents are missing: the enterprise-to-equity "
                             "bridge cannot be completed.")
    if debt is None:
        raise ValuationError("Total debt is missing: the enterprise-to-equity bridge cannot "
                             "be completed.")
    notes = []
    if minority_interest is None:
        notes.append("Minority interest not reported: treated as not applicable.")
        minority_interest = 0.0
    other_senior_claims = other_senior_claims or 0.0
    non_operating_assets = non_operating_assets or 0.0
    equity = (enterprise_value + cash + non_operating_assets - debt - minority_interest
              - other_senior_claims)
    bridge = {"enterprise_value": enterprise_value, "plus_cash": cash,
              "plus_non_operating_assets": non_operating_assets, "minus_debt": debt,
              "minus_minority_interest": minority_interest,
              "minus_other_senior_claims": other_senior_claims, "equity_value": equity}
    return equity, bridge, notes


def value_firm(fcff: Sequence[float], wacc: float, terminal_growth: float,
               cash: float | None, debt: float | None, shares: float | None,
               minority_interest: float | None = 0.0, other_senior_claims: float | None = 0.0,
               non_operating_assets: float | None = 0.0) -> DCFResult:
    """FCFF discounted at WACC -> enterprise value -> equity value -> value per share."""
    result = discount_cash_flows(fcff, wacc, terminal_growth, FCFF, WACC)
    equity, bridge, notes = equity_bridge(result.enterprise_value, cash, debt,
                                          minority_interest, other_senior_claims,
                                          non_operating_assets)
    result.equity_value, result.bridge = equity, bridge
    result.warnings += notes
    _per_share(result, shares)
    return result


def value_equity(fcfe: Sequence[float], cost_of_equity: float, terminal_growth: float,
                 shares: float | None) -> DCFResult:
    """FCFE discounted at the cost of equity -> equity value -> value per share."""
    result = discount_cash_flows(fcfe, cost_of_equity, terminal_growth, FCFE, COST_OF_EQUITY)
    _per_share(result, shares)
    return result


def _per_share(result: DCFResult, shares: float | None) -> None:
    if shares is None or shares <= 0:
        raise ValuationError("Diluted shares outstanding are missing or not positive: value "
                             "per share cannot be calculated.")
    result.shares = shares
    result.value_per_share = result.equity_value / shares
    if result.equity_value <= 0:
        result.warnings.append("Equity value is not positive: debt and other claims exceed "
                               "the value of operations at these inputs.")


def margin_of_safety(value_per_share: float | None, price: float | None) -> float | None:
    """(value - price) / value; None when either side is unavailable (price is optional)."""
    if value_per_share is None or price is None or value_per_share <= 0:
        return None
    return (value_per_share - price) / value_per_share
