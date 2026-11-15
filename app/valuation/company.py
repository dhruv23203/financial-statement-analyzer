"""Company valuation: statement inputs -> FCFF -> forecast -> discount -> equity bridge.

FCFF is derived from the cash-flow statement (the standard indirect approximation):

    FCFF = operating cash flow + interest expense x (1 - tax rate) - capital expenditure

(operating cash flow is after interest, so after-tax interest is added back to obtain the
cash flow available to all capital providers). The tax rate is the market's statutory
rate from ``config/valuation_assumptions.json``.

Two discount-rate options, both applied to FCFF and followed by the same equity bridge:

* ``screening_valuation`` - a fixed, disclosed screening discount rate per market and
  investor profile (used for the universe-wide Quality & Value screen, where a beta is
  not available for every company). It is NOT a computed WACC and is labelled as such.
* ``full_valuation`` - WACC from CAPM (beta estimated from price history), cost of debt
  and market-value capital-structure weights.

Missing inputs are never replaced by zero: the result explains what is missing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from sqlalchemy import select

from app.database import repositories as repo
from app.database.models import RUN_SUCCESS, StandardizedFinancialFact
from app.services.context import AppContext
from app.valuation import capm
from app.valuation.dcf import (DCFResult, ValuationError, margin_of_safety,
                               project_cash_flows, value_firm)

INPUT_METRICS = ("total_revenue", "operating_cash_flow", "capital_expenditure",
                 "interest_expense", "interest_expense_non_operating", "total_debt",
                 "cash_and_equivalents", "cash_and_short_term_investments",
                 "minority_interest", "diluted_average_shares", "ordinary_shares_number",
                 "stockholders_equity")
FINANCIAL_SECTORS = ("Financial Services",)


@dataclass
class ValuationInputs:
    ticker: str
    currency: str | None
    sector: str | None
    periods: dict[date, dict[str, float]]
    price: float | None = None
    price_currency: str | None = None
    market_cap: float | None = None
    price_date: str | None = None

    @property
    def latest(self) -> dict[str, float]:
        return self.periods[max(self.periods)] if self.periods else {}

    @property
    def latest_period(self) -> date | None:
        return max(self.periods) if self.periods else None

    def previous(self) -> dict[str, float]:
        ordered = sorted(self.periods)
        return self.periods[ordered[-2]] if len(ordered) > 1 else {}


@dataclass
class CompanyValuation:
    status: str                       # OK | NOT_APPLICABLE | INSUFFICIENT_DATA | INVALID
    reason: str | None = None
    method: str | None = None         # "screening" | "CAPM/WACC"
    dcf: DCFResult | None = None
    base_fcff: float | None = None
    fcff_components: dict[str, Any] = field(default_factory=dict)
    growth: float | None = None
    growth_source: str | None = None
    discount_rate: float | None = None
    discount_note: str | None = None
    terminal_growth: float | None = None
    shares_source: str | None = None
    value_per_share: float | None = None
    price: float | None = None
    margin_of_safety: float | None = None
    upside: float | None = None
    wacc_detail: Any = None
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "OK"


# ----------------------------------------------------------------------------- loading
def load_inputs(ctx: AppContext, tickers: list[str] | None = None) -> dict[str, ValuationInputs]:
    """Valuation inputs for the latest analysed fetch run of each company (one query per
    table, so it is cheap for the whole universe)."""
    out: dict[str, ValuationInputs] = {}
    wanted = set(tickers) if tickers else None
    with ctx.db.session_scope() as session:
        run_to_ticker: dict[int, str] = {}
        for company in repo.list_companies(session):
            if wanted is not None and company.ticker not in wanted:
                continue
            run = repo.latest_analysis_run(session, company.id)
            if run is None:
                continue
            item = ValuationInputs(company.ticker, run.currency or company.currency,
                                   company.sector, {})
            snap = repo.latest_market_snapshot(session, company.id)
            if snap is not None and snap.status == RUN_SUCCESS:
                item.price = float(snap.price) if snap.price is not None else None
                item.market_cap = float(snap.market_cap) if snap.market_cap is not None else None
                item.price_currency = snap.currency
                item.price_date = snap.fetched_at.date().isoformat() if snap.fetched_at else None
            out[company.ticker] = item
            run_to_ticker[run.fetch_run_id] = company.ticker
        ids = list(run_to_ticker)
        for chunk in (ids[i:i + 500] for i in range(0, len(ids), 500)):
            for f in session.scalars(select(StandardizedFinancialFact).where(
                    StandardizedFinancialFact.fetch_run_id.in_(chunk),
                    StandardizedFinancialFact.metric_code.in_(INPUT_METRICS))):
                if f.value is None:
                    continue
                item = out[run_to_ticker[f.fetch_run_id]]
                item.periods.setdefault(f.fiscal_period_end, {})[f.metric_code] = float(f.value)
    return out


# ----------------------------------------------------------------------------- building blocks
def fcff_base(values: dict[str, float], tax_rate: float) -> tuple[float, dict[str, Any],
                                                                 list[str]]:
    """FCFF of one fiscal year from the cash-flow statement."""
    ocf, capex = values.get("operating_cash_flow"), values.get("capital_expenditure")
    if ocf is None or capex is None:
        missing = [n for n, v in (("operating cash flow", ocf), ("capital expenditure", capex))
                   if v is None]
        raise ValuationError(f"FCFF cannot be calculated: {', '.join(missing)} not reported.")
    interest = values.get("interest_expense", values.get("interest_expense_non_operating"))
    warnings = []
    debt = values.get("total_debt")
    if interest is None:
        if debt:
            warnings.append("Interest expense not reported although the company has debt: "
                            "the after-tax interest add-back is omitted (FCFF understated).")
        add_back = 0.0
    else:
        add_back = abs(interest) * (1 - tax_rate)
    fcff = ocf + add_back - abs(capex)
    parts = {"operating_cash_flow": ocf, "after_tax_interest": add_back,
             "capital_expenditure": -abs(capex), "fcff": fcff, "tax_rate": tax_rate}
    return fcff, parts, warnings


def revenue_growth(inputs: ValuationInputs, min_growth: float, max_growth: float
                   ) -> tuple[float, str, list[str]]:
    """Revenue CAGR between the first and last available year, kept negative when the
    business has shrunk, bounded to [min_growth, max_growth]."""
    points = sorted((p, v["total_revenue"]) for p, v in inputs.periods.items()
                    if v.get("total_revenue") is not None)
    if len(points) < 2:
        raise ValuationError("Fewer than two years of revenue: growth cannot be estimated.")
    (d0, r0), (d1, r1) = points[0], points[-1]
    years = (d1 - d0).days / 365.25
    if r0 <= 0 or r1 <= 0 or years < 1:
        raise ValuationError("Revenue history is not usable for a growth estimate.")
    cagr = (r1 / r0) ** (1 / years) - 1
    bounded = min(max(cagr, min_growth), max_growth)
    note = f"revenue CAGR {cagr:+.1%} over {years:.1f} years"
    warnings = []
    if bounded != cagr:
        warnings.append(f"Historical revenue CAGR {cagr:+.1%} bounded to {bounded:+.1%} "
                        f"(allowed range {min_growth:+.0%} to {max_growth:+.0%}).")
    return bounded, note, warnings


def diluted_shares(inputs: ValuationInputs) -> tuple[float | None, str | None]:
    latest = inputs.latest
    if latest.get("diluted_average_shares"):
        return latest["diluted_average_shares"], "diluted average shares (latest year)"
    if latest.get("ordinary_shares_number"):
        return latest["ordinary_shares_number"], "ordinary shares outstanding (not diluted)"
    return None, None


def cash_balance(values: dict[str, float]) -> float | None:
    """Cash and equivalents plus short-term investments where reported as one line."""
    if values.get("cash_and_short_term_investments") is not None:
        return values["cash_and_short_term_investments"]
    return values.get("cash_and_equivalents")


# ----------------------------------------------------------------------------- valuations
def _value(inputs: ValuationInputs, rate: float, method: str, rate_note: str,
           risk: str | None = None, wacc_detail: Any = None) -> CompanyValuation:
    a = capm.load_assumptions()
    cfg = a["forecast"]
    if inputs.sector in FINANCIAL_SECTORS:
        return CompanyValuation("NOT_APPLICABLE", "DCF is not used for financial companies: "
                                "their operating cash flow includes deposit and loan flows, so "
                                "FCFF is not meaningful.", method)
    if not inputs.periods:
        return CompanyValuation("INSUFFICIENT_DATA", "No stored statements.", method)
    try:
        market = capm.market_assumptions(inputs.currency)
        tax = market["statutory_tax_rate"]
        base, parts, warnings = fcff_base(inputs.latest, tax)
        if base <= 0:
            return CompanyValuation("INSUFFICIENT_DATA",
                                    f"Latest FCFF is negative ({base:,.0f}); a DCF projected "
                                    "from a negative base is not meaningful.", method,
                                    base_fcff=base, fcff_components=parts)
        growth, growth_note, g_warn = revenue_growth(inputs, cfg["min_growth"], cfg["max_growth"])
        if risk is not None:
            growth *= a["screening"]["growth_haircut"].get(risk, 0.8) if growth > 0 else 1.0
            growth_note += f", {risk} haircut on positive growth"
        terminal = market["terminal_growth"]
        flows = project_cash_flows(base, growth, terminal, cfg["explicit_years"],
                                   cfg["fade_years"])
        shares, shares_src = diluted_shares(inputs)
        latest = inputs.latest
        result = value_firm(flows, rate, terminal, cash=cash_balance(latest),
                            debt=latest.get("total_debt"), shares=shares,
                            minority_interest=latest.get("minority_interest"))
    except ValuationError as exc:
        return CompanyValuation("INVALID" if "must be greater" in str(exc)
                                else "INSUFFICIENT_DATA", str(exc), method)
    out = CompanyValuation("OK", None, method, result, base, parts, growth, growth_note, rate,
                           rate_note, terminal, shares_src, result.value_per_share,
                           warnings=warnings + g_warn + result.warnings,
                           wacc_detail=wacc_detail)
    if shares_src and "not diluted" in shares_src:
        out.warnings.append("Diluted share count not reported; ordinary shares used.")
    # Market comparison is optional: intrinsic value does not need the price.
    if inputs.price is not None and inputs.price_currency == inputs.currency:
        out.price = inputs.price
        out.margin_of_safety = margin_of_safety(out.value_per_share, inputs.price)
        if out.value_per_share and inputs.price:
            out.upside = out.value_per_share / inputs.price - 1
    elif inputs.price is not None:
        out.warnings.append(f"Price is in {inputs.price_currency} but statements are in "
                            f"{inputs.currency}: market comparison not shown (no FX "
                            "conversion).")
    return out


def screening_valuation(inputs: ValuationInputs, risk: str = "moderate") -> CompanyValuation:
    """FCFF valuation at a fixed screening discount rate (disclosed, not a computed WACC)."""
    rates = capm.load_assumptions()["screening"]["discount_rates"]
    if inputs.currency not in rates:
        return CompanyValuation("INSUFFICIENT_DATA", f"No screening discount rate for "
                                f"{inputs.currency}.", "screening")
    rate = rates[inputs.currency].get(risk, rates[inputs.currency]["moderate"])
    return _value(inputs, rate, "screening",
                  f"screening discount rate {rate:.0%} ({risk} profile, {inputs.currency}) - a "
                  "fixed assumption, not a computed WACC", risk=risk)


def full_valuation(inputs: ValuationInputs, beta: capm.BetaEstimate | None
                   ) -> CompanyValuation:
    """FCFF at WACC built from CAPM. Without a beta the WACC (and the valuation) is N/A."""
    if beta is None:
        return CompanyValuation("INSUFFICIENT_DATA", "Beta could not be estimated (price "
                                "history unavailable), so CAPM and WACC are not computed.",
                                "CAPM/WACC")
    a = capm.load_assumptions()
    try:
        market = capm.market_assumptions(inputs.currency)
        latest, previous = inputs.latest, inputs.previous()
        ke = capm.cost_of_equity(market["risk_free_rate"], beta.beta,
                                 market["equity_risk_premium"])
        interest = latest.get("interest_expense", latest.get("interest_expense_non_operating"))
        kd = capm.cost_of_debt(interest, latest.get("total_debt"), previous.get("total_debt"),
                               market["risk_free_rate"], a["cost_of_debt"]["max_spread"],
                               a["cost_of_debt"]["fallback_spread"])
        if inputs.market_cap and inputs.price_currency == inputs.currency:
            equity_value, basis = inputs.market_cap, "market capitalisation"
        else:
            equity_value, basis = latest.get("stockholders_equity"), "book equity (fallback)"
        w = capm.wacc(ke, kd.pre_tax, market["statutory_tax_rate"], equity_value,
                      latest.get("total_debt"), basis)
        w.warnings += kd.warnings
        if basis != "market capitalisation":
            w.warnings.append("Market capitalisation unavailable in the statement currency: "
                              "book equity used for the WACC weights.")
    except ValuationError as exc:
        return CompanyValuation("INSUFFICIENT_DATA", str(exc), "CAPM/WACC")
    out = _value(inputs, w.wacc, "CAPM/WACC",
                 f"WACC {w.wacc:.2%} = {w.equity_weight:.0%} x Ke {ke:.2%} + "
                 f"{w.debt_weight:.0%} x Kd after tax {w.cost_of_debt_after_tax:.2%}",
                 wacc_detail=w)
    out.warnings = w.warnings + out.warnings
    return out
