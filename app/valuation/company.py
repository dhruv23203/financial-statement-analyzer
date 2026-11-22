"""Company valuation: statement inputs -> FCFF -> forecast -> discount -> equity bridge.

FCFF (unlevered free cash flow to the firm) is built with the conventional formula when
every input is reported:

    FCFF = EBIT x (1 - tax rate) + D&A - capital expenditure - increase in net working capital

EBIT here is *operating* EBIT (operating income; else provider EBIT less interest income).
The provider's own "EBIT" line includes other and interest income earned on cash, and the
equity bridge adds that cash separately, so using it would count the return on cash twice.

(the cash-flow statement's "change in working capital" already carries the cash sign, so
it is added: a negative value is an investment in working capital). Without EBIT, D&A or
the working-capital change the cash-flow route is used, and interest is added back only
when the reporting standard places interest paid in operating activities:

    FCFF ~ operating cash flow - capital expenditure [+ interest x (1 - tax rate)]

* USD filers (US GAAP, ASC 230): interest paid is an operating cash flow, so after-tax
  interest is added back;
* INR filers (Ind AS 7): non-financial companies normally classify interest paid as a
  financing cash flow, so operating cash flow is already before interest and adding it
  back would double-count - no add-back;
* otherwise the classification is unknown: no add-back, and the result is labelled
  "OCF - CapEx (approximation)" rather than validated FCFF.

The method and the interest classification are recorded with every result. The tax rate
is the market's statutory rate from ``config/valuation_assumptions.json``.

Two discount-rate options, both applied to FCFF and followed by the same equity bridge:

* ``screening_valuation`` - a fixed, disclosed screening discount rate per market and
  investor profile (used for the universe-wide Quality & Value screen, where a beta is
  not available for every company). It is NOT a computed WACC and is labelled as such.
* ``full_valuation`` - WACC from CAPM (beta estimated from price history), cost of debt
  and market-value capital-structure weights.

Missing inputs are never replaced by zero: the result explains what is missing.
"""

from __future__ import annotations

import math
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
                 "stockholders_equity", "ebit", "operating_income", "interest_income",
                 "pretax_income", "tax_provision",
                 "reconciled_depreciation",
                 "depreciation_amortization_cf", "change_in_working_capital")
# Where interest paid sits in the cash-flow statement, by reporting currency (proxy for
# the reporting standard). Anything else is "unknown" and gets no add-back.
INTEREST_CLASSIFICATION = {"USD": ("operating", "US GAAP (ASC 230): interest paid is an "
                                   "operating cash flow"),
                           "INR": ("financing", "Ind AS 7: interest paid by non-financial "
                                   "companies is normally a financing cash flow")}
# The current share count from the market snapshot is used unless it disagrees with the
# reported period-end count by more than this (e.g. a provider quoting one share class).
SHARE_COUNT_TOLERANCE = 0.25
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
    shares_outstanding: float | None = None   # current, from the market snapshot (basic)

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
    # dates: the DCF is computed at the balance-sheet date (valuation_date); when a dated
    # price exists the value is rolled forward to the price date at the discount rate
    # (value_per_share), so value and price refer to the same day.
    valuation_date: date | None = None
    price_date: str | None = None
    roll_years: float = 0.0
    value_per_share_at_valuation_date: float | None = None
    risk_free: float | None = None
    assumption_flags: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "OK"

    @property
    def terminal_share(self) -> float | None:
        """Present value of the terminal value as a share of enterprise value."""
        if self.dcf is None or not self.dcf.enterprise_value:
            return None
        return self.dcf.pv_terminal_value / self.dcf.enterprise_value

    @property
    def net_debt(self) -> float | None:
        b = (self.dcf.bridge if self.dcf else None) or {}
        if b.get("minus_debt") is None or b.get("plus_cash") is None:
            return None
        return b["minus_debt"] - b["plus_cash"]


# ----------------------------------------------------------------------------- loading
def load_inputs(ctx: AppContext, tickers: list[str] | None = None) -> dict[str, ValuationInputs]:
    """Valuation inputs for the latest analysed fetch run of each company (one query per
    table, so it is cheap for the whole universe)."""
    out: dict[str, ValuationInputs] = {}
    wanted = set(tickers) if tickers else None
    with ctx.db.session_scope() as session:
        run_to_ticker: dict[int, str] = {}
        runs = repo.latest_analysis_runs(session)                # batched: one query each
        snaps = repo.latest_market_snapshots(session)
        for company in repo.list_companies(session):
            if wanted is not None and company.ticker not in wanted:
                continue
            run = runs.get(company.id)
            if run is None:
                continue
            item = ValuationInputs(company.ticker, run.currency or company.currency,
                                   company.sector, {})
            snap = snaps.get(company.id)
            if snap is not None and snap.status == RUN_SUCCESS:
                item.price = float(snap.price) if snap.price is not None else None
                item.market_cap = float(snap.market_cap) if snap.market_cap is not None else None
                item.price_currency = snap.currency
                item.price_date = repo.snapshot_price_date(snap)
                item.shares_outstanding = (float(snap.shares_outstanding)
                                           if snap.shares_outstanding is not None else None)
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
def _finite(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def fcff_base(values: dict[str, float], tax_rate: float, currency: str | None = None
              ) -> tuple[float, dict[str, Any], list[str]]:
    """FCFF of one fiscal year. EBIT-based when EBIT, D&A, capex and the working-capital
    change are all reported; otherwise the cash-flow route with the interest add-back
    decided by the interest-paid classification (see the module docstring)."""
    if not 0 <= tax_rate < 1:
        raise ValuationError(f"Tax rate {tax_rate} must be between 0 and 1.")
    v = {k: _finite(x) for k, x in values.items()}
    capex = v.get("capital_expenditure")
    warnings: list[str] = []
    # Operating EBIT: the provider's "EBIT" is pre-tax income + interest expense, so it
    # includes other and interest income earned on cash. The equity bridge adds that cash
    # separately, so starting from it would count the return on cash twice. Operating
    # income is used; failing that, EBIT less reported interest income.
    ebit, ebit_source = v.get("operating_income"), "operating income"
    if ebit is None and v.get("ebit") is not None:
        if v.get("interest_income") is not None:
            ebit = v["ebit"] - abs(v["interest_income"])
            ebit_source = "EBIT less interest income (operating income not reported)"
        else:
            ebit = v["ebit"]
            ebit_source = "reported EBIT (may include non-operating income)"
            warnings.append("Operating income not reported: provider EBIT used, which may "
                            "include non-operating income also captured by the cash in the "
                            "equity bridge (value may be overstated).")
    da = v.get("reconciled_depreciation")
    if da is None:
        da = v.get("depreciation_amortization_cf")
    d_wc = v.get("change_in_working_capital")
    if capex is not None and None not in (ebit, da, d_wc):
        nopat = ebit * (1 - tax_rate) if ebit > 0 else ebit   # no tax shield on a loss
        fcff = nopat + abs(da) - abs(capex) + d_wc
        parts = {"method": "EBIT-based FCFF", "ebit": ebit, "ebit_source": ebit_source,
                 "nopat": nopat,
                 "depreciation_amortization": abs(da), "capital_expenditure": -abs(capex),
                 "change_in_working_capital": d_wc, "fcff": fcff, "tax_rate": tax_rate,
                 "interest_classification": "not needed (starts from EBIT)",
                 "after_tax_interest": 0.0,
                 "formula": "EBIT x (1 - t) + D&A - CapEx + change in working capital "
                            "(cash-flow sign)"}
        if ebit <= 0:
            warnings.append("EBIT is not positive: no tax is applied to the operating loss.")
        return fcff, parts, warnings

    ocf = v.get("operating_cash_flow")
    if ocf is None or capex is None:
        missing = [n for n, x in (("operating cash flow", ocf), ("capital expenditure", capex))
                   if x is None]
        raise ValuationError(f"FCFF cannot be calculated: {', '.join(missing)} not reported.")
    classification, basis = INTEREST_CLASSIFICATION.get(
        currency or "", ("unknown", "reporting standard not identified"))
    interest = v.get("interest_expense")
    if interest is None:
        interest = v.get("interest_expense_non_operating")
    add_back = 0.0
    if classification == "operating":
        if interest is None:
            if v.get("total_debt"):
                warnings.append("Interest expense not reported although the company has debt: "
                                "the after-tax interest add-back is omitted (FCFF understated).")
        else:
            add_back = abs(interest) * (1 - tax_rate)
        method = "Cash-flow FCFF (OCF + after-tax interest - CapEx)"
    elif classification == "financing":
        method = "Cash-flow FCFF (OCF - CapEx; interest already outside OCF)"
    else:
        method = "OCF - CapEx (approximation, not validated FCFF)"
        warnings.append("Interest-paid classification unknown: no interest add-back, so this "
                        "is operating cash flow less capex, not a fully validated FCFF.")
    missing_ebit = [n for n, x in (("EBIT", ebit), ("D&A", da),
                                   ("working-capital change", d_wc)) if x is None]
    warnings.append(f"EBIT-based FCFF not possible ({', '.join(missing_ebit)} not reported): "
                    f"cash-flow route used - {basis}.")
    fcff = ocf + add_back - abs(capex)
    parts = {"method": method, "operating_cash_flow": ocf, "after_tax_interest": add_back,
             "capital_expenditure": -abs(capex), "fcff": fcff, "tax_rate": tax_rate,
             "interest_classification": f"{classification} ({basis})"}
    return fcff, parts, warnings


def fcfe_base(values: dict[str, float], previous: dict[str, float], tax_rate: float,
              currency: str | None = None) -> tuple[float, dict[str, Any], list[str]]:
    """Free cash flow to equity of one fiscal year:

        FCFE = FCFF - interest expense x (1 - t) + net borrowing,
        net borrowing = total debt (this year) - total debt (previous year).

    Both debt balances are required (a missing balance is never treated as zero). FCFE
    pairs only with the cost of equity (``dcf.value_equity``)."""
    fcff, parts, warnings = fcff_base(values, tax_rate, currency)
    debt_now, debt_prev = _finite(values.get("total_debt")), _finite(previous.get("total_debt"))
    if debt_now is None or debt_prev is None:
        raise ValuationError("FCFE needs total debt for this and the previous year to "
                             "measure net borrowing; it is not assumed to be zero.")
    interest = _finite(values.get("interest_expense"))
    if interest is None:
        interest = _finite(values.get("interest_expense_non_operating"))
    if interest is None:
        if debt_now > 0:
            raise ValuationError("FCFE needs the interest expense of a company with debt.")
        interest = 0.0
    after_tax_interest = abs(interest) * (1 - tax_rate)
    net_borrowing = debt_now - debt_prev
    fcfe = fcff - after_tax_interest + net_borrowing
    return fcfe, {**parts, "fcff": fcff, "after_tax_interest_paid": after_tax_interest,
                  "net_borrowing": net_borrowing, "fcfe": fcfe}, warnings


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


def select_share_count(inputs: ValuationInputs) -> dict[str, Any]:
    """Share count for per-share value, in order of preference:

    1. current shares outstanding from the dated market snapshot - unless it differs from
       the reported period-end count by more than SHARE_COUNT_TOLERANCE (a sign that the
       provider quotes a single share class);
    2. ordinary shares outstanding at the latest balance-sheet date;
    3. diluted weighted-average shares of the latest fiscal year (a period average, not a
       point-in-time count - disclosed as such).

    No provider used here reports a current *diluted* count, so the chosen count is basic
    and the result says so. Returns shares, source, as_of, classification and warnings."""
    latest, period = inputs.latest, inputs.latest_period
    period_txt = period.isoformat() if period else "latest period"
    reported = _finite(latest.get("ordinary_shares_number"))
    current = _finite(inputs.shares_outstanding)
    diluted_avg = _finite(latest.get("diluted_average_shares"))
    warnings: list[str] = []
    if current and current > 0:
        if reported and reported > 0 and abs(current / reported - 1) > SHARE_COUNT_TOLERANCE:
            warnings.append(f"Market-snapshot share count ({current:,.0f}) differs from the "
                            f"reported {period_txt} count ({reported:,.0f}) by more than "
                            f"{SHARE_COUNT_TOLERANCE:.0%}; the reported count is used.")
        else:
            return {"shares": current, "as_of": inputs.price_date,
                    "classification": "current, basic (not diluted)",
                    "source": f"current shares outstanding (market snapshot "
                              f"{inputs.price_date or 'date unknown'}; basic, not diluted)",
                    "warnings": warnings + ["Current diluted share count not available: basic "
                                            "current shares used (dilution not included)."]}
    if reported and reported > 0:
        return {"shares": reported, "as_of": period_txt,
                "classification": "period-end, basic (not diluted)",
                "source": f"ordinary shares outstanding at {period_txt} (basic, not diluted)",
                "warnings": warnings + ["Current share count not used: period-end ordinary "
                                        "shares (fallback; dilution not included)."]}
    if diluted_avg and diluted_avg > 0:
        return {"shares": diluted_avg, "as_of": period_txt,
                "classification": "weighted average over the fiscal year, diluted",
                "source": f"diluted weighted-average shares, fiscal year to {period_txt} "
                          "(a period average, not a current count)",
                "warnings": warnings + ["Only a weighted-average share count is reported: it "
                                        "may differ from the current count (fallback)."]}
    return {"shares": None, "as_of": None, "classification": None, "source": None,
            "warnings": warnings}


def diluted_shares(inputs: ValuationInputs) -> tuple[float | None, str | None]:
    """Backward-compatible wrapper: (shares, source) from :func:`select_share_count`."""
    sel = select_share_count(inputs)
    return sel["shares"], sel["source"]


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
        base, parts, warnings = fcff_base(inputs.latest, tax, inputs.currency)
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
        if terminal >= market["risk_free_rate"]:
            raise ValuationError(
                f"Terminal growth {terminal:.2%} must be below the risk-free rate "
                f"{market['risk_free_rate']:.2%}: no business can grow faster than the "
                "economy's risk-free nominal rate forever (economically unreasonable).")
        flows = project_cash_flows(base, growth, terminal, cfg["explicit_years"],
                                   cfg["fade_years"])
        share_sel = select_share_count(inputs)
        shares, shares_src = share_sel["shares"], share_sel["source"]
        warnings = warnings + share_sel["warnings"]
        latest = inputs.latest
        result = value_firm(flows, rate, terminal, cash=cash_balance(latest),
                            debt=latest.get("total_debt"), shares=shares,
                            minority_interest=latest.get("minority_interest"))
    except ValuationError as exc:
        invalid = "must be greater" in str(exc) or "must be below" in str(exc)
        return CompanyValuation("INVALID" if invalid else "INSUFFICIENT_DATA", str(exc), method)
    out = CompanyValuation("OK", None, method, result, base, parts, growth, growth_note, rate,
                           rate_note, terminal, shares_src, result.value_per_share,
                           warnings=warnings + g_warn + result.warnings,
                           wacc_detail=wacc_detail)
    out.fcff_components["share_count"] = {k: share_sel[k] for k in
                                          ("shares", "source", "as_of", "classification")}
    out.valuation_date = inputs.latest_period
    out.value_per_share_at_valuation_date = result.value_per_share
    out.risk_free = market["risk_free_rate"]
    out.price_date = inputs.price_date
    out.assumption_flags = assumption_flags(out, inputs.latest, market)
    # Market comparison is optional: intrinsic value does not need the price.
    if inputs.price is not None and inputs.price_currency == inputs.currency:
        out.price = inputs.price
        out.roll_years = roll_forward_years(inputs.latest_period, inputs.price_date)
        if out.roll_years and result.value_per_share is not None:
            # value at the price date = value at the balance-sheet date grown at the
            # discount rate for the elapsed time (time value of the same cash flows)
            out.value_per_share = result.value_per_share * (1 + rate) ** out.roll_years
        if out.roll_years > 1.25:
            out.assumption_flags.append(
                f"The latest statements end {out.roll_years:.1f} years before the price date: "
                "the valuation rests on old figures.")
        out.margin_of_safety = margin_of_safety(out.value_per_share, inputs.price)
        if out.value_per_share and inputs.price:
            out.upside = out.value_per_share / inputs.price - 1
    elif inputs.price is not None:
        out.warnings.append(f"Price is in {inputs.price_currency} but statements are in "
                            f"{inputs.currency}: market comparison not shown (no FX "
                            "conversion).")
    return out


def roll_forward_years(valuation_date: date | None, price_date: str | None) -> float:
    """Years from the balance-sheet date to the price date, clamped to [0, 2]; 0 when
    either date is unknown (then no roll-forward is applied)."""
    if valuation_date is None or not price_date:
        return 0.0
    try:
        pd_ = date.fromisoformat(str(price_date)[:10])
    except ValueError:
        return 0.0
    return max(0.0, min(2.0, (pd_ - valuation_date).days / 365.25))


def _short(x: float) -> str:
    """Readable amount in the statement currency (e.g. 41.5 bn)."""
    a = abs(x)
    for div, unit in ((1e12, "tn"), (1e9, "bn"), (1e6, "mn")):
        if a >= div:
            return f"{x / div:,.1f} {unit}"
    return f"{x:,.0f}"


TV_SHARE_LIMIT = 0.75
MIN_SPREAD = 0.02          # discount rate - terminal growth below 2 points: very sensitive


def assumption_flags(v: CompanyValuation, latest: dict[str, float],
                     market: dict[str, Any]) -> list[str]:
    """Warnings for assumptions that are unrealistic or make the value fragile."""
    flags: list[str] = []
    share = v.terminal_share
    if share is not None and share > TV_SHARE_LIMIT:
        flags.append(f"The terminal value is {share:.0%} of enterprise value (above "
                     f"{TV_SHARE_LIMIT:.0%}): most of the value lies beyond the forecast, "
                     "so it depends heavily on terminal growth and the discount rate.")
    if v.discount_rate is not None and v.terminal_growth is not None and \
            v.discount_rate - v.terminal_growth < MIN_SPREAD:
        flags.append(f"Discount rate {v.discount_rate:.1%} is less than "
                     f"{MIN_SPREAD:.0%} above terminal growth {v.terminal_growth:.1%}: small "
                     "changes in either move the value a lot.")
    if v.discount_rate is not None and v.discount_rate < market["risk_free_rate"]:
        flags.append(f"Discount rate {v.discount_rate:.1%} is below the risk-free rate "
                     f"{market['risk_free_rate']:.1%}: equity and debt holders are being "
                     "paid less than a government bond.")
    if v.discount_rate is not None and v.discount_rate > 0.20:
        flags.append(f"Discount rate {v.discount_rate:.1%} is above 20 % (very high beta or "
                     "cost of debt): check the inputs.")
    capex, da = _finite(latest.get("capital_expenditure")), \
        _finite(latest.get("reconciled_depreciation"))
    if da is None:
        da = _finite(latest.get("depreciation_amortization_cf"))
    if v.growth and v.growth > 0.03 and capex is not None and da and abs(capex) < abs(da):
        flags.append(f"Capital expenditure ({_short(abs(capex))}) is below depreciation "
                     f"({_short(abs(da))}) while {v.growth:.1%} growth is projected: growth "
                     "normally needs net investment, so the cash flows may be too high.")
    rev = _finite(latest.get("total_revenue"))
    if v.base_fcff is not None and rev and rev > 0 and v.base_fcff / rev > 0.40:
        flags.append(f"Base FCFF is {v.base_fcff / rev:.0%} of revenue - unusually high; "
                     "check for one-off working-capital inflows.")
    return flags


def fcff_history(periods: dict[date, dict[str, float]], tax_rate: float,
                 currency: str | None) -> list[dict[str, Any]]:
    """FCFF of every fiscal year with the same method as the DCF base year, plus the
    drivers (revenue, operating margin, capex, D&A, working capital, effective tax).
    A year whose inputs are missing is kept with FCFF None and the reason."""
    rows = []
    for period in sorted(periods):
        v = periods[period]
        rev, op = _finite(v.get("total_revenue")), _finite(v.get("operating_income"))
        pretax, tax = _finite(v.get("pretax_income")), _finite(v.get("tax_provision"))
        row: dict[str, Any] = {
            "period": period.isoformat(), "revenue": rev, "operating_income": op,
            "operating_margin": (op / rev * 100) if (op is not None and rev and rev > 0)
            else None,
            "operating_cash_flow": _finite(v.get("operating_cash_flow")),
            "capital_expenditure": None if v.get("capital_expenditure") is None
            else -abs(float(v["capital_expenditure"])),
            "effective_tax_rate": (tax / pretax * 100) if (tax is not None and pretax and
                                                           pretax > 0) else None}
        try:
            fcff, parts, _ = fcff_base(v, tax_rate, currency)
            row.update(fcff=fcff, method=parts.get("method"), reason=None)
        except ValuationError as exc:
            row.update(fcff=None, method=None, reason=str(exc))
        rows.append(row)
    return rows


def history_flags(history: list[dict[str, Any]], base_fcff: float | None,
                  statutory_tax: float) -> list[str]:
    """Checks that need several years: an unusual base year, peak margins and a tax
    rate far from the statutory one."""
    flags: list[str] = []
    fcffs = [r["fcff"] for r in history if r.get("fcff") is not None]
    if base_fcff is not None and len(fcffs) >= 3:
        avg = sum(fcffs) / len(fcffs)
        if avg > 0 and abs(base_fcff / avg - 1) > 0.40:
            flags.append(f"The base-year FCFF differs by {base_fcff / avg - 1:+.0%} from the "
                         f"{len(fcffs)}-year average ({_short(avg)}): the base year may not be "
                         "representative.")
    margins = [r["operating_margin"] for r in history if r.get("operating_margin") is not None]
    if len(margins) >= 3 and margins[-1] - sum(margins[:-1]) / len(margins[:-1]) > 5:
        flags.append(f"The latest operating margin ({margins[-1]:.1f} %) is more than 5 points "
                     "above its earlier average: the forecast capitalises a possibly "
                     "peak margin.")
    taxes = [r["effective_tax_rate"] for r in history if r.get("effective_tax_rate") is not None]
    if taxes and abs(taxes[-1] - statutory_tax * 100) > 10:
        flags.append(f"The effective tax rate ({taxes[-1]:.1f} %) differs by more than 10 "
                     f"points from the statutory {statutory_tax:.1%} used in FCFF.")
    return flags


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
