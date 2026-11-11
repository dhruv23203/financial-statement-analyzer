"""Descriptive market multiples, ROCE, dividend payout and the original Altman Z.

These are the headline numbers shown by platforms such as Screener.in, Tickertape and
GuruFocus. Multiples are *descriptive* (what the market currently pays for the latest
reported fundamentals); they are not a valuation model - DCF/intrinsic value is
Semester 2 work.

Rules
-----
* Multiples use the latest market snapshot with the latest annual statements and are
  calculated for the latest fiscal period only.
* If the market data currency differs from the statement currency (e.g. INR share price
  vs USD statements) the multiple is UNAVAILABLE - no currency conversion is performed.
* Altman Z (original, 1968): Z = 1.2 X1 + 1.4 X2 + 3.3 X3 + 0.6 X4 + 1.0 X5 with
  X4 = market value of equity / total liabilities; zones > 2.99 safe, < 1.81 distress.
  Altman, E. I. (1968). Financial ratios, discriminant analysis and the prediction of
  corporate bankruptcy. Journal of Finance, 23(4), 589-609. Designed for public
  manufacturing firms; the book-value Z'' is shown for all companies.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from app.analysis.ratios import (HUNDRED, FinancialDataset, RatioOutcome,
                                 denominator_problem, fallback_note, missing_outcome, ok,
                                 unavailable)
from app.utils.helpers import to_decimal

D = Decimal
# Ratios that need a market snapshot: only calculated for the latest fiscal year.
MARKET_RATIO_CODES = frozenset({"price_to_earnings", "earnings_yield", "price_to_book",
                                "ev_to_ebitda", "dividend_yield", "altman_z_original"})


def return_on_capital_employed(ds: FinancialDataset, period: date,
                               fallback: str = "ending_balance") -> RatioOutcome:
    """ROCE = EBIT / average (total assets - current liabilities) x 100 (Screener.in style)."""
    code = "return_on_capital_employed"
    ebit = ds.get(period, "ebit")
    method_note = None
    if ebit is None:
        ebit = ds.get(period, "operating_income")
        method_note = "Operating Income used as EBIT proxy."

    def capital_employed(p: date | None) -> Decimal | None:
        ta, cl = ds.get(p, "total_assets"), ds.get(p, "current_liabilities")
        return None if ta is None or cl is None else ta - cl

    current, previous_period = capital_employed(period), ds.adjacent_previous(period)
    previous = capital_employed(previous_period)
    inputs = {"ebit": ebit, "capital_employed_end": current,
              "capital_employed_previous": previous}
    if ebit is None or current is None:
        return missing_outcome(code, period, {"ebit": ebit, "capital_employed": current}) or \
            unavailable(code, period, "MISSING_INPUT", "inputs missing", inputs)
    if previous is not None:
        base, method = (current + previous) / 2, "AVERAGE_BALANCE"
    elif fallback == "ending_balance":
        base, method = current, "ENDING_BALANCE_FALLBACK"
    else:
        return unavailable(code, period, "NO_PREVIOUS_BALANCE",
                           "Previous-year capital employed unavailable.", inputs)
    inputs["average_capital_employed"] = base
    problem = denominator_problem(code, period, base, "Capital employed", inputs, method)
    if problem:
        return problem
    explanation = " ".join(filter(None, [method_note, fallback_note(method)])) or None
    return ok(code, period, ebit / base * HUNDRED, inputs, method, explanation)


def dividend_payout_ratio(ds: FinancialDataset, period: date) -> RatioOutcome:
    code = "dividend_payout_ratio"
    inputs = {"cash_dividends_paid": ds.get(period, "cash_dividends_paid"),
              "net_income": ds.get(period, "net_income")}
    problem = missing_outcome(code, period, inputs)
    if problem:
        problem.explanation = (problem.explanation or "") + \
            " (Companies that pay no dividend often have no dividend line.)"
        return problem
    if inputs["net_income"] <= 0:
        return unavailable(code, period, "NON_POSITIVE_NET_INCOME",
                           "Net income is zero or negative; payout ratio not meaningful.",
                           inputs, not_meaningful=True)
    return ok(code, period, abs(inputs["cash_dividends_paid"]) / inputs["net_income"] * HUNDRED,
              inputs)


def calculate_extra_ratios(ds: FinancialDataset, period: date,
                           fallback: str = "ending_balance") -> list[RatioOutcome]:
    return [return_on_capital_employed(ds, period, fallback), dividend_payout_ratio(ds, period)]


# --------------------------------------------------------------------------- market multiples
def _market_cap(ds: FinancialDataset, code: str, period: date) -> tuple[Decimal | None,
                                                                        RatioOutcome | None]:
    market = ds.market or {}
    cap = to_decimal(market.get("market_cap"))
    if cap is None or cap <= 0:
        return None, unavailable(code, period, "MISSING_INPUT", "No market data stored. Run: "
                                 "python main.py market TICKER", {})
    if ds.currency and market.get("currency") and market["currency"] != ds.currency:
        return None, unavailable(
            code, period, "CURRENCY_MISMATCH",
            f"Market data in {market['currency']} but statements in {ds.currency}; multiples "
            "need one currency and no conversion is performed.",
            {"market_currency": market["currency"], "statement_currency": ds.currency})
    return cap, None


def calculate_valuation(ds: FinancialDataset, period: date) -> list[RatioOutcome]:
    market_date = (ds.market or {}).get("fetched_at")
    note = f"Market data as of {market_date} with statements for FY ending {period}."
    out: list[RatioOutcome] = []
    g = ds.get
    ni, equity, ebitda = g(period, "net_income"), g(period, "stockholders_equity"), \
        g(period, "ebitda")
    debt, cash = g(period, "total_debt"), g(period, "cash_and_equivalents")
    dividends = g(period, "cash_dividends_paid")

    def build(code: str, compute) -> None:
        cap, problem = _market_cap(ds, code, period)
        if problem:
            out.append(problem)
            return
        out.append(compute(cap))

    def pe(cap: Decimal) -> RatioOutcome:
        inputs = {"market_cap": cap, "net_income": ni}
        if ni is None:
            return unavailable("price_to_earnings", period, "MISSING_INPUT", "Net income missing.",
                               inputs)
        if ni <= 0:
            return unavailable("price_to_earnings", period, "NON_POSITIVE_NET_INCOME",
                               "Loss-making: P/E not meaningful (see earnings yield).", inputs,
                               not_meaningful=True)
        return ok("price_to_earnings", period, cap / ni, inputs, "MARKET_SNAPSHOT", note)

    def earnings_yield(cap: Decimal) -> RatioOutcome:
        inputs = {"market_cap": cap, "net_income": ni}
        if ni is None:
            return unavailable("earnings_yield", period, "MISSING_INPUT", "Net income missing.",
                               inputs)
        return ok("earnings_yield", period, ni / cap * HUNDRED, inputs, "MARKET_SNAPSHOT", note)

    def pb(cap: Decimal) -> RatioOutcome:
        inputs = {"market_cap": cap, "stockholders_equity": equity}
        if equity is None:
            return unavailable("price_to_book", period, "MISSING_INPUT", "Equity missing.", inputs)
        if equity <= 0:
            return unavailable("price_to_book", period, "NON_POSITIVE_EQUITY",
                               "Equity is zero or negative; P/B not meaningful.", inputs,
                               not_meaningful=True)
        return ok("price_to_book", period, cap / equity, inputs, "MARKET_SNAPSHOT", note)

    def ev_ebitda(cap: Decimal) -> RatioOutcome:
        ebitda_value = ebitda
        if ebitda_value is None and g(period, "operating_income") is not None and \
                g(period, "depreciation_amortization_cf") is not None:
            ebitda_value = g(period, "operating_income") + g(period, "depreciation_amortization_cf")
        inputs = {"market_cap": cap, "total_debt": debt, "cash_and_equivalents": cash,
                  "ebitda": ebitda_value}
        missing = [k for k, v in inputs.items() if v is None]
        if missing:
            return unavailable("ev_to_ebitda", period, "MISSING_INPUT",
                               f"Missing {', '.join(missing)} (not assumed to be zero).", inputs)
        if ebitda_value <= 0:
            return unavailable("ev_to_ebitda", period, "NON_POSITIVE_EBITDA",
                               "EBITDA is zero or negative; EV/EBITDA not meaningful.", inputs,
                               not_meaningful=True)
        ev = cap + debt - cash
        inputs["enterprise_value"] = ev
        return ok("ev_to_ebitda", period, ev / ebitda_value, inputs, "MARKET_SNAPSHOT", note)

    def dividend_yield(cap: Decimal) -> RatioOutcome:
        inputs = {"market_cap": cap, "cash_dividends_paid": dividends}
        if dividends is None:
            return unavailable("dividend_yield", period, "MISSING_INPUT",
                               "No dividend line in the cash-flow statement.", inputs)
        return ok("dividend_yield", period, abs(dividends) / cap * HUNDRED, inputs,
                  "MARKET_SNAPSHOT", note)

    def altman_original(cap: Decimal) -> RatioOutcome:
        ta, tl = g(period, "total_assets"), g(period, "total_liabilities")
        wc = g(period, "working_capital")
        if wc is None and g(period, "current_assets") is not None and \
                g(period, "current_liabilities") is not None:
            wc = g(period, "current_assets") - g(period, "current_liabilities")
        ebit = g(period, "ebit") if g(period, "ebit") is not None else g(period,
                                                                          "operating_income")
        inputs = {"working_capital": wc, "retained_earnings": g(period, "retained_earnings"),
                  "ebit": ebit, "market_cap": cap, "total_liabilities": tl,
                  "total_revenue": g(period, "total_revenue"), "total_assets": ta}
        missing = [k for k, v in inputs.items() if v is None]
        if missing:
            return unavailable("altman_z_original", period, "MISSING_INPUT",
                               f"Missing {', '.join(missing)}.", inputs)
        if ta <= 0 or tl <= 0:
            return unavailable("altman_z_original", period, "ZERO_DENOMINATOR",
                               "Total assets or liabilities not positive.", inputs)
        z = (D("1.2") * wc / ta + D("1.4") * inputs["retained_earnings"] / ta
             + D("3.3") * ebit / ta + D("0.6") * cap / tl + D("1.0") * inputs["total_revenue"] / ta)
        zone = "safe" if z > D("2.99") else "distress" if z < D("1.81") else "grey"
        return ok("altman_z_original", period, z, {**inputs, "zone": zone}, "ALTMAN_1968",
                  f"Z = {float(z):.2f}: {zone} zone (> 2.99 safe, < 1.81 distress). Designed for "
                  "manufacturing firms. " + note)

    for code, fn in (("price_to_earnings", pe), ("earnings_yield", earnings_yield),
                     ("price_to_book", pb), ("ev_to_ebitda", ev_ebitda),
                     ("dividend_yield", dividend_yield), ("altman_z_original", altman_original)):
        build(code, fn)
    return out
