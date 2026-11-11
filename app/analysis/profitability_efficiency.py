"""Day 40 - profitability and efficiency ratios (+ cash-flow ratios used by the score).

Profitability
    Net Profit Margin = Net Income / Total Revenue x 100
    ROA               = Net Income / Average Total Assets x 100
    ROE               = Net Income / Average Shareholders' Equity x 100
Efficiency
    Asset Turnover     = Total Revenue / Average Total Assets
    Inventory Turnover = Cost of Revenue / Average Inventory
Cash flow (supporting the Day 70 Cash Flow category)
    Operating Cash Flow Margin = Operating Cash Flow / Total Revenue x 100
    Cash Conversion            = Operating Cash Flow / Net Income   (Net Income > 0 only)
    Free Cash Flow Margin      = Free Cash Flow / Total Revenue x 100

Average balances = (current year-end + previous year-end) / 2. If the previous
year-end is unavailable the configured policy applies: ``ending_balance`` (the
year-end value, method ENDING_BALANCE_FALLBACK) or ``unavailable``. The method
used is stored with every result, so the two approaches are never mixed silently.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from app.analysis.ratios import (HUNDRED, NO_PREVIOUS_BALANCE, NON_POSITIVE_EQUITY,
                                 NON_POSITIVE_NET_INCOME, FinancialDataset, RatioOutcome,
                                 average_balance, denominator_problem, fallback_note,
                                 missing_outcome, ok, unavailable)
from config.settings import INVENTORY_MATERIALITY_THRESHOLD

IMMATERIAL_INVENTORY = "IMMATERIAL_INVENTORY"


def net_profit_margin(ds: FinancialDataset, period: date) -> RatioOutcome:
    code = "net_profit_margin"
    inputs = {"net_income": ds.get(period, "net_income"),
              "total_revenue": ds.get(period, "total_revenue")}
    problem = missing_outcome(code, period, inputs) or denominator_problem(
        code, period, inputs["total_revenue"], "Total Revenue", inputs)
    if problem:
        return problem
    return ok(code, period, inputs["net_income"] / inputs["total_revenue"] * HUNDRED, inputs)


def _average_based(ds: FinancialDataset, period: date, code: str, numerator_code: str,
                   balance_code: str, label: str, fallback: str,
                   scale: bool) -> RatioOutcome:
    numerator = ds.get(period, numerator_code)
    average = average_balance(ds, period, balance_code, fallback)
    inputs = {numerator_code: numerator, **average.inputs, f"average_{balance_code}":
              average.value}
    if numerator is None:
        return unavailable(code, period, "MISSING_INPUT",
                           f"{numerator_code} missing for FY ending {period}.", inputs)
    if average.value is None:
        reason = "MISSING_INPUT" if average.inputs.get(f"{balance_code}_end") is None \
            else NO_PREVIOUS_BALANCE
        return unavailable(code, period, reason, f"Cannot calculate: {average.problem}.",
                           inputs, average.method)
    problem = denominator_problem(code, period, average.value, label, inputs, average.method)
    if problem:
        return problem
    value = numerator / average.value * (HUNDRED if scale else 1)
    return ok(code, period, value, inputs, average.method, fallback_note(average.method))


def return_on_assets(ds: FinancialDataset, period: date,
                     fallback: str = "ending_balance") -> RatioOutcome:
    return _average_based(ds, period, "return_on_assets", "net_income", "total_assets",
                          "Average Total Assets", fallback, scale=True)


def return_on_equity(ds: FinancialDataset, period: date,
                     fallback: str = "ending_balance") -> RatioOutcome:
    code = "return_on_equity"
    equity_end = ds.get(period, "stockholders_equity")
    previous = ds.get(ds.adjacent_previous(period), "stockholders_equity")
    for label, value in (("year-end", equity_end), ("previous year-end", previous)):
        if value is not None and value <= 0:
            return unavailable(
                code, period, NON_POSITIVE_EQUITY,
                f"Shareholders' equity ({label}) is {value} (zero or negative); ROE is not "
                "meaningful because a loss over negative equity would show a positive ROE.",
                {"net_income": ds.get(period, "net_income"), "stockholders_equity_end":
                 equity_end, "stockholders_equity_previous": previous},
                not_meaningful=True)
    return _average_based(ds, period, code, "net_income", "stockholders_equity",
                          "Average Shareholders' Equity", fallback, scale=True)


def asset_turnover(ds: FinancialDataset, period: date,
                   fallback: str = "ending_balance") -> RatioOutcome:
    return _average_based(ds, period, "asset_turnover", "total_revenue", "total_assets",
                          "Average Total Assets", fallback, scale=False)


def inventory_turnover(ds: FinancialDataset, period: date,
                       fallback: str = "ending_balance") -> RatioOutcome:
    code = "inventory_turnover"
    if ds.get(period, "inventory") is None:
        return unavailable(code, period, "MISSING_INPUT",
                           f"Inventory not reported for FY ending {period} (common for "
                           "service, software and financial companies).",
                           {"inventory_end": None, "cost_of_revenue":
                            ds.get(period, "cost_of_revenue")})
    cogs_code = "cost_of_revenue" if ds.get(period, "cost_of_revenue") is not None \
        else "reconciled_cost_of_revenue"
    outcome = _average_based(ds, period, code, cogs_code, "inventory", "Average Inventory",
                             fallback, scale=False)
    assets = average_balance(ds, period, "total_assets", fallback)
    inventory = outcome.inputs.get("average_inventory")
    if outcome.ok and assets.value and assets.value > 0 and inventory is not None and \
            inventory / assets.value < Decimal(str(INVENTORY_MATERIALITY_THRESHOLD)):
        share = inventory / assets.value * HUNDRED
        return unavailable(
            code, period, IMMATERIAL_INVENTORY,
            f"Inventory is immaterial ({float(share):.3f}% of average total assets, threshold "
            f"{INVENTORY_MATERIALITY_THRESHOLD:.1%}); turnover of {float(outcome.value):,.0f}x "
            "would be misleading.", {**outcome.inputs, "average_total_assets": assets.value},
            outcome.method, not_meaningful=True)
    if cogs_code != "cost_of_revenue" and outcome.ok:
        outcome.explanation = " ".join(filter(None, [
            outcome.explanation, "Cost of Revenue absent; Reconciled Cost of Revenue used."]))
    return outcome


def operating_cash_flow_margin(ds: FinancialDataset, period: date) -> RatioOutcome:
    code = "operating_cash_flow_margin"
    inputs = {"operating_cash_flow": ds.get(period, "operating_cash_flow"),
              "total_revenue": ds.get(period, "total_revenue")}
    problem = missing_outcome(code, period, inputs) or denominator_problem(
        code, period, inputs["total_revenue"], "Total Revenue", inputs)
    if problem:
        return problem
    return ok(code, period, inputs["operating_cash_flow"] / inputs["total_revenue"] * HUNDRED,
              inputs)


def cash_conversion(ds: FinancialDataset, period: date) -> RatioOutcome:
    code = "cash_conversion"
    inputs = {"operating_cash_flow": ds.get(period, "operating_cash_flow"),
              "net_income": ds.get(period, "net_income")}
    problem = missing_outcome(code, period, inputs)
    if problem:
        return problem
    if inputs["net_income"] <= 0:
        return unavailable(code, period, NON_POSITIVE_NET_INCOME,
                           f"Net income is {inputs['net_income']} (zero or a loss); cash "
                           "conversion is not meaningful.", inputs, not_meaningful=True)
    return ok(code, period, inputs["operating_cash_flow"] / inputs["net_income"], inputs)


def free_cash_flow_margin(ds: FinancialDataset, period: date) -> RatioOutcome:
    code = "free_cash_flow_margin"
    revenue = ds.get(period, "total_revenue")
    fcf = ds.get(period, "free_cash_flow")
    method = "PROVIDER_FCF"
    inputs = {"free_cash_flow": fcf, "total_revenue": revenue}
    if fcf is None:
        ocf, capex = ds.get(period, "operating_cash_flow"), ds.get(period, "capital_expenditure")
        inputs.update({"operating_cash_flow": ocf, "capital_expenditure": capex})
        if ocf is not None and capex is not None:
            # Yahoo reports capital expenditure as a negative outflow.
            fcf = ocf - abs(capex)
            method = "OCF_MINUS_CAPEX"
            inputs["free_cash_flow"] = fcf
    problem = missing_outcome(code, period, {"free_cash_flow": fcf, "total_revenue": revenue},
                              method) or denominator_problem(
        code, period, revenue, "Total Revenue", inputs, method)
    if problem:
        return problem
    explanation = None if method == "PROVIDER_FCF" else \
        "Provider FCF absent: computed as Operating Cash Flow - |Capital Expenditure|."
    return ok(code, period, fcf / revenue * HUNDRED, inputs, method, explanation)


def calculate_profitability_efficiency(ds: FinancialDataset, period: date,
                                       fallback: str = "ending_balance") -> list[RatioOutcome]:
    return [net_profit_margin(ds, period), return_on_assets(ds, period, fallback),
            return_on_equity(ds, period, fallback), asset_turnover(ds, period, fallback),
            inventory_turnover(ds, period, fallback)]


def calculate_cash_flow_ratios(ds: FinancialDataset, period: date) -> list[RatioOutcome]:
    return [operating_cash_flow_margin(ds, period), cash_conversion(ds, period),
            free_cash_flow_margin(ds, period)]
