"""Day 30 - liquidity and solvency ratios.

Liquidity
    Current Ratio  = Current Assets / Current Liabilities
    Quick Ratio    = Quick Assets / Current Liabilities
        Quick Assets, in order of preference (the method used is recorded):
        QA1  Cash, Equivalents & Short-Term Investments (combined line) + Accounts Receivable
        QA2  Cash & Equivalents + Short-Term Investments + Accounts Receivable
        QA3  Current Assets - Inventory - Prepaid Expenses
        Every component of the chosen formula must be reported; nothing is assumed 0.
Solvency
    Debt-to-Equity    = Total Debt / Shareholders' Equity  (NOT_MEANINGFUL if equity <= 0)
    Debt Ratio        = Total Liabilities / Total Assets
    Interest Coverage = EBIT / |Interest Expense|
        EBIT = reported EBIT; if absent, Operating Income (method EBIT_PROXY_OPERATING_INCOME).
        Interest Expense = "Interest Expense"; if absent, "Interest Expense Non Operating".
        Zero/absent interest expense -> UNAVAILABLE (coverage is undefined, not infinite).
"""

from __future__ import annotations

from datetime import date

from app.analysis.ratios import (NO_INTEREST_EXPENSE, NON_POSITIVE_EQUITY, FinancialDataset,
                                 RatioOutcome, denominator_problem, missing_outcome, ok,
                                 unavailable)


def current_ratio(ds: FinancialDataset, period: date) -> RatioOutcome:
    code = "current_ratio"
    inputs = {"current_assets": ds.get(period, "current_assets"),
              "current_liabilities": ds.get(period, "current_liabilities")}
    problem = missing_outcome(code, period, inputs) or denominator_problem(
        code, period, inputs["current_liabilities"], "Current Liabilities", inputs)
    if problem:
        return problem
    return ok(code, period, inputs["current_assets"] / inputs["current_liabilities"], inputs)


def quick_ratio(ds: FinancialDataset, period: date) -> RatioOutcome:
    code = "quick_ratio"
    liabilities = ds.get(period, "current_liabilities")
    candidates = [
        ("QA1_CASH_STI_COMBINED_PLUS_AR", {
            "cash_and_short_term_investments": ds.get(period, "cash_and_short_term_investments"),
            "accounts_receivable": ds.get(period, "accounts_receivable")}),
        ("QA2_CASH_PLUS_STI_PLUS_AR", {
            "cash_and_equivalents": ds.get(period, "cash_and_equivalents"),
            "short_term_investments": ds.get(period, "short_term_investments"),
            "accounts_receivable": ds.get(period, "accounts_receivable")}),
        ("QA3_CURRENT_ASSETS_LESS_INVENTORY_PREPAID", {
            "current_assets": ds.get(period, "current_assets"),
            "inventory": ds.get(period, "inventory"),
            "prepaid_assets": ds.get(period, "prepaid_assets")}),
    ]
    for method, components in candidates:
        if any(v is None for v in components.values()):
            continue
        if method.startswith("QA3"):
            quick_assets = (components["current_assets"] - components["inventory"]
                            - components["prepaid_assets"])
        else:
            quick_assets = sum(components.values())
        inputs = {**components, "quick_assets": quick_assets,
                  "current_liabilities": liabilities}
        problem = missing_outcome(code, period, {"current_liabilities": liabilities},
                                  method) or denominator_problem(
            code, period, liabilities, "Current Liabilities", inputs, method)
        if problem:
            return problem
        return ok(code, period, quick_assets / liabilities, inputs, method)
    tried = "; ".join(
        f"{m}: missing {', '.join(k for k, v in c.items() if v is None)}" for m, c in candidates)
    return unavailable(code, period, "MISSING_INPUT",
                       f"No complete quick-assets formula for FY ending {period} ({tried}). "
                       "Unavailable components are not assumed to be zero.",
                       {"current_liabilities": liabilities})


def debt_to_equity(ds: FinancialDataset, period: date) -> RatioOutcome:
    code = "debt_to_equity"
    inputs = {"total_debt": ds.get(period, "total_debt"),
              "stockholders_equity": ds.get(period, "stockholders_equity")}
    problem = missing_outcome(code, period, inputs)
    if problem:
        return problem
    if inputs["stockholders_equity"] <= 0:
        return unavailable(code, period, NON_POSITIVE_EQUITY,
                           f"Shareholders' equity is {inputs['stockholders_equity']} (zero or "
                           "negative); debt-to-equity is not meaningful. See red flag RF07.",
                           inputs, not_meaningful=True)
    return ok(code, period, inputs["total_debt"] / inputs["stockholders_equity"], inputs)


def debt_ratio(ds: FinancialDataset, period: date) -> RatioOutcome:
    code = "debt_ratio"
    inputs = {"total_liabilities": ds.get(period, "total_liabilities"),
              "total_assets": ds.get(period, "total_assets")}
    problem = missing_outcome(code, period, inputs) or denominator_problem(
        code, period, inputs["total_assets"], "Total Assets", inputs)
    if problem:
        return problem
    return ok(code, period, inputs["total_liabilities"] / inputs["total_assets"], inputs)


def interest_coverage(ds: FinancialDataset, period: date) -> RatioOutcome:
    code = "interest_coverage"
    ebit = ds.get(period, "ebit")
    method = "EBIT_REPORTED"
    if ebit is None:
        ebit = ds.get(period, "operating_income")
        method = "EBIT_PROXY_OPERATING_INCOME"
    interest = ds.get(period, "interest_expense")
    interest_source = "interest_expense"
    if interest is None:
        interest = ds.get(period, "interest_expense_non_operating")
        interest_source = "interest_expense_non_operating"
    inputs = {"ebit": ebit, "ebit_source": "ebit" if method == "EBIT_REPORTED"
              else "operating_income", "interest_expense": interest,
              "interest_source": interest_source}
    if interest is None:
        return unavailable(code, period, "MISSING_INPUT",
                           f"Interest expense not reported for FY ending {period}; coverage "
                           "cannot be calculated (it is not assumed to be zero).", inputs, method)
    if ebit is None:
        return unavailable(code, period, "MISSING_INPUT",
                           f"Neither EBIT nor Operating Income reported for FY ending {period}.",
                           inputs, method)
    if interest == 0:
        return unavailable(code, period, NO_INTEREST_EXPENSE,
                           "Interest expense is zero; coverage is undefined (no division by "
                           "zero, no infinite value).", inputs, method)
    explanation = None
    if method == "EBIT_PROXY_OPERATING_INCOME":
        explanation = "EBIT not reported: Operating Income used as an explicit EBIT proxy."
    return ok(code, period, ebit / abs(interest), inputs, method, explanation)


def calculate_liquidity_solvency(ds: FinancialDataset, period: date) -> list[RatioOutcome]:
    return [current_ratio(ds, period), quick_ratio(ds, period), debt_to_equity(ds, period),
            debt_ratio(ds, period), interest_coverage(ds, period)]
