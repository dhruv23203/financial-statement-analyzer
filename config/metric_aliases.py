"""Central metric-mapping dictionary (Day 20).

Every standardized metric has:

* ``code``            - stable identifier used throughout the system
* ``name``            - human-readable name
* ``statement``       - income_statement | balance_sheet | cash_flow
* ``aliases``         - provider metric names that mean the *same* concept, in
                        **precedence order** (first = most preferred). Aliases are
                        compared after normalisation (case, spaces, punctuation and
                        apostrophes ignored), so "Stockholders' Equity",
                        "StockholdersEquity" and "stockholders equity" all match.
* ``expected_sign``   - "non_negative", "non_positive" or "any"; used by the
                        validator to flag unexpected sign conventions.

Only true synonyms are listed as aliases. Related-but-different concepts (for
example "Common Stock Equity" vs "Stockholders Equity") are separate metrics so
that values are never silently substituted.

Conflict rule: if two aliases of the same metric are present for the same period
with *different* values, the higher-precedence alias is used and the fact is
flagged as a mapping conflict. Values are never added together.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

INCOME_STATEMENT = "income_statement"
BALANCE_SHEET = "balance_sheet"
CASH_FLOW = "cash_flow"
STATEMENT_TYPES = (INCOME_STATEMENT, BALANCE_SHEET, CASH_FLOW)

STATEMENT_LABELS = {
    INCOME_STATEMENT: "Income Statement",
    BALANCE_SHEET: "Balance Sheet",
    CASH_FLOW: "Cash Flow Statement",
}


@dataclass(frozen=True)
class MetricDefinition:
    code: str
    name: str
    statement: str
    aliases: tuple[str, ...]
    expected_sign: str = "any"


def _m(code: str, name: str, statement: str, aliases: list[str],
       expected_sign: str = "any") -> MetricDefinition:
    return MetricDefinition(code, name, statement, tuple(aliases), expected_sign)


METRIC_DEFINITIONS: tuple[MetricDefinition, ...] = (
    # ---------------------------------------------------------------- income statement
    _m("total_revenue", "Total Revenue", INCOME_STATEMENT,
       ["Total Revenue", "Revenue", "Revenues", "Total Revenues"], "non_negative"),
    _m("operating_revenue", "Operating Revenue", INCOME_STATEMENT,
       ["Operating Revenue"], "non_negative"),
    _m("cost_of_revenue", "Cost of Revenue", INCOME_STATEMENT,
       ["Cost Of Revenue", "Cost of Goods Sold", "Cost Of Goods And Services Sold"],
       "non_negative"),
    _m("reconciled_cost_of_revenue", "Reconciled Cost of Revenue", INCOME_STATEMENT,
       ["Reconciled Cost Of Revenue"], "non_negative"),
    _m("gross_profit", "Gross Profit", INCOME_STATEMENT, ["Gross Profit"]),
    _m("operating_expense", "Operating Expense", INCOME_STATEMENT, ["Operating Expense"]),
    _m("operating_income", "Operating Income", INCOME_STATEMENT,
       ["Operating Income", "Operating Profit"]),
    _m("ebit", "EBIT", INCOME_STATEMENT, ["EBIT"]),
    _m("ebitda", "EBITDA", INCOME_STATEMENT, ["EBITDA"]),
    _m("normalized_ebitda", "Normalized EBITDA", INCOME_STATEMENT, ["Normalized EBITDA"]),
    _m("interest_expense", "Interest Expense", INCOME_STATEMENT, ["Interest Expense"]),
    _m("interest_expense_non_operating", "Interest Expense (Non-Operating)", INCOME_STATEMENT,
       ["Interest Expense Non Operating"]),
    _m("interest_income", "Interest Income", INCOME_STATEMENT, ["Interest Income"]),
    _m("pretax_income", "Pre-tax Income", INCOME_STATEMENT,
       ["Pretax Income", "Income Before Tax"]),
    _m("tax_provision", "Tax Provision", INCOME_STATEMENT,
       ["Tax Provision", "Income Tax Expense"]),
    _m("net_income", "Net Income", INCOME_STATEMENT, ["Net Income"]),
    _m("net_income_common", "Net Income to Common Stockholders", INCOME_STATEMENT,
       ["Net Income Common Stockholders"]),
    _m("net_income_continuing", "Net Income from Continuing Operations", INCOME_STATEMENT,
       ["Net Income Continuous Operations"]),
    _m("total_expenses", "Total Expenses", INCOME_STATEMENT, ["Total Expenses"]),
    _m("selling_general_administrative", "Selling, General & Administrative Expense",
       INCOME_STATEMENT, ["Selling General And Administration",
                          "Selling General And Administrative Expense"], "non_negative"),
    _m("reconciled_depreciation", "Reconciled Depreciation", INCOME_STATEMENT,
       ["Reconciled Depreciation"], "non_negative"),
    _m("depreciation_amortization_is", "Depreciation & Amortization (Income Statement)",
       INCOME_STATEMENT, ["Depreciation And Amortization In Income Statement"]),
    _m("basic_eps", "Basic EPS", INCOME_STATEMENT, ["Basic EPS"]),
    _m("diluted_eps", "Diluted EPS", INCOME_STATEMENT, ["Diluted EPS"]),
    _m("basic_average_shares", "Basic Average Shares", INCOME_STATEMENT,
       ["Basic Average Shares"], "non_negative"),
    _m("diluted_average_shares", "Diluted Average Shares", INCOME_STATEMENT,
       ["Diluted Average Shares"], "non_negative"),
    # ---------------------------------------------------------------- balance sheet
    _m("total_assets", "Total Assets", BALANCE_SHEET, ["Total Assets"], "non_negative"),
    _m("current_assets", "Current Assets", BALANCE_SHEET,
       ["Current Assets", "Total Current Assets"], "non_negative"),
    _m("non_current_assets", "Non-Current Assets", BALANCE_SHEET,
       ["Total Non Current Assets"], "non_negative"),
    _m("cash_and_equivalents", "Cash and Cash Equivalents", BALANCE_SHEET,
       ["Cash And Cash Equivalents"], "non_negative"),
    _m("cash_and_short_term_investments", "Cash, Equivalents and Short-Term Investments",
       BALANCE_SHEET, ["Cash Cash Equivalents And Short Term Investments"], "non_negative"),
    _m("short_term_investments", "Short-Term Investments", BALANCE_SHEET,
       ["Other Short Term Investments", "Short Term Investments"], "non_negative"),
    _m("accounts_receivable", "Accounts Receivable", BALANCE_SHEET,
       ["Accounts Receivable", "Net Receivables"], "non_negative"),
    _m("receivables", "Total Receivables", BALANCE_SHEET, ["Receivables"], "non_negative"),
    _m("inventory", "Inventory", BALANCE_SHEET, ["Inventory", "Inventories"], "non_negative"),
    _m("prepaid_assets", "Prepaid Expenses", BALANCE_SHEET,
       ["Prepaid Assets", "Prepaid Expenses"], "non_negative"),
    _m("current_liabilities", "Current Liabilities", BALANCE_SHEET,
       ["Current Liabilities", "Total Current Liabilities"], "non_negative"),
    _m("total_liabilities", "Total Liabilities", BALANCE_SHEET,
       ["Total Liabilities Net Minority Interest", "Total Liabilities"], "non_negative"),
    _m("stockholders_equity", "Shareholders' Equity", BALANCE_SHEET,
       ["Stockholders Equity", "Total Stockholder Equity", "Total Stockholders Equity",
        "Shareholders Equity", "Total Shareholders Equity"]),
    _m("common_stock_equity", "Common Stock Equity", BALANCE_SHEET, ["Common Stock Equity"]),
    _m("total_equity_incl_minority", "Total Equity incl. Minority Interest", BALANCE_SHEET,
       ["Total Equity Gross Minority Interest"]),
    _m("minority_interest", "Minority Interest", BALANCE_SHEET, ["Minority Interest"]),
    _m("total_debt", "Total Debt", BALANCE_SHEET, ["Total Debt"], "non_negative"),
    _m("long_term_debt", "Long-Term Debt", BALANCE_SHEET, ["Long Term Debt"], "non_negative"),
    _m("long_term_debt_and_lease", "Long-Term Debt incl. Capital Leases", BALANCE_SHEET,
       ["Long Term Debt And Capital Lease Obligation"], "non_negative"),
    _m("current_debt", "Current Debt", BALANCE_SHEET, ["Current Debt"], "non_negative"),
    _m("net_debt", "Net Debt", BALANCE_SHEET, ["Net Debt"]),
    _m("working_capital", "Working Capital", BALANCE_SHEET, ["Working Capital"]),
    _m("accounts_payable", "Accounts Payable", BALANCE_SHEET, ["Accounts Payable"],
       "non_negative"),
    _m("retained_earnings", "Retained Earnings", BALANCE_SHEET, ["Retained Earnings"]),
    _m("ordinary_shares_number", "Ordinary Shares Outstanding", BALANCE_SHEET,
       ["Ordinary Shares Number"], "non_negative"),
    _m("goodwill", "Goodwill", BALANCE_SHEET, ["Goodwill"], "non_negative"),
    _m("net_ppe", "Net Property, Plant & Equipment", BALANCE_SHEET, ["Net PPE"],
       "non_negative"),
    # ---------------------------------------------------------------- cash flow
    _m("operating_cash_flow", "Operating Cash Flow", CASH_FLOW,
       ["Operating Cash Flow", "Total Cash From Operating Activities",
        "Cash Flow From Continuing Operating Activities"]),
    _m("investing_cash_flow", "Investing Cash Flow", CASH_FLOW,
       ["Investing Cash Flow", "Total Cashflows From Investing Activities",
        "Cash Flow From Continuing Investing Activities"]),
    _m("financing_cash_flow", "Financing Cash Flow", CASH_FLOW,
       ["Financing Cash Flow", "Total Cash From Financing Activities",
        "Cash Flow From Continuing Financing Activities"]),
    _m("capital_expenditure", "Capital Expenditure", CASH_FLOW,
       ["Capital Expenditure", "Capital Expenditures"], "non_positive"),
    _m("free_cash_flow", "Free Cash Flow (provider-reported)", CASH_FLOW, ["Free Cash Flow"]),
    _m("cash_dividends_paid", "Cash Dividends Paid", CASH_FLOW,
       ["Cash Dividends Paid"], "non_positive"),
    _m("depreciation_amortization_cf", "Depreciation & Amortization (Cash Flow)", CASH_FLOW,
       ["Depreciation And Amortization"]),
    _m("change_in_working_capital", "Change in Working Capital", CASH_FLOW,
       ["Change In Working Capital"]),
    _m("repurchase_of_capital_stock", "Repurchase of Capital Stock", CASH_FLOW,
       ["Repurchase Of Capital Stock"], "non_positive"),
    _m("changes_in_cash", "Changes in Cash", CASH_FLOW, ["Changes In Cash"]),
    _m("issuance_of_capital_stock", "Issuance of Capital Stock", CASH_FLOW,
       ["Issuance Of Capital Stock", "Common Stock Issuance"], "non_negative"),
    _m("end_cash_position", "End Cash Position", CASH_FLOW, ["End Cash Position"]),
)

# Fields that are required for the core analysis; missing ones produce warnings.
ESSENTIAL_METRICS: dict[str, tuple[str, ...]] = {
    INCOME_STATEMENT: ("total_revenue", "net_income"),
    BALANCE_SHEET: ("total_assets", "current_assets", "current_liabilities",
                    "stockholders_equity"),
    CASH_FLOW: ("operating_cash_flow",),
}

UNMAPPED_PREFIX = "UNMAPPED:"


def normalize_metric_name(name: str) -> str:
    """Normalise a provider metric name for alias matching.

    Lower-cases, removes apostrophes and every non-alphanumeric character, so
    ``"Stockholders' Equity"`` -> ``"stockholdersequity"``.
    """
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _build_alias_index() -> dict[tuple[str, str], tuple[MetricDefinition, int]]:
    index: dict[tuple[str, str], tuple[MetricDefinition, int]] = {}
    for definition in METRIC_DEFINITIONS:
        for precedence, alias in enumerate(definition.aliases):
            key = (definition.statement, normalize_metric_name(alias))
            if key in index:
                other = index[key][0].code
                raise ValueError(
                    f"Alias {alias!r} is assigned to both {other!r} and {definition.code!r}")
            index[key] = (definition, precedence)
    return index


_ALIAS_INDEX = _build_alias_index()
METRICS_BY_CODE: dict[str, MetricDefinition] = {d.code: d for d in METRIC_DEFINITIONS}


def lookup_metric(statement: str, original_name: str) -> tuple[MetricDefinition, int] | None:
    """Return ``(definition, precedence)`` for a provider metric name, or ``None`` if unmapped."""
    return _ALIAS_INDEX.get((statement, normalize_metric_name(original_name)))


def unmapped_code(original_name: str) -> str:
    """Code used to store a metric that has no standard mapping (original name preserved)."""
    return f"{UNMAPPED_PREFIX}{str(original_name).strip()}"[:200]
