"""Ratio framework shared by the Day 30 / Day 40 engines.

* :class:`FinancialDataset` - standardised annual values indexed by period.
* :class:`RatioOutcome`     - one ratio for one period, with status, formula,
  inputs, method and an explanation when it cannot be calculated.
* :data:`RATIO_DEFINITIONS` - name, category, unit, formula, direction.
* :func:`calculate_all_ratios` - runs every ratio for every available period.

Status semantics
----------------
OK              value calculated
UNAVAILABLE     an input is missing or the denominator is zero/invalid
NOT_MEANINGFUL  inputs exist but the ratio would mislead (e.g. negative equity)

Values are ``Decimal`` rounded to 6 decimal places; ``inf``/``NaN`` are impossible
because every division checks its denominator first.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Any

from app.database.models import CALC_NOT_MEANINGFUL, CALC_OK, CALC_UNAVAILABLE
from app.utils.helpers import quantize_ratio

HUNDRED = Decimal(100)
TWO = Decimal(2)
ADJACENT_MIN_DAYS = 330
ADJACENT_MAX_DAYS = 400

# reason codes
MISSING_INPUT = "MISSING_INPUT"
ZERO_DENOMINATOR = "ZERO_DENOMINATOR"
NEGATIVE_DENOMINATOR = "NEGATIVE_DENOMINATOR"
NON_POSITIVE_EQUITY = "NON_POSITIVE_EQUITY"
NO_INTEREST_EXPENSE = "NO_INTEREST_EXPENSE"
NO_PREVIOUS_BALANCE = "NO_PREVIOUS_BALANCE"
NON_POSITIVE_NET_INCOME = "NON_POSITIVE_NET_INCOME"
# A published model whose estimation sample excluded this kind of company.
MODEL_NOT_APPLICABLE = "MODEL_NOT_APPLICABLE"

# methods
METHOD_DIRECT = "DIRECT"
METHOD_AVERAGE = "AVERAGE_BALANCE"
METHOD_ENDING = "ENDING_BALANCE_FALLBACK"


@dataclass(frozen=True)
class RatioDefinition:
    code: str
    name: str
    category: str
    unit: str
    formula: str
    higher_is_better: bool


def _d(code: str, name: str, category: str, unit: str, formula: str,
       higher: bool) -> RatioDefinition:
    return RatioDefinition(code, name, category, unit, formula, higher)


RATIO_DEFINITIONS: dict[str, RatioDefinition] = {d.code: d for d in (
    _d("current_ratio", "Current Ratio", "LIQUIDITY", "x",
       "Current Assets / Current Liabilities", True),
    _d("quick_ratio", "Quick Ratio", "LIQUIDITY", "x",
       "Quick Assets / Current Liabilities, Quick Assets = Cash & Equivalents + Short-Term "
       "Investments + Accounts Receivable (fallback: Current Assets - Inventory - Prepaid "
       "Expenses)", True),
    _d("debt_to_equity", "Debt-to-Equity", "SOLVENCY", "x",
       "Total Debt / Shareholders' Equity", False),
    _d("debt_ratio", "Debt Ratio", "SOLVENCY", "x", "Total Liabilities / Total Assets", False),
    _d("interest_coverage", "Interest Coverage", "SOLVENCY", "x",
       "EBIT / |Interest Expense| (Operating Income used as EBIT proxy when EBIT is absent)",
       True),
    _d("net_profit_margin", "Net Profit Margin", "PROFITABILITY", "%",
       "Net Income / Total Revenue x 100", True),
    _d("return_on_assets", "Return on Assets (ROA)", "PROFITABILITY", "%",
       "Net Income / Average Total Assets x 100", True),
    _d("return_on_equity", "Return on Equity (ROE)", "PROFITABILITY", "%",
       "Net Income / Average Shareholders' Equity x 100", True),
    _d("asset_turnover", "Asset Turnover", "EFFICIENCY", "x",
       "Total Revenue / Average Total Assets", True),
    _d("inventory_turnover", "Inventory Turnover", "EFFICIENCY", "x",
       "Cost of Revenue / Average Inventory", True),
    _d("operating_cash_flow_margin", "Operating Cash Flow Margin", "CASH_FLOW", "%",
       "Operating Cash Flow / Total Revenue x 100", True),
    _d("cash_conversion", "Cash Conversion", "CASH_FLOW", "x",
       "Operating Cash Flow / Net Income (only when Net Income > 0)", True),
    _d("free_cash_flow_margin", "Free Cash Flow Margin", "CASH_FLOW", "%",
       "Free Cash Flow / Total Revenue x 100 (provider FCF; fallback Operating Cash Flow + "
       "Capital Expenditure)", True),
    _d("return_on_capital_employed", "Return on Capital Employed (ROCE)", "PROFITABILITY", "%",
       "EBIT / Average Capital Employed x 100, Capital Employed = Total Assets - Current "
       "Liabilities (Operating Income used as EBIT proxy when EBIT is absent)", True),
    _d("dividend_payout_ratio", "Dividend Payout Ratio", "SHAREHOLDER", "%",
       "|Cash Dividends Paid| / Net Income x 100 (Net Income > 0)", False),
    _d("price_to_earnings", "Price / Earnings (P/E)", "VALUATION", "x",
       "Market Capitalisation / latest annual Net Income (Net Income > 0)", False),
    _d("earnings_yield", "Earnings Yield", "VALUATION", "%",
       "latest annual Net Income / Market Capitalisation x 100", True),
    _d("price_to_book", "Price / Book (P/B)", "VALUATION", "x",
       "Market Capitalisation / Shareholders' Equity (equity > 0)", False),
    _d("ev_to_ebitda", "EV / EBITDA", "VALUATION", "x",
       "(Market Cap + Total Debt - Cash & Equivalents) / EBITDA (EBITDA > 0)", False),
    _d("dividend_yield", "Dividend Yield", "VALUATION", "%",
       "|Cash Dividends Paid| (latest year) / Market Capitalisation x 100", True),
    # Established third-party scores (see app/analysis/benchmark_scores.py); not in the
    # health score, used for display, red flags and validation.
    _d("piotroski_f_score", "Piotroski F-Score (0-9)", "BENCHMARK", "pts",
       "Sum of 9 binary signals (Piotroski 2000)", True),
    _d("altman_z_score", "Altman Z''-Score", "BENCHMARK", "pts",
       "6.56 WC/TA + 3.26 RE/TA + 6.72 EBIT/TA + 1.05 BookEquity/TL (Altman Z'')", True),
    _d("beneish_m_score", "Beneish M-Score", "BENCHMARK", "pts",
       "-4.84 + 0.920 DSRI + 0.528 GMI + 0.404 AQI + 0.892 SGI + 0.115 DEPI - 0.172 SGAI "
       "+ 4.679 TATA - 0.327 LVGI (Beneish 1999)", False),
    _d("sws_health_checks", "Simply Wall St Health Checks (0-6)", "BENCHMARK", "pts",
       "Count of 6 pass/fail balance-sheet checks (Simply Wall St model)", True),
    _d("altman_z_original", "Altman Z-Score (original, market value)", "BENCHMARK", "pts",
       "1.2 WC/TA + 1.4 RE/TA + 3.3 EBIT/TA + 0.6 MarketCap/TL + 1.0 Sales/TA (Altman 1968)",
       True),
)}


@dataclass
class FinancialDataset:
    """Standardised annual values: ``values[period][metric_code] -> Decimal``."""

    ticker: str
    currency: str | None
    values: dict[date, dict[str, Decimal]] = field(default_factory=dict)
    excluded_periods: list[date] = field(default_factory=list)
    # Latest market snapshot: {"market_cap", "price", "currency", "fetched_at"} or None
    market: dict[str, Any] | None = None

    def exclude_sparse_periods(self) -> list[date]:
        """Drop periods where core metrics are largely missing (see validator)."""
        from app.cleaning.validator import is_sparse_period

        sparse = [p for p in self.periods if is_sparse_period(set(self.values[p]))]
        for period in sparse:
            del self.values[period]
        self.excluded_periods.extend(sparse)
        return sparse

    @property
    def periods(self) -> list[date]:
        """Available fiscal period ends, oldest first."""
        return sorted(self.values)

    def get(self, period: date | None, code: str) -> Decimal | None:
        if period is None:
            return None
        return self.values.get(period, {}).get(code)

    def previous_period(self, period: date) -> date | None:
        earlier = [p for p in self.periods if p < period]
        return earlier[-1] if earlier else None

    def adjacent_previous(self, period: date) -> date | None:
        """The previous period only if it is the immediately preceding fiscal year."""
        previous = self.previous_period(period)
        if previous is None:
            return None
        gap = (period - previous).days
        return previous if ADJACENT_MIN_DAYS <= gap <= ADJACENT_MAX_DAYS else None

    @classmethod
    def from_records(cls, ticker: str, currency: str | None,
                     records: Iterable[tuple[date, str, Decimal | None]]) -> FinancialDataset:
        dataset = cls(ticker=ticker, currency=currency)
        for period, code, value in records:
            dataset.values.setdefault(period, {})
            if value is not None:
                dataset.values[period][code] = value
        return dataset


@dataclass
class RatioOutcome:
    code: str
    period: date
    status: str
    value: Decimal | None = None
    reason_code: str | None = None
    method: str | None = None
    inputs: dict[str, Any] = field(default_factory=dict)
    explanation: str | None = None

    @property
    def definition(self) -> RatioDefinition:
        return RATIO_DEFINITIONS[self.code]

    @property
    def ok(self) -> bool:
        return self.status == CALC_OK


def ok(code: str, period: date, value: Decimal, inputs: dict[str, Any],
       method: str = METHOD_DIRECT, explanation: str | None = None) -> RatioOutcome:
    return RatioOutcome(code, period, CALC_OK, quantize_ratio(value), None, method, inputs,
                        explanation)


def unavailable(code: str, period: date, reason: str, explanation: str,
                inputs: dict[str, Any] | None = None, method: str | None = None,
                not_meaningful: bool = False) -> RatioOutcome:
    status = CALC_NOT_MEANINGFUL if not_meaningful else CALC_UNAVAILABLE
    return RatioOutcome(code, period, status, None, reason, method, inputs or {}, explanation)


def missing_names(inputs: dict[str, Decimal | None]) -> list[str]:
    return [name for name, value in inputs.items() if value is None]


def missing_outcome(code: str, period: date, inputs: dict[str, Decimal | None],
                    method: str | None = None) -> RatioOutcome | None:
    """Return an UNAVAILABLE outcome if any input is missing, else ``None``."""
    missing = missing_names(inputs)
    if not missing:
        return None
    return unavailable(code, period, MISSING_INPUT,
                       f"Missing input(s) for FY ending {period}: {', '.join(missing)}. "
                       "Missing values are never treated as zero.", inputs, method)


def denominator_problem(code: str, period: date, denominator: Decimal, label: str,
                        inputs: dict[str, Any], method: str | None = None) -> RatioOutcome | None:
    """UNAVAILABLE for a zero denominator, NOT_MEANINGFUL for a negative one."""
    if denominator == 0:
        return unavailable(code, period, ZERO_DENOMINATOR,
                           f"{label} is zero; division is undefined.", inputs, method)
    if denominator < 0:
        return unavailable(code, period, NEGATIVE_DENOMINATOR,
                           f"{label} is negative ({denominator}); the ratio is not meaningful.",
                           inputs, method, not_meaningful=True)
    return None


@dataclass
class AverageBalance:
    value: Decimal | None
    method: str | None
    inputs: dict[str, Any]
    problem: str | None = None


def average_balance(dataset: FinancialDataset, period: date, code: str,
                    fallback: str) -> AverageBalance:
    """(current + previous) / 2, or the labelled ending-balance fallback.

    The previous balance is used only if it belongs to the immediately preceding
    fiscal year (330-400 days earlier); otherwise the fallback policy applies.
    """
    current = dataset.get(period, code)
    previous_period = dataset.adjacent_previous(period)
    previous = dataset.get(previous_period, code)
    inputs: dict[str, Any] = {f"{code}_end": current, f"{code}_previous": previous,
                              "previous_period": previous_period}
    if current is None:
        return AverageBalance(None, None, inputs, f"{code} missing for {period}")
    if previous is not None:
        return AverageBalance((current + previous) / TWO, METHOD_AVERAGE, inputs)
    if fallback == "ending_balance":
        return AverageBalance(current, METHOD_ENDING, inputs)
    return AverageBalance(None, None, inputs,
                          f"previous-year {code} unavailable and the ending-balance fallback "
                          "is disabled")


def fallback_note(method: str | None) -> str | None:
    if method == METHOD_ENDING:
        return ("Previous-year balance unavailable: ENDING balance used instead of the "
                "average (labelled fallback).")
    return None


def calculate_all_ratios(dataset: FinancialDataset, fallback: str = "ending_balance",
                         sector: str | None = None) -> list[RatioOutcome]:
    """Every ratio (Day 30 + Day 40 + cash-flow ratios) for every available period.

    ``sector`` only switches off benchmark models that do not apply to the sector
    (see :func:`app.analysis.benchmark_scores.apply_model_applicability`).
    """
    from app.analysis.benchmark_scores import (apply_model_applicability,
                                               calculate_benchmark_scores)
    from app.analysis.valuation import calculate_extra_ratios, calculate_valuation
    from app.analysis.liquidity_solvency import calculate_liquidity_solvency
    from app.analysis.profitability_efficiency import (calculate_cash_flow_ratios,
                                                       calculate_profitability_efficiency)

    outcomes: list[RatioOutcome] = []
    for period in dataset.periods:
        outcomes.extend(calculate_liquidity_solvency(dataset, period))
        outcomes.extend(calculate_profitability_efficiency(dataset, period, fallback))
        outcomes.extend(calculate_cash_flow_ratios(dataset, period))
        outcomes.extend(calculate_benchmark_scores(dataset, period))
        outcomes.extend(calculate_extra_ratios(dataset, period, fallback))
    if dataset.periods and dataset.market:
        outcomes.extend(calculate_valuation(dataset, dataset.periods[-1]))
    return apply_model_applicability(outcomes, sector)


def index_outcomes(outcomes: Iterable[RatioOutcome]) -> dict[tuple[str, date], RatioOutcome]:
    return {(o.code, o.period): o for o in outcomes}
