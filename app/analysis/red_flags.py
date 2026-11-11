"""Day 60 - explainable, rules-based red-flag detection.

Every threshold comes from ``config/red_flag_thresholds.json``. A red flag marks a
*potential* concern worth investigating; it is not evidence of fraud or distress.

Rules (evaluated for every fiscal year that has the inputs; "growth" means
percentage change versus the previous available year and is only computed when
the previous value is positive):

RF01 Profit/cash divergence  Net income up (growth > min) while operating cash flow
                             falls (decline >= min). HIGH if OCF falls by
                             >= high_severity_ocf_decline_pct or turns negative.
RF02 Debt growth             Total debt grows >= debt_growth_pct while revenue growth
                             <= revenue_flat_band_pct (flat or falling). HIGH if debt
                             growth >= high_severity_debt_growth_pct.
RF03 Revenue & profit fall   Revenue and net income both decrease. HIGH if revenue falls
                             >= high_severity_decline_pct or net income turns into a loss.
RF04 Negative OCF            Operating cash flow < 0. HIGH if negative for
                             >= consecutive_years_for_high consecutive years.
RF05 Liquidity concern       Current ratio < current_ratio_threshold. HIGH below
                             high_severity_below.
RF06 Weak interest coverage  Interest coverage < interest_coverage_threshold. HIGH below
                             high_severity_below.
RF07 Negative equity         Shareholders' equity < 0 (always HIGH by default).

Rules whose inputs are missing are reported as NOT_EVALUATED (never as "passed").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Any

from app.analysis.ratios import HUNDRED, FinancialDataset, RatioOutcome
from app.utils.helpers import fmt_number, quantize_ratio

TRIGGERED = "TRIGGERED"
NOT_TRIGGERED = "NOT_TRIGGERED"
NOT_EVALUATED = "NOT_EVALUATED"


@dataclass
class RedFlag:
    rule_code: str
    rule_name: str
    severity: str
    period: date
    previous_period: date | None
    title: str
    triggered_condition: str
    description: str
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class RuleEvaluation:
    rule_code: str
    period: date
    status: str
    note: str | None = None


@dataclass
class RedFlagReport:
    flags: list[RedFlag] = field(default_factory=list)
    evaluations: list[RuleEvaluation] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        result = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
        for flag in self.flags:
            result[flag.severity] += 1
        return result

    def not_evaluated(self) -> list[RuleEvaluation]:
        return [e for e in self.evaluations if e.status == NOT_EVALUATED]


def pct_change(current: Decimal, previous: Decimal) -> Decimal | None:
    """Percentage change, only defined for a positive previous value."""
    if previous is None or current is None or previous <= 0:
        return None
    return quantize_ratio((current - previous) / previous * HUNDRED)


def _pct(value: Decimal | None) -> str:
    return "n/a" if value is None else f"{float(value):+.1f}%"


class RedFlagEngine:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.rules = config["rules"]
        self.report = RedFlagReport()

    # ------------------------------------------------------------------ helpers
    def _enabled(self, code: str) -> bool:
        return code in self.rules and bool(self.rules[code].get("enabled", True))

    def _flag(self, code: str, severity: str, period: date, previous: date | None, title: str,
              condition: str, description: str, evidence: dict[str, Any]) -> None:
        rule = self.rules[code]
        self.report.flags.append(RedFlag(code, rule["name"], severity, period, previous, title,
                                         condition, description, evidence))
        self.report.evaluations.append(RuleEvaluation(code, period, TRIGGERED))

    def _skip(self, code: str, period: date, note: str) -> None:
        self.report.evaluations.append(RuleEvaluation(code, period, NOT_EVALUATED, note))

    def _pass(self, code: str, period: date) -> None:
        self.report.evaluations.append(RuleEvaluation(code, period, NOT_TRIGGERED))

    # ------------------------------------------------------------------ rules
    def rf01(self, ds: FinancialDataset, period: date, prev: date) -> None:
        code = "RF01_PROFIT_CASH_DIVERGENCE"
        rule = self.rules[code]
        ni, ni_prev = ds.get(period, "net_income"), ds.get(prev, "net_income")
        ocf, ocf_prev = ds.get(period, "operating_cash_flow"), ds.get(prev, "operating_cash_flow")
        if None in (ni, ni_prev, ocf, ocf_prev):
            return self._skip(code, period, "net income or operating cash flow missing")
        ni_growth, ocf_growth = pct_change(ni, ni_prev), pct_change(ocf, ocf_prev)
        ni_up = ni > ni_prev and (ni_growth is None or
                                  ni_growth > Decimal(str(rule["min_net_income_growth_pct"])))
        ocf_down = ocf < ocf_prev and (ocf_growth is None or
                                       -ocf_growth >= Decimal(str(rule["min_ocf_decline_pct"])))
        if not (ni_up and ocf_down):
            return self._pass(code, period)
        severe = ocf < 0 or (ocf_growth is not None and -ocf_growth >= Decimal(
            str(rule["high_severity_ocf_decline_pct"])))
        self._flag(code, "HIGH" if severe else rule["default_severity"], period, prev,
                   "Profit rising while operating cash flow falls",
                   f"Net income change {_pct(ni_growth)} > {rule['min_net_income_growth_pct']}% "
                   f"and operating cash flow change {_pct(ocf_growth)} <= "
                   f"-{rule['min_ocf_decline_pct']}%",
                   "Reported profit grew but the cash generated by operations fell. Profits "
                   "not backed by cash can indicate aggressive revenue recognition, rising "
                   "receivables or working-capital strain.",
                   {"net_income": ni, "net_income_previous": ni_prev,
                    "net_income_growth_pct": ni_growth, "operating_cash_flow": ocf,
                    "operating_cash_flow_previous": ocf_prev,
                    "operating_cash_flow_growth_pct": ocf_growth})

    def rf02(self, ds: FinancialDataset, period: date, prev: date) -> None:
        code = "RF02_DEBT_GROWTH"
        rule = self.rules[code]
        debt, debt_prev = ds.get(period, "total_debt"), ds.get(prev, "total_debt")
        rev, rev_prev = ds.get(period, "total_revenue"), ds.get(prev, "total_revenue")
        if None in (debt, debt_prev, rev, rev_prev) or rev_prev <= 0:
            return self._skip(code, period, "debt or revenue missing (or previous revenue <= 0)")
        rev_growth = pct_change(rev, rev_prev)
        if debt_prev > 0:
            debt_growth = pct_change(debt, debt_prev)
            significant = debt_growth >= Decimal(str(rule["debt_growth_pct"]))
        else:
            debt_growth = None
            significant = debt > 0  # debt appeared from zero
        flat = rev_growth <= Decimal(str(rule["revenue_flat_band_pct"]))
        if not (significant and flat):
            return self._pass(code, period)
        severe = debt_growth is None or debt_growth >= Decimal(
            str(rule["high_severity_debt_growth_pct"]))
        self._flag(code, "HIGH" if severe else rule["default_severity"], period, prev,
                   "Debt increasing while revenue is flat or falling",
                   f"Total debt change {_pct(debt_growth)} >= {rule['debt_growth_pct']}% and "
                   f"revenue change {_pct(rev_growth)} <= {rule['revenue_flat_band_pct']}%",
                   "Borrowing grew significantly without matching revenue growth, which can "
                   "pressure future interest payments and solvency.",
                   {"total_debt": debt, "total_debt_previous": debt_prev,
                    "debt_growth_pct": debt_growth, "total_revenue": rev,
                    "total_revenue_previous": rev_prev, "revenue_growth_pct": rev_growth})

    def rf03(self, ds: FinancialDataset, period: date, prev: date) -> None:
        code = "RF03_REVENUE_PROFIT_DECLINE"
        rule = self.rules[code]
        rev, rev_prev = ds.get(period, "total_revenue"), ds.get(prev, "total_revenue")
        ni, ni_prev = ds.get(period, "net_income"), ds.get(prev, "net_income")
        if None in (rev, rev_prev, ni, ni_prev):
            return self._skip(code, period, "revenue or net income missing")
        if not (rev < rev_prev and ni < ni_prev):
            return self._pass(code, period)
        rev_growth, ni_growth = pct_change(rev, rev_prev), pct_change(ni, ni_prev)
        threshold = Decimal(str(rule["high_severity_decline_pct"]))
        severe = (rev_growth is not None and -rev_growth >= threshold) or (ni < 0 <= ni_prev)
        self._flag(code, "HIGH" if severe else rule["default_severity"], period, prev,
                   "Revenue and net income both declined",
                   f"Revenue change {_pct(rev_growth)} < 0 and net income change "
                   f"{_pct(ni_growth)} < 0",
                   "Both the top line and the bottom line fell, suggesting weakening demand, "
                   "pricing pressure or rising costs.",
                   {"total_revenue": rev, "total_revenue_previous": rev_prev,
                    "revenue_growth_pct": rev_growth, "net_income": ni,
                    "net_income_previous": ni_prev, "net_income_growth_pct": ni_growth})

    def rf04(self, ds: FinancialDataset, period: date) -> None:
        code = "RF04_NEGATIVE_OPERATING_CASH_FLOW"
        rule = self.rules[code]
        ocf = ds.get(period, "operating_cash_flow")
        if ocf is None:
            return self._skip(code, period, "operating cash flow missing")
        if ocf >= 0:
            return self._pass(code, period)
        streak, cursor = 0, period
        while cursor is not None and (value := ds.get(cursor, "operating_cash_flow")) \
                is not None and value < 0:
            streak += 1
            cursor = ds.previous_period(cursor)
        previous = ds.previous_period(period)
        severe = streak >= int(rule["consecutive_years_for_high"])
        self._flag(code, "HIGH" if severe else rule["default_severity"], period, previous,
                   "Negative operating cash flow",
                   f"Operating cash flow {fmt_number(ocf, 0)} < 0 "
                   f"({streak} consecutive year(s))",
                   "Core operations consumed cash instead of generating it; the company must "
                   "rely on reserves, borrowing or equity to fund itself.",
                   {"operating_cash_flow": ocf, "operating_cash_flow_previous":
                    ds.get(previous, "operating_cash_flow"), "consecutive_negative_years": streak})

    def _ratio_rule(self, code: str, ratio_code: str, threshold_key: str, title: str,
                    description: str, ratios: dict[tuple[str, date], RatioOutcome],
                    ds: FinancialDataset, period: date) -> None:
        rule = self.rules[code]
        outcome = ratios.get((ratio_code, period))
        if outcome is None or not outcome.ok:
            reason = outcome.explanation if outcome else "ratio not calculated"
            return self._skip(code, period, f"{ratio_code} unavailable: {reason}")
        threshold = Decimal(str(rule[threshold_key]))
        if outcome.value >= threshold:
            return self._pass(code, period)
        previous = ds.previous_period(period)
        previous_outcome = ratios.get((ratio_code, previous)) if previous else None
        severe = outcome.value < Decimal(str(rule["high_severity_below"]))
        self._flag(code, "HIGH" if severe else rule["default_severity"], period, previous,
                   title, f"{ratio_code} {fmt_number(outcome.value)} < {threshold}", description,
                   {ratio_code: outcome.value, f"{ratio_code}_previous":
                    previous_outcome.value if previous_outcome else None,
                    "threshold": threshold, "inputs": outcome.inputs})

    def _score_rule(self, code: str, ratio_code: str, key: str, below: bool,
                    ratios: dict[tuple[str, date], RatioOutcome], ds: FinancialDataset,
                    period: date, title: str, description: str) -> None:
        """Flag when a benchmark score crosses a published threshold."""
        rule = self.rules[code]
        outcome = ratios.get((ratio_code, period))
        if outcome is None or not outcome.ok:
            return self._skip(code, period, f"{ratio_code} unavailable")
        threshold = Decimal(str(rule[key]))
        triggered = outcome.value < threshold if below else outcome.value > threshold
        if not triggered:
            return self._pass(code, period)
        previous = ds.previous_period(period)
        previous_outcome = ratios.get((ratio_code, previous)) if previous else None
        sign = "<" if below else ">"
        self._flag(code, rule["default_severity"], period, previous, title,
                   f"{ratio_code} {float(outcome.value):.2f} {sign} {threshold}", description,
                   {ratio_code: outcome.value, f"{ratio_code}_previous":
                    previous_outcome.value if previous_outcome else None,
                    "threshold": threshold})

    def rf07(self, ds: FinancialDataset, period: date) -> None:
        code = "RF07_NEGATIVE_EQUITY"
        rule = self.rules[code]
        equity = ds.get(period, "stockholders_equity")
        if equity is None:
            return self._skip(code, period, "shareholders' equity missing")
        if equity >= 0:
            return self._pass(code, period)
        previous = ds.previous_period(period)
        self._flag(code, rule["default_severity"], period, previous,
                   "Negative shareholders' equity",
                   f"Shareholders' equity {fmt_number(equity, 0)} < 0",
                   "Total liabilities exceed total assets (accumulated losses or large "
                   "buybacks/dividends). Equity-based ratios such as ROE and debt-to-equity "
                   "are not meaningful.",
                   {"stockholders_equity": equity, "stockholders_equity_previous":
                    ds.get(previous, "stockholders_equity"),
                    "total_assets": ds.get(period, "total_assets"),
                    "total_liabilities": ds.get(period, "total_liabilities")})

    # ------------------------------------------------------------------ driver
    def run(self, ds: FinancialDataset,
            ratios: dict[tuple[str, date], RatioOutcome]) -> RedFlagReport:
        periods = ds.periods
        for index, period in enumerate(periods):
            prev = periods[index - 1] if index else None
            pairwise = (("RF01_PROFIT_CASH_DIVERGENCE", self.rf01),
                        ("RF02_DEBT_GROWTH", self.rf02),
                        ("RF03_REVENUE_PROFIT_DECLINE", self.rf03))
            for code, rule in pairwise:
                if not self._enabled(code):
                    continue
                if prev is None:
                    self._skip(code, period, "no previous year to compare with")
                else:
                    rule(ds, period, prev)
            if self._enabled("RF04_NEGATIVE_OPERATING_CASH_FLOW"):
                self.rf04(ds, period)
            if self._enabled("RF05_LIQUIDITY_CONCERN"):
                self._ratio_rule(
                    "RF05_LIQUIDITY_CONCERN", "current_ratio", "current_ratio_threshold",
                    "Current ratio below threshold",
                    "Current assets may not comfortably cover obligations due within a year.",
                    ratios, ds, period)
            if self._enabled("RF06_WEAK_INTEREST_COVERAGE"):
                self._ratio_rule(
                    "RF06_WEAK_INTEREST_COVERAGE", "interest_coverage",
                    "interest_coverage_threshold", "Weak interest coverage",
                    "Operating earnings provide little cushion over interest costs.",
                    ratios, ds, period)
            if self._enabled("RF07_NEGATIVE_EQUITY"):
                self.rf07(ds, period)
            if self._enabled("RF08_ALTMAN_DISTRESS"):
                self._score_rule("RF08_ALTMAN_DISTRESS", "altman_z_score", "distress_below",
                                 below=True, ratios=ratios, ds=ds, period=period,
                                 title="Altman Z''-Score in the distress zone",
                                 description="Altman's Z'' model combines working capital, "
                                 "retained earnings, EBIT and book leverage; scores below 1.10 "
                                 "have historically been associated with a high probability "
                                 "of financial distress (Altman, Hartzell & Peck 1995).")
            if self._enabled("RF09_BENEISH_MANIPULATION"):
                self._score_rule("RF09_BENEISH_MANIPULATION", "beneish_m_score", "threshold",
                                 below=False, ratios=ratios, ds=ds, period=period,
                                 title="Beneish M-Score above the manipulation threshold",
                                 description="The profile of receivables, margins, asset "
                                 "quality, growth, depreciation, SG&A, leverage and accruals "
                                 "resembles companies that manipulated earnings (Beneish 1999). "
                                 "This is a screening signal with many false positives, not "
                                 "evidence of manipulation.")
        return self.report


def detect_red_flags(ds: FinancialDataset, ratios: dict[tuple[str, date], RatioOutcome],
                     config: dict[str, Any]) -> RedFlagReport:
    return RedFlagEngine(config).run(ds, ratios)
