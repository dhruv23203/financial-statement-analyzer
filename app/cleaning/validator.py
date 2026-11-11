"""Validation of standardised records and of the data set as a whole (pure functions)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from datetime import date
from decimal import Decimal

from app.cleaning.standardizer import StandardizedRecord
from app.database.models import (MAPPING_CONFLICT, MAPPING_UNMAPPED, VALIDATION_INVALID,
                                 VALIDATION_MISSING,
                                 VALIDATION_VALID, VALIDATION_WARNING)
from config.metric_aliases import ESSENTIAL_METRICS, METRICS_BY_CODE, STATEMENT_LABELS
from config.settings import CORE_METRICS, SPARSE_PERIOD_MIN_CORE_METRICS


def is_sparse_period(present_codes: set[str]) -> bool:
    """True when fewer than the configured number of core metrics are reported."""
    return len(present_codes & set(CORE_METRICS)) < SPARSE_PERIOD_MIN_CORE_METRICS

# Amounts larger than this (in reporting-currency units) are flagged for review:
# even the largest companies report revenues well below 1e15 in their currency.
EXTREME_AMOUNT = Decimal("1e15")
# Annual period ends further apart than this indicate a missing fiscal year.
MAX_ANNUAL_GAP_DAYS = 400
# Two period ends closer than this are likely the same fiscal year reported twice.
MIN_DISTINCT_PERIOD_DAYS = 300
# Tolerance for the accounting identity Assets = Liabilities + Equity.
BALANCE_IDENTITY_TOLERANCE = Decimal("0.02")


def validate_record(record: StandardizedRecord) -> tuple[str, list[str]]:
    """Return ``(validation_status, messages)`` for one standardised record."""
    messages = list(record.messages)
    if record.value is None:
        return VALIDATION_MISSING, messages + ["Value missing in source (not treated as 0)."]
    if not record.value.is_finite():
        return VALIDATION_INVALID, messages + ["Non-finite value."]
    status = VALIDATION_WARNING if record.mapping_status == MAPPING_CONFLICT \
        else VALIDATION_VALID
    definition = METRICS_BY_CODE.get(record.metric_code)
    if definition is not None:
        if definition.expected_sign == "non_negative" and record.value < 0:
            status = VALIDATION_WARNING
            messages.append(f"Unexpected negative value for {definition.name}.")
        elif definition.expected_sign == "non_positive" and record.value > 0:
            status = VALIDATION_WARNING
            messages.append(f"Unexpected positive value for {definition.name} "
                            "(provider normally reports outflows as negative).")
        if abs(record.value) >= EXTREME_AMOUNT and not definition.code.endswith("shares"):
            status = VALIDATION_WARNING
            messages.append("Extremely large amount; verify units/currency.")
    return status, messages


def validate_dataset(records: Sequence[StandardizedRecord],
                     available_statements: Sequence[str]) -> list[str]:
    """Data-set level checks; returns human-readable warnings."""
    warnings: list[str] = []
    for statement in ("income_statement", "balance_sheet", "cash_flow"):
        if statement not in available_statements:
            warnings.append(f"{STATEMENT_LABELS[statement]} is unavailable for this fetch run.")

    values: dict[tuple[str, date], Decimal | None] = {}
    periods_by_statement: dict[str, set[date]] = defaultdict(set)
    for record in records:
        periods_by_statement[record.statement_type].add(record.fiscal_period_end)
        if record.mapping_status != MAPPING_UNMAPPED:
            values[(record.metric_code, record.fiscal_period_end)] = record.value

    all_periods = sorted(set().union(*periods_by_statement.values())) if periods_by_statement \
        else []
    present: dict[date, set[str]] = defaultdict(set)
    for (code, period), value in values.items():
        if value is not None:
            present[period].add(code)
    sparse = {p for p in all_periods if is_sparse_period(present[p])}
    for period in sorted(sparse):
        warnings.append(f"Period {period} is sparse ({len(present[period])} mapped values, "
                        "core metrics largely missing); it is kept in the database but "
                        "excluded from analysis.")
    all_periods = [p for p in all_periods if p not in sparse]
    if len(all_periods) == 1:
        warnings.append("Only one fiscal period available: trends and average-balance "
                        "ratios cannot be calculated.")
    for earlier, later in zip(all_periods, all_periods[1:]):
        gap = (later - earlier).days
        if gap > MAX_ANNUAL_GAP_DAYS:
            warnings.append(f"Missing historical year: gap of {gap} days between "
                            f"{earlier} and {later}.")
        elif gap < MIN_DISTINCT_PERIOD_DAYS:
            warnings.append(f"Period ends {earlier} and {later} are only {gap} days apart "
                            "(possible duplicate or changed fiscal year).")
    year_ends = {(p.month, p.day) for p in all_periods}
    if len(year_ends) > 1:
        labels = sorted(f"{m:02d}-{d:02d}" for m, d in year_ends)
        warnings.append(f"Inconsistent fiscal year-end dates within the company: {labels}.")

    for statement, codes in ESSENTIAL_METRICS.items():
        if statement not in available_statements:
            continue
        for period in sorted(periods_by_statement.get(statement, set()) - sparse):
            missing = [c for c in codes if values.get((c, period)) is None]
            if missing:
                warnings.append(f"{STATEMENT_LABELS[statement]} {period}: essential field(s) "
                                f"missing: {', '.join(missing)}.")

    for period in sorted(periods_by_statement.get("balance_sheet", set()) - sparse):
        assets = values.get(("total_assets", period))
        liabilities = values.get(("total_liabilities", period))
        equity = values.get(("total_equity_incl_minority", period))
        if equity is None:
            equity = values.get(("stockholders_equity", period))
        if assets and liabilities is not None and equity is not None:
            difference = abs(assets - (liabilities + equity))
            if difference > abs(assets) * BALANCE_IDENTITY_TOLERANCE:
                warnings.append(
                    f"Balance sheet {period}: Total Assets ({assets}) differ from Liabilities + "
                    f"Equity ({liabilities + equity}) by more than 2%.")
    return warnings
