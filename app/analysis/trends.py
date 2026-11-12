"""Day 50 - historical trend analysis (no forecasting).

Year-over-year (YOY), for each pair of consecutive available fiscal years:
    absolute change   = current - previous                     (always when both exist)
    percentage change = (current - previous) / previous x 100  (only when previous > 0)

Edge cases
    previous == 0  -> percentage undefined; status ABSOLUTE_ONLY
    previous <  0  -> percentage would be misleading (e.g. a loss shrinking from -100
                      to -10 would read as -90%); status ABSOLUTE_ONLY with explanation
    value missing  -> status UNAVAILABLE
    non-adjacent years (gap > 400 days) are compared but the explanation says so.

Multi-year (MULTI_YEAR), first vs latest available year:
    direction  UP / DOWN when the endpoints and the majority of yearly moves agree,
               otherwise MIXED; FLAT when the total change is within +/-1%.
    CAGR       ((last / first) ^ (1 / years) - 1) x 100, only when both are > 0.

Direction compares raw numbers: for outflow lines (investing cash flow is usually
negative) UP means "less negative".
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from app.analysis.ratios import HUNDRED, FinancialDataset
from app.database.models import TREND_ABSOLUTE_ONLY, TREND_OK, TREND_UNAVAILABLE
from app.utils.helpers import quantize_ratio, years_between

FLAT_THRESHOLD_PCT = Decimal("1")
NON_ADJACENT_DAYS = 400

TREND_METRICS: dict[str, tuple[str, str]] = {
    # code: (label, group)
    "total_revenue": ("Revenue", "Revenue"),
    "gross_profit": ("Gross Profit", "Profit"),
    "operating_income": ("Operating Income", "Profit"),
    "net_income": ("Net Income", "Profit"),
    "total_debt": ("Total Debt", "Debt"),
    "total_liabilities": ("Total Liabilities", "Debt"),
    "operating_cash_flow": ("Operating Cash Flow", "Cash Flow"),
    "investing_cash_flow": ("Investing Cash Flow", "Cash Flow"),
    "financing_cash_flow": ("Financing Cash Flow", "Cash Flow"),
    "free_cash_flow": ("Free Cash Flow (provider-reported)", "Cash Flow"),
}


@dataclass
class TrendOutcome:
    metric_code: str
    comparison_type: str  # YOY | MULTI_YEAR
    current_period: date
    previous_period: date | None
    current_value: Decimal | None
    previous_value: Decimal | None
    absolute_change: Decimal | None
    percentage_change: Decimal | None
    direction: str  # UP | DOWN | FLAT | MIXED | UNAVAILABLE
    status: str
    explanation: str | None = None

    @property
    def label(self) -> str:
        return TREND_METRICS.get(self.metric_code, (self.metric_code, ""))[0]


def _direction(change: Decimal, base: Decimal | None) -> str:
    if change == 0:
        return "FLAT"
    if base is not None and base > 0 and abs(change / base * HUNDRED) < FLAT_THRESHOLD_PCT:
        return "FLAT"
    return "UP" if change > 0 else "DOWN"


def year_over_year(metric: str, current_period: date, previous_period: date,
                   current: Decimal | None, previous: Decimal | None) -> TrendOutcome:
    notes = []
    if (current_period - previous_period).days > NON_ADJACENT_DAYS:
        # A change across a missing year is a multi-year change; labelling it
        # "year-over-year" would overstate one year's movement, so it is not reported.
        return TrendOutcome(metric, "YOY", current_period, previous_period, current, previous,
                            None, None, "UNAVAILABLE", TREND_UNAVAILABLE,
                            f"Non-adjacent years ({previous_period} -> {current_period}): an "
                            "intermediate fiscal year is missing, so no year-over-year change "
                            "is reported.")
    if current is None or previous is None:
        which = "current" if current is None else "previous"
        return TrendOutcome(metric, "YOY", current_period, previous_period, current, previous,
                            None, None, "UNAVAILABLE", TREND_UNAVAILABLE,
                            " ".join(notes + [f"The {which}-year value is missing."]))
    change = current - previous
    if previous > 0:
        pct = quantize_ratio(change / previous * HUNDRED)
        status = TREND_OK
        if current < 0:
            notes.append("Value turned negative.")
    else:
        pct = None
        status = TREND_ABSOLUTE_ONLY
        if previous == 0:
            notes.append("Previous value is zero: percentage change is undefined; "
                         "absolute change reported.")
        else:
            notes.append(f"Previous value is negative ({previous}): a percentage change would "
                         "be misleading, so only the absolute change is reported.")
    return TrendOutcome(metric, "YOY", current_period, previous_period, current, previous,
                        quantize_ratio(change), pct, _direction(change, previous), status,
                        " ".join(notes) or None)


def multi_year(metric: str, dataset: FinancialDataset,
               yoy: list[TrendOutcome]) -> TrendOutcome | None:
    points = [(p, dataset.get(p, metric)) for p in dataset.periods
              if dataset.get(p, metric) is not None]
    if not dataset.periods:
        return None
    latest = dataset.periods[-1]
    if len(points) < 2:
        return TrendOutcome(metric, "MULTI_YEAR", latest, points[0][0] if points else None,
                            points[-1][1] if points else None, None, None, None,
                            "UNAVAILABLE", TREND_UNAVAILABLE,
                            "At least two years of data are needed for a multi-year trend.")
    (first_period, first), (last_period, last) = points[0], points[-1]
    change = last - first
    moves = [t.direction for t in yoy if t.status != TREND_UNAVAILABLE]
    ups, downs = moves.count("UP"), moves.count("DOWN")
    years = years_between(first_period, last_period)
    explanation = [f"{len(points)} years ({first_period} to {last_period}); "
                   f"{ups} up / {downs} down year-over-year moves."]
    if change == 0 or (first != 0 and abs(change / abs(first) * HUNDRED) < FLAT_THRESHOLD_PCT):
        direction = "FLAT"
    elif change > 0 and ups >= downs:
        direction = "UP"
    elif change < 0 and downs >= ups:
        direction = "DOWN"
    else:
        direction = "MIXED"
    cagr = None
    status = TREND_OK
    if first > 0 and last > 0 and years > 0:
        cagr = quantize_ratio(Decimal(str(((float(last) / float(first)) ** (1 / years) - 1)
                                          * 100)))
        explanation.append(f"CAGR over {years:.2f} years.")
    else:
        status = TREND_ABSOLUTE_ONLY
        explanation.append("CAGR not calculated because the first or last value is not "
                           "positive.")
    return TrendOutcome(metric, "MULTI_YEAR", last_period, first_period, last, first,
                        quantize_ratio(change), cagr, direction, status, " ".join(explanation))


def analyse_trends(dataset: FinancialDataset,
                   metrics: dict[str, tuple[str, str]] | None = None) -> list[TrendOutcome]:
    """YOY outcomes for every consecutive pair plus one MULTI_YEAR outcome per metric."""
    metrics = metrics or TREND_METRICS
    periods = dataset.periods
    outcomes: list[TrendOutcome] = []
    for metric in metrics:
        yoy = [year_over_year(metric, cur, prev, dataset.get(cur, metric),
                              dataset.get(prev, metric))
               for prev, cur in zip(periods, periods[1:])]
        if all(dataset.get(p, metric) is None for p in periods):
            continue  # metric never reported: nothing to analyse (listed as unavailable)
        outcomes.extend(yoy)
        summary = multi_year(metric, dataset, yoy)
        if summary is not None:
            outcomes.append(summary)
    return outcomes


def unavailable_trend_metrics(dataset: FinancialDataset,
                              metrics: dict[str, tuple[str, str]] | None = None) -> list[str]:
    metrics = metrics or TREND_METRICS
    return [m for m in metrics if all(dataset.get(p, m) is None for p in dataset.periods)]
