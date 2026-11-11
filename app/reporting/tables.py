"""Tabular outputs built from analysis results.

All console text is plain ASCII so it renders in any Windows terminal code page.
Unavailable values are shown as ``N/A`` (missing input / zero denominator) or
``N/M`` (not meaningful, e.g. negative equity) with the reason listed under the
table - never as 0, ``inf`` or a misleading number.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import date
from decimal import Decimal
from typing import Any

import pandas as pd

from app.analysis.health_score import OVERALL, CategoryScore
from app.analysis.ratios import RATIO_DEFINITIONS, RatioOutcome
from app.analysis.red_flags import RedFlag
from app.analysis.trends import TrendOutcome
from app.database.models import CALC_NOT_MEANINGFUL
from app.utils.helpers import fmt_amount, fmt_number

CATEGORY_ORDER = ("LIQUIDITY", "SOLVENCY", "PROFITABILITY", "EFFICIENCY", "CASH_FLOW")
RATIO_TABLE_ORDER = (*CATEGORY_ORDER, "SHAREHOLDER", "VALUATION", "BENCHMARK")
METHOD_MARKERS = {"ENDING_BALANCE_FALLBACK": "e", "EBIT_PROXY_OPERATING_INCOME": "p",
                  "OCF_MINUS_CAPEX": "c"}


def fy_label(period: date) -> str:
    return f"FY{period.year}" if period.month <= 6 else f"FY{period.year} ({period:%b})"


def format_ratio_value(outcome: RatioOutcome) -> str:
    if not outcome.ok:
        return "N/M" if outcome.status == CALC_NOT_MEANINGFUL else "N/A"
    unit = outcome.definition.unit
    suffix = {"%": "%", "x": "x"}.get(unit, "")
    text = f"{float(outcome.value):,.2f}{suffix}"
    marker = METHOD_MARKERS.get(outcome.method or "")
    return f"{text}[{marker}]" if marker else text


def ratio_table(outcomes: Sequence[RatioOutcome]) -> tuple[pd.DataFrame, list[str]]:
    """Wide table (rows = ratios, columns = fiscal years, newest first) + footnotes."""
    periods = sorted({o.period for o in outcomes}, reverse=True)
    index = {(o.code, o.period): o for o in outcomes}
    rows, notes = [], []
    seen_notes: set[str] = set()
    codes = sorted({o.code for o in outcomes},
                   key=lambda c: (RATIO_TABLE_ORDER.index(RATIO_DEFINITIONS[c].category),
                                  list(RATIO_DEFINITIONS).index(c)))
    for code in codes:
        definition = RATIO_DEFINITIONS[code]
        row = {"Category": definition.category.title().replace("_", " "),
               "Ratio": definition.name}
        for period in periods:
            outcome = index.get((code, period))
            row[fy_label(period)] = format_ratio_value(outcome) if outcome else ""
            if outcome is not None and not outcome.ok and outcome.explanation:
                note = f"{definition.name} {fy_label(period)}: {outcome.explanation}"
                if note not in seen_notes:
                    seen_notes.add(note)
                    notes.append(note)
        rows.append(row)
    return pd.DataFrame(rows), notes


def ratio_long_frame(outcomes: Iterable[RatioOutcome]) -> pd.DataFrame:
    return pd.DataFrame([{
        "fiscal_period_end": o.period.isoformat(), "category": o.definition.category,
        "ratio_code": o.code, "ratio_name": o.definition.name, "unit": o.definition.unit,
        "value": None if o.value is None else str(o.value), "status": o.status,
        "reason_code": o.reason_code, "method": o.method, "formula": o.definition.formula,
        "inputs": {k: (str(v) if isinstance(v, Decimal) else v) for k, v in o.inputs.items()},
        "explanation": o.explanation} for o in outcomes])


def trend_frame(trends: Iterable[TrendOutcome], currency: str | None) -> pd.DataFrame:
    rows = []
    for t in trends:
        rows.append({
            "Metric": t.label, "Type": t.comparison_type,
            "From": t.previous_period.isoformat() if t.previous_period else "",
            "To": t.current_period.isoformat(),
            "Previous": fmt_amount(t.previous_value, currency),
            "Current": fmt_amount(t.current_value, currency),
            "Abs change": fmt_amount(t.absolute_change, currency),
            # YOY rows: percentage change; MULTI_YEAR rows: CAGR.
            "% change / CAGR":
                "N/A" if t.percentage_change is None else f"{float(t.percentage_change):+.2f}%",
            "Direction": t.direction, "Status": t.status, "Note": t.explanation or "",
        })
    return pd.DataFrame(rows)


def trend_summary_frame(trends: Iterable[TrendOutcome], currency: str | None) -> pd.DataFrame:
    """One row per metric: latest YOY change and multi-year direction/CAGR."""
    latest_yoy: dict[str, TrendOutcome] = {}
    multi: dict[str, TrendOutcome] = {}
    for t in trends:
        if t.comparison_type == "MULTI_YEAR":
            multi[t.metric_code] = t
        elif t.metric_code not in latest_yoy or t.current_period > \
                latest_yoy[t.metric_code].current_period:
            latest_yoy[t.metric_code] = t
    rows = []
    for code, summary in multi.items():
        yoy = latest_yoy.get(code)
        rows.append({
            "Metric": summary.label,
            "Latest": fmt_amount(summary.current_value, currency),
            "YoY change": "N/A" if yoy is None or yoy.absolute_change is None else
            fmt_amount(yoy.absolute_change, currency),
            "YoY %": "N/A" if yoy is None or yoy.percentage_change is None else
            f"{float(yoy.percentage_change):+.1f}%",
            "Multi-year": summary.direction,
            "CAGR": "N/A" if summary.percentage_change is None else
            f"{float(summary.percentage_change):+.1f}%",
            "Years": f"{summary.previous_period.year}-{summary.current_period.year}"
            if summary.previous_period else "",
        })
    return pd.DataFrame(rows)


def red_flag_frame(flags: Iterable[RedFlag]) -> pd.DataFrame:
    return pd.DataFrame([{
        "Period": fy_label(f.period), "Severity": f.severity, "Rule": f.rule_code,
        "Warning": f.title, "Condition": f.triggered_condition} for f in flags])


def score_frame(scores: dict[date, list[CategoryScore]]) -> pd.DataFrame:
    """Rows = categories (+ overall), columns = fiscal years newest first."""
    periods = sorted(scores, reverse=True)
    rows = []
    for category in (*CATEGORY_ORDER, OVERALL):
        row: dict[str, Any] = {"Category": category.title().replace("_", " ")}
        for period in periods:
            match = next((c for c in scores[period] if c.category == category), None)
            if match is None or match.score is None:
                row[fy_label(period)] = "N/A"
            else:
                row[fy_label(period)] = f"{float(match.score):.1f}"
        rows.append(row)
    return pd.DataFrame(rows)


def score_long_frame(scores: dict[date, list[CategoryScore]]) -> pd.DataFrame:
    return pd.DataFrame([{
        "fiscal_period_end": c.period.isoformat(), "category": c.category,
        "score": None if c.score is None else str(c.score), "status": c.status,
        "data_coverage": c.coverage, "interpretation": c.interpretation,
        "explanation": c.explanation} for cats in scores.values() for c in cats])


def to_text(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(no rows)"
    with pd.option_context("display.max_columns", None, "display.width", 250,
                           "display.max_colwidth", 70):
        return frame.to_string(index=False)


def to_markdown(frame: pd.DataFrame) -> str:
    """Minimal Markdown table writer (avoids an extra 'tabulate' dependency)."""
    if frame.empty:
        return "_No rows._"
    columns = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for _, row in frame.iterrows():
        cells = [str(v).replace("|", "/").replace("\n", " ") for v in row.tolist()]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def legend() -> str:
    return ("N/A = cannot be calculated (missing input or zero denominator); N/M = not "
            "meaningful (e.g. negative equity). [e] ending-balance fallback used instead of "
            "average; [p] Operating Income used as EBIT proxy; [c] FCF = OCF - |CapEx|.")


def fmt_score(score: Decimal | None) -> str:
    return "N/A" if score is None else fmt_number(score, 1)
