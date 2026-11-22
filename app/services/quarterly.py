"""Quarterly and trailing-twelve-month (TTM) view of a company (display only).

Built from the provider's quarterly income statement and cash-flow statement. Columns
are the company's own **fiscal quarter-end dates** (not calendar quarters). Rules:

* QoQ growth  = this quarter / previous quarter - 1 (only when the two are consecutive,
  i.e. 80-100 days apart);
* YoY growth  = this quarter / the same fiscal quarter a year earlier - 1 (only when a
  quarter 350-380 days earlier exists);
* TTM         = sum of the last four quarters, only when all four are reported and
  consecutive - otherwise N/A (a partial sum is never shown as TTM);
* margins     = operating income or net income / revenue of the same quarter;
* free cash flow = operating cash flow - |capital expenditure|.

Missing values stay missing (never zero). Nothing here feeds the health score.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

INCOME_ROWS = {"Total Revenue": "Revenue", "Operating Income": "Operating income",
               "EBITDA": "EBITDA", "Net Income": "Net income"}
CASH_ROWS = {"Operating Cash Flow": "Operating cash flow",
             "Capital Expenditure": "Capital expenditure"}
FLOW_METRICS = ("Revenue", "Operating income", "EBITDA", "Net income", "Operating cash flow",
                "Free cash flow")
CONSECUTIVE_DAYS = (80, 100)
YEAR_DAYS = (350, 380)


def _num(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _rows(frame: pd.DataFrame | None, mapping: dict[str, str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if frame is None or frame.empty:
        return out
    for source, label in mapping.items():
        if source in frame.index:
            out[label] = {pd.Timestamp(c).normalize(): _num(v)
                          for c, v in frame.loc[source].items()}
    return out


def quarterly_table(income: pd.DataFrame | None, cash_flow: pd.DataFrame | None
                    ) -> pd.DataFrame:
    """Rows = fiscal quarter-end dates (oldest first), columns = metrics."""
    data = {**_rows(income, INCOME_ROWS), **_rows(cash_flow, CASH_ROWS)}
    if not data:
        return pd.DataFrame()
    table = pd.DataFrame(data).sort_index()
    table = table.astype(float)
    if {"Operating cash flow", "Capital expenditure"} <= set(table):
        table["Free cash flow"] = table["Operating cash flow"] - table["Capital expenditure"].abs()
    # drop quarters where nothing at all was reported
    return table.dropna(how="all")


def _gap_ok(a: pd.Timestamp, b: pd.Timestamp, bounds: tuple[int, int]) -> bool:
    return bounds[0] <= (b - a).days <= bounds[1]


def growth(table: pd.DataFrame, metric: str) -> pd.DataFrame:
    """QoQ and YoY growth of one metric (decimal; NaN when not comparable)."""
    if metric not in table:
        return pd.DataFrame()
    s, dates = table[metric], list(table.index)
    qoq, yoy = [], []
    for i, d in enumerate(dates):
        prev = dates[i - 1] if i else None
        q = (s[d] / s[prev] - 1 if prev is not None and _gap_ok(prev, d, CONSECUTIVE_DAYS)
             and pd.notna(s[d]) and pd.notna(s[prev]) and s[prev] > 0 else math.nan)
        year_ago = [x for x in dates[:i] if _gap_ok(x, d, YEAR_DAYS)]
        y = math.nan
        if year_ago and pd.notna(s[d]) and pd.notna(s[year_ago[-1]]) and s[year_ago[-1]] > 0:
            y = s[d] / s[year_ago[-1]] - 1
        qoq.append(q)
        yoy.append(y)
    return pd.DataFrame({"QoQ": qoq, "YoY": yoy}, index=table.index)


def margins(table: pd.DataFrame) -> pd.DataFrame:
    if "Revenue" not in table:
        return pd.DataFrame()
    rev = table["Revenue"].where(table["Revenue"] > 0)
    out = {}
    for metric, label in (("Operating income", "Operating margin"),
                          ("Net income", "Net margin")):
        if metric in table:
            out[label] = table[metric] / rev
    return pd.DataFrame(out, index=table.index)


def ttm(table: pd.DataFrame) -> dict[str, Any]:
    """TTM sums of flow metrics; each N/A unless the last four quarters are consecutive
    and all reported."""
    result: dict[str, Any] = {"quarters": [], "values": {}, "note": None}
    if len(table) < 4:
        result["note"] = f"Only {len(table)} quarter(s) available: TTM needs four."
        return result
    last4 = list(table.index[-4:])
    if not all(_gap_ok(a, b, CONSECUTIVE_DAYS) for a, b in zip(last4, last4[1:])):
        result["note"] = "The last four quarters are not consecutive: TTM not computed."
        return result
    result["quarters"] = last4
    for metric in FLOW_METRICS:
        if metric in table:
            vals = table.loc[last4, metric]
            result["values"][metric] = float(vals.sum()) if vals.notna().all() else None
    return result
