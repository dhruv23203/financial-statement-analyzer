"""Sensitivity grid, bear/base/bull scenarios, reverse DCF and the quarterly/TTM rules."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from app.services import quarterly
from app.valuation import scenarios as scen
from app.valuation.company import ValuationInputs, screening_valuation


def _valuation(price=180.0):
    periods = {date(2023, 3, 31): {"total_revenue": 1000.0, "total_debt": 200.0},
               date(2025, 3, 31): {"total_revenue": 1210.0, "operating_cash_flow": 200.0,
                                   "capital_expenditure": -60.0, "total_debt": 200.0,
                                   "cash_and_equivalents": 150.0,
                                   "ordinary_shares_number": 10.0}}
    v = screening_valuation(ValuationInputs("T.NS", "INR", "Industrials", periods, price=price,
                                            price_currency="INR", market_cap=1800.0,
                                            price_date="2026-11-20", shares_outstanding=10.0))
    assert v.ok
    return v


def test_revalue_reproduces_the_base_valuation():
    v = _valuation()
    assert scen.revalue(v) == pytest.approx(v.value_per_share)


def test_sensitivity_grid_centre_is_base_and_invalid_cells_are_na():
    v = _valuation()
    grid = scen.sensitivity_grid(v)
    assert grid.shape == (5, 5)
    assert grid.loc[round(v.discount_rate, 4), round(v.terminal_growth, 4)] == \
        pytest.approx(v.value_per_share)
    # higher discount rate -> lower value; higher terminal growth -> higher value
    col = grid[round(v.terminal_growth, 4)]
    assert col.is_monotonic_decreasing
    row = grid.loc[round(v.discount_rate, 4)]
    assert row.dropna().astype(float).is_monotonic_increasing
    # INR risk-free rate 5.33 %: terminal growth of 5.5 % and 6 % is not economically
    # reasonable for a perpetuity, so those cells are N/A rather than a number
    assert all(row[g] is None or pd.isna(row[g]) for g in row.index if g >= v.risk_free)
    assert all(pd.notna(row[g]) for g in row.index if g < v.risk_free)
    # a rate at or below terminal growth must be N/A, never a number
    tight = scen.sensitivity_grid(v, rate_steps=(-(v.discount_rate - v.terminal_growth),),
                                  terminal_steps=(0.0,))
    assert tight.isna().all().all()


def test_scenarios_order_and_bounds():
    v = _valuation()
    cases = {c.name: c for c in scen.scenarios(v)}
    assert cases["Base"].value_per_share == pytest.approx(v.value_per_share)
    assert cases["Bear"].value_per_share < cases["Base"].value_per_share \
        < cases["Bull"].value_per_share
    assert cases["Bull"].growth <= 0.15 and cases["Bear"].growth >= -0.10
    # vs price = value / price - 1
    assert cases["Base"].vs_price == pytest.approx(v.value_per_share / 180.0 - 1)


def test_reverse_dcf_recovers_a_known_growth_rate():
    v = _valuation()
    target = scen.revalue(v, growth=0.07)           # price set to the value at 7% growth
    rd = scen.reverse_dcf(v, price=target)
    assert rd.implied_growth == pytest.approx(0.07, abs=5e-4)


def test_reverse_dcf_without_price_or_outside_bracket():
    v = _valuation(price=None)
    assert scen.reverse_dcf(v).implied_growth is None
    huge = scen.reverse_dcf(_valuation(), price=1e9)
    assert huge.implied_growth is None and "not explained by growth" in huge.note


# ----------------------------------------------------------------------------- quarterly
def _q(values: dict[str, list], dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame(values, index=pd.to_datetime(dates)).T


DATES = ["2025-06-30", "2025-09-30", "2025-12-31", "2026-03-31", "2026-06-30"]


def test_quarterly_ttm_growth_and_margins():
    inc = _q({"Total Revenue": [100, 110, 120, 130, 125],
              "Net Income": [10, 11, 12, 13, 10]}, DATES)
    cf = _q({"Operating Cash Flow": [20, 20, 20, 20, 20],
             "Capital Expenditure": [-5, -5, -5, -5, -5]}, DATES)
    table = quarterly.quarterly_table(inc, cf)
    t = quarterly.ttm(table)
    # last four quarters: 110 + 120 + 130 + 125 = 485; FCF 4 x 15 = 60
    assert t["values"]["Revenue"] == 485 and t["values"]["Free cash flow"] == 60
    g = quarterly.growth(table, "Revenue")
    assert g["QoQ"].iloc[-1] == pytest.approx(125 / 130 - 1)
    assert g["YoY"].iloc[-1] == pytest.approx(125 / 100 - 1)   # Jun-26 vs Jun-25
    assert np.isnan(g["YoY"].iloc[1])                         # no quarter a year earlier
    assert quarterly.margins(table)["Net margin"].iloc[-1] == pytest.approx(10 / 125)


def test_ttm_is_na_with_a_gap_or_a_missing_quarter():
    gap = ["2025-03-31", "2025-09-30", "2025-12-31", "2026-03-31"]     # Jun-25 missing
    t = quarterly.ttm(quarterly.quarterly_table(
        _q({"Total Revenue": [1, 2, 3, 4]}, gap), None))
    assert not t["values"] and "not consecutive" in t["note"]
    partial = quarterly.quarterly_table(
        _q({"Total Revenue": [1, 2, np.nan, 4], "Net Income": [1, 1, 1, 1]}, DATES[:4]), None)
    vals = quarterly.ttm(partial)["values"]
    assert vals["Revenue"] is None and vals["Net income"] == 4    # never a partial sum
    assert quarterly.ttm(quarterly.quarterly_table(None, None))["values"] == {}
