"""Phase 1 regression tests: invalid multiples, classification states, share count,
FCFF method, finite-number validation, beta inputs and portfolio weight coverage.

Every expected number is worked by hand in the comment next to it."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from app.services import advisor
from app.services.advisor import (FAIR, NO_DATA, UNDERVALUED, _value_row, decide,
                                  meaningful_multiple, peer_median)
from app.valuation import capm
from app.valuation.company import (ValuationInputs, fcff_base, screening_valuation,
                                   select_share_count)
from app.valuation.dcf import (ValuationError, equity_bridge, project_cash_flows,
                               require_finite, value_firm)


# ----------------------------------------------------------------------------- multiples
@pytest.mark.parametrize("value, expected", [
    (15.0, 15.0), (-12.0, None), (0.0, None), (np.nan, None), (np.inf, None),
    (-np.inf, None), (None, None), ("abc", None)])
def test_meaningful_multiple(value, expected):
    assert meaningful_multiple(value) == expected


def _row(**kw):
    base = dict(sector="Industrials", price_to_earnings=None, price_to_book=None,
                market_cap=1000.0, revenue=None, free_cash_flow_margin=None,
                net_income_cagr=None, price_currency="INR", currency="INR")
    base.update(kw)
    return pd.Series(base)


def test_negative_pe_is_never_scored_as_cheap():
    # A loss-maker with P/E -8 against a sector median of 20 must not get a P/E score
    # (the old ratio -8/20 = -0.4 sat below the "cheapest" end of the curve -> 100).
    val = _value_row(_row(price_to_earnings=-8.0, net_income_cagr=10.0), (20.0, "sector"),
                     (None, None), "moderate", None)
    assert "pe_vs_sector" not in val["components"]
    assert val["peg"] is None                       # PEG never from a negative P/E
    assert any("not meaningful" in n for n in val["val_notes"])
    assert val["pe_meaningful"] is None


def test_positive_pe_still_scored():
    # P/E 10 vs median 20 -> ratio 0.5 -> top of the curve (score 100).
    val = _value_row(_row(price_to_earnings=10.0), (20.0, "sector"), (None, None),
                     "moderate", None)
    assert val["components"]["pe_vs_sector"]["score"] == 100


def test_negative_book_equity_pb_not_scored_for_financials():
    val = _value_row(_row(sector="Financial Services", price_to_book=-1.5),
                     (None, None), (1.2, "sector"), "moderate", None)
    assert "pb_vs_sector" not in val["components"]
    assert any("P/B is not meaningful" in n for n in val["val_notes"])


def test_peer_median_excludes_non_positive_and_infinite_multiples():
    # Valid P/Es are 10, 20, 30 -> median 20; -5, 0, inf and NaN are excluded.
    frame = pd.DataFrame({"_mkt": ["INR"] * 7, "sector": ["Tech"] * 7,
                          "price_to_earnings": [10, 20, 30, -5, 0, np.inf, np.nan]})
    old = advisor.MIN_SECTOR_PEERS
    advisor.MIN_SECTOR_PEERS = 3
    try:
        med, basis = peer_median(frame, "price_to_earnings", "INR", "Tech")
    finally:
        advisor.MIN_SECTOR_PEERS = old
    assert med == 20 and "Tech" in basis


# ----------------------------------------------------------------------------- classification
def _val(label, dcf=None, score=55.0):
    return {"val_label": label, "val_score": score if label != NO_DATA else None,
            "dcf": dcf, "dcf_state": "available" if dcf else "unavailable"}


@pytest.mark.parametrize("label", [UNDERVALUED, FAIR])
def test_high_quality_without_dcf_is_not_below_model_value(label):
    out = decide("high", _val(label), "INR")
    assert out["decision"] == "BUY_CHECK"
    assert "not conclusive" in out["headline"]
    assert any("DCF unavailable" in w for w in out["decision_why"])


def test_high_quality_needs_price_at_or_below_margin_of_safety():
    # Model value 100, margin of safety 20% (config) -> buy-below 80.
    dcf = {"fair_value": 100.0, "buy_below": 80.0, "price": 75.0, "growth": 0.05,
           "discount_rate": 0.12}
    assert decide("high", _val(FAIR, dcf), "INR")["decision"] == "BUY"
    dcf_mid = dict(dcf, price=90.0)                 # between buy-below and model value
    assert decide("high", _val(FAIR, dcf_mid), "INR")["decision"] == "BUY_CHECK"
    dcf_hi = dict(dcf, price=120.0)                 # above model value
    assert decide("high", _val(UNDERVALUED, dcf_hi), "INR")["decision"] == "WAIT"
    dcf_np = dict(dcf, price=None)                  # no dated price in statement currency
    assert decide("high", _val(FAIR, dcf_np), "INR")["decision"] == "BUY_CHECK"


# ----------------------------------------------------------------------------- share count
def _inputs(**kw):
    periods = {date(2024, 3, 31): {"total_revenue": 1000.0},
               date(2025, 3, 31): {"total_revenue": 1100.0, "ordinary_shares_number": 100.0,
                                   "diluted_average_shares": 98.0}}
    base = dict(ticker="X.NS", currency="INR", sector="Industrials", periods=periods,
                price=50.0, price_currency="INR", market_cap=5000.0, price_date="2026-11-20",
                shares_outstanding=102.0)
    base.update(kw)
    return ValuationInputs(**base)


def test_share_count_prefers_current_snapshot():
    sel = select_share_count(_inputs())
    assert sel["shares"] == 102.0 and sel["as_of"] == "2026-11-20"
    assert "current" in sel["classification"] and "not diluted" in sel["source"]


def test_share_count_rejects_inconsistent_snapshot():
    # 40 vs reported 100 differs by 60% > 25% tolerance -> reported period-end count.
    sel = select_share_count(_inputs(shares_outstanding=40.0))
    assert sel["shares"] == 100.0 and sel["as_of"] == "2025-03-31"
    assert any("differs" in w for w in sel["warnings"])


def test_share_count_weighted_average_is_last_and_disclosed():
    p = {date(2025, 3, 31): {"diluted_average_shares": 98.0}}
    sel = select_share_count(_inputs(periods=p, shares_outstanding=None))
    assert sel["shares"] == 98.0 and "weighted average" in sel["classification"]
    assert select_share_count(_inputs(periods={date(2025, 3, 31): {}},
                                      shares_outstanding=None))["shares"] is None


# ----------------------------------------------------------------------------- FCFF
def test_ebit_based_fcff_hand_worked():
    # EBIT 300, t 25% -> NOPAT 225; + D&A 50 - capex 80 + WC change -20 = 175.
    v = {"ebit": 300.0, "reconciled_depreciation": 50.0, "capital_expenditure": -80.0,
         "change_in_working_capital": -20.0, "operating_cash_flow": 999.0,
         "interest_expense": 40.0}
    fcff, parts, _ = fcff_base(v, 0.25, "INR")
    assert fcff == pytest.approx(175.0) and parts["method"] == "EBIT-based FCFF"
    assert parts["after_tax_interest"] == 0          # interest never touches this route


def test_ebit_loss_gets_no_tax_shield():
    # EBIT -100 (no tax) + D&A 30 - capex 20 + WC 0 = -90.
    v = {"ebit": -100.0, "reconciled_depreciation": 30.0, "capital_expenditure": -20.0,
         "change_in_working_capital": 0.0}
    fcff, _, warnings = fcff_base(v, 0.25, "USD")
    assert fcff == pytest.approx(-90.0) and any("no tax" in w for w in warnings)


def test_fcff_ignores_non_finite_inputs():
    # Infinite EBIT -> EBIT route unavailable -> cash-flow route (INR, no add-back): 200-60.
    v = {"ebit": np.inf, "reconciled_depreciation": 50.0, "capital_expenditure": -60.0,
         "change_in_working_capital": 0.0, "operating_cash_flow": 200.0}
    fcff, parts, _ = fcff_base(v, 0.25, "INR")
    assert fcff == pytest.approx(140.0) and "Cash-flow" in parts["method"]


# ----------------------------------------------------------------------------- finite checks
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_require_finite_rejects_nan_and_infinity(bad):
    with pytest.raises(ValuationError, match="not a finite"):
        require_finite("x", bad)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_dcf_rejects_non_finite_inputs(bad):
    with pytest.raises(ValuationError):
        project_cash_flows(bad, 0.05, 0.03)
    with pytest.raises(ValuationError):
        value_firm([100, 110, bad], 0.10, 0.03, cash=0, debt=0, shares=10)
    with pytest.raises(ValuationError):
        value_firm([100, 110, 120], bad, 0.03, cash=0, debt=0, shares=10)
    with pytest.raises(ValuationError):
        equity_bridge(1000.0, bad, 100.0)


def test_screening_valuation_reports_share_count_and_method():
    periods = {date(2023, 3, 31): {"total_revenue": 1000.0, "total_debt": 200.0},
               date(2025, 3, 31): {"total_revenue": 1210.0, "operating_cash_flow": 200.0,
                                   "capital_expenditure": -60.0, "total_debt": 200.0,
                                   "cash_and_equivalents": 150.0,
                                   "ordinary_shares_number": 10.0}}
    v = screening_valuation(_inputs(periods=periods, shares_outstanding=10.0))
    assert v.ok and v.fcff_components["share_count"]["shares"] == 10.0
    assert v.base_fcff == pytest.approx(140.0)       # INR: OCF 200 - capex 60, no add-back
    assert np.isfinite(v.value_per_share)


# ----------------------------------------------------------------------------- beta
def test_beta_records_price_basis_and_rejects_infinite_returns():
    idx = pd.date_range("2020-01-31", periods=60, freq="ME")
    rng = np.random.default_rng(0)
    m = pd.Series(100 * np.cumprod(1 + rng.normal(0.01, 0.04, 60)), idx)
    s = pd.Series(100 * np.cumprod(1 + 1.5 * (m.pct_change().fillna(0))), idx)
    b = capm.estimate_beta(s, m, "^NSEI", price_basis="adjusted closes")
    assert b.beta == pytest.approx(1.5, abs=1e-6) and b.price_basis == "adjusted closes"
    assert b.observations == 59 and b.r_squared == pytest.approx(1.0)
    bad = s.copy()
    bad.iloc[10] = 0.0                               # next return = x / 0 -> infinite
    with pytest.raises(ValuationError):
        capm.estimate_beta(bad, m, "^NSEI")
