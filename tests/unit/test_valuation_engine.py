"""Valuation engine: the mandatory manual DCF example, FCFF/FCFE pairing, Gordon-growth
validation, the enterprise-to-equity bridge, CAPM/WACC and company-level edge cases."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from app.valuation import capm
from app.valuation.company import (ValuationInputs, fcff_base, full_valuation,
                                   revenue_growth, screening_valuation)
from app.valuation.dcf import (COST_OF_EQUITY, FCFE, FCFF, WACC, ValuationError,
                               check_pairing, discount_cash_flows, equity_bridge,
                               margin_of_safety, project_cash_flows, terminal_value,
                               value_equity, value_firm)

MANUAL_FCFF = [100.00, 110.00, 121.00, 133.10, 146.41]


# ----------------------------------------------------------------- mandatory manual example
def test_manual_dcf_example_matches_independent_reference():
    """Assumptions from the project brief (millions): WACC 10 %, g 3 %, cash 200, debt 500,
    other senior claims 0, 100 million diluted shares, year-end discounting."""
    r = value_firm(MANUAL_FCFF, wacc=0.10, terminal_growth=0.03, cash=200, debt=500,
                   shares=100, minority_interest=0, other_senior_claims=0)
    assert r.terminal_value == pytest.approx(2154.32, abs=0.01)
    assert r.pv_forecast == pytest.approx(454.55, abs=0.01)
    assert r.pv_terminal_value == pytest.approx(1337.66, abs=0.01)
    assert r.enterprise_value == pytest.approx(1792.21, abs=0.01)
    assert r.equity_value == pytest.approx(1492.21, abs=0.01)
    assert r.value_per_share == pytest.approx(14.92, abs=0.005)
    # every forecast year discounts to 90.909 (cash flows grow at exactly the discount rate)
    assert all(pv == pytest.approx(90.909, abs=0.001) for pv in r.pv_cash_flows)
    assert r.bridge["plus_cash"] == 200 and r.bridge["minus_debt"] == 500


def test_terminal_value_discounted_with_final_year_factor():
    r = discount_cash_flows(MANUAL_FCFF, 0.10, 0.03, FCFF, WACC)
    assert r.pv_terminal_value == pytest.approx(r.terminal_value / 1.1 ** 5)


# ----------------------------------------------------------------- pairing rules
def test_fcff_only_with_wacc_and_fcfe_only_with_cost_of_equity():
    check_pairing(FCFF, WACC)
    check_pairing(FCFE, COST_OF_EQUITY)
    with pytest.raises(ValuationError, match="cannot be discounted"):
        check_pairing(FCFF, COST_OF_EQUITY)
    with pytest.raises(ValuationError, match="cannot be discounted"):
        check_pairing(FCFE, WACC)


def test_fcfe_gives_equity_value_directly_without_bridge():
    r = value_equity([50, 52, 54], cost_of_equity=0.12, terminal_growth=0.04, shares=10)
    assert r.enterprise_value is None and r.bridge == {}
    expected = sum(cf / 1.12 ** t for t, cf in enumerate([50, 52, 54], 1)) \
        + 54 * 1.04 / 0.08 / 1.12 ** 3
    assert r.equity_value == pytest.approx(expected)
    assert r.value_per_share == pytest.approx(expected / 10)


# ----------------------------------------------------------------- Gordon growth validity
@pytest.mark.parametrize("rate, g", [(0.03, 0.05), (0.05, 0.05)])
def test_wacc_less_than_or_equal_to_growth_is_rejected(rate, g):
    with pytest.raises(ValuationError, match="must be greater than terminal growth"):
        terminal_value(100, rate, g)
    with pytest.raises(ValuationError):
        value_firm(MANUAL_FCFF, rate, g, cash=0, debt=0, shares=1)


def test_terminal_growth_close_to_wacc_warns():
    r = discount_cash_flows(MANUAL_FCFF, 0.10, 0.095, FCFF, WACC)
    assert any("extremely sensitive" in w for w in r.warnings)
    assert r.terminal_share > 0.9


# ----------------------------------------------------------------- forecasts
def test_projection_keeps_negative_growth_negative():
    flows = project_cash_flows(100, -0.08, 0.03, explicit_years=5, fade_years=5)
    assert flows[0] == pytest.approx(92.0) and flows[4] == pytest.approx(100 * 0.92 ** 5)
    assert flows[4] < flows[0] < 100  # shrinking in the explicit period


def test_projection_zero_growth_and_fade_to_terminal():
    flows = project_cash_flows(100, 0.0, 0.03, explicit_years=5, fade_years=5)
    assert flows[:5] == pytest.approx([100.0] * 5)
    assert flows[-1] / flows[-2] - 1 == pytest.approx(0.03)  # final year at terminal growth


def test_negative_final_cash_flow_warns():
    r = discount_cash_flows([-10, -5, -1], 0.10, 0.02, FCFF, WACC)
    assert any("not positive" in w for w in r.warnings)


# ----------------------------------------------------------------- bridge and per share
def test_missing_cash_debt_or_shares_never_assumed_zero():
    with pytest.raises(ValuationError, match="Cash"):
        equity_bridge(1000, None, 100)
    with pytest.raises(ValuationError, match="debt"):
        equity_bridge(1000, 50, None)
    for shares in (None, 0):
        with pytest.raises(ValuationError, match="shares"):
            value_firm(MANUAL_FCFF, 0.10, 0.03, cash=200, debt=500, shares=shares)


def test_bridge_subtracts_minority_and_senior_claims():
    equity, bridge, _ = equity_bridge(1000, cash=100, debt=300, minority_interest=50,
                                      other_senior_claims=20, non_operating_assets=10)
    assert equity == 1000 + 100 + 10 - 300 - 50 - 20


def test_margin_of_safety_needs_price_but_value_does_not():
    assert margin_of_safety(14.92, None) is None
    assert margin_of_safety(10.0, 8.0) == pytest.approx(0.2)


# ----------------------------------------------------------------- CAPM, WACC, beta
def test_capm_and_wacc_formulas():
    ke = capm.cost_of_equity(0.072, 1.2, 0.0708)
    assert ke == pytest.approx(0.072 + 1.2 * 0.0708)
    w = capm.wacc(ke, 0.09, 0.2517, equity_value=800, debt_value=200)
    assert w.equity_weight == pytest.approx(0.8) and w.debt_weight == pytest.approx(0.2)
    assert w.wacc == pytest.approx(0.8 * ke + 0.2 * 0.09 * (1 - 0.2517))
    debt_free = capm.wacc(ke, 0.09, 0.25, equity_value=500, debt_value=0)
    assert debt_free.wacc == pytest.approx(ke)
    with pytest.raises(ValuationError):
        capm.wacc(ke, 0.09, 0.25, equity_value=None, debt_value=10)


def test_cost_of_debt_implied_and_fallback():
    ok = capm.cost_of_debt(90, 1000, 800, risk_free=0.072)
    assert ok.pre_tax == pytest.approx(0.10) and not ok.warnings
    absurd = capm.cost_of_debt(500, 1000, 1000, risk_free=0.072)
    assert absurd.pre_tax == pytest.approx(0.092) and absurd.warnings
    missing = capm.cost_of_debt(None, None, None, risk_free=0.0485)
    assert missing.pre_tax == pytest.approx(0.0685) and "not reported" in missing.warnings[0]


def test_beta_regression_recovers_true_beta_and_needs_enough_data():
    rng = np.random.default_rng(4)
    idx = pd.date_range("2021-01-31", periods=61, freq="ME")
    market_r = rng.normal(0.01, 0.04, 60)
    stock_r = 0.002 + 1.5 * market_r + rng.normal(0, 0.01, 60)
    market = pd.Series(100 * np.cumprod(np.r_[1, 1 + market_r]), index=idx)
    stock = pd.Series(50 * np.cumprod(np.r_[1, 1 + stock_r]), index=idx)
    b = capm.estimate_beta(stock, market, "^NSEI")
    assert b.beta == pytest.approx(1.5, abs=0.1) and b.observations == 60
    assert b.blume_adjusted == pytest.approx(0.67 * b.beta + 0.33)
    with pytest.raises(ValuationError, match="observations"):
        capm.estimate_beta(stock.iloc[:20], market.iloc[:20], "^NSEI")


def test_market_assumptions_are_currency_specific():
    inr, usd = capm.market_assumptions("INR"), capm.market_assumptions("USD")
    assert inr["risk_free_rate"] != usd["risk_free_rate"]
    assert inr["equity_risk_premium_source"].startswith("Damodaran")
    with pytest.raises(ValuationError, match="No market assumptions"):
        capm.market_assumptions("JPY")


# ----------------------------------------------------------------- company level
def _inputs(**overrides):
    periods = {
        date(2023, 3, 31): {"total_revenue": 1000.0, "total_debt": 200.0},
        date(2024, 3, 31): {"total_revenue": 1100.0, "total_debt": 200.0},
        date(2025, 3, 31): {"total_revenue": 1210.0, "operating_cash_flow": 200.0,
                            "capital_expenditure": -60.0, "interest_expense": 20.0,
                            "total_debt": 200.0, "cash_and_equivalents": 150.0,
                            "diluted_average_shares": 10.0, "stockholders_equity": 600.0},
    }
    base = dict(ticker="TEST.NS", currency="INR", sector="Industrials", periods=periods,
                price=180.0, price_currency="INR", market_cap=1800.0)
    base.update(overrides)
    return ValuationInputs(**base)


def test_fcff_from_cash_flow_statement():
    fcff, parts, warnings = fcff_base(_inputs().latest, tax_rate=0.25)
    assert fcff == pytest.approx(200 + 20 * 0.75 - 60) and not warnings
    with pytest.raises(ValuationError, match="capital expenditure"):
        fcff_base({"operating_cash_flow": 100}, 0.25)


def test_negative_revenue_growth_is_not_clipped_to_zero():
    shrinking = _inputs(periods={date(2023, 3, 31): {"total_revenue": 1000.0},
                                 date(2025, 3, 31): {"total_revenue": 846.4}})
    g, note, _ = revenue_growth(shrinking, -0.10, 0.15)
    assert g == pytest.approx(-0.08, abs=0.002) and "CAGR" in note


def test_intrinsic_value_without_market_price():
    v = screening_valuation(_inputs(price=None, price_currency=None))
    assert v.ok and v.value_per_share > 0
    assert v.price is None and v.margin_of_safety is None and v.upside is None


def test_screening_rate_is_disclosed_as_not_a_wacc():
    v = screening_valuation(_inputs())
    assert v.ok and "not a computed WACC" in v.discount_note
    assert v.margin_of_safety is not None


def test_company_level_na_cases():
    assert screening_valuation(_inputs(sector="Financial Services")).status == "NOT_APPLICABLE"
    negative = _inputs()
    negative.periods[date(2025, 3, 31)]["capital_expenditure"] = -500.0
    v = screening_valuation(negative)
    assert v.status == "INSUFFICIENT_DATA" and "negative" in v.reason
    no_shares = _inputs()
    del no_shares.periods[date(2025, 3, 31)]["diluted_average_shares"]
    assert "shares" in screening_valuation(no_shares).reason
    no_cash = _inputs()
    del no_cash.periods[date(2025, 3, 31)]["cash_and_equivalents"]
    assert "Cash" in screening_valuation(no_cash).reason
    usd_price = screening_valuation(_inputs(price_currency="USD"))
    assert usd_price.ok and usd_price.margin_of_safety is None
    assert any("no FX" in w for w in usd_price.warnings)


def test_full_valuation_uses_capm_wacc_and_needs_beta():
    assert full_valuation(_inputs(), None).status == "INSUFFICIENT_DATA"
    beta = capm.BetaEstimate(1.1, 0.67 * 1.1 + 0.33, 0.4, 60, "^NSEI", "5y", "1mo")
    v = full_valuation(_inputs(), beta)
    market = capm.market_assumptions("INR")
    ke = market["risk_free_rate"] + 1.1 * market["equity_risk_premium"]
    assert v.ok and v.wacc_detail.cost_of_equity == pytest.approx(ke)
    assert v.wacc_detail.equity_value_basis == "market capitalisation"
    assert v.discount_rate == pytest.approx(v.wacc_detail.wacc)
    assert v.dcf.basis == FCFF and v.dcf.rate_kind == WACC
