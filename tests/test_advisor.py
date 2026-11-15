"""Tests for the Advisor and Portfolio Doctor layers (read-only over the shipped database)."""
import pandas as pd
import pytest

from app.services.advisor import (ACTIONS, NO_DATA, advisor_frame, decide, peer_median,
                                  quality_tier)
from app.services.context import build_context
from app.services.portfolio_doctor import (analyse_portfolio, clean_symbol, parse_holdings,
                                           sample_holdings)


@pytest.fixture(scope="module")
def frame():
    return advisor_frame(build_context(), "moderate")


def test_frame_covers_every_analysed_company(frame):
    assert len(frame) > 0
    assert frame["decision"].isin(ACTIONS).all()
    assert frame["val_score"].dropna().between(0, 100).all()


def test_bottom_quartile_never_buy(frame):
    bad = frame[frame["interpretation"] == "BOTTOM QUARTILE"]
    assert not bad["decision"].isin(["BUY", "BUY_CHECK", "WAIT"]).any()


def test_top_quartile_overvalued_is_wait_not_buy(frame):
    top = frame[(frame["interpretation"] == "TOP QUARTILE") & (frame["high_flags"] == 0)
                & (frame["val_label"] == "Overvalued")]
    assert (top["decision"] == "WAIT").all()


def test_currency_mismatch_keeps_value_but_skips_market_comparison(frame):
    """Intrinsic value does not need the price; comparing it with a price in another
    currency would need FX conversion, so only the comparison is dropped."""
    mixed = frame[frame["price_currency"].notna() & (frame["price_currency"] != frame["currency"])]
    assert len(mixed) > 0
    for _, row in mixed.iterrows():
        assert "dcf_mos" not in row["components"]
        if isinstance(row["dcf"], dict):
            assert row["dcf"]["price"] is None and row["dcf"]["margin_of_safety"] is None


def test_dcf_monotonic_in_growth():
    from app.valuation.dcf import project_cash_flows, value_firm

    def per_share(g):
        return value_firm(project_cash_flows(100, g, 0.04), 0.12, 0.04, cash=0, debt=0,
                          shares=10).value_per_share
    assert per_share(0.12) > per_share(0.05) > per_share(-0.05) > 0


def test_peer_median_never_mixes_markets():
    import numpy as np
    frame = pd.DataFrame({
        "_mkt": ["INR"] * 6 + ["USD"] * 6,
        "sector": ["Tech"] * 5 + ["Energy"] + ["Tech"] * 6,
        "price_to_earnings": [10, 12, 14, 16, 18, 30, 40, 42, 44, 46, 48, 50]})
    assert peer_median(frame, "price_to_earnings", "INR", "Tech") == (14, "INR Tech sector")
    med, basis = peer_median(frame, "price_to_earnings", "INR", "Energy")  # 1 peer only
    assert basis == "market-wide INR" and med == np.median([10, 12, 14, 16, 18, 30])
    tiny = frame[frame["_mkt"] == "USD"].head(3)
    assert peer_median(tiny, "price_to_earnings", "USD", "Tech") == (None, None)


def test_missing_price_never_gives_the_most_positive_result():
    d = decide("high", {"val_label": "Fairly valued", "val_score": 50.0,
                        "dcf": {"fair_value": 100.0, "buy_below": 80.0, "price": None,
                                "upside": None, "growth": 0.05, "discount_rate": 0.12}}, "INR")
    assert d["decision"] == "BUY_CHECK"
    assert any("cannot be compared" in w for w in d["decision_why"])


def test_frame_dcf_uses_fcff_and_bridge(frame):
    with_dcf = frame[frame["dcf"].notna()]
    assert len(with_dcf) > 0
    d = with_dcf.iloc[0]["dcf"]
    assert d["bridge"]["equity_value"] == pytest.approx(d["equity_value"])
    assert "not a computed WACC" in d["discount_note"]


def test_quality_tier_and_red_flags():
    assert quality_tier("TOP QUARTILE", 0)[0] == "high"
    assert quality_tier("TOP QUARTILE", 1)[0] == "mid"
    assert quality_tier("BELOW MEDIAN", 0)[0] == "weak"
    assert quality_tier(None, 0)[0] is None


def test_decide_without_valuation():
    d = decide("high", {"val_label": NO_DATA, "val_score": None, "dcf": None}, "INR")
    assert d["decision"] == "BUY_CHECK"


def _val(label, price, fair=1000.0):
    """Valuation dict with a DCF: buy-below = fair value less the margin of safety."""
    from app.services.advisor import MARGIN_OF_SAFETY
    dcf = {"fair_value": fair, "buy_below": fair * (1 - MARGIN_OF_SAFETY), "price": price,
           "upside": (fair / price - 1) * 100, "growth": 0.05, "discount_rate": 0.12}
    return {"val_label": label, "val_score": 50.0, "dcf": dcf}


@pytest.mark.parametrize("price, expected", [
    (700.0, "BUY"),          # below the buy-below price (800): margin of safety met
    (800.0, "BUY"),          # exactly at buy-below
    (900.0, "BUY_CHECK"),    # below fair value but above buy-below
    (1000.0, "BUY_CHECK"),   # at fair value
    (1243.0, "WAIT"),        # above fair value (TCS case: ~24% above) - never BUY
    (2000.0, "WAIT"),
])
def test_buy_below_price_gates_buy_verdicts(price, expected):
    d = decide("high", _val("Fairly valued", price), "INR")
    assert d["decision"] == expected
    if expected != "BUY":
        assert any("margin-of-safety price" in w or "above the DCF model value" in w
                   for w in d["decision_why"])


def test_gate_also_applies_without_multiples():
    assert decide("high", _val(NO_DATA, 2000.0), "INR")["decision"] == "WAIT"
    assert decide("high", _val(NO_DATA, 700.0), "INR")["decision"] == "BUY_CHECK"


def test_no_buy_above_its_own_buy_below_price(frame):
    has_dcf = frame[frame["buy_below"].notna() & frame["price"].notna()
                    & (frame["price_currency"] == frame["currency"])]
    buys = has_dcf[has_dcf["decision"] == "BUY"]
    assert (buys["price"] <= buys["buy_below"] + 1e-9).all()
    assert not has_dcf[(has_dcf["price"] > has_dcf["fair_value"])]["decision"].isin(
        ["BUY", "BUY_CHECK"]).any()


def test_symbol_cleaning():
    assert clean_symbol("INFY-EQ") == "INFY.NS"
    assert clean_symbol("tcs.ns") == "TCS.NS"
    assert clean_symbol("AAPL", "") == "AAPL"


def test_parse_holdings_merges_duplicates_and_drops_bad_rows():
    df = pd.DataFrame({"Instrument": ["TCS", "TCS", "BAD"], "Qty.": [10, 10, 0],
                       "Avg. cost": [100, 200, 50], "LTP": [0, 0, 0]})
    h = parse_holdings(df)
    assert len(h) == 1 and h.loc[0, "quantity"] == 20 and h.loc[0, "avg_price"] == 150


def test_parse_holdings_missing_column():
    with pytest.raises(ValueError):
        parse_holdings(pd.DataFrame({"a": [1]}))


def test_portfolio_review(frame):
    from datetime import date
    # evaluate on the snapshot date so the stored prices are fresh whatever today is
    snapshot_day = date.fromisoformat(frame["price_date"].dropna().max())
    out = analyse_portfolio(parse_holdings(sample_holdings(frame)), frame, today=snapshot_day)
    assert abs(out["table"]["Weight %"].sum() - 100) < 1e-6
    assert out["summary"]["health"] is not None
    assert all(a["severity"] in ("High", "Medium", "Low") for a in out["alerts"])


def test_unknown_holding_is_flagged(frame):
    h = pd.DataFrame({"symbol": ["NOPE.NS"], "quantity": [1.0], "avg_price": [10.0],
                      "ltp": [float("nan")]})
    out = analyse_portfolio(h, frame)
    assert out["table"].loc[0, "Action"] == "NO DATA"
    assert any("not analysed" in a["title"] for a in out["alerts"])
