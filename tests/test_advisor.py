"""Tests for the Advisor and Portfolio Doctor layers (read-only over the shipped database)."""
import pandas as pd
import pytest

from app.services.advisor import (ACTIONS, NO_DATA, advisor_frame, dcf_fair_value, decide,
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


def test_currency_mismatch_skips_dcf(frame):
    mixed = frame[frame["price_currency"].notna() & (frame["price_currency"] != frame["currency"])]
    assert mixed["dcf"].isna().all()


def test_dcf_monotonic_in_growth():
    assert dcf_fair_value(100, 0.12, 0.12, 0.04, 10) > dcf_fair_value(100, 0.05, 0.12, 0.04, 10) > 0


def test_quality_tier_and_red_flags():
    assert quality_tier("TOP QUARTILE", 0)[0] == "high"
    assert quality_tier("TOP QUARTILE", 1)[0] == "mid"
    assert quality_tier("BELOW MEDIAN", 0)[0] == "weak"
    assert quality_tier(None, 0)[0] is None


def test_decide_without_valuation():
    d = decide("high", {"val_label": NO_DATA, "val_score": None, "dcf": None}, "INR")
    assert d["decision"] == "BUY_CHECK"


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
    out = analyse_portfolio(parse_holdings(sample_holdings(frame)), frame)
    assert abs(out["table"]["Weight %"].sum() - 100) < 1e-6
    assert out["summary"]["health"] is not None
    assert all(a["severity"] in ("High", "Medium", "Low") for a in out["alerts"])


def test_unknown_holding_is_flagged(frame):
    h = pd.DataFrame({"symbol": ["NOPE.NS"], "quantity": [1.0], "avg_price": [10.0],
                      "ltp": [float("nan")]})
    out = analyse_portfolio(h, frame)
    assert out["table"].loc[0, "Action"] == "NO DATA"
    assert any("not analysed" in a["title"] for a in out["alerts"])
