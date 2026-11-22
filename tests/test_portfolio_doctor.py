"""Portfolio Doctor correctness: prices, currencies, coverage, position spread and uploads.

Uses a small synthetic screen frame (no database, no network) so every expected number
can be checked by hand."""

import io
from datetime import date

import numpy as np
import pandas as pd
import pytest

from app.services.portfolio_doctor import (MAX_STOCK_PCT, analyse_portfolio, parse_holdings,
                                           position_spread, select_price, validate_holdings)

TODAY = date(2026, 11, 15)
FRESH, STALE = "2026-11-13", "2026-10-01"


def frame(**changes):
    rows = [
        # ticker, name, sector, price, price_currency, price_date, score, decision
        ("TCS.NS", "TCS", "Technology", 3000.0, "INR", FRESH, 70.0, "WAIT"),
        ("INFY.NS", "Infosys", "Technology", 1500.0, "INR", FRESH, 65.0, "BUY"),
        ("ITC.NS", "ITC", "Consumer Defensive", 400.0, "INR", FRESH, 55.0, "HOLD"),
        ("OLD.NS", "Old Price Ltd", "Industrials", 100.0, "INR", STALE, 50.0, "HOLD"),
        ("BANK.NS", "Some Bank", "Financial Services", 900.0, "INR", FRESH, np.nan, "NODATA"),
        ("MSFT", "Microsoft", "Technology", 500.0, "USD", FRESH, 67.0, "WAIT"),
        ("AAPL", "Apple", "Technology", 200.0, "USD", FRESH, 66.0, "WAIT"),
    ]
    f = pd.DataFrame(rows, columns=["ticker", "name", "sector", "price", "price_currency",
                                    "price_date", "score", "decision"])
    f["val_label"] = "Fairly valued"
    f["headline"] = "label"
    for k, v in changes.items():
        f[k] = v
    return f


def holdings(*rows):
    return pd.DataFrame(rows, columns=["symbol", "quantity", "avg_price", "ltp"])


# ----------------------------------------------------------------- price selection
def test_missing_price_is_na_never_cost():
    h = holdings(("NOPRICE.NS", 100, 500.0, np.nan))
    out = analyse_portfolio(h, frame(), today=TODAY)
    row = out["table"].iloc[0]
    assert pd.isna(row["Price"]) and pd.isna(row["Value"]) and pd.isna(row["P&L"])
    assert row["Invested"] == 50_000 and row["Price status"] == "No reliable price"
    assert out["summary"]["value"] is None and out["summary"]["cost_unpriced"] == 50_000
    assert any(a["title"] == "No reliable current price" for a in out["alerts"])


def test_stale_snapshot_is_not_used_as_current_price():
    out = analyse_portfolio(holdings(("OLD.NS", 10, 90.0, np.nan)), frame(), today=TODAY)
    row = out["table"].iloc[0]
    assert pd.isna(row["Value"]) and "45 days old" in row["Price note"]


def test_price_priority_and_disagreement_warning():
    snap = (100.0, "2026-11-14")
    broker = select_price(110.0, date(2026, 11, 15), None, snap, TODAY)
    assert broker["source"] == "Broker file LTP" and broker["price"] == 110.0
    assert broker["warning"] and "differ by 9%" in broker["warning"]  # |100/110 - 1| = 9%
    undated_broker = select_price(110.0, None, (105.0, date(2026, 11, 14)), snap, TODAY)
    assert undated_broker["source"].startswith("Yahoo Finance") and undated_broker["price"] == 105
    only_snapshot = select_price(None, None, None, snap, TODAY)
    assert only_snapshot["source"] == "Stored snapshot"
    stale_everything = select_price(110.0, date(2026, 9, 1), (105.0, date(2026, 9, 1)),
                                    (100.0, "2026-09-01"), TODAY)
    assert stale_everything["price"] is None and "75 days old" in stale_everything["note"]


def test_stale_broker_ltp_falls_back_to_fresh_live_price():
    h = holdings(("TCS.NS", 10, 2500.0, 2000.0))
    out = analyse_portfolio(h, frame(price=np.nan), today=TODAY,
                            broker_as_of=date(2026, 10, 1),
                            live_prices={"TCS.NS": (3100.0, date(2026, 11, 14))})
    row = out["table"].iloc[0]
    assert row["Price"] == 3100.0 and row["Price source"].startswith("Yahoo")


# ----------------------------------------------------------------- currencies
def test_mixed_currency_portfolio_never_adds_inr_and_usd():
    h = holdings(("TCS.NS", 10, 2500.0, np.nan), ("INFY.NS", 20, 1400.0, np.nan),
                 ("MSFT", 5, 400.0, np.nan), ("AAPL", 10, 150.0, np.nan))
    inr = analyse_portfolio(h, frame(), today=TODAY)
    assert inr["summary"]["currency"] == "INR"        # largest cost basis
    assert set(inr["table"]["Symbol"]) == {"TCS.NS", "INFY.NS"}
    assert inr["summary"]["value"] == pytest.approx(10 * 3000 + 20 * 1500)
    assert inr["summary"]["excluded_other_currency"] == 2
    assert any("another currency" in a["title"] for a in inr["alerts"])
    usd = analyse_portfolio(h, frame(), reporting_currency="USD", today=TODAY)
    assert usd["summary"]["value"] == pytest.approx(5 * 500 + 10 * 200)
    assert set(usd["excluded"]["Symbol"]) == {"TCS.NS", "INFY.NS"}


def test_valid_us_portfolio():
    h = holdings(("MSFT", 5, 400.0, np.nan), ("AAPL", 10, 150.0, np.nan))
    out = analyse_portfolio(h, frame(), today=TODAY)
    assert out["summary"]["currency"] == "USD"
    assert out["table"]["Weight %"].sum() == pytest.approx(100)


# ----------------------------------------------------------------- coverage and weights
def test_weights_and_coverage_with_unanalysed_and_unpriced_holdings():
    h = holdings(("TCS.NS", 10, 2500.0, np.nan),       # value 30,000, analysed
                 ("BANK.NS", 10, 800.0, np.nan),       # value 9,000, no health score
                 ("NOPRICE.NS", 5, 100.0, np.nan))     # no price at all
    out = analyse_portfolio(h, frame(), today=TODAY)
    s, t = out["summary"], out["table"].set_index("Symbol")
    assert t.loc["TCS.NS", "Weight %"] == pytest.approx(30_000 / 39_000 * 100)
    assert pd.isna(t.loc["NOPRICE.NS", "Weight %"])
    assert (s["n"], s["n_priced"], s["n_analysed"]) == (3, 2, 1)
    assert s["analysed_coverage"] == pytest.approx(30_000 / 39_000)
    assert s["unanalysed_coverage"] == pytest.approx(9_000 / 39_000)
    # Phase 1: weights cover priced holdings only - the denominator is disclosed.
    # Cost basis: 10x2500 + 10x800 + 5x100 = 33,500; priced cost 33,000 -> 98.5%.
    assert s["weight_basis"] == "priced holdings only"
    assert s["cost_unpriced"] == pytest.approx(500.0)
    assert s["priced_cost_share"] == pytest.approx(33_000 / 33_500)


def test_composite_withheld_when_coverage_is_low():
    h = holdings(("TCS.NS", 1, 2500.0, np.nan), ("BANK.NS", 100, 800.0, np.nan))
    out = analyse_portfolio(h, frame(), today=TODAY)
    assert out["summary"]["analysed_coverage"] < 0.7
    assert out["summary"]["health"] is None and out["summary"]["composite_withheld"]
    assert any(a["title"] == "Partial assessment" for a in out["alerts"])


def test_missing_sector_is_unknown_not_an_error():
    h = holdings(("TCS.NS", 10, 2500.0, np.nan))
    out = analyse_portfolio(h, frame(sector=np.nan), today=TODAY)
    assert out["table"].iloc[0]["Sector"] == "Unknown"


def test_position_spread_is_not_sector_diversification():
    eight_equal = pd.Series([12.5] * 8)
    score, effective = position_spread(eight_equal)
    assert score == pytest.approx(100) and effective == pytest.approx(8)
    one_heavy = pd.Series([70.0, 10.0, 10.0, 10.0])
    score2, effective2 = position_spread(one_heavy)
    assert effective2 == pytest.approx(1 / (0.49 + 3 * 0.01)) and score2 < 30
    # same position spread, very different sector exposure -> reported separately
    h = holdings(("TCS.NS", 10, 1.0, np.nan), ("INFY.NS", 20, 1.0, np.nan))
    out = analyse_portfolio(h, frame(), today=TODAY)
    assert out["summary"]["n_sectors"] == 1 and out["summary"]["largest_sector_pct"] == 100


def test_concentration_alert_uses_illustrative_limit():
    h = holdings(("TCS.NS", 100, 2500.0, np.nan), ("ITC.NS", 10, 300.0, np.nan))
    out = analyse_portfolio(h, frame(), today=TODAY)
    assert any("Over-concentrated" in a["title"] and f"{MAX_STOCK_PCT:.0f}%" in a["detail"]
               for a in out["alerts"])


# ----------------------------------------------------------------- uploads
def test_validate_holdings_reports_every_rejected_row():
    raw = pd.DataFrame({"Instrument": ["TCS", "INFY", "BAD", "", "ITC", "ZERO", "TCS"],
                        "Qty.": [10, "abc", -5, 3, 7, 0, 5],
                        "Avg. cost": [2500, 1400, 100, 50, None, 10, 2600]})
    h, rejected, counts = validate_holdings(raw)
    assert counts == {"rows": 7, "accepted": 2, "rejected": 5, "holdings": 1,
                      "duplicates_merged": 1}
    assert h.loc[0, "symbol"] == "TCS.NS" and h.loc[0, "quantity"] == 15
    assert h.loc[0, "avg_price"] == pytest.approx((10 * 2500 + 5 * 2600) / 15)
    reasons = dict(zip(rejected["row"], rejected["reason"]))
    assert reasons[3] == "quantity is not a number"
    assert reasons[4].startswith("quantity must be positive")
    assert reasons[5] == "symbol missing"
    assert reasons[6] == "average cost missing or not a number"
    assert reasons[7].startswith("quantity must be positive")


def test_empty_and_wrong_column_files():
    with pytest.raises(ValueError, match="no rows"):
        validate_holdings(pd.DataFrame({"Symbol": [], "Qty": [], "Avg cost": []}))
    with pytest.raises(ValueError, match="Could not find column"):
        validate_holdings(pd.DataFrame({"Name": ["TCS"], "Amount": [1]}))


def test_csv_and_xlsx_uploads_parse_identically():
    raw = pd.DataFrame({"Symbol": ["TCS", "MSFT"], "Quantity": [10, 2], "Avg cost": [2500, 400],
                        "LTP": [3000, 0]})
    csv = parse_holdings(pd.read_csv(io.StringIO(raw.to_csv(index=False))), ".NS")
    buf = io.BytesIO()
    raw.to_excel(buf, index=False)
    buf.seek(0)
    xlsx = parse_holdings(pd.read_excel(buf), ".NS")
    pd.testing.assert_frame_equal(csv, xlsx)
    assert csv.loc[csv["symbol"] == "MSFT.NS", "ltp"].isna().all()  # LTP 0 is not a price


def test_invalid_ticker_is_reported_not_analysed():
    out = analyse_portfolio(holdings(("NOTREAL.NS", 1, 10.0, np.nan)), frame(), today=TODAY)
    assert out["table"].iloc[0]["Action"] == "NO DATA"
    assert any("not analysed" in a["title"] for a in out["alerts"])
