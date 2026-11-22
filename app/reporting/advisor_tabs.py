"""Streamlit rendering for the Quality & Value screen and the Portfolio Doctor tabs.

Kept out of ``dashboard.py`` so the existing tabs stay untouched; the dashboard only
creates the tabs and calls :func:`render_advisor_tab` / :func:`render_portfolio_tab`.
Visual language follows ``app/reporting/ui.py`` (same cards, pills, colours).
"""

from __future__ import annotations

from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.reporting import plots, ui, valuation_view
from app.services.advisor import ACTIONS, NO_DATA, advisor_frame, company_row, ideas
from app.services.advisor import currency_symbol
from app.valuation import capm
from app.valuation import scenarios as scen
from app.valuation.company import full_valuation, load_inputs, screening_valuation
from app.valuation.dcf import ValuationError
from app.services.portfolio_doctor import (CONFIG as PORTFOLIO_CONFIG, MAX_SECTOR_PCT,
                                           MAX_STOCK_PCT, MIN_COVERAGE, PRICE_MAX_AGE_DAYS,
                                           analyse_portfolio, sample_holdings,
                                           validate_holdings)

RISK_PROFILES = ("conservative", "moderate", "aggressive")
SEVERITY = {"High": "#ef5350", "Medium": "#fb8c00", "Low": "#42a5f5"}
VAL_COLORS = {"Undervalued": "#26a69a", "Fairly valued": "#9ccc65", "Overvalued": "#ef5350",
              NO_DATA: "#8b97a8"}
CHART_CONFIG = {"displaylogo": False, "scrollZoom": False, "doubleClick": False,
                "modeBarButtonsToRemove": ["lasso2d", "select2d", "zoom2d", "pan2d", "zoomIn2d",
                                           "zoomOut2d", "autoScale2d", "resetScale2d"]}
TABLE_COLS = ["ticker", "name", "sector", "score", "val_label", "val_score",
              "price_to_earnings", "upside", "headline"]
TABLE_NAMES = {"ticker": "Ticker", "name": "Company", "sector": "Sector", "score": "Health",
               "val_label": "Valuation", "val_score": "Value score",
               "price_to_earnings": "P/E", "upside": "Model value vs price %",
               "headline": "Screening result"}
DISCLAIMER = ("Descriptive screening results from historical statements and a simple model, "
              "not buy/sell recommendations. Educational project - not investment advice; "
              "personalised advice requires a registered adviser (e.g. SEBI RIA in India).")


@st.cache_data(show_spinner="Building the quality & value screen ...")
def cached_advisor(db_path: str, analysis_marker: str, provider: str, risk: str,
                   _ctx) -> pd.DataFrame:
    """Keyed by database, data marker (new analysis or price snapshot), provider and risk
    profile. ``_ctx`` is excluded from the key; it is always the context of ``provider``."""
    return advisor_frame(_ctx, risk)


def _html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def _verdict_card(headline: str, color: str) -> str:
    return (f'<div style="border-left:6px solid {color};padding:.8rem 1.1rem;border-radius:12px;'
            f'background:{color}14;margin:.4rem 0 .8rem 0">'
            f'<div style="font-size:.72rem;letter-spacing:.08em;color:#8b97a8">SCREENING RESULT</div>'
            f'<div style="font-size:1.3rem;font-weight:800;color:{color}">{escape(headline)}</div></div>')


def _fmt(v, decimals=1, suffix=""):
    return "N/A" if v is None or pd.isna(v) else f"{v:,.{decimals}f}{suffix}"


def _val_gauge(score, label) -> go.Figure:
    """Valuation score gauge; a missing score is an empty dial with an N/A message, never
    a needle at zero."""
    missing = score is None or pd.isna(score)
    color = VAL_COLORS.get(label, "#8b97a8")
    fig = go.Figure(go.Indicator(
        mode="gauge" if missing else "gauge+number", value=0 if missing else score,
        number={"suffix": "/100", "font": {"size": 30}},
        gauge={"axis": {"range": [0, 100], "tickcolor": "#8b97a8"},
               "bar": {"color": "rgba(0,0,0,0)" if missing else color,
                       "thickness": 0 if missing else 0.75},
               "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
               "steps": [{"range": [0, 40], "color": "rgba(239,83,80,.18)"},
                         {"range": [40, 65], "color": "rgba(249,168,37,.18)"},
                         {"range": [65, 100], "color": "rgba(38,166,154,.18)"}]}))
    if missing:
        fig.add_annotation(text="N/A<br><span style='font-size:11px'>Insufficient valuation "
                                "data</span>", x=0.5, y=0.1, xref="paper", yref="paper",
                           showarrow=False, font=dict(size=22, color="#8b97a8"))
    fig.update_layout(height=200, margin=dict(l=40, r=45, t=25, b=5),
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e6edf3"))
    return fig


def _fcff_text(c: dict, sym: str) -> str:
    """One-line FCFF build-up for the method actually used (see app/valuation/company.py)."""
    if c.get("method") == "EBIT-based FCFF":
        return (f"{c['method']}: EBIT {sym}{c['ebit']:,.0f} x (1 - {c['tax_rate']:.2%}) + D&A "
                f"{sym}{c['depreciation_amortization']:,.0f} - capex "
                f"{sym}{abs(c['capital_expenditure']):,.0f} + change in working capital "
                f"{sym}{c['change_in_working_capital']:,.0f}")
    return (f"{c.get('method', 'cash-flow FCFF')}: operating cash flow "
            f"{sym}{c.get('operating_cash_flow', 0):,.0f} + after-tax interest "
            f"{sym}{c.get('after_tax_interest', 0):,.0f} - capex "
            f"{sym}{abs(c.get('capital_expenditure', 0)):,.0f}; interest paid classified as "
            f"{c.get('interest_classification', 'unknown')}")


class _PriceUnavailable(Exception):
    """Transient price failure: raised inside cached functions so it is NOT cached."""


@st.cache_data(ttl=86400, show_spinner="Estimating beta from 5 years of prices ...")
def _cached_beta(provider: str, ticker: str, benchmark: str, _ctx) -> dict:
    """Cached only when the inputs were downloaded: a deterministic result (a beta, or a
    model error such as too few common months) is kept for a day; a download failure
    raises and is retried on the next run."""
    stock = _ctx.provider.fetch_price_history(ticker, "5y", adjusted=True, interval="1d")
    market = _ctx.provider.fetch_price_history(benchmark, "5y", adjusted=True, interval="1d")
    if stock.empty or market.empty:
        raise _PriceUnavailable(f"Price history unavailable for {ticker} or {benchmark}.")
    monthly = lambda f: f["Close"].resample("ME").last()  # noqa: E731
    cfg = capm.load_assumptions()["beta"]
    try:
        b = capm.estimate_beta(monthly(stock), monthly(market), benchmark, cfg["period"],
                               cfg["interval"], cfg["min_observations"],
                               "daily adjusted closes (splits and dividends) sampled at "
                               "calendar month-end for the stock; the "
                               "benchmark is a price index (dividends excluded); month-end, "
                               "common months only")
    except ValuationError as exc:
        return {"error": str(exc)}
    return vars(b)


def cached_beta(provider: str, ticker: str, benchmark: str, _ctx) -> dict | None:
    """Monthly-return beta against the market benchmark (5 years). Returns a dict with an
    ``error`` key when no beta is available - never a default beta."""
    if provider != "yfinance":
        return {"error": "Beta needs live prices (Yahoo Finance data source)."}
    try:
        return _cached_beta(provider, ticker, benchmark, _ctx)
    except _PriceUnavailable as exc:
        return {"error": str(exc)}
    except Exception as exc:  # noqa: BLE001 - network or provider failure, shown to the user
        return {"error": f"Price download failed: {exc}"}


@st.cache_data(show_spinner=False)
def cached_history(db_path: str, marker: str, provider: str, ticker: str, _ctx):
    """FCFF history stored with the latest analysis (all fiscal years)."""
    from app.database import repositories as repo

    with _ctx.db.session_scope() as session:
        company = repo.get_company(session, ticker)
        run = repo.latest_analysis_run(session, company.id) if company else None
        return ((run.summary or {}).get("fcff_history") if run else None) or None


def _scale(currency: str | None) -> tuple[float, str]:
    return (1e7, f"{currency} crore") if currency == "INR" else (1e6, f"{currency} million")


@st.cache_data(show_spinner=False)
def cached_inputs(db_path: str, marker: str, provider: str, ticker: str, _ctx):
    return load_inputs(_ctx, [ticker]).get(ticker)


def _table(frame: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in TABLE_COLS if c in frame]
    return frame[cols].rename(columns=TABLE_NAMES).round(1)


# ----------------------------------------------------------------------------- Advisor tab
def render_advisor_tab(ctx, selected: str | None, db_key: str, marker: str, provider: str,
                       names: dict[str, str]) -> None:
    with st.expander("How to read this page"):
        st.markdown(
            "- **Screening result** = financial-health band x valuation; it describes the "
            "numbers, it is not a buy/sell instruction.\n"
            "- **Valuation score** combines P/E and P/B against same-market sector peers, PEG, "
            "free-cash-flow yield and a DCF margin of safety; negative multiples are N/M and "
            "never count as cheap.\n"
            "- **Full valuation** discounts free cash flow to the firm (FCFF) at a WACC built "
            "from CAPM. Below it: the valuation date and price date, the assumptions and why, "
            "warnings, the cash-flow history, a sensitivity table, bear/base/bull scenarios "
            "and a reverse DCF.\n"
            "- A DCF is only as good as its assumptions: read the terminal-value share and "
            "the warnings before the headline value.")
    _html(ui.section("Quality & Value screen",
                     "financial health and valuation side by side - a description, not advice"))
    c1, c2 = st.columns([1, 3])
    risk = c1.selectbox("Investor profile", RISK_PROFILES, index=1, key="advisor_risk",
                        help="Model assumptions only: sets the DCF discount rate and how much "
                             "of past growth is projected forward.")
    frame = cached_advisor(db_key, marker, provider, risk, ctx)
    if frame.empty:
        _html(ui.empty_state("💡", "No analysed companies yet",
                             "Analyse a ticker from the sidebar to screen it."))
        return
    tickers = list(frame["ticker"])
    default = selected if selected in tickers else tickers[0]
    pick = c2.selectbox("Company", tickers, index=tickers.index(default), key="advisor_pick",
                        format_func=lambda t: f"{t} · {names.get(t, '')}"[:48])
    row = company_row(frame, pick)
    if row is None:
        return
    sym = currency_symbol(row.get("currency"))

    left, right = st.columns([1.6, 1])
    with left:
        _html(_verdict_card(row["headline"], row["color"]))
        for line in row["why"]:
            st.markdown(f"- {line}")
    with right:
        st.caption("Valuation score (100 = cheapest)")
        st.plotly_chart(_val_gauge(row["val_score"], row["val_label"]), width="stretch",
                        config=CHART_CONFIG, key="advisor_val_gauge")

    price = row.get("price")
    _html(ui.cards([
        ("Health score", _fmt(row.get("score"), 0), str(row.get("interpretation") or "not scored"),
         ui.score_color(None if pd.isna(row.get("score")) else float(row["score"])),
         "The existing financial-health score (0-100), sector-relative."),
        ("Valuation", str(row["val_label"]),
         f"{int(row.get('val_components_used') or 0)} of "
         f"{int(row.get('val_components_applicable') or 0)} checks · score "
         f"{_fmt(row['val_score'], 0)}", VAL_COLORS.get(row["val_label"], None),
         "Illustrative composite (documented equal weights, not statistically validated) of "
         "the applicable checks: P/E vs same-market sector peers, PEG, FCF yield, DCF margin "
         "of safety (P/B for financials)."),
        ("Price", f"{sym}{_fmt(price, 2)}" if price and not pd.isna(price) else "N/A",
         f"snapshot {row.get('price_date') or 'date n/a'} · not live", None,
         "Last stored market snapshot (not a live quote); its date is shown."),
        ("DCF model value", f"{sym}{_fmt(row.get('fair_value'), 0)}" if row.get("dcf") else "N/A",
         f"vs price {_fmt(row.get('upside'), 0, '%')}" if row.get("dcf") else "not applicable",
         None, "10-year discounted free cash flow: 5 years at the projected growth, 5 years "
               "fading to long-run growth, then a terminal value."),
        ("Margin-of-safety price", f"{sym}{_fmt(row.get('buy_below'), 0)}" if row.get("dcf") else "N/A",
         "model value less 20%", None, "DCF model value less a 20% margin of safety - a "
                                        "reference level, not a price target."),
    ]))

    with st.expander("How the valuation score was built", expanded=False):
        comps = row.get("components") or {}
        if comps:
            st.dataframe(pd.DataFrame([{"Check": c["text"], "Score": c["score"]}
                                       for c in comps.values()]),
                         hide_index=True, width="stretch",
                         column_config={"Score": st.column_config.ProgressColumn(
                             "Score", min_value=0, max_value=100, format="%d")})
        for n in row.get("val_notes") or []:
            st.caption(n)
        d = row.get("dcf")
        if isinstance(d, dict):
            st.markdown("**Screening DCF (FCFF -> enterprise value -> equity value)**")
            c = d.get("fcff_components") or {}
            st.caption(f"Base FCFF {sym}{d['base_fcf']:,.0f} - {_fcff_text(c, sym)}. "
                       f"Growth {d['growth']:+.1%} ({d.get('growth_note')}) for 5 years, fading "
                       f"to {d['terminal_growth']:.0%} terminal growth. Discount: "
                       f"{d.get('discount_note')}.")
            bridge = d.get("bridge") or {}
            if bridge:
                labels = {"enterprise_value": "Enterprise value (PV of FCFF + terminal value)",
                          "plus_cash": "+ Cash and short-term investments",
                          "plus_non_operating_assets": "+ Non-operating assets",
                          "minus_debt": "- Total debt", "minus_minority_interest":
                          "- Minority interest", "minus_other_senior_claims":
                          "- Other senior claims", "equity_value": "= Equity value"}
                st.dataframe(pd.DataFrame([{"Item": labels.get(k, k),
                                            "Amount": ui.compact_number(v, row.get("currency"))}
                                           for k, v in bridge.items()]),
                             hide_index=True, width="stretch")
                sc = c.get("share_count") or {}
                if sc.get("classification"):
                    st.caption(f"Share count: {sc['shares']:,.0f} - {sc['classification']}; "
                               f"as of {sc.get('as_of') or 'unknown date'}.")
                st.caption(f"Equity value / {d.get('shares_source') or 'shares'} "
                           f"({d.get('shares') or 0:,.0f}) = {sym}{d['fair_value']:,.2f} per "
                           f"share. Terminal value share of the total: "
                           f"{d.get('terminal_share', float('nan')):.0%}.")
            for w in d.get("warnings") or []:
                st.caption(f"⚠ {w}")

    render_full_valuation(ctx, provider, db_key, marker, pick, row, sym)

    # ---- universe view
    _html(ui.section("Quality vs Value map",
                     "Top-right = strong business at a good price. Dots coloured by health score."))
    plot = frame.dropna(subset=["score", "val_score"])
    if len(plot):
        fig = plots.scatter_map(plot, "val_score", "score", "Valuation score (100 = cheapest)",
                                "Health score", height=440)
        fig.add_hline(y=50, line_dash="dot", line_color="#8b97a8")
        fig.add_vline(x=40, line_dash="dot", line_color="#8b97a8")
        st.plotly_chart(fig, width="stretch", config=CHART_CONFIG, key="advisor_map")

    _html(ui.section("Screen results", "strong financials with a price at or near model "
                     "value, best health first"))
    f1, f2 = st.columns([2, 1])
    sectors = sorted(s for s in frame["sector"].dropna().unique())
    sec_pick = f1.multiselect("Sector", sectors, default=[], key="advisor_sectors",
                              placeholder="All sectors")
    market = f2.radio("Market", ["All", "India (.NS/.BO)", "Other"], horizontal=True,
                      key="advisor_market")
    pool = frame if not sec_pick else frame[frame["sector"].isin(sec_pick)]
    if market != "All":
        is_in = pool["ticker"].str.upper().str.endswith((".NS", ".BO"))
        pool = pool[is_in] if market.startswith("India") else pool[~is_in]
    best = ideas(pool)
    if best.empty:
        st.caption("No company in this selection matches the screen.")
    else:
        st.dataframe(_table(best), hide_index=True, width="stretch")
    with st.expander("All screening results (filterable)", expanded=False):
        keys = st.multiselect("Screening result", list(ACTIONS), default=[], key="advisor_keys",
                              format_func=lambda k: ACTIONS[k][0], placeholder="All results")
        view = pool if not keys else pool[pool["decision"].isin(keys)]
        st.dataframe(_table(view.sort_values("score", ascending=False)), hide_index=True,
                     width="stretch")
        st.download_button("⬇ Download screen (CSV)", _table(view).to_csv(index=False),
                           file_name="quality_value_screen.csv", mime="text/csv")
    st.caption("Screening result = health-score band × valuation. A strong business can still "
               "be priced above its model value, and a cheap stock with weak financials fits "
               "the value-trap profile. " + DISCLAIMER)


# ----------------------------------------------------------------------------- Portfolio tab
@st.cache_data(ttl=900, show_spinner="Fetching latest closing prices ...")
def cached_live_prices(provider: str, symbols: tuple[str, ...], _ctx) -> dict:
    """Latest daily close per symbol from Yahoo Finance (cached 15 minutes). Missing or
    failed symbols are simply absent - no price is invented."""
    if provider != "yfinance":
        return {}
    out = {}
    for sym in symbols:
        try:
            hist = _ctx.provider.fetch_price_history(sym, "5d")
        except Exception:  # noqa: BLE001 - one bad symbol must not stop the others
            continue
        if not hist.empty:
            out[sym] = (float(hist["Close"].iloc[-1]), hist.index[-1].date())
    return out


def _money(cur: str, v) -> str:
    return "N/A" if v is None or pd.isna(v) else f"{cur}{v:,.0f}"


def render_portfolio_tab(ctx, db_key: str, marker: str, provider: str) -> None:
    _html(ui.section("Portfolio Doctor", "upload a broker holdings file; nothing is stored",
                     kind="risk"))
    with st.expander("How to read this page"):
        st.markdown(
            "- Upload the holdings CSV/Excel from your broker (symbol, quantity, average "
            "cost); every rejected row is listed with the reason.\n"
            "- Prices must be fresh and dated; average cost is never used as a price, so an "
            "unpriced holding shows N/A.\n"
            "- Weights cover **priced holdings only** - the denominator is stated.\n"
            "- Each currency is reviewed separately (no FX conversion).\n"
            "- Alerts are observations on concentration and financial health, not "
            "instructions.")
    st.warning("Privacy: before uploading, remove account numbers, client IDs, names, PAN "
               "and any other personal details from the file. Only symbol, quantity, average "
               "cost (and optionally last price) are needed. The file is processed in memory "
               "and not saved.", icon="🔒")
    risk = st.session_state.get("advisor_risk", "moderate")
    frame = cached_advisor(db_key, marker, provider, risk, ctx)

    c1, c2 = st.columns([2, 1])
    up = c1.file_uploader("Holdings file (CSV or Excel) - needs symbol, quantity and average "
                          "cost columns; Zerodha / Upstox / Groww / ICICI exports work as-is.",
                          type=["csv", "xlsx"], key="pf_upload")
    suffix = c2.selectbox("Default exchange for bare symbols", [".NS", ".BO", ""],
                          format_func=lambda s: {".NS": "NSE (.NS)", ".BO": "BSE (.BO)",
                                                 "": "US / none"}[s], key="pf_suffix")
    use_sample = c2.checkbox("Use the demo portfolio", value=up is None, key="pf_sample")
    broker_as_of = c2.date_input("Prices in my file are as of", value=None, key="pf_asof",
                                 help="Broker LTP is used only when you give its date and it "
                                      f"is at most {PRICE_MAX_AGE_DAYS} days old.")
    raw, demo = None, False
    try:
        if up is not None:
            raw = pd.read_csv(up) if up.name.lower().endswith(".csv") else pd.read_excel(up)
        elif use_sample:
            raw, demo = sample_holdings(frame), True
    except Exception as exc:          # noqa: BLE001 - show the reader's message to the user
        st.error(f"Could not read the file ({type(exc).__name__}): {exc}. Save it again as "
                 "CSV or .xlsx and retry.")
        return
    if raw is None:
        _html(ui.empty_state("🩺", "Upload a holdings file to begin",
                             "Or tick 'Use the demo portfolio' to see how it works."))
        return
    if demo:
        st.info("DEMONSTRATION DATA: the demo portfolio's quantities are invented and its "
                "average costs are synthetic (derived from stored prices), so its P&L is "
                "illustrative only.", icon="🧪")
    try:
        holdings, rejected, counts = validate_holdings(raw, suffix)
    except ValueError as exc:
        st.error(str(exc))
        return
    st.caption(f"File rows: {counts['rows']} · accepted {counts['accepted']} · rejected "
               f"{counts['rejected']} · holdings after merging duplicates "
               f"{counts['holdings']} ({counts['duplicates_merged']} merged)")
    if len(rejected):
        with st.expander(f"{len(rejected)} rejected row(s) and why"):
            st.dataframe(rejected, hide_index=True, width="stretch")

    st.markdown("**Your holdings** (edit if something looks wrong)")
    holdings = st.data_editor(holdings, num_rows="dynamic", hide_index=True, width="stretch",
                              key="pf_editor")
    holdings = holdings.dropna(subset=["symbol", "quantity", "avg_price"])
    holdings = holdings[(holdings["quantity"] > 0) & (holdings["avg_price"] > 0)]
    if holdings.empty:
        st.warning("No valid holdings.")
        return

    live = cached_live_prices(provider, tuple(sorted(holdings["symbol"])), ctx)
    probe = analyse_portfolio(holdings, frame, live_prices=live, broker_as_of=broker_as_of)
    currencies = list(probe["summary"]["currencies"])
    cur_code = probe["summary"]["currency"]
    if len(currencies) > 1:
        st.warning(f"This portfolio holds {', '.join(currencies)} positions. Amounts in "
                   "different currencies are never added together (no FX conversion is "
                   "implemented), so it is analysed one currency at a time.")
        cur_code = st.radio("Reporting currency", currencies, horizontal=True, key="pf_cur")
    out = analyse_portfolio(holdings, frame, reporting_currency=cur_code, live_prices=live,
                            broker_as_of=broker_as_of)
    s, t = out["summary"], out["table"]
    cur = currency_symbol(cur_code)

    g, k = st.columns([1, 2])
    with g:
        cw = PORTFOLIO_CONFIG["composite_weights"]
        st.caption(f"Portfolio score ({cw['health']:.0%} value-weighted health + "
                   f"{cw['position_spread']:.0%} position spread) - illustrative composite")
        st.plotly_chart(plots.score_gauge(s["health"], ""), width="stretch",
                        config=CHART_CONFIG, key="pf_gauge")
        if s["composite_withheld"]:
            st.caption(f"Withheld: only {s['analysed_coverage']:.0%} of the priced value has a "
                       f"health score (minimum {MIN_COVERAGE:.0%}).")
    with k:
        _html(ui.cards([
            ("Invested (cost basis)", _money(cur, s["invested"]), cur_code, None,
             "Quantity x average cost for all holdings in this currency."),
            ("Current value", _money(cur, s["value"]),
             f"{s['pnl_pct']:+.1f}% on priced holdings" if s["pnl_pct"] is not None else
             "no fresh prices", (ui.UP if (s["pnl"] or 0) >= 0 else ui.DOWN)
             if s["value"] is not None else None,
             f"Holdings with a fresh price only (max {PRICE_MAX_AGE_DAYS} days old). Never "
             "uses average cost as a price."),
            ("Without a price", str(s["n_unpriced"]), f"cost basis {_money(cur, s['cost_unpriced'])}",
             ui.DOWN if s["n_unpriced"] else None,
             "No fresh broker, live or stored price: value, P&L and weight are N/A."),
            ("Analysed coverage", f"{s['analysed_coverage']:.0%}" if s["value"] else "N/A",
             f"{s['n_analysed']} of {s['n']} holdings", None,
             "Share of the priced value that has a financial-health score."),
            ("Position spread", f"{s['position_spread']:.0f}/100"
             if s["position_spread"] is not None else "N/A",
             f"{s['effective_positions']} effective positions"
             if s["effective_positions"] is not None else None, None,
             PORTFOLIO_CONFIG["position_spread_note"]),
            ("Largest position", f"{s['largest_position']:.0f}%" if s["largest_position"]
             is not None else "N/A", str(s["largest_position_name"] or ""), ui.DOWN
             if (s["largest_position"] or 0) > MAX_STOCK_PCT else None,
             f"Illustrative limit {MAX_STOCK_PCT:.0f}% (project assumption)."),
            ("Largest sector", f"{s['largest_sector_pct']:.0f}%" if s["largest_sector_pct"]
             is not None else "N/A", f"{s['largest_sector'] or ''} · {s['n_sectors']} sectors",
             ui.DOWN if (s["largest_sector_pct"] or 0) > MAX_SECTOR_PCT else None,
             f"Illustrative limit {MAX_SECTOR_PCT:.0f}% (project assumption). Sector spread "
             "is reported separately from position spread."),
        ]))

    _html(ui.section("Alerts", "what to look at first", kind="risk"))
    if not out["alerts"]:
        st.success("No alerts: position sizes, sectors and fundamentals are within the "
                   "illustrative limits.")
    for a in out["alerts"]:
        _html(ui.flag_card(", ".join(a["tickers"]) or "Portfolio", a["severity"].upper(),
                           a["title"], a["detail"], ""))

    _html(ui.section("Holding-by-holding review", "observations, not instructions"))
    show = t[["Symbol", "Company", "Sector", "Qty", "Avg cost", "Price", "Price source",
              "Price date", "Price status", "Weight %", "P&L %", "Health", "Valuation",
              "Verdict", "Action", "Why", "Price note"]].round(2).rename(
        columns={"Verdict": "Screening result", "Action": "Observation", "Why": "Reason",
                 "Weight %": "Weight of priced holdings (%)"})
    st.dataframe(show, hide_index=True, width="stretch",
                 column_config={"Health": st.column_config.ProgressColumn(
                     "Health", min_value=0, max_value=100, format="%d")})
    if s.get("n_unpriced"):
        st.caption(f"Weights are shares of the {s['n_priced']} priced holding(s) only - "
                   f"{s['n_unpriced']} unpriced holding(s) (cost basis "
                   f"{_money(cur, s['cost_unpriced'])}) are outside the denominator, so "
                   "these weights are not total-portfolio weights.")
    if len(out["excluded"]):
        with st.expander(f"{len(out['excluded'])} holding(s) in another currency (not included "
                         "above)"):
            st.dataframe(out["excluded"][["Symbol", "Company", "Currency", "Currency basis",
                                          "Qty", "Avg cost", "Price", "Price source"]],
                         hide_index=True, width="stretch")

    priced = t[t["Weight %"].notna()]
    if len(priced):
        p1, p2 = st.columns(2)
        sw = s["sector_weights"]
        pie = go.Figure(go.Pie(labels=list(sw.index), values=list(sw.values), hole=0.5,
                               textinfo="label+percent"))
        pie.update_layout(title=dict(text=f"Sector mix ({cur_code}, priced holdings)",
                                     font=dict(size=13)), showlegend=False)
        p1.plotly_chart(plots._style(pie, 330, legend=False), width="stretch",
                        config=CHART_CONFIG, key="pf_pie")
        bars = plots.ranked_bars(dict(zip(priced["Company"], priced["Weight %"])),
                                 f"Weight of priced holdings (%) - dotted line = "
                                 f"{MAX_STOCK_PCT:.0f}% illustrative limit", 330)
        bars.add_vline(x=MAX_STOCK_PCT, line_dash="dot", line_color=ui.DOWN)
        p2.plotly_chart(bars, width="stretch", config=CHART_CONFIG, key="pf_bars")

    st.download_button("⬇ Download portfolio review (CSV)", t.to_csv(index=False),
                       file_name="portfolio_review.csv", mime="text/csv")
    st.caption(f"Price policy: dated broker LTP -> latest Yahoo Finance close -> stored "
               f"snapshot, each only if at most {PRICE_MAX_AGE_DAYS} days old; otherwise N/A. "
               "Thresholds are illustrative project assumptions (config/portfolio_config.json). "
               "Taxes, goals and time horizon are not considered. " + DISCLAIMER)


def render_full_valuation(ctx, provider: str, db_key: str, marker: str, ticker: str, row,
                          sym: str) -> None:
    """FCFF discounted at a CAPM/WACC for one company, with every input and source shown."""
    _html(ui.section("Full valuation: CAPM / WACC", kind="valuation",
                     hint=
                     "FCFF discounted at WACC, then the equity bridge - inputs and sources below"))
    inputs = cached_inputs(db_key, marker, provider, ticker, ctx)
    if inputs is None:
        st.caption("No stored statements for this company.")
        return
    try:
        market = capm.market_assumptions(inputs.currency)
    except ValuationError as exc:
        st.info(str(exc))
        return
    beta_info = cached_beta(provider, ticker, market["benchmark_index"], ctx)
    beta = None
    if beta_info and "error" not in beta_info:
        beta = capm.BetaEstimate(**beta_info)
    v = full_valuation(inputs, beta)
    rows = [("Risk-free rate", f"{market['risk_free_rate']:.2%}", market["risk_free_source"]),
            ("Equity risk premium", f"{market['equity_risk_premium']:.2%}",
             market["equity_risk_premium_source"]),
            ("Beta", f"{beta.beta:.2f}" if beta else "N/A",
             (f"OLS on {beta.observations} monthly returns vs {market['benchmark_name']} "
              f"({beta.period}, {beta.price_basis}); R² {beta.r_squared:.2f}; Blume-adjusted "
              f"{beta.blume_adjusted:.2f}"
              f" (reference)") if beta else (beta_info or {}).get("error", "unavailable"))]
    w = v.wacc_detail
    if w is not None:
        rows += [("Cost of equity (CAPM)", f"{w.cost_of_equity:.2%}", "Rf + beta x ERP"),
                 ("Pre-tax cost of debt", f"{w.cost_of_debt_pre_tax:.2%}",
                  "interest / average debt (or disclosed fallback)"),
                 ("Tax rate", f"{w.tax_rate:.2%}", market["statutory_tax_source"]),
                 ("Weights E / D", f"{w.equity_weight:.0%} / {w.debt_weight:.0%}",
                  w.equity_value_basis + " / book debt"),
                 ("WACC", f"{w.wacc:.2%}", "E/(D+E) x Ke + D/(D+E) x Kd x (1 - t)"),
                 ("Terminal growth", f"{market['terminal_growth']:.1%}",
                  market["terminal_growth_note"])]
    st.dataframe(pd.DataFrame(rows, columns=["Input", "Value", "Source / formula"]),
                 hide_index=True, width="stretch")
    history = cached_history(db_key, marker, provider, ticker, ctx)
    scale = _scale(inputs.currency)
    if not v.ok:
        st.info(f"Full valuation N/A: {v.reason}")
        screen = screening_valuation(inputs)
        if screen.ok:
            st.markdown("**Screening DCF instead** (fixed screening discount rate, not a WACC)")
            valuation_view.render(screen, market, sym, inputs.currency, history, scale)
            render_valuation_research(screen, sym, inputs.currency,
                                      "screening DCF (fixed screening discount rate, not WACC)")
        elif history:
            valuation_view.render_history_only(history, scale)
        return
    valuation_view.render(v, market, sym, inputs.currency, history, scale)
    st.caption("Market inputs (risk-free rate, equity risk premium, tax) are dated in "
               "config/valuation_assumptions.json; beta uses live Yahoo Finance prices. "
               + DISCLAIMER)
    render_valuation_research(v, sym, inputs.currency, "CAPM / WACC DCF")


def render_valuation_research(v, sym: str, currency: str | None, basis: str) -> None:
    """Sensitivity grid, bear/base/bull and reverse DCF for one valuation (same engine,
    same base FCFF, bridge and share count - only the named inputs change)."""
    _html(ui.section("Valuation research", kind="valuation", hint=f"how much the answer "
                     f"depends on the inputs - "
                                           f"{basis}"))
    st.markdown("**Sensitivity: value per share by discount rate and terminal growth**")
    grid = scen.sensitivity_grid(v)
    if not grid.empty:
        shown = grid.copy()
        shown.index = [f"{r:.1%}" + (" (base)" if abs(r - v.discount_rate) < 1e-9 else "")
                       for r in grid.index]
        shown.columns = [f"g {g:.1%}" + (" (base)" if abs(g - v.terminal_growth) < 1e-9
                                        else "") for g in grid.columns]
        shown.index.name = "Discount rate"
        st.dataframe(shown.map(lambda x: f"{sym}{x:,.0f}" if x is not None and pd.notna(x)
                               else "N/A"), width="stretch")
        st.caption("Each cell re-runs the same DCF with only the discount rate and terminal "
                   "growth changed (values at the same date as the headline value). N/A = the "
                   "rate is not above terminal growth (Gordon model undefined), or terminal "
                   f"growth is at or above the risk-free rate ({v.risk_free:.2%}), which is "
                   "not economically reasonable for a perpetuity.")
    st.markdown("**Scenarios** (assumption sets, not forecasts or probabilities)")
    cases = scen.scenarios(v)
    if cases:
        st.dataframe(pd.DataFrame([{
            "Case": c.name, "Growth (yrs 1-5)": f"{c.growth:+.1%}",
            "Discount rate": f"{c.rate:.2%}", "Terminal growth": f"{c.terminal:.1%}",
            "Value / share": f"{sym}{c.value_per_share:,.0f}" if c.value_per_share is not None
            else "N/A",
            "vs price": f"{c.vs_price:+.0%}" if c.vs_price is not None else "N/A"}
            for c in cases]), hide_index=True, width="stretch")
        st.caption("Bear: growth -3 pts, rate +1 pt, terminal -1 pt. Bull: growth +3 pts, "
                   "rate -1 pt, terminal +0.5 pt. Growth stays inside the forecast bounds and "
                   "terminal growth stays below the risk-free rate. Defined in "
                   "config/valuation_assumptions.json.")
    st.markdown("**Reverse DCF: what growth does the current price imply?**")
    rd = scen.reverse_dcf(v)
    if rd.implied_growth is not None:
        _html(ui.cards([
            ("Implied growth (yrs 1-5)", f"{rd.implied_growth:+.1%}",
             f"at price {sym}{rd.price:,.0f}", None),
            ("Model growth", f"{rd.model_growth:+.1%}", v.growth_source or "", None),
            ("Gap", f"{(rd.implied_growth - rd.model_growth) * 100:+.1f} pts",
             "price expects more" if rd.implied_growth > rd.model_growth else
             "price expects less", None)]))
    st.caption(rd.note + " Implied growth describes market expectations under this model; "
               "it is not a forecast.")
