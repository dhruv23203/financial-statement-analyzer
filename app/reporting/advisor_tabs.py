"""Streamlit rendering for the two investor tabs: Advisor and Portfolio Doctor.

Kept out of ``dashboard.py`` so the existing tabs stay untouched; the dashboard only
creates the tabs and calls :func:`render_advisor_tab` / :func:`render_portfolio_tab`.
Visual language follows ``app/reporting/ui.py`` (same cards, pills, colours).
"""

from __future__ import annotations

from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.reporting import plots, ui
from app.services.advisor import ACTIONS, NO_DATA, advisor_frame, company_row, ideas
from app.services.advisor import currency_symbol
from app.services.portfolio_doctor import (MAX_STOCK_PCT, analyse_portfolio, parse_holdings,
                                           sample_holdings)

RISK_PROFILES = ("conservative", "moderate", "aggressive")
SEVERITY = {"High": "#ef5350", "Medium": "#fb8c00", "Low": "#42a5f5"}
VAL_COLORS = {"Undervalued": "#26a69a", "Fairly valued": "#9ccc65", "Overvalued": "#ef5350",
              NO_DATA: "#8b97a8"}
CHART_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}
TABLE_COLS = ["ticker", "name", "sector", "score", "val_label", "val_score",
              "price_to_earnings", "upside", "headline"]
TABLE_NAMES = {"ticker": "Ticker", "name": "Company", "sector": "Sector", "score": "Health",
               "val_label": "Valuation", "val_score": "Value score",
               "price_to_earnings": "P/E", "upside": "Upside to DCF %", "headline": "Verdict"}


@st.cache_data(show_spinner="Building advisor verdicts ...")
def cached_advisor(db_path: str, analysis_marker: int, provider: str, risk: str,
                   _ctx) -> pd.DataFrame:
    """Cached like the dashboard's other reads: any new analysis changes the marker and
    refreshes it. ``_ctx`` is excluded from the cache key (leading underscore)."""
    return advisor_frame(_ctx, risk)


def _html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def _verdict_card(headline: str, color: str) -> str:
    return (f'<div style="border-left:6px solid {color};padding:.8rem 1.1rem;border-radius:12px;'
            f'background:{color}14;margin:.4rem 0 .8rem 0">'
            f'<div style="font-size:.72rem;letter-spacing:.08em;color:#8b97a8">ADVISOR VERDICT</div>'
            f'<div style="font-size:1.45rem;font-weight:800;color:{color}">{escape(headline)}</div></div>')


def _fmt(v, decimals=1, suffix=""):
    return "N/A" if v is None or pd.isna(v) else f"{v:,.{decimals}f}{suffix}"


def _val_gauge(score, label) -> go.Figure:
    color = VAL_COLORS.get(label, "#8b97a8")
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=0 if score is None or pd.isna(score) else score,
        number={"suffix": "/100", "font": {"size": 30}},
        gauge={"axis": {"range": [0, 100], "tickcolor": "#8b97a8"}, "bar": {"color": color},
               "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
               "steps": [{"range": [0, 40], "color": "rgba(239,83,80,.18)"},
                         {"range": [40, 65], "color": "rgba(249,168,37,.18)"},
                         {"range": [65, 100], "color": "rgba(38,166,154,.18)"}]}))
    fig.update_layout(height=200, margin=dict(l=40, r=45, t=25, b=5),
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e6edf3"))
    return fig


def _table(frame: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in TABLE_COLS if c in frame]
    return frame[cols].rename(columns=TABLE_NAMES).round(1)


# ----------------------------------------------------------------------------- Advisor tab
def render_advisor_tab(ctx, selected: str | None, db_key: str, marker: int, provider: str,
                       names: dict[str, str]) -> None:
    _html(ui.section("Investment Advisor",
                     "Quality (the health score) and price (valuation) combined into one call"))
    c1, c2 = st.columns([1, 3])
    risk = c1.selectbox("Investor profile", RISK_PROFILES, index=1, key="advisor_risk",
                        help="Sets the DCF discount rate and how much of past growth is "
                             "projected forward.")
    frame = cached_advisor(db_key, marker, provider, risk, ctx)
    if frame.empty:
        _html(ui.empty_state("💡", "No analysed companies yet",
                             "Analyse a ticker from the sidebar to get a verdict."))
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
        ("Valuation", str(row["val_label"]), f"score {_fmt(row['val_score'], 0)}",
         VAL_COLORS.get(row["val_label"], None),
         "Average of the checks below: P/E vs sector, PEG, FCF yield, DCF margin of safety."),
        ("Price", f"{sym}{_fmt(price, 2)}" if price and not pd.isna(price) else "N/A",
         "latest stored snapshot", None, "From the market snapshot saved with the analysis."),
        ("DCF fair value", f"{sym}{_fmt(row.get('fair_value'), 0)}" if row.get("dcf") else "N/A",
         f"upside {_fmt(row.get('upside'), 0, '%')}" if row.get("dcf") else "not applicable",
         None, "10-year discounted free cash flow: 5 years at the projected growth, 5 years "
               "fading to long-run growth, then a terminal value."),
        ("Buy below", f"{sym}{_fmt(row.get('buy_below'), 0)}" if row.get("dcf") else "N/A",
         "20% margin of safety", None, "Fair value less a 20% margin of safety."),
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
        if d:
            st.caption(f"DCF assumptions: growth {d['growth'] * 100:.1f}% for 5 years fading to "
                       f"{d['terminal_growth'] * 100:.0f}%, discount rate {d['discount_rate'] * 100:.0f}%, "
                       f"base free cash flow {sym}{d['base_fcf']:,.0f}.")

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

    _html(ui.section("Ideas", "Top-quartile health with a buy-type verdict, best first"))
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
        st.caption("No buy-type verdicts in this selection.")
    else:
        st.dataframe(_table(best), hide_index=True, width="stretch")
    with st.expander("All verdicts (filterable)", expanded=False):
        keys = st.multiselect("Verdict", list(ACTIONS), default=[], key="advisor_keys",
                              format_func=lambda k: ACTIONS[k][0], placeholder="All verdicts")
        view = pool if not keys else pool[pool["decision"].isin(keys)]
        st.dataframe(_table(view.sort_values("score", ascending=False)), hide_index=True,
                     width="stretch")
        st.download_button("⬇ Download verdicts (CSV)", _table(view).to_csv(index=False),
                           file_name="advisor_verdicts.csv", mime="text/csv")
    st.caption("Verdict = health-score band × valuation. A great business at a high price is "
               "'wait', not 'buy'; a cheap stock with weak financials is a possible value trap. "
               "Educational - not investment advice.")


# ----------------------------------------------------------------------------- Portfolio tab
def render_portfolio_tab(ctx, db_key: str, marker: int, provider: str) -> None:
    _html(ui.section("Portfolio Doctor",
                     "Upload your broker's holdings file; nothing is stored"))
    risk = st.session_state.get("advisor_risk", "moderate")
    frame = cached_advisor(db_key, marker, provider, risk, ctx)

    c1, c2 = st.columns([2, 1])
    up = c1.file_uploader("Holdings file (CSV or Excel) - needs symbol, quantity and average "
                          "cost columns; Zerodha / Upstox / Groww / ICICI exports work as-is.",
                          type=["csv", "xlsx"], key="pf_upload")
    suffix = c2.selectbox("Default exchange for bare symbols", [".NS", ".BO", ""],
                          format_func=lambda s: {".NS": "NSE (.NS)", ".BO": "BSE (.BO)",
                                                 "": "US / none"}[s], key="pf_suffix")
    use_sample = c2.checkbox("Use a sample portfolio", value=up is None, key="pf_sample")
    raw = None
    try:
        if up is not None:
            raw = pd.read_csv(up) if up.name.lower().endswith(".csv") else pd.read_excel(up)
        elif use_sample:
            raw = sample_holdings(frame)
    except Exception as exc:          # noqa: BLE001 - show the reader's message to the user
        st.error(f"Could not read the file: {exc}")
        return
    if raw is None:
        _html(ui.empty_state("🩺", "Upload a holdings file to begin",
                             "Or tick 'Use a sample portfolio' to see how it works."))
        return
    try:
        holdings = parse_holdings(raw, suffix)
    except ValueError as exc:
        st.error(str(exc))
        return

    st.markdown("**Your holdings** (edit if something looks wrong)")
    holdings = st.data_editor(holdings, num_rows="dynamic", hide_index=True, width="stretch",
                              key="pf_editor")
    holdings = holdings.dropna(subset=["symbol", "quantity", "avg_price"])
    if holdings.empty:
        st.warning("No valid holdings.")
        return
    out = analyse_portfolio(holdings, frame)
    s, t = out["summary"], out["table"]
    cur = "₹" if suffix in (".NS", ".BO") else "$"

    g, k = st.columns([1, 2])
    with g:
        st.caption("Portfolio health (70% value-weighted health score + 30% diversification)")
        st.plotly_chart(plots.score_gauge(s["health"], ""), width="stretch",
                        config=CHART_CONFIG, key="pf_gauge")
    with k:
        _html(ui.cards([
            ("Invested", f"{cur}{s['invested']:,.0f}", None, None),
            ("Current value", f"{cur}{s['value']:,.0f}", f"{s['pnl_pct']:+.1f}%",
             ui.UP if s["pnl"] >= 0 else ui.DOWN),
            ("Holdings", str(s["n"]), None, None),
            ("Diversification", f"{s['diversification']:.0f}/100",
             "8 equal positions = 100", None),
            ("In weak stocks", f"{s['weak_pct']:.0f}%", "of value", ui.DOWN if s["weak_pct"] > 20 else None),
            ("In overvalued stocks", f"{s['expensive_pct']:.0f}%", "of value", None),
        ]))

    _html(ui.section("Alerts", "What to look at first"))
    if not out["alerts"]:
        st.success("No alerts: position sizes, sectors and fundamentals look reasonable.")
    for a in out["alerts"]:
        _html(ui.flag_card(", ".join(a["tickers"]) or "Portfolio", a["severity"].upper(),
                           a["title"], a["detail"], ""))

    _html(ui.section("What to do with each holding"))
    show = t[["Symbol", "Company", "Sector", "Weight %", "P&L %", "Health", "Valuation",
              "Verdict", "Action", "Why"]].round(1)
    st.dataframe(show, hide_index=True, width="stretch",
                 column_config={"Health": st.column_config.ProgressColumn(
                     "Health", min_value=0, max_value=100, format="%d")})

    p1, p2 = st.columns(2)
    sw = s["sector_weights"]
    pie = go.Figure(go.Pie(labels=list(sw.index), values=list(sw.values), hole=0.5,
                           textinfo="label+percent"))
    pie.update_layout(title=dict(text="Sector mix", font=dict(size=13)), showlegend=False)
    p1.plotly_chart(plots._style(pie, 330, legend=False), width="stretch", config=CHART_CONFIG,
                    key="pf_pie")
    bars = plots.ranked_bars(dict(zip(t["Company"], t["Weight %"])),
                             f"Position size (%) - dotted line = {MAX_STOCK_PCT:.0f}% limit", 330)
    bars.add_vline(x=MAX_STOCK_PCT, line_dash="dot", line_color=ui.DOWN)
    p2.plotly_chart(bars, width="stretch", config=CHART_CONFIG, key="pf_bars")

    st.download_button("⬇ Download portfolio review (CSV)", t.to_csv(index=False),
                       file_name="portfolio_review.csv", mime="text/csv")
    st.caption("Prices are the stored market snapshots (not live). Taxes, goals and time "
               "horizon are not considered. Educational - not investment advice.")
