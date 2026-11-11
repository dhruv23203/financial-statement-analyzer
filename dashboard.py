"""Web dashboard (Streamlit) for the Financial Statement Analyzer.

    streamlit run dashboard.py

A thin presentation layer over the same services as the CLI: it reads stored
results from the database (``app.services.queries``) and calls ``run_pipeline`` /
``compare_companies`` for new work. No financial logic lives here. The look (dark
theme, quote header, gauge, snowflake, market map) follows common trading-platform
conventions; see ``app/reporting/ui.py`` and ``app/reporting/plots.py``.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from app.analysis.comparison import (amounts_frame, compare_companies, ratios_frame,
                                     summary_frame)
from app.analysis.ratios import RATIO_DEFINITIONS
from app.reporting import plots, tables, ui
from app.services.context import AppContext, build_context
from app.services.pipeline import company_history, run_pipeline
from app.services.screener import (field_descriptions, find_peers, latest_analysis_id,
                                   run_screen, screen_frame)
from app.services.queries import (StoredAnalysis, key_financials, list_company_summaries,
                                  load_stored_analysis)
from app.utils.exceptions import AnalyzerError
from config.settings import MAX_COMPARISON_COMPANIES, PEER_GROUPS, get_settings

st.set_page_config(page_title="Financial Statement Analyzer", page_icon="📊", layout="wide")
st.markdown(ui.CSS, unsafe_allow_html=True)

PROVIDERS = {"Yahoo Finance (live)": "yfinance", "Offline sample data": "sample"}
METRIC_LABELS = {"total_revenue": "Revenue", "net_income": "Net income",
                 "operating_cash_flow": "Operating cash flow", "total_debt": "Total debt"}
CATEGORIES = ("PROFITABILITY", "CASH_FLOW", "EFFICIENCY", "SOLVENCY", "LIQUIDITY")
PRICE_RANGES = {"1M": "1mo", "6M": "6mo", "1Y": "1y", "5Y": "5y", "Max": "max"}
KEY_RATIOS = (("return_on_equity", "ROE"), ("return_on_capital_employed", "ROCE"),
              ("net_profit_margin", "Net margin"), ("debt_to_equity", "Debt / equity"),
              ("current_ratio", "Current ratio"), ("interest_coverage", "Interest cover"),
              ("piotroski_f_score", "Piotroski F"), ("altman_z_score", "Altman Z''"))
CHART_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}


@st.cache_resource
def _cached_context(provider: str, db_path: str) -> AppContext:
    return build_context(get_settings().with_overrides(provider=provider))


def get_context(provider: str) -> AppContext:
    """One shared context per (provider, database file)."""
    return _cached_context(provider, str(get_settings().db_path))


# Expensive reads are cached; the newest analysis id is part of the key, so any new
# analysis (from the sidebar, CLI or another process) refreshes them automatically.
@st.cache_data(show_spinner="Loading company data ...")
def cached_screen_frame(db_path: str, analysis_marker: int) -> pd.DataFrame:
    return screen_frame(get_context(PROVIDERS[provider_label]))


@st.cache_data(show_spinner=False)
def cached_summaries(db_path: str, analysis_marker: int) -> list[dict]:
    return list_company_summaries(get_context(PROVIDERS[provider_label]))


@st.cache_data(ttl=3600, show_spinner="Loading prices ...")
def cached_prices(provider: str, ticker: str, period: str) -> pd.DataFrame:
    """Live price history for the chart (display only, refreshed hourly)."""
    return get_context(provider).provider.fetch_price_history(ticker, period)


def scale_for(currency: str | None) -> tuple[float, str]:
    return (1e7, "INR crore") if currency == "INR" else (1e6, f"{currency or '?'} millions")


def show_table(frame: pd.DataFrame, **kwargs) -> None:
    if frame.empty:
        st.caption("No rows.")
    else:
        st.dataframe(frame, hide_index=True, width="stretch", **kwargs)


def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def chart(fig, key: str | None = None, **kwargs):
    return st.plotly_chart(fig, width="stretch", config=CHART_CONFIG, key=key, **kwargs)


def fmt_ratio(value: float | None, code: str) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    if code == "piotroski_f_score":
        return f"{value:.0f} / 9"
    unit = RATIO_DEFINITIONS[code].unit if code in RATIO_DEFINITIONS else ""
    return f"{value:,.1f}%" if unit == "%" else f"{value:,.2f}"


# ---------------------------------------------------------------------------- sidebar
st.sidebar.markdown('<div class="fsa-brand">📊 Financial Statement <span>Analyzer</span>'
                    '</div><div class="fsa-sub">NIFTY 500 + S&amp;P 500 · historical '
                    'financial health</div>', unsafe_allow_html=True)
provider_label = st.sidebar.radio("Data source", list(PROVIDERS), index=0, horizontal=True,
                                  help="Offline sample data needs no internet "
                                       "(DEMOGOOD, DEMOWEAK, DEMOUSD, DEMOPARTIAL).")
PROVIDER = PROVIDERS[provider_label]
ctx = get_context(PROVIDER)

with st.sidebar.form("analyse"):
    ticker_input = st.text_input("Analyse a company", placeholder="TCS.NS, INFY.NS, AAPL ...")
    submitted = st.form_submit_button("Fetch & analyse", type="primary", width="stretch")
st.sidebar.caption("Indian stocks need .NS (NSE) or .BO (BSE).")

if submitted:
    with st.sidebar.status(f"Running pipeline for {ticker_input or '...'}",
                           expanded=True) as status:
        try:
            result = run_pipeline(ctx, ticker_input, progress=st.write)
        except AnalyzerError as exc:
            status.update(label="Failed", state="error")
            st.error(str(exc))
        else:
            if result.succeeded:
                status.update(label=f"{result.ticker} analysed", state="complete")
                st.session_state["selected"] = result.ticker
            else:
                status.update(label="Stopped", state="error")
                st.error(result.error)

DB_KEY = str(ctx.settings.db_path)
MARKER = latest_analysis_id(ctx)
companies = cached_summaries(DB_KEY, MARKER)
names = {c["ticker"]: c["name"] or "" for c in companies}
analysed = [c["ticker"] for c in companies if c["analysed"]]
if analysed:
    default = st.session_state.get("selected")
    selected = st.sidebar.selectbox(
        f"Watchlist · {len(analysed)} companies", analysed,
        index=analysed.index(default) if default in analysed else 0,
        format_func=lambda t: f"{t} · {names.get(t, '')}"[:48])
    st.session_state["selected"] = selected
else:
    selected = None
    st.sidebar.info("No analysed companies yet. Enter a ticker above, or switch to "
                    "'Offline sample data' and analyse DEMOGOOD.")
st.sidebar.divider()
st.sidebar.caption("Educational analysis - not a credit rating or investment advice. "
                   "Data: Yahoo Finance.")

tab_company, tab_compare, tab_screen, tab_map, tab_history, tab_db = st.tabs(
    ["📈 Company", "⚖️ Compare", "🔎 Screener", "🗺️ Market map", "🕘 History",
     "🗄️ Database"])


# ---------------------------------------------------------------------------- company tab
def render_header(analysis: StoredAnalysis, prices: pd.DataFrame, range_label: str) -> None:
    market = analysis.summary.get("market") or {}
    latest = analysis.latest_period
    chips = [ui.chip(analysis.ticker, accent=True)]
    chips += [ui.chip(x) for x in (analysis.sector, analysis.industry, analysis.exchange) if x]
    chips += [ui.chip(f"Reports in {analysis.currency or '?'}")]
    if latest:
        chips.append(ui.chip(f"Latest {tables.fy_label(latest)}"))
    price = currency = None
    change = ""
    note = f"snapshot {market.get('fetched_at')}" if market else ""
    if not prices.empty:
        close = prices["Close"]
        price = f"{float(close.iloc[-1]):,.2f}"
        if len(close) > 1 and float(close.iloc[0]):
            change = ui.change_pill((float(close.iloc[-1]) / float(close.iloc[0]) - 1) * 100)
        note = f"{range_label} change · {prices.index[-1]:%d %b %Y}"
        currency = market.get("currency") or analysis.currency
    elif market.get("price"):
        price, currency = f"{float(market['price']):,.2f}", market.get("currency")
    html(ui.hero(analysis.company_name or analysis.ticker, chips, price, currency, change,
                 note))


def render_scorecard(analysis: StoredAnalysis) -> None:
    overall = analysis.overall()
    score = None if overall is None or overall.score is None else float(overall.score)
    cats = {c.category: c for c in analysis.latest_scores() if c.category != "OVERALL"}
    grades = {k: v for k, v in (analysis.summary.get("grades") or {}).items() if v}
    g_col, r_col, k_col = st.columns([1, 1.15, 1.25], gap="medium")
    with g_col:
        chart(plots.score_gauge(score, "Financial health score"), key="gauge")
        interpretation = overall.interpretation if overall and overall.interpretation else \
            "Not scored"
        html(f'<div style="text-align:center;margin-top:-.6rem">'
             f'{ui.pill(interpretation, ui.score_color(score))}</div>')
    with r_col:
        snowflake = {c: (None if cats.get(c) is None or cats[c].score is None
                         else float(cats[c].score)) for c in CATEGORIES}
        chart(plots.category_radar({analysis.ticker: snowflake}, height=300), key="radar")
    with k_col:
        if grades:
            html(ui.grade_badges(grades))
        flags_latest = [f for f in analysis.flags if f.period == analysis.latest_period]
        ok = sum(1 for o in analysis.ratios if o.ok)
        html(ui.cards([
            ("Health score", "N/A" if score is None else f"{score:.1f}", interpretation,
             ui.score_color(score)),
            ("Data coverage", f"{overall.coverage:.0%}" if overall else "N/A", None, None),
            ("Red flags (latest)", str(len(flags_latest)),
             f"{len(analysis.flags)} across all years",
             ui.DOWN if flags_latest else ui.UP),
            ("Ratios calculated", f"{ok} / {len(analysis.ratios)}", None, None)]))
    if overall is not None:
        with st.expander("How the overall score was built"):
            st.write(overall.explanation)
            for c in cats.values():
                if c.score is None:
                    st.caption(f"{c.category.title()}: {c.explanation[:220]}")


def render_key_numbers(analysis: StoredAnalysis) -> None:
    market = analysis.summary.get("market") or {}
    latest = {o.code: o for o in analysis.ratios if o.period == analysis.latest_period}

    def ratio(code: str) -> float | None:
        o = latest.get(code)
        return float(o.value) if o is not None and o.ok else None

    items = []
    if market.get("market_cap"):
        items.append(("Market cap", ui.compact_number(float(market["market_cap"]),
                                                      market.get("currency")), None, None))
    for code, label in (("price_to_earnings", "P/E"), ("price_to_book", "P/B"),
                        ("ev_to_ebitda", "EV / EBITDA"), ("dividend_yield", "Div. yield")):
        if code in latest:
            items.append((label, fmt_ratio(ratio(code), code), None, None))
    for code, label in KEY_RATIOS:
        value = ratio(code)
        items.append((label, fmt_ratio(value, code), None, None))
    html(ui.section("Key numbers"))
    html(ui.cards(items))
    if market:
        st.caption(f"Market multiples use the {market.get('fetched_at')} price - descriptive, "
                   "not a valuation or recommendation.")


def render_company(analysis: StoredAnalysis) -> None:
    check = analysis.currency_check
    live = PROVIDER == "yfinance"
    # widgets below set these keys; reading them first lets the header use the same range
    range_label = st.session_state.get("price_range") or "1Y"
    style = st.session_state.get("price_style") or "Line"
    prices = (cached_prices(PROVIDER, analysis.ticker, PRICE_RANGES[range_label]) if live
              else pd.DataFrame())
    render_header(analysis, prices, range_label)
    if check.get("status") in ("FAILED", "OVERRIDDEN"):
        (st.error if check["status"] == "FAILED" else st.info)(check.get("message", ""))
    if live:
        c1, c2 = st.columns([3, 2], gap="small")
        c1.segmented_control("Range", list(PRICE_RANGES), default="1Y", key="price_range",
                             label_visibility="collapsed")
        c2.segmented_control("Style", ["Line", "Candles"], default="Line", key="price_style",
                             label_visibility="collapsed")
        if prices.empty:
            st.caption("Live price chart unavailable right now (Yahoo Finance did not "
                       "respond).")
        else:
            chart(plots.price_chart(prices, style), key="price")

    render_scorecard(analysis)
    render_key_numbers(analysis)

    pros, cons = analysis.summary.get("pros") or [], analysis.summary.get("cons") or []
    if pros or cons:
        html(ui.section("Strengths & weaknesses"))
        p_col, c_col = st.columns(2)
        with p_col:
            html(ui.bullet_box("✅ Pros", pros, "pros"))
        with c_col:
            html(ui.bullet_box("⚠️ Cons", cons, "cons"))
        st.caption("Generated by explicit rules from the numbers below (peer percentiles, "
                   "trends, red flags, published scores).")

    html(ui.section("Financials"))
    t_fin, t_ratio, t_score, t_trend = st.tabs(
        ["Key financials", "Ratios", "Score history", "Trends"])
    with t_fin:
        factor, unit = scale_for(analysis.currency)
        data = key_financials(analysis)
        bars = {METRIC_LABELS[k]: {tables.fy_label(p): (None if v is None else float(v) / factor)
                                   for p, v in series.items()} for k, series in data.items()}
        chart(plots.grouped_bars(bars, unit), key="financials")
    with t_ratio:
        codes = sorted({o.code for o in analysis.ratios}, key=list(RATIO_DEFINITIONS).index)
        pick = st.multiselect("Plot ratios over time", codes,
                              default=[c for c in ("net_profit_margin", "return_on_assets",
                                                   "return_on_equity") if c in codes],
                              format_func=lambda c: RATIO_DEFINITIONS[c].name)
        if pick:
            lines = {RATIO_DEFINITIONS[c].name: {tables.fy_label(o.period): float(o.value)
                                                 for o in analysis.ratios
                                                 if o.code == c and o.ok} for c in pick}
            chart(plots.line_chart(lines), key="ratio_lines")
        wide, notes = tables.ratio_table(analysis.ratios)
        show_table(wide)
        st.caption(tables.legend())
        if notes:
            with st.expander(f"Why {len(notes)} value(s) are N/A or N/M"):
                for note in notes:
                    st.markdown(f"- {note}")
    with t_score:
        history = {}
        for category in ("OVERALL", *CATEGORIES):
            points = {tables.fy_label(p): float(c.score) for p, cs in analysis.scores.items()
                      for c in cs if c.category == category and c.score is not None}
            if points:
                history[category.replace("_", " ").title()] = points
        if history:
            chart(plots.line_chart(history, "Score (0-100)"), key="score_history")
        show_table(tables.score_frame(analysis.scores))
    with t_trend:
        show_table(tables.trend_summary_frame(analysis.trends, analysis.currency))
        with st.expander("Year-over-year detail"):
            show_table(tables.trend_frame(analysis.trends, analysis.currency))

    html(ui.section(f"Red flags · {len(analysis.flags)}"))
    st.caption("Potential concerns that deserve a closer look - not proof of fraud or distress.")
    if not analysis.flags:
        st.success("No red-flag rule was triggered.")
    recent = [f for f in analysis.flags if f.period == analysis.latest_period]
    older = [f for f in analysis.flags if f.period != analysis.latest_period]

    def flag_html(flags) -> str:
        return "".join(ui.flag_card(tables.fy_label(f.period), f.severity, f.title,
                                    f.triggered_condition, f.description) for f in flags)

    if recent:
        html(flag_html(recent))
    if older:
        with st.expander(f"{len(older)} flag(s) in earlier years", expanded=not recent):
            html(flag_html(older))

    notes = analysis.summary.get("notes") or []
    if notes:
        with st.expander("Notes on the data"):
            for note in notes:
                st.markdown(f"- {note}")
    st.download_button("⬇ Download ratio detail (CSV)",
                       tables.ratio_long_frame(analysis.ratios).to_csv(index=False),
                       file_name=f"{analysis.ticker}_ratios.csv", mime="text/csv")
    st.caption(f"Analysis run #{analysis.analysis_run_id} · "
               f"{analysis.executed_at:%Y-%m-%d %H:%M} UTC · scoring "
               f"v{analysis.scoring_version}")


with tab_company:
    if selected is None:
        st.info("Analyse a company from the sidebar to see its results here.")
    else:
        try:
            render_company(load_stored_analysis(ctx, selected))
        except AnalyzerError as exc:
            st.error(str(exc))


# ---------------------------------------------------------------------------- compare tab
with tab_compare:
    html(ui.section("Compare companies"))
    st.caption(f"Choose 2-{MAX_COMPARISON_COMPANIES} analysed companies. Ratios are compared "
               "across currencies; absolute amounts never are.")
    groups = {"(custom)": []} | {k: v for k, v in PEER_GROUPS.items()}
    g1, g2 = st.columns([2, 1])
    group = g1.selectbox("Peer group (optional)", list(groups))
    preset = [t for t in groups[group] if t in analysed]
    if group != "(custom)" and len(preset) < len(groups[group]):
        missing = [t for t in groups[group] if t not in analysed]
        st.caption(f"Not analysed yet (analyse them first): {', '.join(missing)}")
    g2.write("")
    if selected and g2.button(f"Find peers of {selected}", width="stretch"):
        try:
            st.session_state["peer_preset"] = [selected] + find_peers(
                ctx, selected, MAX_COMPARISON_COMPANIES - 1,
                frame=cached_screen_frame(DB_KEY, MARKER))
        except AnalyzerError as exc:
            st.error(str(exc))
    if st.session_state.get("peer_preset"):
        preset = [t for t in st.session_state["peer_preset"] if t in analysed]
    chosen = st.multiselect("Companies", analysed, default=preset,
                            max_selections=MAX_COMPARISON_COMPANIES,
                            format_func=lambda t: f"{t} · {names.get(t, '')}"[:40])
    col_a, col_b, col_c = st.columns([1, 2, 1])
    use_year = col_a.checkbox("Specific fiscal year")
    fiscal_year = col_b.number_input("Fiscal year ending in", 2000, 2100,
                                     date.today().year - 1, disabled=not use_year)
    col_c.write("")
    if col_c.button("Compare", type="primary", disabled=len(chosen) < 2, width="stretch"):
        try:
            st.session_state["comparison"] = compare_companies(
                ctx, chosen, int(fiscal_year) if use_year else None)
        except AnalyzerError as exc:
            st.session_state.pop("comparison", None)
            st.error(str(exc))
    report = st.session_state.get("comparison")
    if report is not None:
        st.markdown(f"**Comparison #{report.comparison_run_id}** · status `{report.status}` · "
                    f"currency `{report.currency_compatibility}` · periods "
                    f"`{report.period_alignment}`")
        for warning in report.warnings:
            st.warning(warning)
        tickers = [s.ticker for s in report.included]
        left, right = st.columns([1.2, 1])
        with left:
            snowflakes = {}
            for t in tickers:
                try:
                    cats = {c.category: c.score for c in
                            load_stored_analysis(ctx, t).latest_scores()}
                except AnalyzerError:
                    continue
                snowflakes[t] = {c: None if cats.get(c) is None else float(cats[c])
                                 for c in CATEGORIES}
            if snowflakes:
                chart(plots.category_radar(snowflakes, height=380), key="cmp_radar")
                st.caption("Category scores from each company's latest stored analysis.")
        with right:
            scores = {s.ticker: float(s.overall_score) for s in report.included
                      if s.overall_score is not None}
            if scores:
                chart(plots.ranked_bars(scores, "Overall health score", height=380,
                                        score_colors=True), key="cmp_scores")
        show_table(summary_frame(report))
        html(ui.section("Ratios (rank in brackets)"))
        show_table(ratios_frame(report))
        ratio_pick = st.selectbox("Chart a ratio", list(RATIO_DEFINITIONS)[:10],
                                  format_func=lambda c: RATIO_DEFINITIONS[c].name)
        values = {s.ticker: float(s.ratios[ratio_pick].value) for s in report.included
                  if ratio_pick in s.ratios and s.ratios[ratio_pick].value is not None}
        if values:
            chart(plots.ranked_bars(values, RATIO_DEFINITIONS[ratio_pick].name),
                  key="cmp_ratio")
        else:
            st.caption("This ratio is only shown for the 8 comparison ratios with data.")
        html(ui.section("Absolute amounts (own currency, millions)"))
        show_table(amounts_frame(report))


# ---------------------------------------------------------------------------- screener tab
SCREEN_COLUMNS = {
    "score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100,
                                             format="%.1f"),
    "ticker": st.column_config.TextColumn("Ticker", pinned=True),
    "market_cap": st.column_config.NumberColumn("Market cap", format="compact"),
    "revenue": st.column_config.NumberColumn("Revenue", format="compact"),
}

with tab_screen:
    html(ui.section("Stock screener"))
    st.caption("Filter every analysed company with conditions joined by AND. Short names: "
               "roe, roa, roce, npm, de, pe, pb, cr, qr, piotroski, altman, beneish; text fields: "
               "sector, industry, currency. Missing values never pass a condition.")
    examples = {
        "Quality at low leverage": "roe > 15 and de < 0.5 and piotroski >= 7",
        "Financially safe (Altman Z'' > 2.6, score > 60)": "altman > 2.6 and score > 60",
        "Indian IT top scorers": 'sector == "Technology" and currency == INR and score > 55',
        "Possible earnings-manipulation profile": "beneish > -1.78",
    }
    example = st.selectbox("Examples", ["(write your own)"] + list(examples))
    query = st.text_input("Query", value=examples.get(example, "score > 60"))
    s1, s2, s3 = st.columns([2, 1, 1])
    sort_by = s1.text_input("Sort by", value="score")
    limit = s2.number_input("Max rows", 5, 500, 50)
    s3.write("")
    if s3.button("Run screen", type="primary", width="stretch"):
        try:
            st.session_state["screen"] = run_screen(
                ctx, query, sort_by, False, int(limit),
                frame=cached_screen_frame(DB_KEY, MARKER))
        except AnalyzerError as exc:
            st.session_state.pop("screen", None)
            st.error(str(exc))
    if st.session_state.get("screen") is not None:
        result, total = st.session_state["screen"]
        st.success(f"{len(result)} shown of {total} companies screened")
        lead = [c for c in ("ticker", "name", "sector", "score") if c in result]
        ordered = result[lead + [c for c in result.columns if c not in lead]].round(2)
        show_table(ordered, column_config=SCREEN_COLUMNS, height=420)
        st.download_button("⬇ Download results (CSV)", result.to_csv(index=False),
                           file_name="screen_results.csv", mime="text/csv")
    with st.expander("All screenable fields"):
        frame_fields = cached_screen_frame(DB_KEY, MARKER)
        descriptions = field_descriptions()
        show_table(pd.DataFrame({"Field": sorted(frame_fields.columns),
                                 "Meaning": [descriptions.get(f, "") for f in
                                             sorted(frame_fields.columns)]}))


# ---------------------------------------------------------------------------- market map tab
COLOR_FIELDS = {"score": "Overall score", "profitability_score": "Profitability",
                "cash_flow_score": "Cash flow", "solvency_score": "Solvency",
                "efficiency_score": "Efficiency", "liquidity_score": "Liquidity"}
SCATTER_FIELDS = {"price_to_earnings": "P/E", "return_on_equity": "ROE %",
                  "return_on_capital_employed": "ROCE %", "net_profit_margin": "Net margin %",
                  "debt_to_equity": "Debt / equity", "price_to_book": "P/B",
                  "ev_to_ebitda": "EV / EBITDA", "free_cash_flow_margin": "FCF margin %",
                  "piotroski_f_score": "Piotroski F", "score": "Overall score"}

with tab_map:
    frame = cached_screen_frame(DB_KEY, MARKER)
    if frame.empty:
        st.info("No analysed companies yet.")
    else:
        html(ui.section("Market map"))
        markets = {"India (INR)": "INR", "United States (USD)": "USD", "All": None}
        m1, m2, m3 = st.columns(3)
        market_pick = m1.selectbox("Market", list(markets))
        color_by = m2.selectbox("Colour by", list(COLOR_FIELDS), format_func=COLOR_FIELDS.get)
        size_options = ["market_cap", "revenue"] if "market_cap" in frame else ["revenue"]
        size_by = m3.selectbox("Box size", size_options,
                               format_func=lambda s: s.replace("_", " ").title())
        currency = markets[market_pick]
        view = frame if currency is None else frame[frame["currency"] == currency]
        if currency is None:
            st.caption("'All' mixes currencies, so box sizes are only comparable within a "
                       "market. Pick India or United States for a true size comparison.")
        st.caption("Box = company, grouped by sector. Colour bands match the score gauge: "
                   "teal 70+, green 55-70, amber 40-55, orange 25-40, red below 25, grey = "
                   "not scored (e.g. banks). Click a sector to zoom in; hover for details.")
        if size_by in view and view[size_by].notna().any() and color_by in view:
            chart(plots.market_treemap(view, size=size_by, color=color_by), key="treemap")

        html(ui.section("Explore: every company as a dot"))
        x1, x2 = st.columns(2)
        x_field = x1.selectbox("X axis", list(SCATTER_FIELDS), index=0,
                               format_func=SCATTER_FIELDS.get)
        y_field = x2.selectbox("Y axis", list(SCATTER_FIELDS), index=1,
                               format_func=SCATTER_FIELDS.get)
        if {x_field, y_field, "score"} <= set(view.columns):
            event = chart(plots.scatter_map(view, x_field, y_field, SCATTER_FIELDS[x_field],
                                            SCATTER_FIELDS[y_field]),
                          key="scatter", on_select="rerun", selection_mode="points")
            st.caption("Colour = overall score. Extreme 2% outliers are trimmed. Click a dot "
                       "to open that company in the Company tab.")
            points = (event or {}).get("selection", {}).get("points", []) if event else []
            if points:
                custom = points[0].get("customdata") or []
                clicked = custom[2] if len(custom) > 2 else points[0].get("text")
                if clicked in analysed and clicked != st.session_state.get("selected"):
                    st.session_state["selected"] = clicked
                    st.toast(f"Opened {clicked} - see the Company tab")
                    st.rerun()

        html(ui.section("Sector overview"))
        sectors = (view.groupby(view["sector"].fillna("Other"))
                   .agg(Companies=("ticker", "count"), Median_score=("score", "median"),
                        Median_ROE=("return_on_equity", "median"),
                        Median_net_margin=("net_profit_margin", "median"),
                        Median_PE=("price_to_earnings", "median"))
                   .round(1).sort_values("Median_score", ascending=False).reset_index())
        show_table(sectors, column_config={
            "Median_score": st.column_config.ProgressColumn("Median score", min_value=0,
                                                            max_value=100, format="%.1f")})


# ---------------------------------------------------------------------------- history tab
with tab_history:
    html(ui.section("Run history"))
    if selected is None:
        st.info("No company selected.")
    else:
        history = company_history(ctx, selected)
        for title, key in (("Fetch runs", "fetch_runs"), ("Cleaning runs", "cleaning_runs"),
                           ("Analysis runs", "analysis_runs"),
                           ("Comparison runs", "comparison_runs")):
            st.subheader(title)
            show_table(pd.DataFrame(history[key]).astype(str) if history[key]
                       else pd.DataFrame())


# ---------------------------------------------------------------------------- database tab
with tab_db:
    html(ui.section("Database"))
    st.caption(f"{ctx.settings.db_path} · schema revision {ctx.db.current_revision()}")
    # The full check scans the whole file (PRAGMA integrity_check), so it runs on demand.
    if st.button("Run full health check"):
        health = ctx.db.health_check()
        (st.success if health.ok else st.error)(
            f"{'Healthy' if health.ok else 'Problems found'} · schema revision "
            f"{health.current_revision} · integrity {health.integrity} · "
            f"{health.foreign_key_violations} foreign-key violations")
        for problem in health.problems:
            st.write(f"- {problem}")
        show_table(pd.DataFrame({"Table": list(health.table_counts),
                                 "Rows": list(health.table_counts.values())}))
    st.subheader("All companies")
    show_table(pd.DataFrame(companies))
    if st.button("Create backup"):
        st.success(f"Backup created: {ctx.db.backup(label='dashboard')}")
