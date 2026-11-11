"""Web dashboard (Streamlit) for the Financial Statement Analyzer.

    streamlit run dashboard.py

A thin presentation layer over the same services as the CLI: it reads stored
results from the database (``app.services.queries``) and calls ``run_pipeline`` /
``compare_companies`` for new work. No financial logic lives here.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from app.analysis.comparison import (amounts_frame, compare_companies, ratios_frame,
                                     summary_frame)
from app.analysis.ratios import RATIO_DEFINITIONS
from app.reporting import tables
from app.services.context import AppContext, build_context
from app.services.pipeline import company_history, run_pipeline
from app.services.screener import (field_descriptions, find_peers, latest_analysis_id,
                                   run_screen, screen_frame)
from app.services.queries import (StoredAnalysis, key_financials, list_company_summaries,
                                  load_stored_analysis)
from app.utils.exceptions import AnalyzerError
from config.settings import MAX_COMPARISON_COMPANIES, PEER_GROUPS, get_settings

st.set_page_config(page_title="Financial Statement Analyzer", page_icon="📊", layout="wide")

PROVIDERS = {"Yahoo Finance (live)": "yfinance", "Offline sample data": "sample"}
SEVERITY_STYLE = {"HIGH": st.error, "MEDIUM": st.warning, "LOW": st.info}
METRIC_LABELS = {"total_revenue": "Revenue", "net_income": "Net income",
                 "operating_cash_flow": "Operating cash flow", "total_debt": "Total debt"}


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


def scale_for(currency: str | None) -> tuple[float, str]:
    return (1e7, "INR crore") if currency == "INR" else (1e6, f"{currency or '?'} millions")


def show_table(frame: pd.DataFrame) -> None:
    if frame.empty:
        st.caption("No rows.")
    else:
        st.dataframe(frame, hide_index=True, width="stretch")


# ---------------------------------------------------------------------------- sidebar
st.sidebar.title("📊 Financial Statement Analyzer")
st.sidebar.caption("Semester 1 - historical financial health analysis")
provider_label = st.sidebar.radio("Data source", list(PROVIDERS), index=0,
                                  help="Offline sample data needs no internet "
                                       "(DEMOGOOD, DEMOWEAK, DEMOUSD, DEMOPARTIAL).")
ctx = get_context(PROVIDERS[provider_label])

st.sidebar.subheader("Analyse a company")
with st.sidebar.form("analyse"):
    ticker_input = st.text_input("Ticker", placeholder="TCS.NS, INFY.NS, AAPL ...")
    submitted = st.form_submit_button("Fetch & analyse", type="primary",
                                      width="stretch")
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
analysed = [c["ticker"] for c in companies if c["analysed"]]
st.sidebar.subheader("Stored companies")
if analysed:
    default = st.session_state.get("selected")
    selected = st.sidebar.selectbox(
        "Company", analysed, index=analysed.index(default) if default in analysed else 0,
        format_func=lambda t: next(f"{t} - {c['name'] or ''}"[:45] for c in companies
                                   if c["ticker"] == t))
    st.session_state["selected"] = selected
else:
    selected = None
    st.sidebar.info("No analysed companies yet. Enter a ticker above, or switch to "
                    "'Offline sample data' and analyse DEMOGOOD.")
st.sidebar.divider()
st.sidebar.caption("Educational analysis - not a credit rating or investment advice.")

tab_company, tab_compare, tab_screen, tab_history, tab_db = st.tabs(
    ["Company analysis", "Compare companies", "Screener", "History", "Database"])


# ---------------------------------------------------------------------------- company tab
def render_company(analysis: StoredAnalysis) -> None:
    st.header(f"{analysis.company_name or analysis.ticker}")
    st.caption(f"{analysis.ticker} · {analysis.sector or 'sector n/a'} · "
               f"{analysis.industry or ''} · reporting currency {analysis.currency or '?'} · "
               f"analysis run #{analysis.analysis_run_id} "
               f"({analysis.executed_at:%Y-%m-%d %H:%M} UTC) · scoring v{analysis.scoring_version}")
    check = analysis.currency_check
    if check.get("status") in ("FAILED", "OVERRIDDEN"):
        (st.error if check["status"] == "FAILED" else st.info)(check.get("message", ""))

    overall = analysis.overall()
    latest = analysis.latest_period
    flags_latest = [f for f in analysis.flags if f.period == latest]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Financial health score",
              "N/A" if overall is None or overall.score is None else f"{float(overall.score):.1f}",
              overall.interpretation if overall and overall.interpretation else None,
              delta_color="off")
    c2.metric("Data coverage", f"{overall.coverage:.0%}" if overall else "N/A")
    c3.metric(f"Red flags ({tables.fy_label(latest)})" if latest else "Red flags",
              len(flags_latest), f"{len(analysis.flags)} across all years", delta_color="off")
    ok = sum(1 for o in analysis.ratios if o.ok)
    c4.metric("Ratios calculated", f"{ok} / {len(analysis.ratios)}")
    if overall is not None:
        with st.expander("How the overall score was built"):
            st.write(overall.explanation)

    market = analysis.summary.get("market")
    valuation = {o.code: o for o in analysis.ratios
                 if o.definition.category == "VALUATION" and o.period == latest}
    if market:
        st.caption(f"Market snapshot as of {market.get('fetched_at')} - descriptive multiples, "
                   "not a valuation or recommendation")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Market cap", f"{market.get('currency', '')} "
                                f"{float(market['market_cap']) / 1e9:,.1f}bn")
        for column, code in ((m2, "price_to_earnings"), (m3, "price_to_book"),
                             (m4, "ev_to_ebitda"), (m5, "dividend_yield")):
            o = valuation.get(code)
            column.metric(RATIO_DEFINITIONS[code].name,
                          tables.format_ratio_value(o) if o else "N/A")

    grades = {k: v for k, v in (analysis.summary.get("grades") or {}).items() if v}
    if grades:
        st.markdown("**Grades** (A = top 10% ... F = bottom 10% of NIFTY 500 + S&P 500): " +
                    " · ".join(f"{k.title().replace('_', ' ')} **{v}**"
                               for k, v in grades.items()))
    pros, cons = analysis.summary.get("pros") or [], analysis.summary.get("cons") or []
    if pros or cons:
        p_col, c_col = st.columns(2)
        with p_col:
            st.subheader("Pros")
            for item in pros or ["None stood out."]:
                st.markdown(f"- :green[{item}]")
        with c_col:
            st.subheader("Cons")
            for item in cons or ["None stood out."]:
                st.markdown(f"- :red[{item}]")
        st.caption("Generated by explicit rules from the numbers below (peer percentiles, "
                   "trends, red flags, published scores).")

    left, right = st.columns([1, 1.3])
    with left:
        st.subheader("Category scores (latest year)")
        cats = [c for c in analysis.latest_scores() if c.category != "OVERALL"]
        frame = pd.DataFrame({"Category": [c.category.title().replace("_", " ") for c in cats],
                              "Score": [None if c.score is None else float(c.score)
                                        for c in cats]}).set_index("Category")
        st.bar_chart(frame, horizontal=True, height=260)
        for c in cats:
            if c.score is None:
                st.caption(f"{c.category.title()}: {c.explanation[:220]}")
    with right:
        st.subheader("Key financials")
        factor, unit = scale_for(analysis.currency)
        data = key_financials(analysis)
        chart = pd.DataFrame({METRIC_LABELS[k]: {str(p): (None if v is None else float(v) / factor)
                                                 for p, v in series.items()}
                              for k, series in data.items()})
        st.bar_chart(chart, stack=False, height=260, y_label=unit)

    st.subheader("Financial health score by year")
    show_table(tables.score_frame(analysis.scores))

    st.subheader("Financial ratios")
    wide, notes = tables.ratio_table(analysis.ratios)
    show_table(wide)
    st.caption(tables.legend())
    if notes:
        with st.expander(f"Why {len(notes)} value(s) are N/A or N/M"):
            for note in notes:
                st.markdown(f"- {note}")
    codes = sorted({o.code for o in analysis.ratios}, key=list(RATIO_DEFINITIONS).index)
    pick = st.multiselect("Plot ratios over time", codes,
                          default=["net_profit_margin", "return_on_assets",
                                   "return_on_equity"],
                          format_func=lambda c: RATIO_DEFINITIONS[c].name)
    if pick:
        lines = pd.DataFrame({RATIO_DEFINITIONS[c].name: {
            str(o.period): float(o.value) for o in analysis.ratios if o.code == c and o.ok}
            for c in pick}).sort_index()
        st.line_chart(lines, height=280)

    st.subheader("Trends")
    show_table(tables.trend_summary_frame(analysis.trends, analysis.currency))
    with st.expander("Year-over-year detail"):
        show_table(tables.trend_frame(analysis.trends, analysis.currency))

    st.subheader("Red flags")
    st.caption("Potential concerns that deserve a closer look - not proof of fraud or distress.")
    if not analysis.flags:
        st.success("No red-flag rule was triggered.")
    for flag in analysis.flags:
        SEVERITY_STYLE[flag.severity](
            f"**{tables.fy_label(flag.period)} · {flag.severity} · {flag.title}**  \n"
            f"{flag.triggered_condition}  \n{flag.description}")

    notes = analysis.summary.get("notes") or []
    if notes:
        st.subheader("Notes")
        for note in notes:
            st.markdown(f"- {note}")
    st.download_button("Download ratio detail (CSV)",
                       tables.ratio_long_frame(analysis.ratios).to_csv(index=False),
                       file_name=f"{analysis.ticker}_ratios.csv", mime="text/csv")


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
    st.header("Compare companies")
    st.caption(f"Choose 2-{MAX_COMPARISON_COMPANIES} analysed companies. Ratios are compared "
               "across currencies; absolute amounts never are.")
    groups = {"(custom)": []} | {k: v for k, v in PEER_GROUPS.items()}
    group = st.selectbox("Peer group (optional)", list(groups))
    preset = [t for t in groups[group] if t in analysed]
    if group != "(custom)" and len(preset) < len(groups[group]):
        missing = [t for t in groups[group] if t not in analysed]
        st.caption(f"Not analysed yet (analyse them first): {', '.join(missing)}")
    if selected and st.button(f"Find peers of {selected}"):
        try:
            st.session_state["peer_preset"] = [selected] + find_peers(
                ctx, selected, MAX_COMPARISON_COMPANIES - 1,
                frame=cached_screen_frame(DB_KEY, MARKER))
        except AnalyzerError as exc:
            st.error(str(exc))
    if st.session_state.get("peer_preset"):
        preset = [t for t in st.session_state["peer_preset"] if t in analysed]
    chosen = st.multiselect("Companies", analysed, default=preset,
                            max_selections=MAX_COMPARISON_COMPANIES)
    col_a, col_b = st.columns([1, 3])
    use_year = col_a.checkbox("Specific fiscal year")
    fiscal_year = col_b.number_input("Fiscal year ending in", 2000, 2100,
                                     date.today().year - 1, disabled=not use_year)
    if st.button("Compare", type="primary", disabled=len(chosen) < 2):
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
        show_table(summary_frame(report))
        scores = pd.DataFrame({"Score": {s.ticker: float(s.overall_score)
                                         for s in report.included
                                         if s.overall_score is not None}})
        if not scores.empty:
            st.bar_chart(scores, height=240)
        st.subheader("Ratios (rank in brackets)")
        show_table(ratios_frame(report))
        ratio_pick = st.selectbox("Chart a ratio", list(RATIO_DEFINITIONS)[:10],
                                  format_func=lambda c: RATIO_DEFINITIONS[c].name)
        values = {s.ticker: float(s.ratios[ratio_pick].value) for s in report.included
                  if ratio_pick in s.ratios and s.ratios[ratio_pick].value is not None}
        if values:
            st.bar_chart(pd.DataFrame({RATIO_DEFINITIONS[ratio_pick].name: values}), height=240)
        else:
            st.caption("This ratio is only shown for the 8 comparison ratios with data.")
        st.subheader("Absolute amounts (own currency, millions)")
        show_table(amounts_frame(report))


# ---------------------------------------------------------------------------- screener tab
with tab_screen:
    st.header("Stock screener")
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
    s1, s2 = st.columns(2)
    sort_by = s1.text_input("Sort by", value="score")
    limit = s2.number_input("Max rows", 5, 500, 50)
    if st.button("Run screen", type="primary"):
        try:
            result, total = run_screen(ctx, query, sort_by, False, int(limit),
                                       frame=cached_screen_frame(DB_KEY, MARKER))
            st.success(f"{len(result)} shown of {total} companies screened")
            show_table(result.round(2))
            st.download_button("Download results (CSV)", result.to_csv(index=False),
                               file_name="screen_results.csv", mime="text/csv")
        except AnalyzerError as exc:
            st.error(str(exc))
    with st.expander("All screenable fields"):
        frame_fields = cached_screen_frame(DB_KEY, MARKER)
        descriptions = field_descriptions()
        show_table(pd.DataFrame({"Field": sorted(frame_fields.columns),
                                 "Meaning": [descriptions.get(f, "") for f in
                                             sorted(frame_fields.columns)]}))


# ---------------------------------------------------------------------------- history tab
with tab_history:
    st.header("Run history")
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
    st.header("Database")
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
