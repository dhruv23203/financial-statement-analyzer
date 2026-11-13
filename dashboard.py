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
from pathlib import Path

import pandas as pd
import streamlit as st

from app.analysis.comparison import (amounts_frame, compare_companies, ratios_frame,
                                     summary_frame)
from app.analysis.ratios import RATIO_DEFINITIONS
from app.reporting import plots, tables, ui
from app.reporting.advisor_tabs import render_advisor_tab, render_portfolio_tab
from app.reporting.methodology_tab import render_methodology_tab
from app.services.context import AppContext, build_context
from app.services.pipeline import company_history, run_pipeline
from app.services.screener import (field_descriptions, find_peers, latest_analysis_id,
                                   run_screen, screen_frame)
from app.services.queries import (StoredAnalysis, key_financials, list_company_summaries,
                                  load_stored_analysis)
from app.analysis.config_loader import load_red_flag_config
from app.analysis.ratios import CALC_NOT_MEANINGFUL
from app.utils.exceptions import (AnalyzerError, DataNotFoundError, InvalidTickerError,
                                  ProviderNetworkError, ProviderRateLimitError)
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
              ("piotroski_f_score", "Piotroski F"), ("altman_z_score", "Altman Z''"),
              ("beneish_m_score", "Beneish M"))
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


@st.cache_resource(max_entries=64)
def cached_analysis(provider: str, db_path: str, ticker: str, marker: int) -> StoredAnalysis:
    """Stored analysis per company; the newest analysis id in the key refreshes it."""
    return load_stored_analysis(get_context(provider), ticker)


@st.cache_resource
def startup_marker(provider: str, db_path: str) -> int:
    """Newest analysis id when this server process started. Anything newer was added
    while the app was running (temporary storage on a hosted instance)."""
    return latest_analysis_id(get_context(provider))


@st.cache_data(show_spinner=False)
def scoring_settings(path: str) -> dict:
    from app.analysis.config_loader import load_scoring_config

    return load_scoring_config(Path(path))


@st.cache_data(show_spinner=False)
def red_flag_config(path: str) -> dict:
    return load_red_flag_config(Path(path))


@st.cache_data(show_spinner=False)
def red_flag_rule_names(path: str) -> dict[str, str]:
    return {code: rule.get("name", code) for code, rule in
            load_red_flag_config(Path(path))["rules"].items()}


def friendly_error(exc: Exception) -> str:
    """Plain-language message for the errors a user can cause or wait out."""
    if isinstance(exc, InvalidTickerError):
        return f"Ticker not accepted: {exc}"
    if isinstance(exc, ProviderRateLimitError):
        return ("Yahoo Finance is limiting requests right now (rate limit). Wait a few "
                "minutes and try again. Stored companies are still available.")
    if isinstance(exc, ProviderNetworkError):
        return ("Could not reach Yahoo Finance (network problem). Check the connection and "
                "try again; stored companies are still available.")
    if isinstance(exc, DataNotFoundError):
        return (f"No usable financial statements were found: {exc} Check the ticker "
                "(Indian stocks need .NS or .BO) - some listings have no annual statements "
                "on Yahoo Finance.")
    return str(exc)


METRIC_HELP = {
    "Market cap": "Share price x shares outstanding, in the trading currency, at the "
                  "snapshot date.",
    "P/E": "Price / earnings: market cap / latest annual net income. N/A for losses.",
    "P/B": "Price / book: market cap / shareholders' equity.",
    "EV / EBITDA": "Enterprise value (market cap + debt - cash) / EBITDA.",
    "Div. yield": "Dividends paid in the latest year / market cap.",
    "ROE": "Return on equity = net income / average shareholders' equity x 100. N/M when "
           "equity is zero or negative.",
    "ROCE": "Return on capital employed = EBIT / average (total assets - current "
            "liabilities) x 100.",
    "Net margin": "Net income / total revenue x 100.",
    "Debt / equity": "Total debt / shareholders' equity. Lower = less leverage. N/M when "
                     "equity is zero or negative.",
    "Current ratio": "Current assets / current liabilities. Below 1 = short-term "
                     "obligations exceed short-term assets.",
    "Interest cover": "EBIT / interest expense. How many times operating profit covers "
                      "interest. N/A when there is no interest expense (debt-free).",
    "Piotroski F": "Piotroski (2000) F-Score: 9 pass/fail signals of profitability, "
                   "leverage/liquidity and efficiency. 8-9 strong, 0-2 weak. Not our score.",
    "Altman Z''": "Altman Z''-Score (book-value, non-manufacturer version): > 2.60 safe, "
                  "1.10-2.60 grey, < 1.10 distress zone. A statistical warning sign, not a "
                  "default prediction. Not applicable to banks/insurers.",
    "Beneish M": "Beneish (1999) M-Score: above -1.78 resembles past earnings manipulators. "
                 "Many false positives (fast growth, acquisitions) - a screening signal, "
                 "never proof of manipulation. Not applicable to banks/insurers.",
    "Health score": "Our 0-100 Financial Health Score: weighted average of the category "
                    "scores; each ratio is scored by its percentile among sector peers.",
    "Data coverage": "Share of the score's weighted inputs that were available. Missing "
                     "data never earns points; it lowers coverage.",
    "Red flags (latest)": "Rules triggered in the latest fiscal year. See the Red flags "
                          "section for rules that could not be evaluated.",
    "Ratios calculated": "Ratios with a value across all years. The rest are N/A (missing "
                         "input) or N/M (mathematically not meaningful).",
}


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
st.sidebar.markdown(ui.brand(), unsafe_allow_html=True)
provider_label = st.sidebar.radio("Data source", list(PROVIDERS), index=0, horizontal=True,
                                  help="Offline sample data needs no internet "
                                       "(DEMOGOOD, DEMOWEAK, DEMOUSD, DEMOPARTIAL).")
PROVIDER = PROVIDERS[provider_label]
ctx = get_context(PROVIDER)

st.sidebar.markdown(ui.side_label("Analyse a company"), unsafe_allow_html=True)
with st.sidebar.form("analyse"):
    ticker_input = st.text_input("Ticker", placeholder="TCS.NS, INFY.NS, AAPL ...",
                                 label_visibility="collapsed")
    submitted = st.form_submit_button("Fetch & analyse", type="primary", width="stretch")
st.sidebar.caption("Indian stocks need .NS (NSE) or .BO (BSE). On the hosted demo, "
                   "companies you add are kept only until the app restarts (see Database).")

if submitted:
    with st.sidebar.status(f"Running pipeline for {ticker_input or '...'}",
                           expanded=True) as status:
        try:
            result = run_pipeline(ctx, ticker_input, progress=st.write)
        except AnalyzerError as exc:
            status.update(label="Failed", state="error")
            st.error(friendly_error(exc))
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
st.sidebar.markdown(ui.side_label("Watchlist"), unsafe_allow_html=True)
# A well-known, fully scored company makes a better first impression than the first
# ticker in alphabetical order (often a bank, which has no overall score).
PREFERRED_DEFAULTS = ("TCS.NS", "INFY.NS", "MSFT", "AAPL", "DEMOGOOD")
if analysed:
    default = st.session_state.get("selected") or next(
        (t for t in PREFERRED_DEFAULTS if t in analysed), None)
    selected = st.sidebar.selectbox(
        f"{len(analysed)} companies", analysed,
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

_tape_frame = cached_screen_frame(DB_KEY, MARKER)
if not _tape_frame.empty and "score" in _tape_frame:
    _top = _tape_frame.dropna(subset=["score"]).nlargest(30, "score")
    html(ui.ticker_tape([(t, float(v), ui.score_color(float(v)))
                         for t, v in zip(_top["ticker"], _top["score"])],
                        "Top financial health scores"))

(tab_company, tab_compare, tab_screen, tab_map, tab_advisor, tab_portfolio, tab_method,
 tab_history, tab_db) = st.tabs(
    ["📈 Company", "⚖️ Compare", "🔎 Screener", "🗺️ Market map", "💡 Quality & Value",
     "🩺 Portfolio", "📘 Methodology", "🕘 History", "🗄️ Database"])


# ---------------------------------------------------------------------------- company tab
def render_header(analysis: StoredAnalysis, prices: pd.DataFrame, range_label: str) -> None:
    market = analysis.summary.get("market") or {}
    latest = analysis.latest_period
    chips = [ui.chip(analysis.ticker, accent=True)]
    chips += [ui.chip(x) for x in (analysis.sector, analysis.industry, analysis.exchange) if x]
    chips += [ui.chip(f"Reports in {analysis.currency or '?'}")]
    if latest:
        chips.append(ui.chip(f"Latest {tables.fy_label(latest)} (year ended "
                             f"{latest:%d %b %Y})"))
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
                 note, ticker=analysis.ticker))
    render_quote_stats(analysis, market)


def render_quote_stats(analysis: StoredAnalysis, market: dict) -> None:
    """Quote statistics strip: 52-week range (live prices) and key market figures."""
    year = cached_prices(PROVIDER, analysis.ticker, "1y") if PROVIDER == "yfinance" \
        else pd.DataFrame()
    latest = {o.code: o for o in analysis.ratios if o.period == analysis.latest_period}

    def ratio_text(code: str) -> str:
        o = latest.get(code)
        return fmt_ratio(float(o.value), code) if o is not None and o.ok else "N/A"

    items, range_html = [], ""
    if not year.empty:
        close = year["Close"]
        high = float(year["High"].max()) if "High" in year else float(close.max())
        low = float(year["Low"].min()) if "Low" in year else float(close.min())
        range_html = ui.range_bar(low, high, float(close.iloc[-1]))
        if len(close) > 1:
            items.append(("Prev. close", f"{float(close.iloc[-2]):,.2f}"))
        if "Volume" in year and year["Volume"].tail(63).sum() > 0:
            items.append(("Avg. volume (3M)", ui.compact_number(float(
                year["Volume"].tail(63).mean()))))
    if market.get("market_cap"):
        items.append(("Market cap", ui.compact_number(float(market["market_cap"]),
                                                      market.get("currency"))))
    items += [("P/E", ratio_text("price_to_earnings")),
              ("Div. yield", ratio_text("dividend_yield"))]
    if items or range_html:
        html(ui.stats_strip(items, range_html))


def render_scorecard(analysis: StoredAnalysis) -> None:
    overall = analysis.overall()
    score = None if overall is None or overall.score is None else float(overall.score)
    cats = {c.category: c for c in analysis.latest_scores() if c.category != "OVERALL"}
    grades = {k: v for k, v in (analysis.summary.get("grades") or {}).items() if v}
    scored_cats = [c for c in cats.values() if c.score is not None]
    min_categories = int(scoring_settings(str(ctx.settings.scoring_config_file))[
        "min_categories_for_overall"])
    g_col, r_col, k_col = st.columns([1, 1.15, 1.25], gap="medium")
    with g_col:
        chart(plots.score_gauge(score, "Health score"), key="gauge")
        interpretation = overall.interpretation if overall and overall.interpretation else \
            ui.NA_TEXT
        html(f'<div style="text-align:center;margin-top:-.6rem">'
             f'{ui.pill(interpretation, ui.score_color(score))}</div>')
        if score is None and analysis.sector and "financial" in analysis.sector.lower():
            st.caption("Banks, insurers and other lenders are not scored: deposits and loans "
                       "are their operating assets and liabilities, so working-capital "
                       "liquidity, asset turnover and operating cash flow do not measure "
                       "their health. Bank-specific measures (capital adequacy, NPA ratio, "
                       "net interest margin) are planned for Semester 2.")
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
            ("Health score", "N/A" if score is None else f"{score:.1f}",
             interpretation if score is not None else "insufficient applicable data",
             ui.score_color(score), METRIC_HELP["Health score"]),
            ("Data coverage", f"{overall.coverage:.0%}" if overall else "N/A",
             None if score is not None else
             f"of applicable inputs; {len(scored_cats)} of {min_categories} required "
             "categories scored", None, METRIC_HELP["Data coverage"]),
            ("Red flags (latest)", str(len(flags_latest)),
             f"{len(analysis.flags)} across all years",
             ui.DOWN if flags_latest else None, METRIC_HELP["Red flags (latest)"]),
            ("Ratios calculated", f"{ok} / {len(analysis.ratios)}", None, None,
             METRIC_HELP["Ratios calculated"])]))
    if overall is not None:
        with st.expander("How the overall score was built"):
            st.write(overall.explanation)
            for c in cats.values():
                if c.score is None:
                    st.caption(f"{c.category.title()}: {c.explanation[:220]}")


def render_glance(analysis: StoredAnalysis) -> None:
    """One-line summary first (task-oriented hierarchy): score, best and weakest
    category, latest red flags and coverage."""
    overall = analysis.overall()
    score = None if overall is None or overall.score is None else float(overall.score)
    cats = [c for c in analysis.latest_scores() if c.category != "OVERALL" and c.score is not None]
    grades = analysis.summary.get("grades") or {}
    items = [("Health", f"{score:.1f} · {overall.interpretation.title()}" if score is not None
              and overall.interpretation else "N/A", ui.score_color(score))]
    if cats:
        best = max(cats, key=lambda c: float(c.score))
        worst = min(cats, key=lambda c: float(c.score))
        name = lambda c: c.category.replace("_", " ").title()  # noqa: E731
        items.append(("Strongest", f"{name(best)} {grades.get(best.category) or ''}".strip(),
                      ui.UP))
        if worst.category != best.category:
            items.append(("Weakest", f"{name(worst)} {grades.get(worst.category) or ''}".strip(),
                          ui.DOWN))
    flags = [f for f in analysis.flags if f.period == analysis.latest_period]
    items.append(("Red flags", f"{len(flags)} in {tables.fy_label(analysis.latest_period)}",
                  ui.DOWN if flags else ui.UP))
    if overall is not None:
        items.append(("Coverage", f"{overall.coverage:.0%}", None))
    html(ui.glance(items))


def render_key_numbers(analysis: StoredAnalysis) -> None:
    market = analysis.summary.get("market") or {}
    latest = {o.code: o for o in analysis.ratios if o.period == analysis.latest_period}

    def text(code: str) -> tuple[str, str | None]:
        """Formatted value, or N/A (missing input) / N/M (not meaningful) with the reason."""
        o = latest.get(code)
        if o is not None and o.ok:
            return fmt_ratio(float(o.value), code), None
        if o is not None and o.status == CALC_NOT_MEANINGFUL:
            return "N/M", "not applicable" if o.reason_code == "MODEL_NOT_APPLICABLE" \
                else "not meaningful"
        return "N/A", "data missing" if o is not None else None

    items = []
    if market.get("market_cap"):
        items.append(("Market cap", ui.compact_number(float(market["market_cap"]),
                                                      market.get("currency")),
                      market.get("currency"), None, METRIC_HELP["Market cap"]))
    for code, label in (("price_to_earnings", "P/E"), ("price_to_book", "P/B"),
                        ("ev_to_ebitda", "EV / EBITDA"), ("dividend_yield", "Div. yield")):
        if code in latest:
            value, sub = text(code)
            items.append((label, value, sub, None, METRIC_HELP[label]))
    for code, label in KEY_RATIOS:
        value, sub = text(code)
        items.append((label, value, sub, None, METRIC_HELP[label]))
    html(ui.section("Key numbers", "latest fiscal year · tap a card for its formula"))
    html(ui.cards(items))
    if market:
        st.caption(f"Latest fiscal year: {analysis.latest_period:%d %b %Y}. Market multiples "
                   f"use the {market.get('fetched_at')} price - descriptive, not a valuation "
                   "or recommendation. Hover a card for its formula.")
    with st.expander("About the benchmark scores (Piotroski, Altman, Beneish, Simply Wall St)"):
        st.markdown(
            "These are **published third-party models**, shown for reference and used to "
            "validate our score. They are **not** our Financial Health Score and do not "
            "feed into it.\n\n"
            "* **Piotroski F-Score (0-9)** - nine yes/no tests of profitability, leverage, "
            "liquidity and efficiency (Piotroski 2000). Designed for value stocks; a low "
            "score is a weakness signal, not a forecast.\n"
            "* **Altman Z''-Score** - book-value version for non-manufacturers (Altman, "
            "Hartzell & Peck 1995): > 2.60 safe, 1.10-2.60 grey, < 1.10 distress zone. Many "
            "healthy asset-light or highly leveraged-but-stable firms score low (false "
            "positives).\n"
            "* **Beneish M-Score** - above -1.78 the accounting profile resembles past "
            "earnings manipulators (Beneish 1999). Fast growth, acquisitions or restructuring "
            "also raise it. **A screening signal, never proof of fraud.**\n"
            "* **Simply Wall St checks (0-6)** - pass/fail balance-sheet tests from their "
            "open Company Analysis Model.\n\n"
            "**Applicability:** the Altman and Beneish models and the Simply Wall St checks "
            "were built for non-financial companies, so for banks, insurers and other "
            "financial companies they are shown as **N/M (not applicable)** instead of "
            "producing false distress or manipulation signals. With fewer than two "
            "consecutive fiscal years, models that need prior-year data are N/A.")


def render_flag_status(analysis: StoredAnalysis) -> None:
    """Triggered / not triggered / not evaluated for the latest year, so that
    'no flags' is never confused with 'rules could not be checked'."""
    latest = analysis.latest_period
    evaluations = [e for e in analysis.summary.get("red_flag_evaluations") or []
                   if latest is not None and e.get("period") == latest.isoformat()]
    if not evaluations:
        # Runs stored before per-rule statuses were recorded (analysis v2.2 and older).
        skipped = analysis.summary.get("rules_not_evaluated")
        st.info("Per-rule statuses were not recorded for this older analysis run, so it "
                "cannot show which rules were checked"
                + (f" ({skipped} rule checks across all years could not be evaluated)"
                   if skipped else "") + ". Re-analyse the company to see them.")
        if not analysis.flags:
            st.caption("No triggered flags are stored for this run.")
        return
    triggered = [e for e in evaluations if e["status"] == "TRIGGERED"]
    passed = [e for e in evaluations if e["status"] == "NOT_TRIGGERED"]
    skipped = [e for e in evaluations if e["status"] == "NOT_EVALUATED"]
    not_applicable = [e for e in skipped if (e.get("note") or "").startswith("not applicable")]
    applicable = len(evaluations) - len(not_applicable)
    evaluated = len(triggered) + len(passed)
    coverage = evaluated / applicable if applicable else 0.0
    html(ui.cards([
        ("Triggered", str(len(triggered)), tables.fy_label(latest),
         ui.DOWN if triggered else None, "Rules whose condition was met in the latest year."),
        ("Not triggered", str(len(passed)), "checked and clear", ui.UP if passed else None,
         "Rules evaluated with complete data whose condition was not met."),
        ("Not evaluated", str(len(skipped)),
         f"{len(not_applicable)} not applicable to sector" if not_applicable else
         "missing data", "#f9a825" if len(skipped) > len(not_applicable) else None,
         "Rules that could not be checked: inputs missing, no immediately preceding fiscal "
         "year, or not applicable to the company's sector."),
        ("Rule coverage", f"{coverage:.0%}", f"{evaluated} of {applicable} applicable rules",
         None, "Evaluated rules / rules applicable to this company in the latest year.")]))
    if not triggered and len(skipped) == len(not_applicable):
        st.success(f"All {evaluated} applicable rules were checked for "
                   f"{tables.fy_label(latest)}; none was triggered."
                   + (f" {len(not_applicable)} other rules do not apply to "
                      f"{analysis.sector} companies, so this covers fewer risks than for "
                      "other sectors." if not_applicable else ""))
    elif not triggered:
        st.warning(f"No rule was triggered, but {len(skipped) - len(not_applicable)} of "
                   f"{applicable} applicable rules could not be evaluated because data is "
                   "missing - this is an insufficient-data result, not a clean bill of "
                   "health.")
    names = red_flag_rule_names(str(ctx.settings.red_flag_config_file))
    with st.expander(f"Status of every rule for {tables.fy_label(latest)}"):
        show_table(pd.DataFrame([{"Rule": names.get(e["rule"], e["rule"]),
                                  "Code": e["rule"].split("_")[0],
                                  "Status": e["status"].replace("_", " ").title(),
                                  "Reason": e.get("note") or ""} for e in evaluations]))


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

    render_glance(analysis)
    html(ui.section("Financial health", "our 0-100 score, its five categories and grades"))
    render_scorecard(analysis)
    render_key_numbers(analysis)

    pros, cons = analysis.summary.get("pros") or [], analysis.summary.get("cons") or []
    if pros or cons:
        html(ui.section("Strengths & weaknesses", "generated by explicit rules"))
        p_col, c_col = st.columns(2)
        with p_col:
            html(ui.bullet_box("✅ Pros", pros, "pros"))
        with c_col:
            html(ui.bullet_box("⚠️ Cons", cons, "cons"))
        st.caption("Generated by explicit rules from the numbers below (peer percentiles, "
                   "trends, red flags, published scores).")

    html(ui.section("Financials", f"amounts in {scale_for(analysis.currency)[1]}"))
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

    html(ui.section(f"Red flags · {len(analysis.flags)} triggered (all years)",
                    "potential concerns, not proof of fraud or distress"))
    st.caption("Potential concerns that deserve a closer look - not proof of fraud or distress.")
    render_flag_status(analysis)
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
        html(ui.empty_state("📈", "No company selected",
                            "Analyse a company from the sidebar to see its results here."))
    else:
        try:
            render_company(cached_analysis(PROVIDER, DB_KEY, selected, MARKER))
        except AnalyzerError as exc:
            st.error(friendly_error(exc))


# ---------------------------------------------------------------------------- compare tab
with tab_compare:
    html(ui.section("Compare companies", "side-by-side ratios, scores and snowflakes"))
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
    if len(chosen) < 2:
        st.caption("Pick at least two companies (or a peer group) to enable Compare.")
    col_a, col_b, col_c = st.columns([1, 2, 1])
    use_year = col_a.checkbox("Specific fiscal year",
                              help="Off: each company's latest fiscal year. On: the fiscal "
                                   "year ending in the chosen calendar year (fiscal year "
                                   "ends can differ, e.g. March in India, December in the "
                                   "US).")
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
                            cached_analysis(PROVIDER, DB_KEY, t, MARKER).latest_scores()}
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
    html(ui.section("Stock screener", "filter all analysed companies"))
    st.caption("Filter every analysed company with conditions joined by AND. Short names: "
               "roe, roa, roce, npm, de, pe, pb, cr, qr, piotroski, altman, beneish; text fields: "
               "sector, industry, currency. Missing values never pass a condition.")
    examples = {
        "Quality at low leverage": "roe > 15 and de < 0.5 and piotroski >= 7",
        "Financially safe (Altman Z'' > 2.6, score > 60)": "altman > 2.6 and score > 60",
        "Indian IT top scorers": 'sector == "Technology" and currency == INR and score > 55',
        "Possible earnings-manipulation profile": "beneish > -1.78",
    }
    example = st.pills("Popular screens", list(examples), selection_mode="single",
                       key="screen_preset")
    query = st.text_input("Query", value=examples.get(example or "", "score > 60"),
                          help="Conditions joined by AND, e.g. roe > 15 and de < 0.5")
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
        html(ui.section("Market map", "every company, grouped by sector"))
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
        st.caption("Box = company, grouped by sector. Colours are the score's "
                   f"interpretation bands: {ui.band_legend()}; grey = not scored (e.g. "
                   "banks). Click a sector to zoom in; hover for details.")
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
MAX_DOWNLOAD_MB = 250  # larger files: use the server-side copy or the CLI

with tab_db:
    html(ui.section("Database"))
    db_file = ctx.settings.db_path
    size_mb = db_file.stat().st_size / 1_048_576 if db_file.exists() else 0.0
    st.caption(f"SQLite file `{db_file.name}` · {size_mb:,.1f} MB · schema revision "
               f"{ctx.db.current_revision()}")

    st.subheader("Data persistence")
    added = [c for c in companies if c["analysed"]
             and (c.get("analysis_run_id") or 0) > startup_marker(PROVIDER, DB_KEY)]
    preloaded = len(analysed) - len(added)
    p1, p2 = st.columns(2)
    p1.metric("Companies stored when the app started", preloaded,
              help="Records in the database file that shipped with the app.")
    p2.metric("Companies added or re-analysed since then", len(added),
              help="Fetched or analysed through the sidebar while this server has been "
                   "running.")
    st.caption(
        "Where the data lives: the app reads and writes one SQLite file on the server's own "
        "disk. Running locally, that disk is yours and everything persists. On Streamlit "
        "Community Cloud (the public demo) the disk is temporary: no persistent storage is "
        "configured, so companies added there are lost when the app restarts, sleeps or is "
        "redeployed, and the preloaded database is restored. Download a backup below to keep "
        "a copy.")

    st.subheader("Backup")
    st.caption("Creates a consistent copy with SQLite's online-backup API (safe while the app "
               "is running), verifies it, and offers it for download.")
    if st.button("Create backup"):
        try:
            backup_path = ctx.db.backup(label="dashboard")
        except AnalyzerError as exc:
            st.error(friendly_error(exc))
        else:
            st.session_state["backup_path"] = str(backup_path)
    backup_path = st.session_state.get("backup_path")
    if backup_path and Path(backup_path).is_file():
        backup_mb = Path(backup_path).stat().st_size / 1_048_576
        if backup_mb <= MAX_DOWNLOAD_MB:
            st.download_button(f"⬇ Download backup ({backup_mb:,.1f} MB)",
                               Path(backup_path).read_bytes(), file_name=Path(backup_path).name,
                               mime="application/vnd.sqlite3", type="primary")
        else:
            st.warning(f"The backup ({backup_mb:,.0f} MB) is too large to download through "
                       f"the browser (limit {MAX_DOWNLOAD_MB} MB). It is saved on the server "
                       f"as `{Path(backup_path).name}` in the backups folder.")
    st.caption("Restoring replaces the whole database, so it is deliberately not offered in "
               "the public web app (any visitor could overwrite the data). Restore locally "
               "with: `python main.py restore-db <backup file>` - the current database is "
               "backed up first and the file is verified before it is used.")

    st.subheader("Health check")
    # The full check scans the whole file (PRAGMA integrity_check), so it runs on demand.
    if st.button("Run full health check"):
        with st.spinner("Checking integrity and foreign keys ..."):
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

# ---------------------------------------------------------------------------- advisor tabs
# Added on top of the existing dashboard: they only read the stored analyses
# (see app/services/advisor.py and app/services/portfolio_doctor.py).
with tab_advisor:
    render_advisor_tab(ctx, selected, DB_KEY, MARKER, PROVIDER, names)

with tab_portfolio:
    render_portfolio_tab(ctx, DB_KEY, MARKER, PROVIDER)

with tab_method:
    render_methodology_tab(scoring_settings(str(ctx.settings.scoring_config_file)),
                           red_flag_config(str(ctx.settings.red_flag_config_file)))

html(ui.footer(f"v{scoring_settings(str(ctx.settings.scoring_config_file))['methodology_version']}"))
