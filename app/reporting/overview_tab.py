"""Overview: the first page - what the app does, quick start, the selected company at a
glance, where to go next, what the stored universe looks like, and a glossary.

Every number comes from the stored analyses (``summaries``) or the stored market
snapshots (``freshness``); the chart data are produced by the small pure functions below
so they can be tested."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.reporting import glossary, plots, ui

QUICK_PICKS = ("TCS.NS", "INFY.NS", "RELIANCE.NS", "HDFCBANK.NS", "AAPL", "MSFT", "NVDA",
               "DEMOGOOD")
PAGES = [
    ("📈", "Company", "Health", "Is this company financially healthy, and are there "
     "warning signs?"),
    ("💡", "Quality & Value", "Valuation", "Is the price reasonable? DCF, sensitivity, "
     "scenarios, reverse DCF."),
    ("⚖️", "Compare", "Health", "How does it stack up against chosen peers?"),
    ("🔎", "Screener", "Data", "Which companies meet my conditions?"),
    ("🗺️", "Market map", "Data", "How do sectors and companies look side by side?"),
    ("🩺", "Portfolio", "Risk", "Where is my portfolio concentrated or weak?"),
    ("🤖", "ML prediction", "Risk", "Could a healthy company slip into losses next year?"),
    ("📘", "Methodology", "Data", "How is every number calculated - and how reliable is it?"),
]
AREA_KIND = {"Health": "health", "Valuation": "valuation", "Risk": "risk", "Data": "data"}
CSS = """<style>
.fsa-hero2 {padding: 1.2rem 1.4rem; border-radius: 14px; margin: .2rem 0 .8rem 0;
  background: linear-gradient(135deg, rgba(41,98,255,.16), rgba(38,166,154,.08));
  border: 1px solid var(--fsa-border);}
.fsa-hero2 h2 {margin: 0 0 .3rem 0; font-size: 1.45rem; letter-spacing: -.02em;}
.fsa-hero2 p {margin: 0; color: #b8c2cf; font-size: .92rem; line-height: 1.5; max-width: 60rem;}
.fsa-hero2 .facts {display: flex; flex-wrap: wrap; gap: .45rem; margin-top: .75rem;}
.fsa-hero2 .facts span {font-size: .78rem; padding: .22rem .65rem; border-radius: 999px;
  background: rgba(255,255,255,.04); border: 1px solid var(--fsa-border); color: #cfd8e3;}
.fsa-tiles {display: grid; gap: .6rem; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
  margin: .2rem 0 .6rem 0;}
.fsa-tile {padding: .8rem .95rem; border-radius: 12px; background: var(--fsa-surface);
  border: 1px solid var(--fsa-border); border-left: 3px solid #8b97a8;}
.fsa-tile.k-health {border-left-color: #42a5f5;} .fsa-tile.k-valuation {border-left-color: #ab47bc;}
.fsa-tile.k-risk {border-left-color: #ef5350;} .fsa-tile.k-data {border-left-color: #8b97a8;}
.fsa-tile .t {font-weight: 700; font-size: .95rem;}
.fsa-tile .q {color: #b8c2cf; font-size: .82rem; margin-top: .25rem; line-height: 1.4;}
</style>"""


def _html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def _pick_company() -> None:
    """Quick-start button: make it the sidebar selection (acts only on a click)."""
    choice = st.session_state.get("overview_pick")
    if choice:
        st.session_state["selected"] = choice
    st.session_state["overview_pick"] = None


# ----------------------------------------------------------------------------- data (pure)
def sample_tickers() -> set[str]:
    """Tickers of the synthetic offline sample companies (sample_data/<TICKER>/)."""
    from config.settings import get_settings
    base = Path(get_settings().sample_data_dir)
    return {d.name.upper() for d in base.iterdir() if d.is_dir()} if base.is_dir() else set()


def real_universe(summaries: list[dict[str, Any]],
                  synthetic: set[str]) -> tuple[list[dict[str, Any]], int]:
    """Summaries without the synthetic demo companies - unless they are all there is
    (offline sample mode). Returns (summaries, number excluded)."""
    real = [s for s in summaries if str(s.get("ticker", "")).upper() not in synthetic]
    if not any(s.get("analysed") for s in real):
        return summaries, 0
    return real, sum(1 for s in summaries if s.get("analysed")) -         sum(1 for s in real if s.get("analysed"))


def _frame(summaries: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame([s for s in summaries if s.get("analysed")])
    if df.empty:
        return df
    for col in ("score", "red_flags_latest"):
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
    df["market"] = df["currency"].map({"INR": "India (INR)", "USD": "US (USD)"}).fillna(
        "Other")
    return df


def snapshot(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    analysed = [s for s in summaries if s.get("analysed")]
    scores = [float(s["score"]) for s in analysed if s.get("score") is not None]
    years = sorted(str(s["latest_fy"])[:10] for s in analysed if s.get("latest_fy"))
    return {
        "companies": len(analysed),
        "scored": len(scores),
        "median_score": float(pd.Series(scores).median()) if scores else None,
        "with_flags": sum(1 for s in analysed if (s.get("red_flags_latest") or 0) > 0),
        "latest_fy": years[-1] if years else None,
        "markets": sorted({s.get("currency") for s in analysed if s.get("currency")}),
    }


def score_distribution(summaries: list[dict[str, Any]],
                       bands: list[dict[str, Any]]) -> pd.DataFrame:
    """Scored companies per 10-point bucket and band label (for the histogram)."""
    df = _frame(summaries)
    if df.empty or df["score"].dropna().empty:
        return pd.DataFrame(columns=["bucket", "band", "companies"])
    scored = df.dropna(subset=["score"]).copy()
    scored["bucket"] = (scored["score"].clip(0, 99.999) // 10 * 10).astype(int)
    ordered = sorted(bands, key=lambda b: -float(b["min"]))

    def band(v):
        return next((b["label"] for b in ordered if v >= float(b["min"])), "n/a")

    scored["band"] = scored["score"].map(band)
    return (scored.groupby(["bucket", "band"]).size().reset_index(name="companies")
            .sort_values("bucket"))


def sector_medians(summaries: list[dict[str, Any]], min_companies: int = 5) -> pd.DataFrame:
    """Median health score and company count per sector (sectors with enough members)."""
    df = _frame(summaries)
    if df.empty:
        return pd.DataFrame(columns=["sector", "median", "companies"])
    g = df.dropna(subset=["score"]).groupby("sector")["score"].agg(["median", "count"])
    g = g[g["count"] >= min_companies].reset_index().rename(columns={"count": "companies"})
    return g.sort_values("median", ascending=False)


def market_split(summaries: list[dict[str, Any]]) -> pd.DataFrame:
    df = _frame(summaries)
    if df.empty:
        return pd.DataFrame(columns=["market", "companies", "median", "with_flags"])
    g = df.groupby("market").agg(companies=("ticker", "count"),
                                 median=("score", "median"),
                                 with_flags=("red_flags_latest", lambda s: int((s > 0).sum())))
    return g.reset_index()


# ----------------------------------------------------------------------------- charts
BAND_ORDER = ("BOTTOM QUARTILE", "BELOW MEDIAN", "ABOVE MEDIAN", "TOP QUARTILE")


def distribution_figure(dist: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    colors = {label: c for _, label, c in ui.score_bands()}
    for band in BAND_ORDER:
        part = dist[dist["band"] == band]
        if part.empty:
            continue
        fig.add_bar(x=[f"{b}-{b + 9}" for b in part["bucket"]], y=part["companies"],
                    name=band.title(), marker_color=colors.get(band, ui.MUTED),
                    hovertemplate="%{x}: %{y} companies<extra>" + band.title() + "</extra>")
    fig.update_layout(barmode="stack", xaxis_title="Health score", yaxis_title="Companies")
    fig.update_xaxes(type="category", categoryorder="array",
                     categoryarray=[f"{b}-{b + 9}" for b in range(0, 100, 10)])
    return plots._style(fig, 300)


# ----------------------------------------------------------------------------- render
def render(summaries: list[dict[str, Any]], selected: str | None,
           freshness: dict[str, Any] | None = None, bands: list[dict] | None = None) -> None:
    _html(CSS)
    universe, n_demo = real_universe(summaries, sample_tickers())
    snap = snapshot(universe)
    facts = [f"{snap['companies']:,} companies analysed",
             f"markets: {', '.join(snap['markets']) or 'none yet'}",
             f"newest statements: FY to {snap['latest_fy'] or 'n/a'}"]
    if freshness and freshness.get("latest_price_date"):
        facts.append(f"prices as of {freshness['latest_price_date']}")
    _html('<div class="fsa-hero2"><h2>Financial research workspace</h2>'
          "<p>From published annual statements to ratios, trends, red flags, an "
          "industry-aware 0-100 Financial Health Score, DCF valuation and an experimental ML "
          "risk model - every number traceable to its formula and source. It describes "
          "companies; it does not tell you what to buy or sell.</p>"
          '<div class="facts">' + "".join(f"<span>{escape(f)}</span>" for f in facts)
          + "</div></div>")

    df = _frame(summaries)
    picks = [t for t in QUICK_PICKS if not df.empty and t in set(df["ticker"])]
    if picks:
        st.pills("Quick start - open a well-known company", picks, selection_mode="single",
                 key="overview_pick", on_change=_pick_company)
    render_selected(df, selected)

    _html(ui.section("Where to go", "each page answers one question", kind="data"))
    _html('<div class="fsa-tiles">' + "".join(
        f'<div class="fsa-tile k-{AREA_KIND[area]}"><div class="t">{icon} {escape(name)}</div>'
        f'<div class="q">{escape(question)}</div></div>'
        for icon, name, area, question in PAGES) + "</div>")
    st.caption("Section colours: blue = financial health, red = risk, purple = valuation, "
               "green = profitability, amber = growth, grey = data and method.")

    render_universe(universe, snap, freshness, bands or [], n_demo)

    c1, c2 = st.columns([1.1, 1])
    with c1:
        with st.expander("New here? A 5-step guide", expanded=selected is None):
            _html(ui.guide("Start here", [
                "Pick a company: a quick-start button above, the sidebar watchlist, or type "
                "a ticker (TCS.NS, AAPL; Indian stocks need .NS or .BO) and press "
                "'Fetch & analyse'.",
                "Company page, top to bottom: price, data status (source, dates, "
                "completeness), Financial Health breakdown, key numbers, statements, red "
                "flags. Hover the (i) cards for formulas.",
                "Quality & Value: DCF value with its assumptions, sensitivity table, "
                "scenarios and reverse DCF.",
                "Context: Compare, Screener and Market map; Portfolio reviews your own "
                "holdings file.",
                "Unsure about a term? Search the glossary. Every method is in Methodology.",
            ]))
    with c2:
        render_glossary("overview", compact=True)


def render_selected(df: pd.DataFrame, selected: str | None) -> None:
    if not selected or df.empty or selected not in set(df["ticker"]):
        st.caption("Pick a company to see it at a glance here.")
        return
    row = df[df["ticker"] == selected].iloc[0]
    score = None if pd.isna(row.get("score")) else float(row["score"])
    flags = None if pd.isna(row.get("red_flags_latest")) else int(row["red_flags_latest"])
    _html(ui.section(f"{row.get('name') or selected} at a glance",
                     f"{selected} · {row.get('sector') or 'sector n/a'} · open the 📈 Company "
                     "tab for the full analysis", kind="health"))
    _html(ui.cards([
        ("Health score", "N/A" if score is None else f"{score:.0f}/100",
         (row.get("interpretation") or "not scored").title(), ui.score_color(score),
         glossary.explain("Financial Health Score")),
        ("Red flags (latest year)", "N/A" if flags is None else str(flags),
         "potential concerns to read", ui.DOWN if flags else None,
         glossary.explain("Red flag")),
        ("Latest fiscal year-end", str(row.get("latest_fy") or "n/a")[:10],
         "statements analysed", None, glossary.explain("Fiscal year (FY)")),
        ("Reporting currency", str(row.get("currency") or "n/a"), "no FX conversion is done",
         None, "Amounts are always shown in the company's own reporting currency."),
    ]))


def render_universe(summaries, snap, freshness, bands, n_demo: int = 0) -> None:
    _html(ui.section("The analysed universe", "what is stored right now", kind="data"))
    if n_demo:
        st.caption(f"{n_demo} synthetic demo companies (offline sample data) are left out of "
                   "these statistics.")
    stale = (freshness or {}).get("stale_prices")
    _html(ui.cards([
        ("Companies analysed", f"{snap['companies']:,}",
         f"{snap['scored']:,} with a health score · {', '.join(snap['markets']) or '-'}",
         None, "Companies with a stored analysis (INR = India, USD = US reporting). The "
         "ones without a score lack enough data for the framework."),
        ("Median health score", "N/A" if snap["median_score"] is None
         else f"{snap['median_score']:.0f}/100", "half the companies score above", None,
         None),
        ("Red flags in latest year", f"{snap['with_flags']:,}", "companies with at least one",
         ui.DOWN if snap["with_flags"] else None, glossary.explain("Red flag")),
        (f"Prices older than {(freshness or {}).get('stale_days', 10)} days", "N/A" if stale is None else f"{stale:,}",
         f"of {freshness.get('priced', 0):,} priced · newest {freshness.get('latest_price_date') or 'n/a'}"
         if freshness else "",
         ui.DOWN if stale else None,
         "Market prices are stored snapshots; multiples and DCF comparisons use them."),
    ]))
    if not snap["scored"]:
        return
    c1, c2 = st.columns([1.15, 1])
    with c1:
        _html(ui.section("Health score distribution", "colour = band", kind="health"))
        dist = score_distribution(summaries, bands)
        st.plotly_chart(distribution_figure(dist), width="stretch",
                        config={"displaylogo": False, "scrollZoom": False,
                                "doubleClick": False}, key="overview_dist")
        st.caption("Bands are the universe quartiles of the score, so about a quarter of the "
                   "companies fall in each.")
    with c2:
        _html(ui.section("Median score by sector", "sectors with 5+ companies", kind="health"))
        sectors = sector_medians(summaries)
        if not sectors.empty:
            fig = plots.ranked_bars(dict(zip(sectors["sector"], sectors["median"])), "",
                                    height=300, score_colors=True)
            st.plotly_chart(fig, width="stretch", config={"displaylogo": False,
                                                          "scrollZoom": False,
                                                          "doubleClick": False},
                            key="overview_sectors")
            st.caption("Scores are relative to each company's own industry peers, so "
                       "sector medians sit near the middle by design.")
    split = market_split(summaries)
    if not split.empty:
        st.dataframe(split.rename(columns={"market": "Market", "companies": "Companies",
                                           "median": "Median score",
                                           "with_flags": "With red flags (latest)"})
                     .round(1), hide_index=True, width="stretch")

    df = _frame(summaries)
    score_col = {"Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100,
                                                          format="%.0f")}
    c1, c2 = st.columns(2)
    with c1:
        _html(ui.section("Highest health scores", kind="health"))
        top = df.dropna(subset=["score"]).nlargest(10, "score")
        st.dataframe(top[["ticker", "score", "market", "name"]].rename(columns={
            "ticker": "Ticker", "name": "Company", "market": "Market", "score": "Score"}),
            hide_index=True, width="stretch", column_config=score_col)
    with c2:
        _html(ui.section("Most red flags (latest year)", kind="risk"))
        flags = df.dropna(subset=["red_flags_latest"])
        flags = flags[flags["red_flags_latest"] > 0].nlargest(10, "red_flags_latest")
        if flags.empty:
            st.caption("No red flags in the latest year for any company.")
        else:
            st.dataframe(flags[["ticker", "red_flags_latest", "market", "name"]].rename(
                columns={"ticker": "Ticker", "name": "Company", "market": "Market",
                         "red_flags_latest": "Flags"}), hide_index=True, width="stretch")
    st.caption("Rankings use stored analyses. A high score describes past financial "
               "position, not future returns.")


def render_glossary(key: str, compact: bool = False) -> None:
    _html(ui.section("Glossary", "plain-language explanations", kind="data"))
    query = st.text_input("Search a term", key=f"glossary_{key}",
                          placeholder="e.g. WACC, current ratio, red flag")
    rows = glossary.search(query)
    if compact and not query:
        st.caption(f"{len(glossary.TERMS)} terms - type to search, e.g. 'margin' or 'debt'.")
        return
    if not rows:
        st.caption("No matching term.")
        return
    st.dataframe(pd.DataFrame(rows, columns=["Term", "Area", "Meaning"]), hide_index=True,
                 width="stretch", height=min(38 + 35 * len(rows), 420))
