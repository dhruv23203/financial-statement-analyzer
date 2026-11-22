"""Overview: the first page a new user sees - what the app does, how to use it, what each
page answers, a snapshot of the stored data, and a searchable glossary."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from app.reporting import glossary, ui

PAGES = [
    ("📈 Company", "Health", "How financially healthy is this company, how has it changed, "
     "and are there warning signs?"),
    ("⚖️ Compare", "Health", "How does it compare with chosen peers, ratio by ratio?"),
    ("🔎 Screener", "Data", "Which companies meet my conditions (e.g. ROE > 15 and low debt)?"),
    ("🗺️ Market map", "Data", "How do sectors and companies look side by side?"),
    ("💡 Quality & Value", "Valuation", "Is the price reasonable for the quality? DCF, "
     "sensitivity, scenarios and reverse DCF."),
    ("🩺 Portfolio", "Risk", "What does my uploaded portfolio hold, and where is the "
     "concentration or weakness?"),
    ("🤖 ML prediction", "Risk", "How likely is a currently healthy company to report a loss "
     "or cash burn next year (a statistical model)?"),
    ("📘 Methodology", "Data", "Exactly how every number is calculated, with evidence and "
     "limitations."),
    ("🕘 History / 🗄️ Database", "Data", "When was the data fetched and analysed; data "
     "checks, archives and backups."),
]


def _html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


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


def render(summaries: list[dict[str, Any]], selected: str | None) -> None:
    _html(ui.section("Welcome", "a financial research workspace for NIFTY 500 and S&P 500 "
                                "companies"))
    st.markdown(
        "This app turns published annual statements into **ratios, trends, red flags, a "
        "0-100 Financial Health Score, valuations and an experimental ML risk model** - with "
        "every number traceable to its formula and source. It describes companies; it does "
        "not tell you what to buy or sell.")
    _html(ui.guide("Start here", [
        "Pick a company in the sidebar watchlist (or type a ticker such as TCS.NS or AAPL "
        "and press 'Fetch & analyse'; Indian stocks need .NS or .BO).",
        "Read the Company page from the top: the price, the data-status line (source, "
        "dates, completeness), the Financial Health breakdown, key numbers, statements and "
        "red flags. Hover the (i) cards for formulas.",
        "Check the price in 'Quality & Value': the DCF value, the sensitivity table and "
        "scenarios, and the reverse DCF.",
        "Use 'Compare', 'Screener' and 'Market map' to put the company in context; "
        "'Portfolio' reviews your own holdings file.",
        "Unsure about a term? Search the glossary below. Every method is in 'Methodology'.",
    ]))
    if selected:
        st.caption(f"Currently selected: **{selected}** - open the 📈 Company tab.")

    _html(ui.section("What each page answers", kind="data"))
    st.dataframe(pd.DataFrame(PAGES, columns=["Page", "Area", "Question it answers"]),
                 hide_index=True, width="stretch")
    st.caption("Colour bars on section headings mark the area: blue = financial health, "
               "red = risk, purple = valuation, green = profitability, amber = growth, "
               "grey = data and method.")

    snap = snapshot(summaries)
    _html(ui.section("Data snapshot", "what is stored right now", kind="data"))
    _html(ui.cards([
        ("Companies analysed", f"{snap['companies']:,}",
         ", ".join(snap["markets"]) or "no data yet", None,
         "Companies with a stored analysis (INR = India, USD = US reporting)."),
        ("With a health score", f"{snap['scored']:,}",
         "the rest lack enough data", None, glossary.explain("Financial Health Score")),
        ("Median health score", "N/A" if snap["median_score"] is None
         else f"{snap['median_score']:.0f}/100", "across scored companies", None,
         "Half of the scored companies are above this value."),
        ("Red flags in latest year", f"{snap['with_flags']:,}", "companies with at least one",
         ui.DOWN if snap["with_flags"] else None, glossary.explain("Red flag")),
        ("Latest fiscal year-end", snap["latest_fy"] or "N/A", "newest statements stored",
         None, glossary.explain("Fiscal year (FY)")),
    ]))

    analysed = pd.DataFrame([s for s in summaries if s.get("analysed")])
    if not analysed.empty and "score" in analysed:
        analysed["score"] = pd.to_numeric(analysed["score"], errors="coerce")
        c1, c2 = st.columns(2)
        with c1:
            _html(ui.section("Highest health scores", kind="health"))
            top = analysed.dropna(subset=["score"]).nlargest(10, "score")
            st.dataframe(top[["ticker", "name", "sector", "score"]].rename(columns={
                "ticker": "Ticker", "name": "Company", "sector": "Sector",
                "score": "Score"}).round(1), hide_index=True, width="stretch")
        with c2:
            _html(ui.section("Most red flags (latest year)", kind="risk"))
            flags = analysed.assign(red_flags_latest=pd.to_numeric(
                analysed.get("red_flags_latest"), errors="coerce")).dropna(
                subset=["red_flags_latest"])
            flags = flags[flags["red_flags_latest"] > 0].nlargest(10, "red_flags_latest")
            if flags.empty:
                st.caption("No red flags in the latest year for any company.")
            else:
                st.dataframe(flags[["ticker", "name", "red_flags_latest"]].rename(columns={
                    "ticker": "Ticker", "name": "Company",
                    "red_flags_latest": "Flags"}), hide_index=True, width="stretch")
        st.caption("Rankings use stored analyses only. A high score describes past "
                   "financial position, not future returns.")

    render_glossary("overview")


def render_glossary(key: str) -> None:
    _html(ui.section("Glossary", "plain-language explanations of the terms used", kind="data"))
    query = st.text_input("Search a term", key=f"glossary_{key}",
                          placeholder="e.g. WACC, current ratio, red flag")
    rows = glossary.search(query)
    if not rows:
        st.caption("No matching term.")
        return
    st.dataframe(pd.DataFrame(rows, columns=["Term", "Area", "Meaning"]), hide_index=True,
                 width="stretch")
