"""'Methodology' tab: how every number in the dashboard is produced.

Written for evaluators and first-time users. Tables are generated from the code and the
active configuration files (ratio definitions, scoring weights, red-flag thresholds), so
the page always describes exactly what the running app does.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from app.analysis.ratios import RATIO_DEFINITIONS
from app.reporting import ui

PIPELINE = (
    ("1", "Fetch", "Annual statements, metadata and a market snapshot from Yahoo Finance "
                   "(yfinance), with retries and typed errors."),
    ("2", "Store", "Raw facts saved unchanged in SQLite (SQLAlchemy + Alembic); every run is "
                   "kept for audit."),
    ("3", "Clean", "60+ provider line items mapped to standard metrics; sparse years and "
                   "currency mislabels detected; missing values stay missing."),
    ("4", "Ratios", "Liquidity, solvency, profitability, efficiency and cash-flow ratios with "
                    "N/A (missing input) and N/M (not meaningful) states."),
    ("5", "Trends", "Year-over-year changes (adjacent years only) and multi-year CAGR."),
    ("6", "Red flags", "Rule engine: every rule ends as triggered, not triggered or not "
                       "evaluated, with the reason."),
    ("7", "Score", "Sector-relative 0-100 health score with category scores, coverage and "
                   "letter grades."),
    ("8", "Use", "Company view, comparison, screener, market map, history and database tools."),
)
LIMITATIONS = (
    "Yahoo Finance is a secondary source: 104 of 110 US values checked against SEC 10-K "
    "filings matched within 0.5 %; Indian figures have not yet been checked against annual "
    "reports.",
    "About four fiscal years per company, so trends are short.",
    "Scores are relative to large NIFTY 500 / S&P 500 companies in 2021-2026, not absolute "
    "measures of safety; banks and insurers get no overall score.",
    "The reference universe is today's index members (survivorship bias).",
    "Descriptive analysis only - not a credit rating, not a recommendation, not investment "
    "advice.",
)


def _pipeline_html() -> str:
    steps = "".join(
        f'<div class="fsa-card"><div class="lbl">Step {n}</div>'
        f'<div class="val" style="font-size:1.05rem">{name}</div>'
        f'<div class="sub" style="line-height:1.45">{text}</div></div>'
        for n, name, text in PIPELINE)
    return f'<div class="fsa-grid">{steps}</div>'


def ratio_table() -> pd.DataFrame:
    """Every ratio the engine calculates, with its formula (from RATIO_DEFINITIONS)."""
    order = ("LIQUIDITY", "SOLVENCY", "PROFITABILITY", "EFFICIENCY", "CASH_FLOW",
             "SHAREHOLDER", "VALUATION", "BENCHMARK")
    rows = [{"Category": d.category.replace("_", " ").title(), "Ratio": d.name,
             "Formula": d.formula, "Unit": d.unit,
             "Better when": "higher" if d.higher_is_better else "lower"}
            for d in RATIO_DEFINITIONS.values()]
    frame = pd.DataFrame(rows)
    frame["_o"] = frame["Category"].str.upper().str.replace(" ", "_").map(
        {c: i for i, c in enumerate(order)}).fillna(len(order))
    return frame.sort_values(["_o", "Ratio"]).drop(columns="_o").reset_index(drop=True)


def weight_table(scoring: dict[str, Any]) -> pd.DataFrame:
    """Category weights and the ratio weights inside each category (active config)."""
    rows = []
    for category, spec in scoring["categories"].items():
        ratios = ", ".join(f"{RATIO_DEFINITIONS[c].name if c in RATIO_DEFINITIONS else c} "
                           f"{float(m['weight']):.2f}" for c, m in spec["metrics"].items())
        rows.append({"Category": category.replace("_", " ").title(),
                     "Category weight": round(float(spec["weight"]), 3),
                     "Ratios (weight inside the category)": ratios})
    return pd.DataFrame(rows)


def rule_table(red_flags: dict[str, Any]) -> pd.DataFrame:
    """Red-flag rules with their thresholds (active config)."""
    skip = {"name", "enabled", "default_severity", "description", "evidence", "source"}
    rows = []
    for code, rule in red_flags["rules"].items():
        params = "; ".join(f"{k.replace('_', ' ')} = {v}" for k, v in rule.items()
                           if k not in skip) or "-"
        rows.append({"Code": code.split("_")[0], "Rule": rule.get("name", code),
                     "Thresholds": params, "Default severity": rule.get("default_severity")})
    return pd.DataFrame(rows)


def render_methodology_tab(scoring: dict[str, Any], red_flags: dict[str, Any]) -> None:
    st.markdown(ui.section("How this app works",
                           "every number below comes from the code and active configuration"),
                unsafe_allow_html=True)
    st.markdown(_pipeline_html(), unsafe_allow_html=True)

    st.markdown(ui.section("Financial ratios", f"{len(RATIO_DEFINITIONS)} ratios"),
                unsafe_allow_html=True)
    st.dataframe(ratio_table(), hide_index=True, width="stretch")
    st.caption("Missing inputs are never treated as zero: the ratio is N/A. Ratios that are "
               "mathematically misleading (e.g. ROE with negative equity) are N/M. Averages "
               "use the immediately preceding fiscal year only.")

    calibration = scoring.get("calibration") or {}
    st.markdown(ui.section("Financial health score",
                           f"methodology v{scoring.get('methodology_version', '?')}"),
                unsafe_allow_html=True)
    bands = ", ".join(f"{b['label'].title()} >= {b['min']}" for b in
                      sorted(scoring["interpretation_bands"], key=lambda b: -b["min"]))
    st.markdown(
        f"1. **Ratio -> points (0-100):** percentile breakpoints of the reference universe "
        f"({calibration.get('companies', '?')} companies, {calibration.get('company_years', '?')} "
        "company-years), within the company's sector when the sector is large enough.\n"
        f"2. **Category score:** weighted average of the available ratio points (ratio weights: "
        f"{calibration.get('ratio_weight_method', 'CRITIC')}). A category needs at least "
        f"{float(scoring['min_category_coverage']):.0%} of its weight or of its ratios.\n"
        f"3. **Overall score:** weighted average of the scored categories (category weights: "
        f"{calibration.get('category_weight_method', '?')}); at least "
        f"{scoring['min_categories_for_overall']} categories are required, otherwise N/A.\n"
        f"4. **Bands** (quartiles of the universe): {bands}.")
    st.dataframe(weight_table(scoring), hide_index=True, width="stretch")
    st.caption("Liquidity has weight 0: in the calibration it added no predictive information "
               "once the other categories were known (coefficient CI spans zero). It is still "
               "shown, graded and checked by red flag RF05.")

    st.markdown(ui.section("Red-flag rules", f"thresholds v{red_flags.get('version', '?')}"),
                unsafe_allow_html=True)
    st.dataframe(rule_table(red_flags), hide_index=True, width="stretch")
    excluded = red_flags.get("sector_exclusions") or {}
    for sector, codes in excluded.items():
        st.caption(f"Not applicable to {sector}: {', '.join(c.split('_')[0] for c in codes)} - "
                   "reported as 'not evaluated' with the reason.")

    st.markdown(ui.section("Data and limitations"), unsafe_allow_html=True)
    for item in LIMITATIONS:
        st.markdown(f"- {item}")
    st.caption("Full documentation in the project repository: docs/SCORING_METHODOLOGY.md, "
               "docs/RED_FLAG_METHODOLOGY.md, docs/FINANCIAL_FORMULAS.md, "
               "docs/DATA_VALIDATION.md, docs/AUDIT_REPORT.md.")
