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
from app.services.portfolio_doctor import CONFIG as PORTFOLIO_CONFIG
from app.valuation.capm import load_assumptions
from app.valuation.dcf import value_firm

RATIO_STATUSES = pd.DataFrame([
    {"Shown as": "a number", "Status": "OK",
     "Meaning": "Calculated from reported inputs. A genuine value of zero is shown as 0."},
    {"Shown as": "N/A", "Status": "UNAVAILABLE",
     "Meaning": "An input is missing, a denominator is zero, or a previous year is needed "
                "and not available. Missing inputs are never treated as zero."},
    {"Shown as": "N/M", "Status": "NOT_MEANINGFUL",
     "Meaning": "Mathematically possible but misleading, e.g. ROE or debt-to-equity with "
                "negative equity, cash conversion with a loss, inventory turnover with "
                "immaterial inventory, Altman/Beneish for banks."},
    {"Shown as": "number + note", "Status": "OK (fallback method)",
     "Meaning": "Calculated with a labelled fallback, e.g. ending balance instead of the "
                "average when the previous year is missing, or operating income as an EBIT "
                "proxy. The method is stored with the value."},
])

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
    st.caption("Averages (ROA, ROE, turnover ratios) use the immediately preceding fiscal "
               "year only (330-400 days earlier).")
    st.markdown("**How to read a ratio cell**")
    st.dataframe(RATIO_STATUSES, hide_index=True, width="stretch")

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

    render_valuation_method()
    render_portfolio_method()

    st.markdown(ui.section("Data and limitations"), unsafe_allow_html=True)
    for item in LIMITATIONS:
        st.markdown(f"- {item}")
    st.caption("Full documentation in the project repository: docs/SCORING_METHODOLOGY.md, "
               "docs/RED_FLAG_METHODOLOGY.md, docs/FINANCIAL_FORMULAS.md, "
               "docs/DATA_VALIDATION.md, docs/AUDIT_REPORT.md.")


def render_valuation_method() -> None:
    a = load_assumptions()
    st.markdown(ui.section("Valuation (Quality & Value tab)",
                           "FCFF discounted at WACC, then the equity bridge"), unsafe_allow_html=True)
    st.markdown(
        "1. **FCFF** = operating cash flow + interest x (1 - tax rate) - capital expenditure "
        "(latest fiscal year).\n"
        f"2. **Forecast**: {a['forecast']['explicit_years']} years at the historical revenue CAGR "
        f"bounded to [{a['forecast']['min_growth']:+.0%}, {a['forecast']['max_growth']:+.0%}] "
        "(negative history stays negative), then a linear fade to terminal growth over "
        f"{a['forecast']['fade_years']} years.\n"
        "3. **Discount rate**: the universe screen uses a fixed, disclosed *screening discount "
        "rate* (not a computed WACC). The selected company's full valuation uses **WACC** = "
        "E/(D+E) x Ke + D/(D+E) x Kd x (1 - t), with **Ke = Rf + beta x ERP** (CAPM) and beta "
        "from 5 years of monthly returns against NIFTY 50 / S&P 500.\n"
        "4. **Terminal value** (Gordon growth) = FCFF_N x (1 + g) / (WACC - g), valid only if "
        "WACC > g; otherwise the valuation is reported as invalid.\n"
        "5. **Enterprise value** = PV of forecast FCFF + PV of terminal value (year-end "
        "discounting).\n"
        "6. **Equity value** = EV + cash and short-term investments - total debt - minority "
        "interest; **value per share** = equity value / diluted shares.\n"
        "7. FCFF is never discounted at the cost of equity, and FCFE never at WACC (the engine "
        "rejects those pairings). Intrinsic value does not need the market price; the price is "
        "only used for the comparison.")
    rows = [{"Market": cur, "Risk-free rate": f"{m['risk_free_rate']:.2%}",
             "Equity risk premium": f"{m['equity_risk_premium']:.2%}",
             "Tax rate": f"{m['statutory_tax_rate']:.2%}",
             "Terminal growth": f"{m['terminal_growth']:.1%}",
             "Benchmark": m["benchmark_name"],
             "Sources": f"{m['risk_free_source']}; {m['equity_risk_premium_source']}"}
            for cur, m in a["markets"].items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    check = value_firm([100, 110, 121, 133.1, 146.41], 0.10, 0.03, cash=200, debt=500,
                       shares=100)
    st.caption(
        f"Engine check (live, manual reference example): FCFF 100, 110, 121, 133.1, 146.41; "
        f"WACC 10%, g 3%, cash 200, debt 500, 100 shares -> TV {check.terminal_value:,.2f}, "
        f"PV of forecast {check.pv_forecast:,.2f}, PV of TV {check.pv_terminal_value:,.2f}, "
        f"EV {check.enterprise_value:,.2f}, equity {check.equity_value:,.2f}, value per share "
        f"{check.value_per_share:,.2f} (reference: 2,154.32 / 454.55 / 1,337.66 / 1,792.21 / "
        f"1,492.21 / 14.92). Assumptions as of {a['as_of']}.")
    sc = a["valuation_score"]
    st.caption(f"Valuation score: {sc['note']} Weights: "
               + ", ".join(f"{k} {v:g}" for k, v in sc["weights"].items())
               + f". Peer medians: same market and sector (at least {sc['min_sector_peers']} "
               "peers), else the same market overall (flagged), never a mixed India + US median.")


def render_portfolio_method() -> None:
    c = PORTFOLIO_CONFIG
    st.markdown(ui.section("Portfolio Doctor", "illustrative project assumptions"),
                unsafe_allow_html=True)
    st.markdown(
        f"* **Prices**: dated broker LTP -> latest Yahoo Finance close -> stored snapshot, each "
        f"used only if at most {c['price_max_age_days']} days old; otherwise value, P&L and "
        "weight are N/A. Average cost is never used as a market price. Sources that differ by "
        f"more than {c['price_disagreement_pct']:g}% are flagged.\n"
        "* **Currencies**: each holding keeps its own currency; INR and USD are never added "
        "(no FX conversion), so the portfolio is reviewed one currency at a time.\n"
        f"* **Position spread score** = min(100, effective positions / "
        f"{c['position_spread_target']} x 100), effective positions = 1 / sum(w²). It measures "
        "how evenly money is spread across positions, not sector diversification or return "
        "correlation (largest sector and number of sectors are reported separately).\n"
        f"* **Portfolio score** = {c['composite_weights']['health']:.0%} value-weighted health "
        f"+ {c['composite_weights']['position_spread']:.0%} position spread, shown only when at "
        f"least {c['min_analysed_coverage']:.0%} of the priced value is analysed.\n"
        f"* **Alert limits**: one stock above {c['max_stock_pct']:g}%, one sector above "
        f"{c['max_sector_pct']:g}%, fewer than {c['min_holdings']} holdings. {c['note']}")
