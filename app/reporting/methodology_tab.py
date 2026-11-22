"""'Methodology' tab: how every number in the dashboard is produced.

Written for evaluators and first-time users. Tables are generated from the code and the
active configuration files (ratio definitions, scoring weights, red-flag thresholds), so
the page always describes exactly what the running app does.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import streamlit as st

from app.analysis.ratios import RATIO_DEFINITIONS
from app.reporting import ui
from app.services.portfolio_doctor import CONFIG as PORTFOLIO_CONFIG
from app.valuation.capm import load_assumptions
from config.settings import CONFIG_DIR
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

    render_health_framework()

    st.markdown(ui.section("Red-flag rules", f"thresholds v{red_flags.get('version', '?')}",
                           kind="risk"),
                unsafe_allow_html=True)
    st.dataframe(rule_table(red_flags), hide_index=True, width="stretch")
    excluded = red_flags.get("sector_exclusions") or {}
    for sector, codes in excluded.items():
        st.caption(f"Not applicable to {sector}: {', '.join(c.split('_')[0] for c in codes)} - "
                   "reported as 'not evaluated' with the reason.")

    render_validation_evidence()
    render_valuation_method()
    render_portfolio_method()

    st.markdown(ui.section("Data and limitations", kind="data"), unsafe_allow_html=True)
    for item in LIMITATIONS:
        st.markdown(f"- {item}")
    from app.reporting.overview_tab import render_glossary
    render_glossary("methodology")
    st.caption("Full documentation in the project repository: docs/HEALTH_SCORE_FRAMEWORK.md, "
               "docs/RED_FLAG_METHODOLOGY.md, docs/FINANCIAL_FORMULAS.md, "
               "docs/DATA_VALIDATION.md, docs/AUDIT_REPORT.md.")


def framework_table(config: dict[str, Any]) -> pd.DataFrame:
    """Each industry framework: pillar budgets and the metrics (with their points)."""
    rows = []
    for key, fw in config["frameworks"].items():
        row = {"Framework": fw["label"]}
        for pillar in config["pillars"]:
            spec = fw["pillars"][pillar]
            if not spec.get("max"):
                row[pillar.title()] = "not assessed"
                continue
            total = sum(spec["metrics"].values())
            row[pillar.title()] = f"{spec['max']}: " + ", ".join(
                f"{config['metrics'][m]['name']} {spec['max'] * w / total:.0f}"
                for m, w in spec["metrics"].items())
        rows.append(row)
    return pd.DataFrame(rows)


def guardrail_table(config: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for fw in config["frameworks"].values():
        for g in fw.get("guardrails", []):
            cond = f"< {g['below']}" if "below" in g else f"> {g['above']}"
            rows.append({"Framework": fw["label"], "Metric": config["metrics"][g["metric"]]["name"],
                         "Condition": cond, "Max share of the metric's points": f"{g['cap']:.0%}",
                         "Why": g["reason"]})
    return pd.DataFrame(rows)


def render_health_framework() -> None:
    from app.analysis.health_framework import load_framework

    cfg = load_framework()
    st.markdown(ui.section("Financial Health Score", kind="health", hint=f"methodology v{cfg['version']} - "
                                                     "industry frameworks"),
                unsafe_allow_html=True)
    bands = ", ".join(f"{b['label'].title()} >= {b['min']}" for b in
                      sorted(cfg.get("interpretation_bands") or [], key=lambda b: -b["min"]))
    st.markdown(
        "1. **Framework:** the company's industry decides the metrics, budgets and guardrails "
        "(table below). Banks, insurers and other financial companies are scored on "
        "profitability, capital and growth; their liquidity and efficiency are *not "
        "assessed* because the data provider does not report deposits, loans, NPAs or "
        "claims - nothing is guessed.\n"
        "2. **Metric points:** the metric's percentile among industry peers (same sector, or "
        f"same financial framework; at least {cfg['min_peer_observations']} peer observations, "
        "else the wider group) times the metric's points. Lower-is-better metrics are mirrored.\n"
        "3. **Guardrails:** absolute minimum standards can only lower points (so a weak "
        "company does not score well just because its peers are weak).\n"
        f"4. **Pillar:** budget x weighted average of the measured metrics, if at least "
        f"{cfg['min_pillar_coverage']:.0%} of its metric weight is measured. Missing data never "
        "earns points.\n"
        f"5. **Overall:** points earned / points assessable x 100; withheld when less than "
        f"{cfg['min_assessed_share']:.0%} of the budget or fewer than {cfg['min_pillars']} "
        "pillars can be assessed. **Confidence** = share of the framework's weight measured.\n"
        f"6. **Bands** (universe quartiles): {bands}.")
    st.dataframe(framework_table(cfg), hide_index=True, width="stretch")
    st.caption("Budgets (25/20/20/15/20 for non-financial companies) are a declared design, "
               "not fitted to any outcome. Metrics sharing a pillar are kept below |0.7| "
               "Spearman correlation to avoid double counting.")
    with st.expander("Guardrails (absolute thresholds)"):
        st.dataframe(guardrail_table(cfg), hide_index=True, width="stretch")
    ev_path = CONFIG_DIR / "validation_evidence.json"
    ev = json.loads(ev_path.read_text(encoding="utf-8")).get("health_framework_v3")         if ev_path.is_file() else None
    if ev:
        d, sens = ev["discrimination"], ev["sensitivity"]
        quint = " -> ".join(f"{v:.1%}" for _, v in sorted(d["distress_rate_by_quintile"].items(),
                                                          key=lambda kv: int(kv[0])))
        st.markdown(ui.cards([
            ("Companies scored", f"{ev['scored_share']:.0%}",
             f"{ev['company_years']:,} company-years, {ev['companies']} companies", None),
            ("Distress rate by score quintile", quint.split(" -> ")[0] + " ... " +
             quint.split(" -> ")[-1], "lowest -> highest quintile", None),
            ("AUC (in-sample)", f"{d['auc_score']:.2f}",
             f"two-ratio baseline {d['auc_two_ratio_baseline']:.2f}", None),
            ("Budget sensitivity", f"{sens['spearman_rank_correlation']:.3f}",
             "rank correlation with equal budgets", None)]), unsafe_allow_html=True)
        max_rho = max((abs(r["spearman"]) for r in ev["redundancy"]), default=0)
        st.caption(
            f"Validation ({ev['generated_at']}): next-year distress (net loss or negative OCF) "
            f"falls across score quintiles ({quint}), so the score orders risk monotonically. "
            f"Its AUC of {d['auc_score']:.2f} is below a simple two-ratio rule "
            f"({d['auc_two_ratio_baseline']:.2f}): the score is a broad, explainable health "
            "summary, not a tuned distress predictor, and no predictive accuracy is claimed. "
            f"Largest within-pillar metric correlation: {max_rho:.2f}. Benchmarks are pooled "
            "over all years (in-sample); universe = today's index members (survivorship).")


def render_validation_evidence() -> None:
    """Does the score mean anything, and is the data right? Numbers come from
    config/validation_evidence.json (copied from the calibration and validation runs)."""
    path = CONFIG_DIR / "validation_evidence.json"
    if not path.is_file():
        return
    ev = json.loads(path.read_text(encoding="utf-8"))
    sv, dv = ev["score_validation"], ev["data_validation"]
    st.markdown(ui.section("Evidence: what the score test does and does not show",
                           "out-of-sample test, naive baselines and filing checks"),
                unsafe_allow_html=True)
    base = sv["baseline_auc"]
    naive = sv.get("naive_baselines") or {}
    counts = dv["status_counts"]
    st.markdown(ui.cards([
        ("Score, out-of-sample AUC", f"{sv['oof_auc']:.2f}",
         f"95% CI {sv['oof_auc_ci'][0]:.2f}-{sv['oof_auc_ci'][1]:.2f}; equal weights "
         f"{base.get('EQUAL', float('nan')):.2f}", None),
        ("Two-ratio baseline AUC", f"{naive['two_ratio_auc']:.2f}" if naive.get(
            "two_ratio_auc") is not None else "N/A",
         "min(net margin, OCF margin) this year", None),
        ("Last-year-status baseline", f"{naive['persistence_auc']:.2f}" if naive.get(
            "persistence_auc") is not None else "N/A",
         "this year's loss / negative OCF", None),
        ("Filing checks matching", f"{counts.get('MATCH', 0)} / {dv['checks']}",
         f"{counts.get('CLOSE', 0)} close, {counts.get('DIFFERENT', 0)} different", None),
    ]), unsafe_allow_html=True)
    st.caption(
        f"Score test: {sv['method']} ({sv['observations']:,} company-years, "
        f"{sv['companies']} companies). Label: {sv['label']}; {sv['positive_rate']:.1%} of "
        "company-years are positive. AUC 0.5 = no better than chance, 1.0 = perfect ranking. "
        f"Generated {sv['generated_at'][:10]}.")
    if naive.get("two_ratio_auc") is not None:
        st.caption(
            "Limitation: the label is next year's loss or negative operating cash flow, and "
            "the profitability and cash-flow categories measure this year's version of the "
            "same thing, so much of the AUC is persistence. A two-ratio rule on the same "
            f"kind of sample ({naive['observations']:,} company-years) reaches "
            f"{naive['two_ratio_auc']:.2f}, at least as high as the full score: this test "
            "does not show that the score adds information beyond those two ratios. It "
            "describes relative financial position; it is not a validated distress "
            "predictor.")
    st.caption(
        f"Data check: {dv['checks']} stored values for {dv['companies']} companies compared "
        f"with {dv['reference']} ({dv['thresholds']}); accessed {dv['accessed_on']}. "
        f"Scope: {dv['scope_note']}.")


def render_valuation_method() -> None:
    a = load_assumptions()
    st.markdown(ui.section("Valuation (Quality & Value tab)", kind="valuation", hint=
                           "FCFF discounted at WACC, then the equity bridge"), unsafe_allow_html=True)
    st.markdown(
        "1. **FCFF** = EBIT x (1 - tax rate) + D&A - capital expenditure + change in working "
        "capital (latest fiscal year). If any of these is missing: operating cash flow - capex, "
        "plus after-tax interest only where interest paid is an operating cash flow (US GAAP); "
        "Indian (Ind AS 7) filers report interest paid under financing, so it is not added "
        "back. The method used is shown with every valuation.\n"
        f"2. **Forecast**: {a['forecast']['explicit_years']} years at the historical revenue CAGR "
        f"bounded to [{a['forecast']['min_growth']:+.0%}, {a['forecast']['max_growth']:+.0%}] "
        "(negative history stays negative), then a linear fade to terminal growth over "
        f"{a['forecast']['fade_years']} years.\n"
        "3. **Discount rate**: the universe screen uses a fixed, disclosed *screening discount "
        "rate* (not a computed WACC). The selected company's full valuation uses **WACC** = "
        "E/(D+E) x Ke + D/(D+E) x Kd x (1 - t), with **Ke = Rf + beta x ERP** (CAPM) and beta "
        "from 5 years of monthly returns (prices adjusted for splits and dividends) against "
        "NIFTY 50 / S&P 500.\n"
        "4. **Terminal value** (Gordon growth) = FCFF_N x (1 + g) / (WACC - g), valid only if "
        "WACC > g; otherwise the valuation is reported as invalid.\n"
        "5. **Enterprise value** = PV of forecast FCFF + PV of terminal value (year-end "
        "discounting).\n"
        "6. **Equity value** = EV + cash and short-term investments - total debt - minority "
        "interest; **value per share** = equity value / share count (current shares from the "
        "dated market snapshot, else period-end ordinary shares, else diluted weighted-average "
        "shares - the count, source and date are shown).\n"
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
