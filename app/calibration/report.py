"""Markdown report of a calibration run (docs/calibration/CALIBRATION_REPORT.md)."""

from __future__ import annotations

from typing import Any

from app.analysis.ratios import RATIO_DEFINITIONS
from app.calibration.calibrate import (CATEGORIES, DAMODARAN_COVERAGE, MIN_SECTOR_OBSERVATIONS,
                                       CalibrationResult)

LABEL_TEXT = {
    "distress": ("PRIMARY (pre-declared, may set weights)",
                 "1 if next fiscal year shows a net loss or negative operating cash flow "
                 "(distress proxy)"),
    "earnings_decline": ("SECONDARY (reference only)",
                         "1 if next fiscal year's net income is lower (direction of earnings, "
                         "Ou & Penman 1989)"),
}


def _fmt(points: list[list[float]]) -> str:
    return " ".join(f"[{v:g}, {p:g}]" for v, p in points)


def _supervised_lines(label: str, sup: dict[str, Any]) -> list[str]:
    role, definition = LABEL_TEXT[label]
    lines = [f"#### Label `{label}` - {role}", "", f"Definition: {definition}.", ""]
    if "skipped" in sup:
        return lines + [f"Not fitted: {sup['skipped']}.", ""]
    ci = sup["coefficient_ci"]
    return lines + [
        f"* Observations: {sup['n_observations']} company-years from {sup['n_companies']} "
        f"companies; event rate {sup['positive_rate']:.1%}",
        "* Standardised coefficients (negative = higher score, lower risk) with "
        "company-bootstrap 95 % CI: " + "; ".join(
            f"{k} {v:+.3f} [{ci[k][0]:+.3f}, {ci[k][1]:+.3f}]"
            for k, v in sup["coefficients"].items()),
        f"* Out-of-fold AUC: **{sup['oof_auc']:.3f}** (95 % CI {sup['oof_auc_ci'][0]:.3f}-"
        f"{sup['oof_auc_ci'][1]:.3f})",
        "* AUC of simple composites on the same rows: " + ", ".join(
            f"{k} {v:.3f}" for k, v in sup["candidate_auc"].items()),
        f"* Model weights (negative coefficients, normalised): {sup['weights']}",
        f"* Accepted: **{sup['accepted']}**" + (
            "" if sup["accepted"] else " - reasons: " + "; ".join(sup["reasons"])),
        ""]


def render_report(result: CalibrationResult) -> str:
    lines = [
        "# Calibration report (generated)", "",
        f"Generated {result.generated_at} by `python main.py calibrate`.", "",
        f"* Reference universe: {result.universe_size} tickers "
        "(`config/reference_universe.json`: NIFTY 500 + S&P 500)",
        f"* Usable non-financial companies: **{len(result.companies)}**, company-years: "
        f"**{result.observations}**",
        f"* Excluded: {len(result.excluded)} (reasons in section 8)", "",
        "## 1. Ratio breakpoints = percentiles (universe and sector)", "",
        "Each ratio is scored by its percentile rank (P5..P95 -> 5..95 points; reversed for "
        "lower-is-better ratios), following rank-based scoring of fundamentals (Asness, "
        "Frazzini & Pedersen 2019). A sector with at least "
        f"{MIN_SECTOR_OBSERVATIONS} observations of a ratio gets its own curve, so companies "
        "are compared with sector peers (the approach of Tickertape's sector-relative score "
        "and Seeking Alpha's sector-relative factor grades); other companies use the "
        "universe curve.", "",
        "| Ratio | n | Median | Universe breakpoints [value, points] | Sector curves |",
        "|---|---:|---:|---|---|",
    ]
    for code, info in result.breakpoints.items():
        note = " (kept previous: too few observations)" if info["kept_previous"] else ""
        sectors = ", ".join(info.get("sector_curves", [])) or "none"
        lines.append(f"| {RATIO_DEFINITIONS[code].name}{note} | {info['n']} | "
                     f"{info.get('median', '')} | {_fmt(info['points'])} | {sectors} |")

    lines += ["", "## 2. Ratio weights inside each category (CRITIC)", "",
              "CRITIC (Diakoulaki, Mavrotas & Papayannakis 1995): weight proportional to the "
              "spread of a ratio's scores times its lack of correlation with the other ratios "
              "(less double counting).", "",
              "| Category | Method | Complete rows | Weights |", "|---|---|---:|---|"]
    for category, info in result.metric_weights.items():
        weights = ", ".join(f"{RATIO_DEFINITIONS[c].name} {w:.3f}"
                            for c, w in info["weights"].items())
        lines.append(f"| {category} | {info['method']} | {info['rows']} | {weights} |")

    cw = result.category_weights
    distress = result.supervised.get("distress", {})
    sup_w = distress.get("weights") or {}
    lines += ["", "## 3. Category weights", "",
              f"Company-years with all five category scores: {cw['rows']}.", "",
              "| Category | Equal | CRITIC | Entropy | Logistic (distress label) |",
              "|---|---:|---:|---:|---:|"]
    for name in CATEGORIES:
        lines.append(f"| {name} | {cw['candidates']['EQUAL'][name]:.3f} | "
                     f"{cw['candidates']['CRITIC'][name]:.3f} | "
                     f"{cw['candidates']['ENTROPY'][name]:.3f} | "
                     f"{sup_w.get(name, float('nan')):.3f} |")
    lines += ["", f"**Chosen: {cw['chosen']}.**", "",
              "### Supervised models (machine learning)", "",
              "Standardised logistic regression on the five category scores; 5-fold GroupKFold "
              "by company (a company never appears in both training and test folds); out-of-fold "
              "ROC AUC with a company-level bootstrap 95 % CI. Pre-declared acceptance rule: no "
              "coefficient significantly of the wrong sign (95 % CI entirely above 0), AUC CI "
              "excluding 0.5, and an AUC at least 0.02 above the best simple composite. Only the "
              "primary label may set the weights; it was fixed before the data was analysed.", ""]
    for label in ("distress", "earnings_decline"):
        if label in result.supervised:
            lines += _supervised_lines(label, result.supervised[label])

    lines += ["## 4. Interpretation bands = quartiles of the universe's overall scores", "",
              "| Label | Minimum score |", "|---|---:|"]
    lines += [f"| {b['label']} | {b['min']} |" for b in result.bands]

    rf = result.red_flag_config["rules"]
    ev = result.red_flags
    lines += [
        "", "## 5. Red-flag thresholds", "",
        "| Rule | Parameter | Value | Source |", "|---|---|---:|---|",
        f"| RF01 | high_severity_ocf_decline_pct | "
        f"{rf['RF01_PROFIT_CASH_DIVERGENCE']['high_severity_ocf_decline_pct']} | worst 10 % of "
        f"universe OCF changes (P10 = {ev.get('ocf_growth_p10')}%); rule: Sloan (1996) |",
        f"| RF02 | debt_growth_pct | {rf['RF02_DEBT_GROWTH']['debt_growth_pct']} | universe P75 |",
        f"| RF02 | high_severity_debt_growth_pct | "
        f"{rf['RF02_DEBT_GROWTH']['high_severity_debt_growth_pct']} | universe P90 |",
        f"| RF02 | revenue_flat_band_pct | {rf['RF02_DEBT_GROWTH']['revenue_flat_band_pct']} | "
        "universe P25 of revenue growth |",
        f"| RF03 | high_severity_decline_pct | "
        f"{rf['RF03_REVENUE_PROFIT_DECLINE']['high_severity_decline_pct']} | worst 10 % of "
        f"revenue changes (P10 = {ev.get('revenue_growth_p10')}%) |",
        f"| RF05 | current_ratio_threshold | "
        f"{rf['RF05_LIQUIDITY_CONCERN']['current_ratio_threshold']} | definitional (negative "
        "working capital) |",
        f"| RF05 | high_severity_below | {rf['RF05_LIQUIDITY_CONCERN']['high_severity_below']} "
        f"| universe P10 ({ev.get('current_ratio_p10')}) |",
        "| RF06 | interest_coverage_threshold / HIGH | 1.5 / 1.25 | Damodaran (Jan 2026): B- / "
        "CCC boundaries |",
        "| RF08 | Altman Z'' distress | < 1.10 | Altman, Hartzell & Peck (1995) |",
        "| RF09 | Beneish M | > -1.78 | Beneish (1999) |",
        "", "Damodaran large-firm table: " + ", ".join(
            f">{lo} {r}" for lo, r in DAMODARAN_COVERAGE[1:]) + ".", ""]

    val = result.diagnostics.get("validation", {})
    lines += ["## 6. Validation against independent published scores", "",
              "Spearman rank correlation between our overall score (latest year) and scores "
              "computed with the published formulas. A positive correlation with Piotroski, "
              "Altman and Simply Wall St and a negative one with Beneish (higher M = more "
              "manipulation risk) indicate agreement.", "",
              "| Benchmark | n | Spearman rho | p-value |", "|---|---:|---:|---:|"]
    for code in ("piotroski_f_score", "altman_z_score", "beneish_m_score", "sws_health_checks"):
        if code in val:
            v = val[code]
            lines.append(f"| {RATIO_DEFINITIONS[code].name} | {v['n']} | {v['spearman_rho']} | "
                         f"{v['p_value']} |")
    zones = val.get("mean_score_by_altman_zone", {})
    if zones:
        lines += ["", "Mean overall score by Altman Z'' zone: " + "; ".join(
            f"{k}: {z['mean_score']} (n={z['n']})" for k, z in zones.items()) + ".", ""]

    d = result.diagnostics
    inv = d["inventory_share_of_assets_pct"]
    cur = d["currency_check_ratio"]
    lines += [
        "## 7. Diagnostics for engineering rules", "",
        f"* **Sparse-period rule**: core-metric counts per stored period "
        f"{d['core_metric_count_per_period']}; periods classified sparse: "
        f"{d['sparse_periods']}.",
        f"* **Inventory materiality** (< 0.5 % of assets): {inv['n']} company-years; min "
        f"{inv['min']}, P5 {inv['p5']}, P10 {inv['p10']}, median {inv['median']}; "
        f"{inv['below_0_5_pct']} below 0.5 %, {inv['between_0_5_and_2_pct']} between 0.5 % "
        "and 2 %.",
        f"* **Currency plausibility band** (0.2-5x): {cur['n']} companies; min {cur['min']}, "
        f"max {cur['max']}; outside the band: {cur['outside_0_2_to_5']}.",
        "", "## 8. Excluded tickers", ""]
    lines += [f"* {t}: {reason[:160]}" for t, reason in sorted(result.excluded.items())]
    lines += ["", "## References", "",
              "* Altman, E. I., Hartzell, J. & Peck, M. (1995). Emerging markets corporate "
              "bonds: a scoring system. Salomon Brothers.",
              "* Asness, C., Frazzini, A. & Pedersen, L. H. (2019). Quality minus junk. Review "
              "of Accounting Studies, 24(1), 34-112.",
              "* Beneish, M. D. (1999). The detection of earnings manipulation. Financial "
              "Analysts Journal, 55(5), 24-36.",
              "* Damodaran, A. (2026, January). Ratings, interest coverage ratios and default "
              "spreads. NYU Stern.",
              "* Diakoulaki, D., Mavrotas, G. & Papayannakis, L. (1995). Determining objective "
              "weights in multiple criteria problems: The CRITIC method. Computers & Operations "
              "Research, 22(7), 763-770.",
              "* Ou, J. A. & Penman, S. H. (1989). Financial statement analysis and the "
              "prediction of stock returns. Journal of Accounting and Economics, 11(4), 295-329.",
              "* Piotroski, J. D. (2000). Value investing: The use of historical financial "
              "statement information to separate winners from losers. Journal of Accounting "
              "Research, 38 (Supplement), 1-41.",
              "* Simply Wall St. Company Analysis Model (MODEL.markdown), CC BY-NC-SA 4.0.",
              "* Sloan, R. G. (1996). Do stock prices fully reflect information in accruals and "
              "cash flows about future earnings? The Accounting Review, 71(3), 289-315.", ""]
    return "\n".join(lines)
