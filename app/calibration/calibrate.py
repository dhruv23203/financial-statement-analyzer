"""Calibrate the scoring and red-flag configuration from a reference universe.

Steps (each documented in the generated report):
1. Panel: every non-financial company-year of the universe with its 13 ratios,
   recomputed in memory from stored standardised data.
2. Breakpoints: each ratio's points = its percentile rank in the universe
   (Asness, Frazzini & Pedersen 2019 rank-based scoring).
3. Ratio weights inside each category: CRITIC (Diakoulaki et al. 1995).
4. Category weights: candidates equal / CRITIC / entropy / supervised logistic
   regression (Ou & Penman 1989 style label: next-year net income decline). The
   supervised weights are used only if they pass the pre-declared acceptance rule
   (see ``methods.supervised_weights``); otherwise CRITIC is used.
5. Interpretation bands: quartiles of the universe's overall scores.
6. Red-flag growth thresholds: percentiles of the universe's year-over-year changes;
   interest-coverage thresholds from Damodaran's synthetic rating table.
7. Diagnostics for engineering rules (sparse years, inventory materiality,
   currency plausibility band).
"""

from __future__ import annotations

import copy
import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from app.analysis.health_score import OVERALL, curve_for, interpolate, score_period
from app.analysis.ratios import RATIO_DEFINITIONS, FinancialDataset, RatioOutcome, \
    calculate_all_ratios, index_outcomes
from app.analysis.service import load_dataset
from app.calibration import methods
from app.cleaning.validator import is_sparse_period
from app.database import repositories as repo
from app.database.models import FREQ_ANNUAL, FetchRun
from app.services.context import AppContext
from app.utils.exceptions import AnalyzerError, ConfigError
from app.utils.helpers import utcnow
from config.settings import CORE_METRICS

logger = logging.getLogger(__name__)
CATEGORIES = ("LIQUIDITY", "SOLVENCY", "PROFITABILITY", "EFFICIENCY", "CASH_FLOW")
FINANCIAL_SECTORS = {"Financial Services"}
MIN_RATIO_OBSERVATIONS = 30
MIN_WEIGHT_ROWS = 30
MIN_SUPERVISED_ROWS = 60
SCORING_VERSION = "2.1.0"
MIN_SECTOR_OBSERVATIONS = 120  # ~30 companies x 4 years before a sector gets its own curve
DESCRIPTION = (
    "Data-calibrated Financial Health Score (python main.py calibrate). Each ratio's "
    "breakpoints are percentiles (P5-P95) of a reference universe (NIFTY 500 + S&P 500), "
    "computed within the company's sector when the sector has enough observations (sector-"
    "relative grading, as used by Tickertape and Seeking Alpha), so points approximate the "
    "company's percentile rank among its peers (Asness, Frazzini & Pedersen 2019). Ratio weights within a category and category weights are CRITIC "
    "objective weights (Diakoulaki et al. 1995); a supervised logistic model was tested and "
    "used only if it passed a pre-declared acceptance rule. Interpretation bands are the "
    "universe quartiles of the overall score, so labels describe relative position among "
    "large listed companies. Missing data never receives points; it lowers data coverage. "
    "Academic indicator only - not a credit rating or investment advice. Full evidence: "
    "docs/calibration/CALIBRATION_REPORT.md")
RED_FLAG_VERSION = "2.2.0"

# Damodaran, A. (January 2026). Ratings, interest coverage ratios and default spreads,
# large non-financial firms. pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/ratings.html
DAMODARAN_COVERAGE = [  # (coverage lower bound, rating)
    (-100000, "D"), (0.2, "C"), (0.65, "CC"), (0.8, "CCC"), (1.25, "B-"), (1.5, "B"),
    (1.75, "B+"), (2.0, "BB"), (2.25, "BB+"), (2.5, "BBB"), (3.0, "A-"), (4.25, "A"),
    (5.5, "A+"), (6.5, "AA"), (8.5, "AAA"),
]


@dataclass
class CompanyData:
    ticker: str
    sector: str | None
    dataset: FinancialDataset
    ratios: dict[tuple[str, date], RatioOutcome]


@dataclass
class CalibrationResult:
    universe_size: int
    companies: list[str]
    excluded: dict[str, str]
    observations: int
    breakpoints: dict[str, dict[str, Any]]
    metric_weights: dict[str, dict[str, Any]]
    category_weights: dict[str, Any]
    supervised: dict[str, Any]
    bands: list[dict[str, Any]]
    red_flags: dict[str, Any]
    diagnostics: dict[str, Any]
    scoring_config: dict[str, Any]
    red_flag_config: dict[str, Any]
    generated_at: str = field(default_factory=lambda: utcnow().isoformat())


# ---------------------------------------------------------------------- panel
def build_panel(ctx: AppContext, tickers: list[str]) -> tuple[list[CompanyData], dict[str, str]]:
    companies, excluded = [], {}
    for ticker in tickers:
        try:
            dataset, info = load_dataset(ctx, ticker)
        except AnalyzerError as exc:
            excluded[ticker] = str(exc)
            continue
        if info.get("sector") in FINANCIAL_SECTORS:
            excluded[ticker] = "financial sector (different ratio model)"
            continue
        if (info.get("currency_check") or {}).get("status") == "FAILED":
            excluded[ticker] = "currency label failed plausibility check"
            continue
        ratios = calculate_all_ratios(dataset, ctx.settings.average_balance_fallback)
        companies.append(CompanyData(ticker, info.get("sector"), dataset,
                                     index_outcomes(ratios)))
    return companies, excluded


def _ratio_values(companies: list[CompanyData], code: str) -> list[float]:
    return [float(o.value) for c in companies for (k, _), o in c.ratios.items()
            if k == code and o.ok]


def _pct_changes(companies: list[CompanyData], code: str) -> list[float]:
    changes = []
    for c in companies:
        for period in c.dataset.periods:
            previous = c.dataset.adjacent_previous(period)
            cur, prev = c.dataset.get(period, code), c.dataset.get(previous, code)
            if cur is not None and prev is not None and prev > 0:
                changes.append(float((cur - prev) / prev * 100))
    return changes


def _q(values: list[float], p: float) -> float:
    return float(round(np.percentile(values, p), 2))


# ---------------------------------------------------------------------- calibration
def calibrate(ctx: AppContext, tickers: list[str], base_scoring: dict[str, Any],
              base_red_flags: dict[str, Any]) -> CalibrationResult:
    companies, excluded = build_panel(ctx, tickers)
    if len(companies) < 20:
        raise ConfigError(f"Only {len(companies)} usable companies; at least 20 are needed "
                          "for a meaningful calibration. Run: python main.py calibrate --fetch")
    scoring = copy.deepcopy(base_scoring)
    observations = sum(len(c.dataset.periods) for c in companies)

    # 2) breakpoints ------------------------------------------------------------
    breakpoints: dict[str, dict[str, Any]] = {}
    for category in scoring["categories"].values():
        for code, spec in category["metrics"].items():
            values = _ratio_values(companies, code)
            higher = RATIO_DEFINITIONS[code].higher_is_better
            if len(values) < MIN_RATIO_OBSERVATIONS:
                breakpoints[code] = {"n": len(values), "kept_previous": True,
                                     "points": spec["points"]}
                continue
            spec["points"] = methods.percentile_breakpoints(values, higher)
            spec["higher_is_better"] = higher
            spec.pop("note", None)  # drop "provisional" notes once calibrated
            sector_points: dict[str, list[list[float]]] = {}
            sector_n: dict[str, int] = {}
            for sector in sorted({c.sector for c in companies if c.sector}):
                sector_values = _ratio_values([c for c in companies if c.sector == sector], code)
                sector_n[sector] = len(sector_values)
                if len(sector_values) >= MIN_SECTOR_OBSERVATIONS:
                    try:
                        sector_points[sector] = methods.percentile_breakpoints(sector_values,
                                                                               higher)
                    except ValueError:
                        continue
            if sector_points:
                spec["sector_points"] = sector_points
            else:
                spec.pop("sector_points", None)
            breakpoints[code] = {"n": len(values), "kept_previous": False,
                                 "points": spec["points"], "median": _q(values, 50),
                                 "sector_curves": sorted(sector_points),
                                 "sector_n": sector_n}

    # 3) ratio weights inside each category (CRITIC on ratio points) --------------
    metric_weights: dict[str, dict[str, Any]] = {}
    for name, category in scoring["categories"].items():
        codes = list(category["metrics"])
        rows = []
        for c in companies:
            for period in c.dataset.periods:
                outcomes = [c.ratios.get((code, period)) for code in codes]
                if all(o is not None and o.ok for o in outcomes):
                    rows.append([interpolate(float(o.value),
                                             curve_for(category["metrics"][code], c.sector)[0])
                                 for code, o in zip(codes, outcomes)])
        if len(rows) >= MIN_WEIGHT_ROWS:
            weights = methods.critic_weights(np.array(rows))
            method = "CRITIC"
        else:
            weights = np.full(len(codes), 1.0 / len(codes))
            method = f"EQUAL (only {len(rows)} complete rows < {MIN_WEIGHT_ROWS})"
        for code, weight in zip(codes, weights):
            category["metrics"][code]["weight"] = round(float(weight), 4)
        metric_weights[name] = {"method": method, "rows": len(rows),
                                "weights": {c: round(float(w), 4) for c, w in zip(codes, weights)}}

    # 4) category weights ------------------------------------------------------------
    for name in CATEGORIES:
        scoring["categories"][name]["weight"] = 0.2  # neutral while computing category scores
    category_rows = []
    label_rows: dict[str, list[tuple[list[float], int, str]]] = {"distress": [],
                                                                 "earnings_decline": []}
    for c in companies:
        for period in c.dataset.periods:
            cats = {s.category: s.score for s in score_period(period, c.ratios, scoring,
                                                              c.sector)}
            if not all(cats.get(k) is not None for k in CATEGORIES):
                continue
            vector = [float(cats[k]) for k in CATEGORIES]
            category_rows.append(vector)
            following = [p for p in c.dataset.periods if c.dataset.adjacent_previous(p) == period]
            if not following:
                continue
            nxt = following[0]
            ni, ni_next = c.dataset.get(period, "net_income"), c.dataset.get(nxt, "net_income")
            ocf_next = c.dataset.get(nxt, "operating_cash_flow")
            # PRIMARY (pre-declared): next-year distress proxy = net loss or negative OCF.
            if ni_next is not None and ocf_next is not None:
                label_rows["distress"].append((vector, int(ni_next < 0 or ocf_next < 0),
                                               c.ticker))
            # SECONDARY (reference only): next-year earnings decline (Ou & Penman 1989).
            if ni is not None and ni_next is not None:
                label_rows["earnings_decline"].append((vector, int(ni_next < ni), c.ticker))
    if len(category_rows) < MIN_WEIGHT_ROWS:
        raise ConfigError(f"Only {len(category_rows)} company-years with all five categories.")
    matrix = np.array(category_rows)
    candidates = {"EQUAL": np.full(5, 0.2), "CRITIC": methods.critic_weights(matrix),
                  "ENTROPY": methods.entropy_weights(matrix)}
    supervised: dict[str, Any] = {}
    chosen_name, chosen = "CRITIC", candidates["CRITIC"]
    for label, rows in label_rows.items():
        events = sum(r[1] for r in rows)
        if len(rows) < MIN_SUPERVISED_ROWS or events < 20 or events > len(rows) - 20:
            supervised[label] = {"skipped": f"{len(rows)} rows, {events} events (need >= "
                                            f"{MIN_SUPERVISED_ROWS} rows and >= 20 per class)"}
            continue
        result = methods.supervised_weights(
            np.array([r[0] for r in rows]), np.array([r[1] for r in rows]),
            np.array([r[2] for r in rows]), list(CATEGORIES), candidates)
        supervised[label] = {k: v for k, v in vars(result).items()}
        # Only the pre-declared primary (distress) label may set the weights.
        if label == "distress" and result.accepted and result.weights:
            chosen_name = "SUPERVISED_LOGISTIC (distress label)"
            chosen = np.array([result.weights[k] for k in CATEGORIES])
    for name, weight in zip(CATEGORIES, chosen):
        scoring["categories"][name]["weight"] = round(float(weight), 4)
    category_weights = {"chosen": chosen_name, "rows": len(category_rows),
                        "candidates": {k: dict(zip(CATEGORIES, np.round(v, 4).tolist()))
                                       for k, v in candidates.items()}}

    # 5) interpretation bands from the universe's overall scores ----------------------
    overall_scores = []
    for c in companies:
        latest = c.dataset.periods[-1]
        overall = next(s for s in score_period(latest, c.ratios, scoring, c.sector)
                       if s.category == OVERALL)
        if overall.score is not None:
            overall_scores.append(float(overall.score))
    # Scores are percentile-based, so labels describe the position in the universe.
    bands = [{"min": _q(overall_scores, 75), "label": "TOP QUARTILE"},
             {"min": _q(overall_scores, 50), "label": "ABOVE MEDIAN"},
             {"min": _q(overall_scores, 25), "label": "BELOW MEDIAN"},
             {"min": 0, "label": "BOTTOM QUARTILE"}]
    scoring["interpretation_bands"] = bands
    scoring["grade_bands"] = _grade_bands(companies, scoring)
    scoring["grade_bands_note"] = (
        "Letter grades follow Morningstar's documented Financial Health grade distribution "
        "(top 10% A, next 20% B, middle 40% C, next 20% D, bottom 10% F); cut-offs are the "
        "universe P90/P70/P30/P10 of each category score.")
    scoring["methodology_version"] = SCORING_VERSION
    scoring["description"] = DESCRIPTION

    # 6) red-flag thresholds --------------------------------------------------------
    red_flags, red_flag_evidence = _calibrate_red_flags(companies, base_red_flags)

    # 7) diagnostics ---------------------------------------------------------------
    diagnostics = _diagnostics(ctx, companies, tickers)
    diagnostics["validation"] = _validate_against_benchmarks(companies, scoring)
    scoring["calibration"] = {
        "generated_at": utcnow().isoformat(), "companies": len(companies),
        "company_years": observations, "category_weight_method": chosen_name,
        "breakpoint_method": "universe percentiles P5-P95 (rank scoring)",
        "ratio_weight_method": "CRITIC within category",
        "report": "docs/calibration/CALIBRATION_REPORT.md",
    }
    return CalibrationResult(len(tickers), [c.ticker for c in companies], excluded, observations,
                             breakpoints, metric_weights, category_weights, supervised, bands,
                             red_flag_evidence, diagnostics, scoring, red_flags)


def _grade_bands(companies: list[CompanyData],
                 scoring: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Per-category letter-grade cut-offs at universe percentiles P90/P70/P30/P10."""
    from app.analysis.insights import GRADE_LEVELS

    scores: dict[str, list[float]] = {}
    for c in companies:
        for s in score_period(c.dataset.periods[-1], c.ratios, scoring, c.sector):
            if s.score is not None:
                scores.setdefault(s.category, []).append(float(s.score))
    return {category: [{"grade": grade, "min": _q(values, pct)} for grade, pct in GRADE_LEVELS]
            for category, values in scores.items() if len(values) >= MIN_WEIGHT_ROWS}


def _calibrate_red_flags(companies: list[CompanyData],
                         base: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    config = copy.deepcopy(base)
    rules = config["rules"]
    revenue = _pct_changes(companies, "total_revenue")
    debt = _pct_changes(companies, "total_debt")
    ocf = _pct_changes(companies, "operating_cash_flow")
    current = _ratio_values(companies, "current_ratio")
    evidence = {"n_revenue_changes": len(revenue), "n_debt_changes": len(debt),
                "n_ocf_changes": len(ocf), "n_current_ratios": len(current)}

    # RF01: any profit rise with any OCF fall (Sloan 1996: accrual-driven earnings are less
    # persistent); HIGH when the OCF fall is in the worst 10 % of the universe.
    if ocf:
        rules["RF01_PROFIT_CASH_DIVERGENCE"]["high_severity_ocf_decline_pct"] = \
            max(0.0, -_q(ocf, 10))
        evidence["ocf_growth_p10"] = _q(ocf, 10)
    # RF02: debt growth in the top quartile while revenue growth is in the bottom quartile.
    if debt and revenue:
        rules["RF02_DEBT_GROWTH"]["debt_growth_pct"] = _q(debt, 75)
        rules["RF02_DEBT_GROWTH"]["high_severity_debt_growth_pct"] = _q(debt, 90)
        rules["RF02_DEBT_GROWTH"]["revenue_flat_band_pct"] = _q(revenue, 25)
        evidence.update(debt_growth_p75=_q(debt, 75), debt_growth_p90=_q(debt, 90),
                        revenue_growth_p25=_q(revenue, 25))
    # RF03: HIGH when the revenue fall is in the worst 10 % of the universe.
    if revenue:
        evidence["revenue_growth_p10"] = _q(revenue, 10)
        rules["RF03_REVENUE_PROFIT_DECLINE"]["high_severity_decline_pct"] = \
            max(0.0, -_q(revenue, 10)) if _q(revenue, 10) < 0 else \
            rules["RF03_REVENUE_PROFIT_DECLINE"]["high_severity_decline_pct"]
    # RF05: current ratio < 1 means current liabilities exceed current assets (negative
    # working capital) - a definitional boundary; HIGH below the universe's 10th percentile.
    if current:
        evidence["current_ratio_p10"] = _q(current, 10)
        rules["RF05_LIQUIDITY_CONCERN"]["current_ratio_threshold"] = 1.0
        rules["RF05_LIQUIDITY_CONCERN"]["high_severity_below"] = min(1.0, _q(current, 10))
    # RF06: Damodaran (Jan 2026) large-firm synthetic ratings: coverage < 1.5 maps to B- or
    # worse ("highly speculative"); < 1.25 maps to CCC or worse ("substantial risk").
    rules["RF06_WEAK_INTEREST_COVERAGE"]["interest_coverage_threshold"] = 1.5
    rules["RF06_WEAK_INTEREST_COVERAGE"]["high_severity_below"] = 1.25
    config["version"] = RED_FLAG_VERSION
    config["calibration"] = {"generated_at": utcnow().isoformat(),
                             "companies": len(companies),
                             "report": "docs/calibration/CALIBRATION_REPORT.md"}
    return config, evidence


def _validate_against_benchmarks(companies: list[CompanyData],
                                 scoring: dict[str, Any]) -> dict[str, Any]:
    """Rank correlation of our overall score with independent published scores."""
    from scipy.stats import spearmanr

    codes = ("piotroski_f_score", "altman_z_score", "beneish_m_score", "sws_health_checks")
    rows = []
    for c in companies:
        latest = c.dataset.periods[-1]
        overall = next(s for s in score_period(latest, c.ratios, scoring, c.sector)
                       if s.category == OVERALL)
        if overall.score is None:
            continue
        row: dict[str, float | None] = {"score": float(overall.score)}
        for code in codes:
            outcome = c.ratios.get((code, latest))
            row[code] = float(outcome.value) if outcome is not None and outcome.ok else None
        rows.append(row)
    result: dict[str, Any] = {"companies": len(rows)}
    for code in codes:
        pairs = [(r["score"], r[code]) for r in rows if r[code] is not None]
        if len(pairs) >= 10:
            rho, p_value = spearmanr([a for a, _ in pairs], [b for _, b in pairs])
            result[code] = {"n": len(pairs), "spearman_rho": round(float(rho), 3),
                            "p_value": float(f"{p_value:.3g}")}
    zones: dict[str, list[float]] = {"distress (<1.10)": [], "grey (1.10-2.60)": [],
                                     "safe (>2.60)": []}
    for r in rows:
        z = r["altman_z_score"]
        if z is not None:
            key = "distress (<1.10)" if z < 1.10 else "safe (>2.60)" if z > 2.60 \
                else "grey (1.10-2.60)"
            zones[key].append(r["score"])
    result["mean_score_by_altman_zone"] = {
        k: {"n": len(v), "mean_score": round(float(np.mean(v)), 1) if v else None}
        for k, v in zones.items()}
    return result


def _diagnostics(ctx: AppContext, companies: list[CompanyData],
                 tickers: list[str]) -> dict[str, Any]:
    shares = []
    for c in companies:
        for period in c.dataset.periods:
            inv, assets = c.dataset.get(period, "inventory"), c.dataset.get(period, "total_assets")
            if inv is not None and assets and assets > 0:
                shares.append(float(inv / assets * 100))
    core_counts: dict[int, int] = {}
    sparse_periods = 0
    currency_ratios = []
    with ctx.db.session_scope() as session:
        for ticker in tickers:
            company = repo.get_company(session, ticker)
            run = repo.latest_cleaned_fetch_run(session, company.id) if company else None
            if run is None:
                continue
            present: dict[date, set[str]] = {}
            for fact in repo.standardized_facts_for_run(session, run.id, FREQ_ANNUAL,
                                                        mapped_only=True):
                if fact.value is not None:
                    present.setdefault(fact.fiscal_period_end, set()).add(fact.metric_code)
            for codes in present.values():
                count = len(codes & set(CORE_METRICS))
                core_counts[count] = core_counts.get(count, 0) + 1
                sparse_periods += is_sparse_period(codes)
            fetch = session.get(FetchRun, run.id)
            check = (fetch.fetch_metadata or {}).get("currency_check") or {}
            if check.get("ratio"):
                currency_ratios.append(float(check["ratio"]))
    shares.sort()
    return {
        "inventory_share_of_assets_pct": {
            "n": len(shares), "min": round(shares[0], 4) if shares else None,
            "p5": _q(shares, 5) if shares else None, "p10": _q(shares, 10) if shares else None,
            "median": _q(shares, 50) if shares else None,
            "below_0_5_pct": sum(1 for s in shares if s < 0.5),
            "between_0_5_and_2_pct": sum(1 for s in shares if 0.5 <= s < 2)},
        "core_metric_count_per_period": dict(sorted(core_counts.items())),
        "sparse_periods": sparse_periods,
        "currency_check_ratio": {
            "n": len(currency_ratios),
            "min": round(min(currency_ratios), 3) if currency_ratios else None,
            "max": round(max(currency_ratios), 3) if currency_ratios else None,
            "outside_0_2_to_5": sorted(round(r, 2) for r in currency_ratios
                                       if not 0.2 <= r <= 5)},
    }


# ---------------------------------------------------------------------- outputs
def write_outputs(result: CalibrationResult, output_dir: Path, apply: bool,
                  scoring_path: Path, red_flag_path: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "results": output_dir / "calibration_results.json",
        "proposed_scoring": output_dir / "proposed_scoring_config.json",
        "proposed_red_flags": output_dir / "proposed_red_flag_thresholds.json",
        "report": output_dir / "CALIBRATION_REPORT.md",
    }
    serialisable = {k: v for k, v in vars(result).items()
                    if k not in ("scoring_config", "red_flag_config")}
    outputs["results"].write_text(json.dumps(serialisable, indent=2, default=str),
                                  encoding="utf-8")
    outputs["proposed_scoring"].write_text(json.dumps(result.scoring_config, indent=2),
                                           encoding="utf-8")
    outputs["proposed_red_flags"].write_text(json.dumps(result.red_flag_config, indent=2),
                                             encoding="utf-8")
    from app.calibration.report import render_report

    outputs["report"].write_text(render_report(result), encoding="utf-8")
    if apply:
        archive = scoring_path.parent / "archive"
        archive.mkdir(exist_ok=True)
        for path in (scoring_path, red_flag_path):
            if path.exists():
                old = json.loads(path.read_text(encoding="utf-8"))
                version = old.get("methodology_version") or old.get("version") or "old"
                shutil.copy2(path, archive / f"{path.stem}_v{version}.json")
        shutil.copy2(outputs["proposed_scoring"], scoring_path)
        shutil.copy2(outputs["proposed_red_flags"], red_flag_path)
    return {k: v.as_posix() for k, v in outputs.items()}
