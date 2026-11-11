"""Day 70 - explainable Financial Health Score.

Method (all numbers from ``config/scoring_config.json``)
-------------------------------------------------------
1. Ratio -> points (0-100) by linear interpolation between configured
   ``[ratio value, points]`` breakpoints. Breakpoints encode the direction: for
   "lower is better" ratios (debt-to-equity, debt ratio) points fall as the ratio
   rises. Values outside the range get the nearest end-point score.
2. Negative equity: debt-to-equity and ROE are NOT_MEANINGFUL; when the metric has
   ``negative_equity_points`` (default 0) that score is applied because negative
   equity is itself a solvency weakness. Any other unavailable ratio scores
   nothing - it is excluded and lowers *data coverage*.
3. Category score = weighted average of the available ratio points;
   coverage = available weight / total weight. A category with coverage below
   ``min_category_coverage`` is UNAVAILABLE (no score is invented).
4. Overall score = weighted average of the scored categories (weights
   re-normalised). Overall coverage = sum(category weight x category coverage) /
   sum(category weights). At least ``min_categories_for_overall`` scored
   categories are required.

The score is an academic indicator of historical financial health, not a credit
rating and not an investment recommendation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Any

from app.analysis.ratios import NON_POSITIVE_EQUITY, RATIO_DEFINITIONS, RatioOutcome
from app.database.models import SCORE_SCORED, SCORE_UNAVAILABLE

OVERALL = "OVERALL"
NOT_APPLICABLE = "NOT_APPLICABLE"
# Ratio statuses meaning "does not apply to this business" rather than "data missing".
NOT_APPLICABLE_REASONS = frozenset({"IMMATERIAL_INVENTORY"})
SCORE_PLACES = Decimal("0.01")


def interpolate(value: float, points: list[list[float]]) -> float:
    """Piecewise-linear score for *value*; clamps outside the breakpoint range."""
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    if value <= xs[0]:
        return ys[0]
    if value >= xs[-1]:
        return ys[-1]
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        if x0 <= value <= x1:
            return y0 + (y1 - y0) * (value - x0) / (x1 - x0)
    raise AssertionError("unreachable: value inside range but no segment matched")


@dataclass
class MetricScore:
    ratio_code: str
    weight: float
    value: Decimal | None
    points: float | None
    status: str  # SCORED | UNAVAILABLE
    note: str


@dataclass
class CategoryScore:
    category: str
    period: date
    score: Decimal | None
    status: str
    coverage: float
    explanation: str
    components: list[MetricScore] = field(default_factory=list)
    interpretation: str | None = None


def interpretation_for(score: Decimal | None, bands: list[dict[str, Any]]) -> str | None:
    if score is None:
        return None
    for band in bands:
        if float(score) >= float(band["min"]):
            return str(band["label"])
    return str(bands[-1]["label"])


def curve_for(metric: dict[str, Any], sector: str | None) -> tuple[list[list[float]], str]:
    """Sector-specific breakpoints when calibrated for the sector, else the universe curve."""
    sector_points = metric.get("sector_points") or {}
    if sector and sector in sector_points:
        return sector_points[sector], f"vs {sector} peers"
    return metric["points"], "vs all companies"


def score_category(category: str, spec: dict[str, Any], period: date,
                   ratios: dict[tuple[str, date], RatioOutcome],
                   min_coverage: float, sector: str | None = None) -> CategoryScore:
    components: list[MetricScore] = []
    for code, metric in spec["metrics"].items():
        outcome = ratios.get((code, period))
        weight = float(metric["weight"])
        name = RATIO_DEFINITIONS[code].name
        if outcome is not None and outcome.ok:
            curve, basis = curve_for(metric, sector)
            points = interpolate(float(outcome.value), curve)
            direction = "higher is better" if metric.get("higher_is_better", True) \
                else "lower is better"
            components.append(MetricScore(code, weight, outcome.value, points, SCORE_SCORED,
                                          f"{name} = {float(outcome.value):.4g} -> "
                                          f"{points:.1f} pts ({direction}, {basis})"))
        elif outcome is not None and outcome.reason_code == NON_POSITIVE_EQUITY and \
                "negative_equity_points" in metric:
            points = float(metric["negative_equity_points"])
            components.append(MetricScore(code, weight, None, points, SCORE_SCORED,
                                          f"{name} not meaningful (equity <= 0) -> {points:.0f}"
                                          " pts (negative-equity rule)"))
        elif outcome is not None and outcome.reason_code in NOT_APPLICABLE_REASONS:
            # The ratio does not apply to this business model; it is not missing data,
            # so it is removed from the denominator instead of lowering coverage.
            components.append(MetricScore(code, 0.0, None, None, NOT_APPLICABLE,
                                          f"{name} not applicable ({outcome.explanation}) - "
                                          "weight removed"))
        else:
            reason = outcome.explanation if outcome else "not calculated"
            components.append(MetricScore(code, weight, None, None, SCORE_UNAVAILABLE,
                                          f"{name} unavailable ({reason}) - excluded"))
    total_weight = sum(c.weight for c in components)
    scored = [c for c in components if c.points is not None]
    applicable = [c for c in components if c.status != NOT_APPLICABLE]
    scored_weight = sum(c.weight for c in scored)
    coverage = scored_weight / total_weight if total_weight else 0.0
    # Enough data = at least half of the weight OR at least half of the applicable ratios.
    # The count condition keeps two-ratio categories stable when data-derived weights are
    # close to 50/50 (e.g. 0.498 vs 0.502).
    enough = coverage >= min_coverage or 2 * len(scored) >= len(applicable)
    if not scored or not enough or scored_weight <= 0:
        return CategoryScore(category, period, None, SCORE_UNAVAILABLE, round(coverage, 4),
                             f"Insufficient data: coverage {coverage:.0%} < required "
                             f"{min_coverage:.0%}. " + "; ".join(c.note for c in components),
                             components)
    value = sum(c.points * c.weight for c in scored) / scored_weight
    score = Decimal(str(value)).quantize(SCORE_PLACES)
    return CategoryScore(category, period, score, SCORE_SCORED, round(coverage, 4),
                         f"Weighted average of {len(scored)}/{len(components)} ratios "
                         f"(coverage {coverage:.0%}): " + "; ".join(c.note for c in components),
                         components)


def excluded_categories(config: dict[str, Any], sector: str | None) -> list[str]:
    """Categories configured as not applicable to *sector* (e.g. banks)."""
    if not sector:
        return []
    return list(config.get("sector_exclusions", {}).get(sector, []))


def score_period(period: date, ratios: dict[tuple[str, date], RatioOutcome],
                 config: dict[str, Any], sector: str | None = None) -> list[CategoryScore]:
    """All five category scores plus the OVERALL score for one fiscal period."""
    min_coverage = float(config["min_category_coverage"])
    bands = config["interpretation_bands"]
    not_applicable = excluded_categories(config, sector)
    categories = []
    for name, spec in config["categories"].items():
        if name in not_applicable:
            categories.append(CategoryScore(
                name, period, None, SCORE_UNAVAILABLE, 0.0,
                f"Not applicable to {sector} companies (configured sector exclusion); "
                "ratios are still reported but not scored."))
            continue
        category = score_category(name, spec, period, ratios, min_coverage, sector)
        category.interpretation = interpretation_for(category.score, bands)
        categories.append(category)

    weights = {name: float(spec["weight"]) for name, spec in config["categories"].items()
               if name not in not_applicable}
    total_weight = sum(weights.values())
    applicable = [c for c in categories if c.category in weights]
    # Zero-weight categories are still shown but do not count towards the overall score.
    scored = [c for c in applicable if c.score is not None and weights[c.category] > 0]
    coverage = sum(weights[c.category] * c.coverage for c in applicable) / total_weight
    minimum = int(config["min_categories_for_overall"])
    parts = "; ".join(
        f"{c.category} {c.score if c.score is not None else 'N/A'} (weight {weights[c.category]})"
        for c in applicable)
    if not_applicable:
        parts += f"; not applicable to {sector}: {', '.join(not_applicable)}"
    if len(scored) < minimum:
        overall = CategoryScore(OVERALL, period, None, SCORE_UNAVAILABLE, round(coverage, 4),
                                f"Only {len(scored)} of {len(categories)} categories could be "
                                f"scored (minimum {minimum}). {parts}")
    else:
        scored_weight = sum(weights[c.category] for c in scored)
        value = sum(float(c.score) * weights[c.category] for c in scored) / scored_weight
        overall_score = Decimal(str(value)).quantize(SCORE_PLACES)
        missing = [c.category for c in applicable if c.score is None]
        note = f" Unscored categories excluded and weights re-normalised: {missing}." \
            if missing else ""
        overall = CategoryScore(OVERALL, period, overall_score, SCORE_SCORED, round(coverage, 4),
                                f"Weighted average of {len(scored)} categories: {parts}.{note}")
    overall.interpretation = interpretation_for(overall.score, bands)
    return categories + [overall]


def score_all_periods(periods: list[date], ratios: dict[tuple[str, date], RatioOutcome],
                      config: dict[str, Any],
                      sector: str | None = None) -> dict[date, list[CategoryScore]]:
    return {period: score_period(period, ratios, config, sector) for period in periods}
