"""Load and validate the JSON rule/scoring configuration files.

Invalid configuration is reported with :class:`ConfigError` before any analysis
runs, so a typo in a threshold can never silently change results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.utils.exceptions import ConfigError

SEVERITIES = ("LOW", "MEDIUM", "HIGH")
SCORE_CATEGORIES = ("LIQUIDITY", "SOLVENCY", "PROFITABILITY", "EFFICIENCY", "CASH_FLOW")

RED_FLAG_REQUIRED: dict[str, tuple[str, ...]] = {
    "RF01_PROFIT_CASH_DIVERGENCE": ("min_net_income_growth_pct", "min_ocf_decline_pct",
                                    "high_severity_ocf_decline_pct"),
    "RF02_DEBT_GROWTH": ("debt_growth_pct", "revenue_flat_band_pct",
                         "high_severity_debt_growth_pct"),
    "RF03_REVENUE_PROFIT_DECLINE": ("high_severity_decline_pct",),
    "RF04_NEGATIVE_OPERATING_CASH_FLOW": ("consecutive_years_for_high",),
    "RF05_LIQUIDITY_CONCERN": ("current_ratio_threshold", "high_severity_below"),
    "RF06_WEAK_INTEREST_COVERAGE": ("interest_coverage_threshold", "high_severity_below"),
    "RF07_NEGATIVE_EQUITY": (),
}
# Optional rules (added in v2.1.0): validated only when present in the file.
OPTIONAL_RED_FLAGS: dict[str, tuple[str, ...]] = {
    "RF08_ALTMAN_DISTRESS": ("distress_below",),
    "RF09_BENEISH_MANIPULATION": ("threshold",),
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Configuration file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a JSON object")
    return data


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{where} must be a number, got {value!r}")
    return float(value)


def load_red_flag_config(path: Path) -> dict[str, Any]:
    data = _read_json(path)
    rules = data.get("rules")
    if not isinstance(rules, dict):
        raise ConfigError(f"{path}: 'rules' object missing")
    optional = {c: k for c, k in OPTIONAL_RED_FLAGS.items() if c in rules}
    for code, required in {**RED_FLAG_REQUIRED, **optional}.items():
        rule = rules.get(code)
        if not isinstance(rule, dict):
            raise ConfigError(f"{path}: rule {code} missing")
        if rule.get("default_severity") not in SEVERITIES:
            raise ConfigError(f"{path}: {code}.default_severity must be one of {SEVERITIES}")
        for key in required:
            _number(rule.get(key), f"{path}: {code}.{key}")
        rule.setdefault("enabled", True)
        rule.setdefault("name", code)
    data.setdefault("version", "unversioned")
    return data


def load_scoring_config(path: Path) -> dict[str, Any]:
    from app.analysis.ratios import RATIO_DEFINITIONS

    data = _read_json(path)
    if not isinstance(data.get("methodology_version"), str):
        raise ConfigError(f"{path}: methodology_version (string) is required")
    categories = data.get("categories")
    if not isinstance(categories, dict) or set(categories) != set(SCORE_CATEGORIES):
        raise ConfigError(f"{path}: categories must be exactly {SCORE_CATEGORIES}")
    coverage = _number(data.get("min_category_coverage"), f"{path}: min_category_coverage")
    if not 0 < coverage <= 1:
        raise ConfigError(f"{path}: min_category_coverage must be in (0, 1]")
    minimum = data.get("min_categories_for_overall")
    if not isinstance(minimum, int) or not 1 <= minimum <= len(SCORE_CATEGORIES):
        raise ConfigError(f"{path}: min_categories_for_overall must be 1..5")
    # Weights may be 0 (a data-driven method can find that a category or ratio adds no
    # information - the category is then still shown but does not enter the overall score),
    # but they may not be negative and each set of weights must have a positive total.
    category_total = 0.0
    for name, category in categories.items():
        weight = _number(category.get("weight"), f"{path}: {name}.weight")
        if weight < 0:
            raise ConfigError(f"{path}: {name}.weight must not be negative")
        category_total += weight
        metrics = category.get("metrics")
        if not isinstance(metrics, dict) or not metrics:
            raise ConfigError(f"{path}: {name}.metrics must be a non-empty object")
        metric_total = 0.0
        for code, spec in metrics.items():
            where = f"{path}: {name}.{code}"
            if code not in RATIO_DEFINITIONS:
                raise ConfigError(f"{where}: unknown ratio code")
            if RATIO_DEFINITIONS[code].category != name:
                raise ConfigError(f"{where}: ratio belongs to "
                                  f"{RATIO_DEFINITIONS[code].category}")
            metric_weight = _number(spec.get("weight"), f"{where}.weight")
            if metric_weight < 0:
                raise ConfigError(f"{where}.weight must not be negative")
            metric_total += metric_weight
            points = spec.get("points")
            if not isinstance(points, list) or len(points) < 2:
                raise ConfigError(f"{where}.points needs at least two [value, score] pairs")
            xs = [_number(p[0], f"{where}.points") for p in points]
            ys = [_number(p[1], f"{where}.points") for p in points]
            if any(b <= a for a, b in zip(xs, xs[1:])):
                raise ConfigError(f"{where}.points values must be strictly increasing")
            if any(not 0 <= y <= 100 for y in ys):
                raise ConfigError(f"{where}.points scores must be within 0-100")
            for sector, curve in (spec.get("sector_points") or {}).items():
                if not isinstance(curve, list) or len(curve) < 2:
                    raise ConfigError(f"{where}.sector_points.{sector} needs >= 2 breakpoints")
                sxs = [_number(p[0], f"{where}.sector_points") for p in curve]
                if any(b <= a for a, b in zip(sxs, sxs[1:])):
                    raise ConfigError(f"{where}.sector_points.{sector} must be increasing")
            if "negative_equity_points" in spec:
                nep = _number(spec["negative_equity_points"], f"{where}.negative_equity_points")
                if not 0 <= nep <= 100:
                    raise ConfigError(f"{where}.negative_equity_points must be within 0-100")
        if metric_total <= 0:
            raise ConfigError(f"{path}: {name} ratio weights must have a positive total")
    if category_total <= 0:
        raise ConfigError(f"{path}: category weights must have a positive total")
    exclusions = data.setdefault("sector_exclusions", {})
    if not isinstance(exclusions, dict):
        raise ConfigError(f"{path}: sector_exclusions must be an object")
    for sector, excluded in exclusions.items():
        if not isinstance(excluded, list) or not set(excluded) <= set(SCORE_CATEGORIES):
            raise ConfigError(f"{path}: sector_exclusions.{sector} must list categories from "
                              f"{SCORE_CATEGORIES}")
    bands = data.get("interpretation_bands")
    if not isinstance(bands, list) or not bands:
        raise ConfigError(f"{path}: interpretation_bands required")
    data["interpretation_bands"] = sorted(bands, key=lambda b: -_number(b["min"], "band"))
    return data
