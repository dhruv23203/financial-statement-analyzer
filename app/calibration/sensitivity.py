"""Sensitivity of the overall score to the category weights (audit item: LIQUIDITY = 0).

Recomputes every stored company's overall score from its latest category scores under
alternative weight vectors and reports how much the ranking and the interpretation
bands change. It re-uses the stored category scores only (no recalculation of ratios),
so it isolates the effect of the category weights.

    python main.py sensitivity
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

CATEGORIES = ("LIQUIDITY", "SOLVENCY", "PROFITABILITY", "EFFICIENCY", "CASH_FLOW")


def overall_scores(frame: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Weighted average of the available category scores, weights re-normalised per
    company exactly like ``health_score.score_period`` (zero-weight categories ignored).
    Companies with fewer than three weighted categories get NaN (as in the app)."""
    columns = {c: f"{c.lower()}_score" for c in CATEGORIES}
    values = frame[[columns[c] for c in CATEGORIES]].to_numpy(dtype=float)
    w = np.array([weights[c] for c in CATEGORIES], dtype=float)
    available = ~np.isnan(values) & (w > 0)
    weighted = np.where(available, np.nan_to_num(values) * w, 0.0).sum(axis=1)
    total = np.where(available, w, 0.0).sum(axis=1)
    enough = available.sum(axis=1) >= 3
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(enough & (total > 0), weighted / total, np.nan)
    return pd.Series(result, index=frame.index)


def band_of(score: float, bands: list[dict[str, Any]]) -> str | None:
    if np.isnan(score):
        return None
    return next(b["label"] for b in sorted(bands, key=lambda b: -b["min"])
                if score >= b["min"])


def compare_weightings(frame: pd.DataFrame, scenarios: dict[str, dict[str, float]],
                       baseline: str, bands: list[dict[str, Any]]) -> pd.DataFrame:
    """One row per scenario: rank correlation with the baseline, mean / max absolute
    score change, and the share of companies whose interpretation band changes."""
    from scipy.stats import spearmanr

    base = overall_scores(frame, scenarios[baseline])
    rows = []
    for name, weights in scenarios.items():
        scores = overall_scores(frame, weights)
        both = base.notna() & scores.notna()
        diff = (scores[both] - base[both]).abs()
        changed = sum(band_of(a, bands) != band_of(b, bands)
                      for a, b in zip(base[both], scores[both]))
        rows.append({
            "scenario": name,
            "weights": ", ".join(f"{c[:4]} {weights[c]:.3f}" for c in CATEGORIES),
            "companies": int(both.sum()),
            "spearman_vs_baseline": round(float(spearmanr(base[both], scores[both])[0]), 4)
            if both.sum() > 2 else None,
            "mean_abs_change": round(float(diff.mean()), 2) if len(diff) else None,
            "max_abs_change": round(float(diff.max()), 2) if len(diff) else None,
            "band_changes_pct": round(100 * changed / both.sum(), 1) if both.sum() else None,
        })
    return pd.DataFrame(rows)


def default_scenarios(config: dict[str, Any],
                      critic: dict[str, float] | None = None) -> dict[str, dict[str, float]]:
    """Shipped weights, liquidity given its CRITIC weight (others scaled down
    proportionally), and equal weights."""
    shipped = {c: float(config["categories"][c]["weight"]) for c in CATEGORIES}
    scenarios = {"shipped (logistic, liquidity 0)": shipped}
    if critic:
        liq = critic["LIQUIDITY"]
        scaled = {c: (shipped[c] * (1 - liq) if c != "LIQUIDITY" else liq) for c in CATEGORIES}
        scenarios[f"liquidity at its CRITIC weight ({liq:.3f})"] = scaled
        scenarios["CRITIC for all categories"] = dict(critic)
    scenarios["equal weights (0.2 each)"] = {c: 0.2 for c in CATEGORIES}
    return scenarios
