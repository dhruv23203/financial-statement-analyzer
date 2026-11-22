"""Distress-onset prediction - a supervised ML model, separate from the Health Score.

Target (defensible proxy, not default)
--------------------------------------
Real distress events (default, bankruptcy, delisting) are not in the data: the provider
does not report them and the universe is today's index members (survivors). The model
therefore predicts **distress onset**: for a non-financial company that is *healthy in
fiscal year t* (net income > 0 and operating cash flow > 0), will fiscal year t+1 show
a **net loss or negative operating cash flow**? Conditioning on "healthy now" removes
the trivial persistence (a loss-maker staying a loss-maker) that inflated earlier
validations.

No leakage
----------
* Features use only fiscal year t and the adjacent year t-1 (changes); the label is the
  next adjacent fiscal year (330-400 days later). No later data enters a feature.
* The split is **chronological** by the label's fiscal year: train <= 2024, validation
  2025 (model, hyper-parameter and threshold selection), test 2026 (untouched until the
  end).
* Every preprocessing parameter (clip bounds, medians, means, standard deviations,
  correlation filter) is fitted on the training years only.

Features come from stored ratio outcomes and the statement lines the hosted database
keeps, so the same function serves training and live prediction.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from config.settings import PROJECT_ROOT

MODEL_FILE = PROJECT_ROOT / "models" / "distress_model.json"
FINANCIAL_SECTORS = ("Financial Services",)
ADJACENT = (330, 400)

FEATURES: dict[str, str] = {
    "net_profit_margin": "Net profit margin (%)",
    "operating_profit_margin": "Operating margin (%)",
    "return_on_assets": "Return on assets (%)",
    "operating_cash_flow_margin": "Operating cash-flow margin (%)",
    "free_cash_flow_margin": "Free-cash-flow margin (%)",
    "cash_conversion": "Cash conversion (OCF / net income)",
    "current_ratio": "Current ratio",
    "debt_ratio": "Debt ratio (liabilities / assets)",
    "interest_coverage_log": "Interest coverage (signed log)",
    "asset_turnover": "Asset turnover",
    "revenue_growth": "Revenue growth (%)",
    "d_operating_margin": "Change in operating margin (pts)",
    "d_ocf_margin": "Change in OCF margin (pts)",
    "accruals_to_assets": "Accruals (NI - OCF) / total assets (%)",
}
RATIO_FEATURES = ("net_profit_margin", "operating_profit_margin", "return_on_assets",
                  "operating_cash_flow_margin", "free_cash_flow_margin", "cash_conversion",
                  "current_ratio", "debt_ratio", "asset_turnover")


def _f(v) -> float | None:
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _adjacent_previous(periods: list[date], period: date) -> date | None:
    earlier = [p for p in periods if p < period]
    if not earlier:
        return None
    prev = earlier[-1]
    return prev if ADJACENT[0] <= (period - prev).days <= ADJACENT[1] else None


def _adjacent_next(periods: list[date], period: date) -> date | None:
    later = [p for p in periods if p > period]
    if not later:
        return None
    nxt = later[0]
    return nxt if ADJACENT[0] <= (nxt - period).days <= ADJACENT[1] else None


def _ratio(ratios: dict, code: str, period: date | None) -> tuple[float | None, str | None]:
    o = ratios.get((code, period)) if period else None
    if o is None:
        return None, None
    return (_f(o.value) if o.ok else None), o.reason_code


def is_healthy(get, period: date) -> bool | None:
    """Healthy = net income > 0 and operating cash flow > 0 (None if unknown)."""
    ni, ocf = _f(get(period, "net_income")), _f(get(period, "operating_cash_flow"))
    if ni is None or ocf is None:
        return None
    return ni > 0 and ocf > 0


def features_for(get, periods: list[date], ratios: dict, period: date) -> dict[str, float | None]:
    """Feature vector for fiscal year ``period`` (uses only ``period`` and the adjacent
    previous year). ``get(period, metric_code)`` reads a statement value."""
    out: dict[str, float | None] = {}
    for code in RATIO_FEATURES:
        out[code] = _ratio(ratios, code, period)[0]
    ic, reason = _ratio(ratios, "interest_coverage", period)
    if ic is not None:
        ic = max(-100.0, min(100.0, ic))
        out["interest_coverage_log"] = math.copysign(math.log1p(abs(ic)), ic)
    elif reason == "NO_INTEREST_EXPENSE":
        out["interest_coverage_log"] = math.log1p(100.0)     # nothing to cover
    else:
        out["interest_coverage_log"] = None
    prev = _adjacent_previous(periods, period)
    rev, rev_prev = _f(get(period, "total_revenue")), _f(get(prev, "total_revenue")) \
        if prev else None
    out["revenue_growth"] = ((rev / rev_prev - 1) * 100) if (rev is not None and rev_prev
                                                             and rev_prev > 0) else None
    for name, code in (("d_operating_margin", "operating_profit_margin"),
                       ("d_ocf_margin", "operating_cash_flow_margin")):
        now, before = _ratio(ratios, code, period)[0], _ratio(ratios, code, prev)[0]
        out[name] = (now - before) if (now is not None and before is not None) else None
    ni, ocf, ta = (_f(get(period, c)) for c in ("net_income", "operating_cash_flow",
                                                 "total_assets"))
    out["accruals_to_assets"] = ((ni - ocf) / ta * 100) if (None not in (ni, ocf, ta)
                                                            and ta > 0) else None
    return out


def label_for(get, periods: list[date], period: date) -> tuple[int | None, date | None]:
    """(label, label period): 1 if the next adjacent year shows a loss or negative OCF,
    for a company healthy in ``period``; (None, None) when not eligible or unknown."""
    if not is_healthy(get, period):
        return None, None
    nxt = _adjacent_next(periods, period)
    if nxt is None:
        return None, None
    healthy_next = is_healthy(get, nxt)
    if healthy_next is None:
        return None, None
    return (0 if healthy_next else 1), nxt


# ----------------------------------------------------------------------------- preprocessing
@dataclass
class Preprocessor:
    """Winsorise -> median-impute -> standardise, with parameters fitted on training rows."""
    features: list[str]
    lo: dict[str, float]
    hi: dict[str, float]
    median: dict[str, float]
    mean: dict[str, float]
    std: dict[str, float]

    @classmethod
    def fit(cls, X: np.ndarray, features: list[str]) -> Preprocessor:
        lo, hi, med, mean, std = {}, {}, {}, {}, {}
        for j, name in enumerate(features):
            col = X[:, j][np.isfinite(X[:, j])]
            lo[name], hi[name] = (float(np.percentile(col, 1)), float(np.percentile(col, 99))) \
                if len(col) else (0.0, 0.0)
            clipped = np.clip(col, lo[name], hi[name]) if len(col) else col
            med[name] = float(np.median(clipped)) if len(col) else 0.0
            filled = np.where(np.isfinite(X[:, j]), np.clip(X[:, j], lo[name], hi[name]),
                              med[name])
            mean[name], std[name] = float(filled.mean()), float(filled.std() or 1.0)
        return cls(list(features), lo, hi, med, mean, std)

    def transform(self, X: np.ndarray, scale: bool = True) -> np.ndarray:
        out = np.empty_like(X, dtype=float)
        for j, name in enumerate(self.features):
            col = np.where(np.isfinite(X[:, j]), np.clip(X[:, j], self.lo[name], self.hi[name]),
                           self.median[name])
            out[:, j] = (col - self.mean[name]) / self.std[name] if scale else col
        return out

    def to_json(self) -> dict[str, Any]:
        return {"features": self.features, "lo": self.lo, "hi": self.hi,
                "median": self.median, "mean": self.mean, "std": self.std}


def matrix(rows: list[dict[str, float | None]], features: list[str]) -> np.ndarray:
    return np.array([[np.nan if r.get(f) is None else float(r[f]) for f in features]
                     for r in rows], dtype=float)


# ----------------------------------------------------------------------------- prediction
def load_model(path: Path = MODEL_FILE) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


@dataclass
class Prediction:
    status: str                       # OK | NOT_APPLICABLE | INSUFFICIENT_DATA | NO_MODEL
    reason: str = ""
    probability: float | None = None
    flagged: bool | None = None
    period: date | None = None
    contributions: list[tuple[str, float, float | None]] | None = None  # (label, logit, value)
    missing: list[str] | None = None


def predict(model: dict[str, Any] | None, get, periods: list[date], ratios: dict,
            sector: str | None) -> Prediction:
    """Probability of distress onset next year for the latest fiscal year, with each
    feature's contribution to the log-odds (logistic model: coefficient x standardised
    value). Returns a non-OK status with the reason when the model does not apply."""
    if model is None:
        return Prediction("NO_MODEL", "No trained model file (python main.py train-distress).")
    if sector in FINANCIAL_SECTORS:
        return Prediction("NOT_APPLICABLE", "Financial companies are excluded: their operating "
                          "cash flow includes deposit and loan flows, so the target is not "
                          "meaningful.")
    if not periods:
        return Prediction("INSUFFICIENT_DATA", "No fiscal years stored.")
    period = periods[-1]
    healthy = is_healthy(get, period)
    if healthy is None:
        return Prediction("INSUFFICIENT_DATA", "Net income or operating cash flow is missing "
                          "for the latest year.", period=period)
    if not healthy:
        return Prediction("NOT_APPLICABLE", "The company already shows a net loss or negative "
                          "operating cash flow in the latest year. The model predicts the "
                          "*onset* of distress for currently healthy companies, so it does not "
                          "apply.", period=period)
    feats = features_for(get, periods, ratios, period)
    names = model["preprocessor"]["features"]
    missing = [FEATURES[n] for n in names if feats.get(n) is None]
    if len(missing) > model.get("max_missing_features", 4):
        return Prediction("INSUFFICIENT_DATA", f"{len(missing)} of {len(names)} features are "
                          "missing - too many to predict reliably.", period=period,
                          missing=missing)
    pre = Preprocessor(**model["preprocessor"])
    if model.get("type") == "xgboost":
        booster = _booster(model.get("xgboost_file"))
        if booster is None:
            return Prediction("NO_MODEL", "The XGBoost runtime or model file is not available "
                              "on this server.", period=period)
        import xgboost as xgb

        x = xgb.DMatrix(pre.transform(matrix([feats], names), scale=False))
        prob = float(booster.predict(x)[0])
        # exact per-feature contributions to the log-odds (TreeSHAP); last column = bias
        shap = booster.predict(x, pred_contribs=True)[0][:-1]
        contrib = sorted(((FEATURES[n], float(c), feats.get(n))
                          for n, c in zip(names, shap)), key=lambda t: -abs(t[1]))
    else:
        z = pre.transform(matrix([feats], names))[0]
        coefs = np.array(model["coefficients"], dtype=float)
        logit = float(model["intercept"] + coefs @ z)
        prob = 1 / (1 + math.exp(-logit))
        contrib = sorted(((FEATURES[n], float(c * zz), feats.get(n))
                          for n, c, zz in zip(names, coefs, z)), key=lambda t: -abs(t[1]))
    return Prediction("OK", "", prob, prob >= model["threshold"], period, contrib, missing)


_BOOSTERS: dict[str, Any] = {}


def _booster(filename: str | None):
    """Load an XGBoost model saved in XGBoost's own JSON format (no pickle)."""
    if not filename:
        return None
    if filename not in _BOOSTERS:
        try:
            import xgboost as xgb
        except ImportError:
            return None
        path = MODEL_FILE.parent / filename
        if not path.is_file():
            return None
        booster = xgb.Booster()
        booster.load_model(str(path))
        _BOOSTERS[filename] = booster
    return _BOOSTERS[filename]


def predict_for_analysis(model: dict[str, Any] | None, analysis) -> Prediction:
    """Prediction from a stored analysis (works on the hosted database: it needs only the
    stored ratio outcomes and the revenue, net income, OCF and total-asset lines)."""
    from app.analysis.ratios import index_outcomes

    ds = analysis.dataset
    return predict(model, ds.get, ds.periods, index_outcomes(analysis.ratios), analysis.sector)
