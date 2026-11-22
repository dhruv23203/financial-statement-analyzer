"""Train and evaluate the distress-onset models (``python main.py train-distress``).

Protocol (declared before looking at the test year):

* Rows: healthy non-financial company-years with a known next-year outcome
  (``app.ml.distress.label_for``). Real statements only - nothing is synthesised.
* Chronological split by the label's fiscal year: **train <= 2024, validation 2025,
  test 2026**.
* Preprocessing fitted on train: 1st/99th percentile clipping, median imputation,
  standardisation; a Spearman |rho| > 0.85 filter removes redundant features.
* Class imbalance (~5 % positives): balanced class weights (logistic regression, random
  forest) and ``scale_pos_weight`` (XGBoost); the decision threshold maximises F1 on the
  validation year, then stays fixed.
* Model choice on **validation average precision** (PR-AUC, the right summary under
  imbalance). If logistic regression is within 0.02 of the best, it is chosen for its
  transparency (exact per-feature contributions).
* The chosen settings are refitted on train + validation and evaluated **once** on the
  test year, next to baselines: no-skill, and the two-ratio rule
  ``-min(net margin, OCF margin)``.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from app.analysis.ratios import calculate_all_ratios, index_outcomes
from app.analysis.service import load_dataset
from app.database import repositories as repo
from app.ml import distress as dm
from app.services.context import AppContext
from app.utils.exceptions import AnalyzerError

logger = logging.getLogger(__name__)
TRAIN_MAX_YEAR, VALID_YEAR, TEST_YEAR = 2024, 2025, 2026
CORR_LIMIT = 0.85
LR_TOLERANCE = 0.02
SEED = 42


# ----------------------------------------------------------------------------- data
def build_rows(ctx: AppContext, progress=lambda _m: None) -> pd.DataFrame:
    with ctx.db.session_scope() as session:
        tickers = [c.ticker for c in repo.list_companies(session)]
    rows = []
    for i, ticker in enumerate(tickers, 1):
        try:
            ds, info = load_dataset(ctx, ticker)
        except AnalyzerError:
            continue
        if info["sector"] in dm.FINANCIAL_SECTORS:
            continue
        ds.market = None
        ratios = index_outcomes(calculate_all_ratios(ds, ctx.settings.average_balance_fallback,
                                                     info["sector"]))
        for period in ds.periods:
            label, label_period = dm.label_for(ds.get, ds.periods, period)
            if label is None:
                continue
            rows.append({"ticker": ticker, "sector": info["sector"], "currency": ds.currency,
                         "period": period, "label_year": label_period.year, "label": label,
                         **dm.features_for(ds.get, ds.periods, ratios, period)})
        if i % 200 == 0:
            progress(f"[{i}/{len(tickers)}] rows {len(rows)}")
    return pd.DataFrame(rows)


def correlation_filter(train: pd.DataFrame, features: list[str]) -> tuple[list[str], list[dict]]:
    corr = train[features].corr(method="spearman").abs()
    kept, dropped = [], []
    for f in features:
        clash = [k for k in kept if corr.loc[f, k] > CORR_LIMIT]
        if clash:
            dropped.append({"feature": f, "correlated_with": clash[0],
                            "spearman": round(float(corr.loc[f, clash[0]]), 3)})
        else:
            kept.append(f)
    return kept, dropped


# ----------------------------------------------------------------------------- models
def _models(n_pos: int, n_neg: int):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    grids = {
        "Logistic regression": [
            (dict(C=c), lambda c=c: LogisticRegression(C=c, class_weight="balanced",
                                                       max_iter=5000))
            for c in (0.01, 0.1, 1.0, 10.0)],
        "Random forest": [
            (dict(max_depth=d, min_samples_leaf=leaf),
             lambda d=d, leaf=leaf: RandomForestClassifier(
                 n_estimators=500, max_depth=d, min_samples_leaf=leaf,
                 class_weight="balanced_subsample", random_state=SEED, n_jobs=-1))
            for d in (3, 5, None) for leaf in (5, 20)],
    }
    try:
        from xgboost import XGBClassifier
        grids["XGBoost"] = [
            (dict(max_depth=d, n_estimators=n),
             lambda d=d, n=n: XGBClassifier(
                 n_estimators=n, max_depth=d, learning_rate=0.05, subsample=0.8,
                 colsample_bytree=0.8, reg_lambda=1.0, min_child_weight=5,
                 scale_pos_weight=n_neg / max(n_pos, 1), random_state=SEED,
                 eval_metric="logloss", verbosity=0))
            for d in (2, 3) for n in (100, 300)]
    except ImportError:
        logger.info("xgboost not installed: XGBoost skipped")
    return grids


def _threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Threshold maximising F1 on the given (validation) data."""
    from sklearn.metrics import precision_recall_curve

    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.where(prec + rec == 0, 1, prec + rec)
    return float(thr[int(np.argmax(f1[:-1]))]) if len(thr) else 0.5


def _metrics(y: np.ndarray, p: np.ndarray, threshold: float, boot: bool = True) -> dict:
    from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                                 f1_score, precision_score, recall_score, roc_auc_score)

    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = {"roc_auc": round(float(roc_auc_score(y, p)), 3),
           "pr_auc": round(float(average_precision_score(y, p)), 3),
           "precision": round(float(precision_score(y, pred, zero_division=0)), 3),
           "recall": round(float(recall_score(y, pred, zero_division=0)), 3),
           "f1": round(float(f1_score(y, pred, zero_division=0)), 3),
           "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
           "threshold": round(float(threshold), 4), "n": int(len(y)), "positives": int(y.sum())}
    if 0 < threshold < 1 and np.all((p >= 0) & (p <= 1)):
        out["brier"] = round(float(brier_score_loss(y, p)), 4)
    if boot:
        rng = np.random.default_rng(SEED)
        aucs = []
        for _ in range(1000):
            idx = rng.integers(0, len(y), len(y))
            if y[idx].min() != y[idx].max():
                aucs.append(roc_auc_score(y[idx], p[idx]))
        out["roc_auc_ci95"] = [round(float(np.percentile(aucs, 2.5)), 3),
                               round(float(np.percentile(aucs, 97.5)), 3)]
    return out


def _proba(model, X):
    return model.predict_proba(X)[:, 1]


def run_experiment(df: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any], dict]:
    """Return (evaluation report, deployable artifact, fitted models). The deployed model
    is the one the pre-declared validation rule chose - never the test-set winner."""
    feats = list(dm.FEATURES)
    train = df[df["label_year"] <= TRAIN_MAX_YEAR]
    valid = df[df["label_year"] == VALID_YEAR]
    test = df[df["label_year"] == TEST_YEAR]
    if min(train["label"].sum(), valid["label"].sum(), test["label"].sum()) < 10:
        raise AnalyzerError("Fewer than 10 positive events in a split: not enough labelled "
                            "data for a reliable model.")
    kept, dropped = correlation_filter(train, feats)

    def prep(fit_rows):
        pre = dm.Preprocessor.fit(dm.matrix(fit_rows[kept].to_dict("records"), kept), kept)
        return pre, (lambda rows, scale=True: pre.transform(
            dm.matrix(rows[kept].to_dict("records"), kept), scale))

    pre_tr, tx = prep(train)
    y_tr, y_va, y_te = (s["label"].to_numpy() for s in (train, valid, test))
    grids = _models(int(y_tr.sum()), int((1 - y_tr).sum()))

    # 1) hyper-parameters and model family on the validation year
    selection = {}
    for name, grid in grids.items():
        best = None
        for params, make in grid:
            m = make().fit(tx(train, name == "Logistic regression"), y_tr)
            p = _proba(m, tx(valid, name == "Logistic regression"))
            from sklearn.metrics import average_precision_score
            ap = float(average_precision_score(y_va, p))
            if best is None or ap > best["val_pr_auc"]:
                best = {"params": params, "val_pr_auc": ap, "make": make,
                        "threshold": _threshold(y_va, p), "val_p": p}
        selection[name] = best
    top = max(v["val_pr_auc"] for v in selection.values())
    lr_sel = selection["Logistic regression"]
    chosen = "Logistic regression" if lr_sel["val_pr_auc"] >= top - LR_TOLERANCE else \
        max(selection, key=lambda k: selection[k]["val_pr_auc"])

    # 2) refit on train + validation, evaluate once on the test year
    trval = pd.concat([train, valid])
    pre_tv, tx2 = prep(trval)
    y_tv = trval["label"].to_numpy()
    results, fitted = {}, {}
    for name, sel in selection.items():
        scale = name == "Logistic regression"
        model = sel["make"]().fit(tx2(trval, scale), y_tv)
        fitted[name] = model
        results[name] = {"params": sel["params"],
                         "validation_pr_auc": round(sel["val_pr_auc"], 3),
                         "test": _metrics(y_te, _proba(model, tx2(test, scale)),
                                          sel["threshold"])}
    # baselines (no fitting): two-ratio rule, and no skill
    def two_ratio(rows):
        return -np.fmin(rows["net_profit_margin"].fillna(np.inf).to_numpy(),
                        rows["operating_cash_flow_margin"].fillna(np.inf).to_numpy())
    rule_va, rule_te = two_ratio(valid), two_ratio(test)
    rule_te = np.where(np.isfinite(rule_te), rule_te, np.nanmin(rule_te[np.isfinite(rule_te)]))
    rule_va = np.where(np.isfinite(rule_va), rule_va, np.nanmin(rule_va[np.isfinite(rule_va)]))
    rule_test = _metrics(y_te, rule_te, _threshold(y_va, rule_va))
    rule_test.pop("brier", None)
    baseline = {
        "Two-ratio rule (-min(net margin, OCF margin))": rule_test,
        "No skill (base rate)": {"roc_auc": 0.5, "pr_auc": round(float(y_te.mean()), 3),
                                 "note": "predicting the base rate for everyone"},
    }

    # importance
    lr = fitted["Logistic regression"]
    importance = {
        "Logistic regression (standardised coefficient)":
            {dm.FEATURES[f]: round(float(c), 3) for f, c in zip(kept, lr.coef_[0])},
        "Random forest (impurity importance)":
            {dm.FEATURES[f]: round(float(v), 3)
             for f, v in zip(kept, fitted["Random forest"].feature_importances_)},
    }
    if "XGBoost" in fitted:
        importance["XGBoost (gain importance)"] = {
            dm.FEATURES[f]: round(float(v), 3)
            for f, v in zip(kept, fitted["XGBoost"].feature_importances_)}

    def span(rows):
        return {"fiscal_years": [str(rows["period"].min()), str(rows["period"].max())],
                "label_years": sorted(int(y) for y in rows["label_year"].unique()),
                "rows": int(len(rows)), "positives": int(rows["label"].sum()),
                "companies": int(rows["ticker"].nunique())}

    report = {
        "generated_at": datetime.now(timezone.utc).date().isoformat(),
        "target": "distress onset: healthy company (net income > 0 and OCF > 0) in year t "
                  "shows a net loss or negative OCF in fiscal year t+1 - a proxy, not "
                  "default or bankruptcy",
        "split": {"train": span(train), "validation": span(valid), "test": span(test)},
        "features_used": [dm.FEATURES[f] for f in kept],
        "features_dropped": dropped,
        "selection_rule": f"highest validation PR-AUC; logistic regression preferred when "
                          f"within {LR_TOLERANCE}",
        "chosen_model": chosen,
        "models": results, "baselines": baseline, "importance": importance,
        "test_base_rate": round(float(y_te.mean()), 4),
    }

    # deployable artifact: the protocol's choice. XGBoost is stored in its own JSON
    # format (no pickle); logistic regression as coefficients. Random forest cannot be
    # stored without pickle, so if it were chosen, logistic regression is deployed and
    # the report says so.
    deployed = chosen if chosen in ("Logistic regression", "XGBoost") else "Logistic regression"
    artifact = {
        "model": {"Logistic regression": "Logistic regression (L2, balanced class weights)",
                  "XGBoost": "XGBoost (gradient-boosted trees, depth "
                             f"{selection.get('XGBoost', {}).get('params', {}).get('max_depth')}"
                             ", class-weighted)"}[deployed],
        "type": "xgboost" if deployed == "XGBoost" else "logistic",
        "xgboost_file": "distress_xgb.json" if deployed == "XGBoost" else None,
        "version": "1.0.0", "trained_at": report["generated_at"],
        "preprocessor": pre_tv.to_json(),
        "coefficients": [float(c) for c in lr.coef_[0]],
        "intercept": float(lr.intercept_[0]),
        "threshold": float(selection[deployed]["threshold"]),
        "params": selection[deployed]["params"],
        "max_missing_features": 4,
        "test": results[deployed]["test"],
        "training_period": {"fiscal_years": [str(trval["period"].min()),
                                             str(trval["period"].max())],
                            "label_years": sorted(int(y) for y in trval["label_year"].unique()),
                            "rows": int(len(trval)), "positives": int(y_tv.sum())},
        "chosen_by_protocol": deployed == chosen,
    }
    report["deployed_model"] = deployed
    return report, artifact, fitted


def train(ctx: AppContext, progress=lambda _m: None, write: bool = True) -> dict[str, Any]:
    df = build_rows(ctx, progress)
    report, artifact, fitted = run_experiment(df)
    if write:
        dm.MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
        if artifact["type"] == "xgboost":
            fitted["XGBoost"].get_booster().save_model(
                str(dm.MODEL_FILE.parent / artifact["xgboost_file"]))
        dm.MODEL_FILE.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        (dm.MODEL_FILE.parent / "distress_evaluation.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


__all__ = ["build_rows", "run_experiment", "train", "TRAIN_MAX_YEAR", "VALID_YEAR",
           "TEST_YEAR", "date"]
