"""Calibration methods (pure functions, no I/O).

References
----------
* Percentile (cross-sectional rank) scoring of fundamentals: Asness, C., Frazzini, A. &
  Pedersen, L. H. (2019). Quality minus junk. Review of Accounting Studies, 24(1), 34-112.
* CRITIC objective weights: Diakoulaki, D., Mavrotas, G. & Papayannakis, L. (1995).
  Determining objective weights in multiple criteria problems: The CRITIC method.
  Computers & Operations Research, 22(7), 763-770.
* Entropy weights: Shannon, C. E. (1948). A mathematical theory of communication. Bell
  System Technical Journal, 27, 379-423; applied to criteria weighting in Zeleny, M.
  (1982). Multiple Criteria Decision Making. McGraw-Hill.
* Predicting the direction of next-year earnings from financial-statement data with a
  statistical model: Ou, J. A. & Penman, S. H. (1989). Financial statement analysis and
  the prediction of stock returns. Journal of Accounting and Economics, 11(4), 295-329.
* Equal weights as a robust benchmark: Dawes, R. M. (1979). The robust beauty of
  improper linear models in decision making. American Psychologist, 34(7), 571-582.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

PERCENTILES = (5, 10, 25, 50, 75, 90, 95)


def percentile_breakpoints(values: list[float], higher_is_better: bool,
                           percentiles: tuple[int, ...] = PERCENTILES) -> list[list[float]]:
    """``[[value, points], ...]`` so that points approximate the percentile rank.

    For a higher-is-better ratio the value at the p-th percentile receives p points;
    for lower-is-better it receives 100 - p. Linear interpolation between breakpoints
    then approximates the empirical CDF. Tied percentile values are merged (the
    breakpoint values must be strictly increasing).
    """
    data = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if data.size == 0:
        raise ValueError("no finite values to calibrate")
    quantiles = np.percentile(data, percentiles)
    points: list[list[float]] = []
    for p, q in zip(percentiles, quantiles):
        score = float(p) if higher_is_better else float(100 - p)
        value = float(round(q, 6))
        if points and value <= points[-1][0]:
            points[-1][1] = round((points[-1][1] + score) / 2, 2)  # merge tie
            continue
        points.append([value, score])
    if len(points) < 2:
        raise ValueError("values are (almost) constant; cannot build breakpoints")
    return points


def _minmax(matrix: np.ndarray) -> np.ndarray:
    lo, hi = matrix.min(axis=0), matrix.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    return (matrix - lo) / span


def critic_weights(matrix: np.ndarray) -> np.ndarray:
    """CRITIC: w_j proportional to sigma_j * sum_k (1 - r_jk) on min-max normalised data.

    Criteria with more spread (contrast intensity) and less correlation with the
    others (conflict, i.e. less double counting) receive more weight.
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 3 or matrix.shape[1] < 2:
        raise ValueError("CRITIC needs at least 3 observations and 2 criteria")
    normalised = _minmax(matrix)
    sigma = normalised.std(axis=0, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(normalised, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    information = sigma * (1.0 - corr).sum(axis=0)
    if information.sum() <= 0:
        return np.full(matrix.shape[1], 1.0 / matrix.shape[1])
    return information / information.sum()


def entropy_weights(matrix: np.ndarray) -> np.ndarray:
    """Entropy weight method: criteria whose values differ more carry more information."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        raise ValueError("entropy weighting needs at least 2 observations")
    shifted = _minmax(matrix) + 1e-9
    proportions = shifted / shifted.sum(axis=0)
    k = 1.0 / np.log(matrix.shape[0])
    entropy = -k * (proportions * np.log(proportions)).sum(axis=0)
    divergence = 1.0 - entropy
    if divergence.sum() <= 0:
        return np.full(matrix.shape[1], 1.0 / matrix.shape[1])
    return divergence / divergence.sum()


@dataclass
class SupervisedResult:
    n_observations: int
    n_companies: int
    positive_rate: float
    coefficients: dict[str, float]
    coefficient_ci: dict[str, tuple[float, float]]
    oof_auc: float
    oof_auc_ci: tuple[float, float]
    candidate_auc: dict[str, float]
    weights: dict[str, float] | None
    accepted: bool
    reasons: list[str] = field(default_factory=list)


def _auc(y: np.ndarray, score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, score))


def supervised_weights(features: np.ndarray, labels: np.ndarray, groups: np.ndarray,
                       names: list[str], candidate_weights: dict[str, np.ndarray],
                       min_auc_gain: float = 0.02, n_bootstrap: int = 1000,
                       seed: int = 7) -> SupervisedResult:
    """Logistic regression of an adverse next-year outcome on category scores.

    * Features are standardised; 5-fold GroupKFold keeps all years of one company in
      the same fold (no leakage across a company's own history).
    * Out-of-fold AUC is compared with the AUC of each candidate composite score
      (equal, CRITIC, entropy weights) computed on the same observations.
    * Pre-declared acceptance rule: every coefficient must have the expected sign
      (no coefficient whose company-bootstrap 95 % CI lies entirely above 0, i.e.
      significantly 'higher health -> higher risk'), the 95 % bootstrap CI of
      the out-of-fold AUC must lie above 0.5, and the AUC must beat the best
      candidate composite by at least ``min_auc_gain``. Otherwise the weights are
      reported but not used.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(features, dtype=float)
    y = np.asarray(labels, dtype=int)
    groups = np.asarray(groups)
    n_companies = len(set(groups.tolist()))
    folds = min(5, n_companies)
    oof = np.zeros(len(y))
    for train, test in GroupKFold(n_splits=folds).split(X, y, groups):
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        model.fit(X[train], y[train])
        oof[test] = model.predict_proba(X[test])[:, 1]
    oof_auc = _auc(y, oof)

    rng = np.random.default_rng(seed)
    unique = np.array(sorted(set(groups.tolist())))
    boot = []
    for _ in range(n_bootstrap):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        index = np.concatenate([np.flatnonzero(groups == g) for g in chosen])
        if len(set(y[index].tolist())) == 2:
            boot.append(_auc(y[index], oof[index]))
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))

    def fit_coefficients(index: np.ndarray) -> np.ndarray:
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        return model.fit(X[index], y[index])[-1].coef_[0]

    coefficients = dict(zip(names, fit_coefficients(np.arange(len(y))).round(4).tolist()))
    coef_boot = []
    for _ in range(max(100, n_bootstrap // 5)):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        index = np.concatenate([np.flatnonzero(groups == g) for g in chosen])
        if len(set(y[index].tolist())) == 2:
            coef_boot.append(fit_coefficients(index))
    coef_boot = np.array(coef_boot)
    coefficient_ci = {n: (round(float(np.percentile(coef_boot[:, i], 2.5)), 4),
                          round(float(np.percentile(coef_boot[:, i], 97.5)), 4))
                      for i, n in enumerate(names)}

    # Composite health scores predict LOW risk, so their AUC for the adverse label uses -score.
    candidate_auc = {name: _auc(y, -(X @ np.asarray(w))) for name, w in
                     candidate_weights.items()}
    best_candidate = max(candidate_auc.values())

    reasons = []
    # Significantly wrong sign: the whole 95 % bootstrap CI is above zero (higher score ->
    # higher risk). Small, insignificant positive coefficients are treated as zero weight.
    wrong_sign = [n for n, (low, _high) in coefficient_ci.items() if low > 0]
    if wrong_sign:
        reasons.append(f"coefficients significantly of the wrong sign (higher score -> higher "
                       f"risk, 95% CI above 0): {wrong_sign}")
    if ci[0] <= 0.5:
        reasons.append(f"95% CI of out-of-fold AUC includes 0.5 ({ci[0]:.3f}-{ci[1]:.3f})")
    if oof_auc < best_candidate + min_auc_gain:
        reasons.append(f"out-of-fold AUC {oof_auc:.3f} does not beat the best simple "
                       f"composite ({best_candidate:.3f}) by {min_auc_gain}")
    negative = np.clip(-np.array([coefficients[n] for n in names]), 0, None)
    weights = None
    if negative.sum() > 0:
        weights = dict(zip(names, (negative / negative.sum()).round(4).tolist()))
    return SupervisedResult(len(y), n_companies, float(y.mean()), coefficients, coefficient_ci,
                            oof_auc, ci,
                            candidate_auc, weights, accepted=not reasons, reasons=reasons)
