"""Calibration and validation of the industry-specific Financial Health Score (3.0.0).

``python main.py calibrate-framework`` (or :func:`run`) does, on the stored statements:

1. **Benchmarks** - for every metric, percentiles P1-P99 per peer group (each sector,
   each framework, and all non-financial / all financial companies) over every analysed
   company-year. These are *descriptive* distributions: no outcome is used, so the
   weights and thresholds are not tuned to make any company look better.
2. **Bands and grades** - interpretation bands are the universe quartiles of the overall
   score; letter grades per pillar follow the Morningstar 10/20/40/20/10 % distribution.
3. **Validation evidence** (written to ``config/validation_evidence.json``):
   * metric redundancy - Spearman correlation of every metric pair inside each pillar
     (the design keeps pairs below |0.7| to avoid double counting);
   * discrimination - AUC of the overall score for a next-year distress proxy (net loss or
     negative operating cash flow), next to naive baselines on the same rows, and the
     distress rate per score quintile; reported as found, without any tuning;
   * sensitivity - rank correlation between the declared pillar budgets and equal budgets.

Limitations stated with the results: benchmarks are pooled over all years (in-sample);
the distress proxy is mechanically related to the profitability pillar; the universe is
today's index members (survivorship).
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from app.analysis import health_framework as hf
from app.analysis.ratios import calculate_all_ratios, index_outcomes
from app.analysis.service import load_dataset
from app.database import repositories as repo
from app.services.context import AppContext
from app.utils.exceptions import AnalyzerError
from config.settings import CONFIG_DIR

logger = logging.getLogger(__name__)
PERCENTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)
EVIDENCE_FILE = CONFIG_DIR / "validation_evidence.json"


# ----------------------------------------------------------------------------- data
def collect(ctx: AppContext, config: dict[str, Any], progress=lambda _m: None) -> list[dict]:
    """One record per company-year: framework, peer group, metric values and the facts
    needed for the validation label."""
    with ctx.db.session_scope() as session:
        tickers = [c.ticker for c in repo.list_companies(session)]
    records = []
    for i, ticker in enumerate(tickers, 1):
        try:
            ds, info = load_dataset(ctx, ticker)
        except AnalyzerError:
            continue
        ds.market = None                      # no market multiples needed
        ratios = index_outcomes(calculate_all_ratios(ds, ctx.settings.average_balance_fallback,
                                                     info["sector"]))
        framework = hf.classify(info["sector"], info.get("industry"), config)
        group = hf.peer_group(framework, info["sector"], config)
        for period in ds.periods:
            values = hf.metric_values(ds, period, ratios, config)
            records.append({"ticker": ticker, "sector": info["sector"],
                            "industry": info.get("industry"), "framework": framework,
                            "group": group, "period": period, "dataset": ds,
                            "ratios": ratios, "values": values})
        if i % 100 == 0:
            progress(f"[{i}/{len(tickers)}] collected")
    return records


def benchmarks(records: list[dict], config: dict[str, Any]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    frame_rows = []
    for r in records:
        fin = bool(config["frameworks"][r["framework"]].get("financial"))
        groups = {r["group"], f"FW:{r['framework']}",
                  "ALL:financial" if fin else "ALL:non_financial"}
        for metric, mv in r["values"].items():
            if mv.status == "OK" and mv.value is not None:
                for g in groups:
                    frame_rows.append((g, metric, mv.value))
    frame = pd.DataFrame(frame_rows, columns=["group", "metric", "value"])
    for (group, metric), values in frame.groupby(["group", "metric"])["value"]:
        arr = values.to_numpy(dtype=float)
        out.setdefault(group, {})[metric] = {
            **{f"p{p}": round(float(np.percentile(arr, p)), 6) for p in PERCENTILES},
            "n": int(len(arr))}
    return out


def score_records(records: list[dict], config: dict[str, Any],
                  budgets: dict[str, dict[str, float]] | None = None) -> pd.DataFrame:
    """Score every company-year (optionally with replacement pillar budgets)."""
    rows = []
    for r in records:
        cfg = config
        if budgets is not None:
            cfg = json.loads(json.dumps(config))
            for fw, pillars in budgets.items():
                for p, mx in pillars.items():
                    if cfg["frameworks"][fw]["pillars"][p].get("max"):
                        cfg["frameworks"][fw]["pillars"][p]["max"] = mx
        res = hf.score_company(r["dataset"], r["period"], r["ratios"], r["sector"],
                               r["industry"], cfg)
        row = {"ticker": r["ticker"], "period": r["period"], "framework": r["framework"],
               "score": res.score, "confidence": res.confidence}
        for name, p in res.pillars.items():
            row[name] = p.percent
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- validation
def redundancy(records: list[dict], config: dict[str, Any]) -> list[dict]:
    """Spearman correlation of metric pairs that share a pillar in any framework."""
    values = pd.DataFrame([{m: v.value for m, v in r["values"].items() if v.status == "OK"}
                           for r in records])
    pairs = set()
    for fw in config["frameworks"].values():
        for spec in fw["pillars"].values():
            ms = sorted(spec.get("metrics", {}))
            pairs |= {(a, b) for i, a in enumerate(ms) for b in ms[i + 1:]}
    out = []
    for a, b in sorted(pairs):
        if a in values and b in values:
            sub = values[[a, b]].dropna()
            if len(sub) >= 30:
                out.append({"pair": f"{a} ~ {b}", "spearman": round(float(
                    sub[a].corr(sub[b], method="spearman")), 3), "n": int(len(sub))})
    return out


def discrimination(scores: pd.DataFrame, records: list[dict]) -> dict[str, Any]:
    from sklearn.metrics import roc_auc_score

    facts = {}
    for r in records:
        ds = r["dataset"]
        ni, ocf = ds.get(r["period"], "net_income"), ds.get(r["period"], "operating_cash_flow")
        rev = ds.get(r["period"], "total_revenue")
        facts[(r["ticker"], r["period"])] = (
            None if ni is None or ocf is None else int(ni < 0 or ocf < 0),
            None if (ni is None or ocf is None or not rev or rev <= 0)
            else float(min(ni / rev, ocf / rev)))
    df = scores[~scores["framework"].isin(["banking", "insurance", "financial_services"])].copy()
    df = df.sort_values(["ticker", "period"])
    df["distress"] = [facts[(t, p)][0] for t, p in zip(df["ticker"], df["period"])]
    df["two_ratio"] = [facts[(t, p)][1] for t, p in zip(df["ticker"], df["period"])]
    nxt = df.groupby("ticker").shift(-1)
    gap = (pd.to_datetime(nxt["period"]) - pd.to_datetime(df["period"])).dt.days
    df["label"] = nxt["distress"]
    rows = df[gap.between(330, 400) & df["label"].notna() & df["score"].notna()
              & df["distress"].notna() & df["two_ratio"].notna()].copy()
    if rows.empty or rows["label"].nunique() < 2:
        return {"note": "not enough labelled rows"}
    y = rows["label"].astype(int)
    rows["quintile"] = pd.qcut(rows["score"], 5, labels=False, duplicates="drop") + 1
    return {
        "label": "next-year net loss or negative operating cash flow (a proxy, not default)",
        "observations": int(len(rows)), "companies": int(rows["ticker"].nunique()),
        "positive_rate": round(float(y.mean()), 4),
        "auc_score": round(float(roc_auc_score(y, -rows["score"])), 3),
        "auc_two_ratio_baseline": round(float(roc_auc_score(y, -rows["two_ratio"])), 3),
        "auc_last_year_status_baseline": round(float(roc_auc_score(y, rows["distress"])), 3),
        "distress_rate_by_quintile": {int(q): round(float(g["label"].mean()), 4)
                                      for q, g in rows.groupby("quintile")},
        "note": "In-sample: benchmarks use the same company-years. No weight or threshold "
                "was tuned on this label.",
    }


def bands_and_grades(scores: pd.DataFrame, config: dict[str, Any]) -> tuple[list, dict]:
    latest = scores.dropna(subset=["score"]).sort_values("period").groupby("ticker").tail(1)
    q = latest["score"].quantile([0.75, 0.5, 0.25]).round(2).tolist()
    bands = [{"min": q[0], "label": "TOP QUARTILE"}, {"min": q[1], "label": "ABOVE MEDIAN"},
             {"min": q[2], "label": "BELOW MEDIAN"}, {"min": 0, "label": "BOTTOM QUARTILE"}]
    grades = {}
    for pillar in config["pillars"]:
        col = latest[pillar].dropna()
        if len(col) >= 30:
            p90, p70, p30, p10 = col.quantile([0.9, 0.7, 0.3, 0.1]).round(2).tolist()
            grades[pillar] = [{"grade": "A", "min": p90}, {"grade": "B", "min": p70},
                              {"grade": "C", "min": p30}, {"grade": "D", "min": p10},
                              {"grade": "F", "min": 0}]
    return bands, grades


def sensitivity(records: list[dict], config: dict[str, Any], base: pd.DataFrame) -> dict:
    equal = {fw: {p: 20 for p in config["pillars"]} for fw, spec in config["frameworks"].items()
             if not spec.get("financial")}
    alt = score_records(records, config, equal)
    merged = base[["ticker", "period", "score"]].merge(
        alt[["ticker", "period", "score"]], on=["ticker", "period"], suffixes=("", "_equal"))
    merged = merged.dropna()
    latest = merged.sort_values("period").groupby("ticker").tail(1)
    top = latest["score"] >= latest["score"].quantile(0.75)
    top_eq = latest["score_equal"] >= latest["score_equal"].quantile(0.75)
    return {"alternative": "equal 20-point budgets for the five non-financial pillars",
            "spearman_rank_correlation": round(float(
                merged["score"].corr(merged["score_equal"], method="spearman")), 3),
            "top_quartile_overlap": round(float((top & top_eq).sum() / max(top.sum(), 1)), 3),
            "max_abs_score_change": round(float((merged["score"] - merged["score_equal"])
                                                .abs().max()), 2)}


# ----------------------------------------------------------------------------- driver
def run(ctx: AppContext, write: bool = True, progress=lambda _m: None) -> dict[str, Any]:
    hf.load_framework.cache_clear()
    config = hf.load_framework()
    records = collect(ctx, config, progress)
    config["benchmarks"] = benchmarks(records, config)
    scores = score_records(records, config)
    config["interpretation_bands"], config["grade_bands"] = bands_and_grades(scores, config)
    evidence = {
        "generated_at": date.today().isoformat(),
        "company_years": int(len(scores)), "companies": int(scores["ticker"].nunique()),
        "scored_share": round(float(scores["score"].notna().mean()), 3),
        "frameworks": scores.groupby("framework")["ticker"].nunique().to_dict(),
        "redundancy": redundancy(records, config),
        "discrimination": discrimination(scores, records),
        "sensitivity": sensitivity(records, config, scores),
    }
    if write:
        path = hf.FRAMEWORK_FILE
        path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        ev = json.loads(EVIDENCE_FILE.read_text(encoding="utf-8")) if EVIDENCE_FILE.is_file() \
            else {}
        ev["health_framework_v3"] = evidence
        EVIDENCE_FILE.write_text(json.dumps(ev, indent=2, default=str) + "\n", encoding="utf-8")
        hf.load_framework.cache_clear()
    return evidence


__all__ = ["run", "collect", "benchmarks", "score_records", "redundancy", "discrimination",
           "bands_and_grades", "sensitivity", "replace"]
