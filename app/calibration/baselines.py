"""Naive baselines for the distress-proxy label used to validate the health score.

The calibration label is "next-year net loss or negative operating cash flow". Two of the
score's categories (profitability and cash flow) measure *this* year's profit and cash
flow, so part of any AUC on this label is simple persistence. These baselines make that
visible:

* ``persistence``  - this year's distress status (net loss or negative OCF) as the score;
* ``two_ratio``    - min(net margin, OCF margin) of this year, lower = riskier.

Both use the same universe rules as the calibration: non-financial companies, latest
fetch run per company, adjacent fiscal years (330-400 days apart), revenue > 0.

Run ``python -m app.calibration.baselines`` to refresh ``config/validation_evidence.json``
(needs scikit-learn, a development dependency only).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.settings import CONFIG_DIR, get_settings

EVIDENCE_FILE = CONFIG_DIR / "validation_evidence.json"
FINANCIAL_SECTORS = ("Financial Services",)

QUERY = """
SELECT co.ticker, co.sector, s.fiscal_period_end AS period, s.metric_code AS metric,
       s.value AS value
FROM standardized_financial_facts s
JOIN companies co ON co.id = s.company_id
JOIN (SELECT company_id, MAX(fetch_run_id) AS fetch_id
      FROM standardized_financial_facts GROUP BY company_id) latest
  ON latest.company_id = s.company_id AND latest.fetch_id = s.fetch_run_id
WHERE s.metric_code IN ('net_income', 'operating_cash_flow', 'total_revenue')
  AND s.value IS NOT NULL
"""


def label_frame(db_path: Path) -> pd.DataFrame:
    """One row per company-year with this year's figures and next year's distress label."""
    with sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True) as conn:
        raw = pd.read_sql(QUERY, conn)
    if raw.empty:
        return raw
    wide = raw.pivot_table(index=["ticker", "sector", "period"], columns="metric",
                           values="value").reset_index()
    wide["period"] = pd.to_datetime(wide["period"])
    wide = wide[~wide["sector"].isin(FINANCIAL_SECTORS)].sort_values(["ticker", "period"])
    wide = wide.dropna(subset=["net_income", "operating_cash_flow", "total_revenue"])
    wide["distress"] = ((wide["net_income"] < 0) | (wide["operating_cash_flow"] < 0)).astype(int)
    grouped = wide.groupby("ticker")
    wide["label_next"] = grouped["distress"].shift(-1)
    wide["gap_days"] = (grouped["period"].shift(-1) - wide["period"]).dt.days
    rows = wide[wide["label_next"].notna() & wide["gap_days"].between(330, 400)
                & (wide["total_revenue"] > 0)].copy()
    rows["label_next"] = rows["label_next"].astype(int)
    return rows


def baseline_aucs(rows: pd.DataFrame) -> dict[str, Any]:
    from sklearn.metrics import roc_auc_score

    if rows.empty or rows["label_next"].nunique() < 2:
        return {"observations": int(len(rows)), "note": "not enough labelled rows"}
    margin = rows["net_income"] / rows["total_revenue"]
    ocf_margin = rows["operating_cash_flow"] / rows["total_revenue"]
    two_ratio = -np.minimum(margin, ocf_margin)
    return {
        "observations": int(len(rows)), "companies": int(rows["ticker"].nunique()),
        "positive_rate": round(float(rows["label_next"].mean()), 4),
        "persistence_auc": round(float(roc_auc_score(rows["label_next"], rows["distress"])), 3),
        "two_ratio_auc": round(float(roc_auc_score(rows["label_next"], two_ratio)), 3),
    }


def update_evidence(db_path: Path | None = None, path: Path = EVIDENCE_FILE) -> dict[str, Any]:
    result = baseline_aucs(label_frame(db_path or get_settings().db_path))
    evidence = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    evidence.setdefault("score_validation", {})["naive_baselines"] = {
        **result,
        "definitions": {
            "persistence": "this year's status (net loss or negative operating cash flow) "
                           "used as the prediction",
            "two_ratio": "min(net margin, operating cash flow margin) this year, lower = "
                         "riskier",
        },
        "source": "app/calibration/baselines.py on the stored statements",
    }
    path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(update_evidence())
