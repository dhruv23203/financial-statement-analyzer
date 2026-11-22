"""ML prediction tab: distress-onset model, kept clearly apart from the rule-based
Financial Health Score."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import streamlit as st

from app.ml import distress as dm
from app.reporting import ui

LIMITATIONS = [
    "The target is a **proxy**: a net loss or negative operating cash flow next year for a "
    "company that is profitable and cash-generating now. It is not default, bankruptcy or "
    "delisting - those events are not in the data.",
    "Small sample: about 2,200 labelled company-years with about 100 onset events; the test "
    "year has only 33. Confidence intervals are wide, and the three models are within each "
    "other's intervals.",
    "Scores are **not calibrated probabilities**: class weighting (to learn the rare class) "
    "raises them. Read them as a ranking; the test-year precision shows how often a flag "
    "was right.",
    "Survivorship: the universe is today's NIFTY 500 / S&P 500 members, so companies that "
    "failed and left the indices are missing - real risk is understated.",
    "The test year (fiscal 2025 -> 2026) is dominated by Indian March year-ends (US "
    "December 2025 statements are mostly not yet followed by a 2026 year).",
    "Statements come from Yahoo Finance; a provider error flows into the features. "
    "Financial companies are excluded (deposits and loans make the target meaningless).",
    "Educational model - not investment advice and not a credit rating.",
]


def _html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def load_report() -> dict[str, Any] | None:
    path = dm.MODEL_FILE.parent / "distress_evaluation.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def metrics_table(report: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for name, r in report["models"].items():
        m = r["test"]
        ci = m.get("roc_auc_ci95") or [None, None]
        rows.append({"Model": name + (" (deployed)" if name == report["deployed_model"] else ""),
                     "ROC-AUC": m["roc_auc"], "95% CI": f"{ci[0]}-{ci[1]}",
                     "PR-AUC": m["pr_auc"], "Precision": m["precision"], "Recall": m["recall"],
                     "F1": m["f1"], "Validation PR-AUC": r["validation_pr_auc"]})
    for name, m in report["baselines"].items():
        ci = m.get("roc_auc_ci95") or ["", ""]
        rows.append({"Model": f"Baseline: {name}", "ROC-AUC": m.get("roc_auc"),
                     "95% CI": f"{ci[0]}-{ci[1]}" if ci[0] != "" else "",
                     "PR-AUC": m.get("pr_auc"), "Precision": m.get("precision", ""),
                     "Recall": m.get("recall", ""), "F1": m.get("f1", ""),
                     "Validation PR-AUC": ""})
    return pd.DataFrame(rows)


def confusion_frame(cm: dict[str, int]) -> pd.DataFrame:
    return pd.DataFrame({"Predicted: no distress": [cm["tn"], cm["fn"]],
                         "Predicted: distress": [cm["fp"], cm["tp"]]},
                        index=["Actual: no distress", "Actual: distress"])


def render_prediction(pred: dm.Prediction, model: dict[str, Any], ticker: str) -> None:
    if pred.status != "OK":
        label = {"NOT_APPLICABLE": "Model not applicable", "INSUFFICIENT_DATA":
                 "Not enough data", "NO_MODEL": "No model available"}.get(pred.status,
                                                                           pred.status)
        st.info(f"**{label} for {ticker}.** {pred.reason}")
        if pred.missing:
            st.caption("Missing features: " + ", ".join(pred.missing))
        return
    test = model.get("test") or {}
    _html(ui.cards([
        ("Predicted distress-onset risk", f"{pred.probability:.0%}",
         f"model score for fiscal year to {pred.period}", ui.DOWN if pred.flagged else None,
         "Uncalibrated model score (class-weighted training raises it): read it as a "
         "ranking, not as a literal probability."),
        ("Model flag", "Flagged" if pred.flagged else "Not flagged",
         f"threshold {model['threshold']:.2f} (chosen on the validation year)",
         ui.DOWN if pred.flagged else ui.UP, None),
        ("How often a flag was right", f"{test.get('precision', 0):.0%}",
         f"precision in the test year (base rate {test.get('positives', 0)} of "
         f"{test.get('n', 0)})", None, None),
        ("Model", model["model"].split(" (")[0], f"trained {model['trained_at']}", None,
         model["model"]),
    ]))
    rows = [{"Feature": name,
             "Value": "N/A" if value is None else f"{value:,.2f}",
             "Effect on the risk score": f"{'raises' if c > 0 else 'lowers'} "
                                         f"({c:+.2f} log-odds)"}
            for name, c, value in (pred.contributions or [])[:8]]
    st.markdown("**Most important contributing features for this company**")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Contributions are exact per-feature effects on the model's log-odds for "
               "this company (TreeSHAP for XGBoost; coefficient x standardised value for "
               "logistic regression). Missing features are filled with the training median "
               "and listed: " + (", ".join(pred.missing) if pred.missing else "none") + ".")


def render(analysis, ticker: str | None) -> None:
    _html(ui.section("ML prediction: distress onset (experimental)", kind="risk", hint=
                     "a statistical model - separate from the rule-based Financial Health "
                     "Score"))
    st.warning("This is a **machine-learning model trained on historical outcomes**, not "
               "the rule-based Financial Health Score. The two answer different questions: "
               "the Health Score describes the company's position today against its peers; "
               "this model estimates the risk that a currently healthy company reports a "
               "loss or negative operating cash flow next year.", icon="🤖")
    model, report = dm.load_model(), load_report()
    if model is None or report is None:
        st.info("No trained model. Run `python main.py train-distress` on the full database.")
        return
    if analysis is not None:
        render_prediction(dm.predict_for_analysis(model, analysis), model, ticker or "")
    else:
        st.caption("Select an analysed company in the sidebar to see its prediction.")

    _html(ui.section("Model evaluation", "untouched test year, next to baselines", kind="data"))
    split = report["split"]
    st.caption(
        f"Chronological split by the outcome year - train: outcomes "
        f"{split['train']['label_years']} ({split['train']['rows']} company-years, "
        f"{split['train']['positives']} events); validation: {split['validation']['label_years']}"
        f" ({split['validation']['rows']}, {split['validation']['positives']} events); test: "
        f"{split['test']['label_years']} ({split['test']['rows']}, "
        f"{split['test']['positives']} events). Features come from fiscal year t (and t-1), "
        "the outcome from year t+1, so no future information is used. Preprocessing is "
        "fitted on the training years only.")
    st.dataframe(metrics_table(report), hide_index=True, width="stretch")
    st.caption(f"Model choice ({report['selection_rule']}), declared before the test year was "
               f"scored: **{report['chosen_model']}** had the best validation PR-AUC and is "
               "deployed, even though another model scored higher on the test year - "
               "switching after seeing the test would leak it into the choice. The "
               "differences between models are within their confidence intervals.")
    c1, c2 = st.columns([1, 1.4])
    with c1:
        st.markdown(f"**Confusion matrix - {report['deployed_model']}, test year**")
        st.dataframe(confusion_frame(report["models"][report["deployed_model"]]["test"]
                                     ["confusion_matrix"]), width="stretch")
    with c2:
        key = next((k for k in report["importance"] if k.startswith(
            report["deployed_model"])), next(iter(report["importance"])))
        imp = sorted(report["importance"][key].items(), key=lambda kv: -abs(kv[1]))[:8]
        st.markdown(f"**Most important features overall** ({key})")
        st.dataframe(pd.DataFrame(imp, columns=["Feature", "Importance"]), hide_index=True,
                     width="stretch")
    if report.get("features_dropped"):
        st.caption("Removed as redundant (Spearman |rho| > 0.85 on the training years): " +
                   "; ".join(f"{dm.FEATURES.get(d['feature'], d['feature'])} (with "
                             f"{dm.FEATURES.get(d['correlated_with'], d['correlated_with'])}, "
                             f"{d['spearman']})" for d in report["features_dropped"]))

    _html(ui.section("Limitations", kind="data"))
    for item in LIMITATIONS:
        st.markdown(f"- {item}")
