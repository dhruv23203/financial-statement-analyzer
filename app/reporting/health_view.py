"""Transparent Financial Health breakdown (score 3.0) for the Company tab.

Shows the headline score, the points earned in each pillar against its budget, each
metric's value, peer percentile, points and any guardrail, and a plain-language reason
for every pillar. Everything comes from the stored score components - nothing is
recomputed or estimated in the view.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from app.reporting import ui

PILLAR_ORDER = ("PROFITABILITY", "LIQUIDITY", "SOLVENCY", "EFFICIENCY", "GROWTH")


def _component(score) -> dict[str, Any]:
    comps = getattr(score, "components", None) or []
    return comps[0] if comps and isinstance(comps[0], dict) else {}


def pillar_rows(latest: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per pillar: name, points, max, status, reason text and metric table."""
    rows = []
    for name in PILLAR_ORDER:
        c = latest.get(name)
        if c is None:
            continue
        comp = _component(c)
        metrics = comp.get("metrics") or []
        rows.append({"pillar": name, "points": comp.get("points"),
                     "max": comp.get("max_points") or 0, "status": comp.get("status"),
                     "explanation": c.explanation, "metrics": metrics,
                     "reason": reason(name, comp, c.explanation)})
    return rows


def ordinal(n: float) -> str:
    n = int(round(n))
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _lower_first(text: str) -> str:
    """Lower-case only a leading capital word (keeps acronyms such as ROCE and CAGR)."""
    return text if text[:2].isupper() else text[:1].lower() + text[1:]


def reason(pillar: str, comp: dict[str, Any], explanation: str) -> str:
    """Why the pillar scored what it did: strongest and weakest metric, guardrails and
    missing inputs - in words."""
    if comp.get("status") == "NOT_ASSESSED":
        return explanation
    metrics = [m for m in comp.get("metrics") or [] if m.get("status") == "SCORED"]
    missing = [m["name"] for m in comp.get("metrics") or [] if m.get("status") == "MISSING"]
    if not metrics:
        return explanation
    parts = []
    best = max(metrics, key=lambda m: m.get("fraction") or 0)
    worst = min(metrics, key=lambda m: m.get("fraction") or 0)
    if best.get("peer_percentile") is not None:
        parts.append(f"strongest: {_lower_first(best['name'])} at the "
                     f"{ordinal(best['peer_percentile'])} percentile of {best.get('peer_group')}")
    if worst is not best and worst.get("peer_percentile") is not None and             (worst.get("fraction") or 0) < (best.get("fraction") or 0):
        parts.append(f"weakest: {_lower_first(worst['name'])} at the "
                     f"{ordinal(worst['peer_percentile'])} percentile")
    for m in metrics:
        if m.get("guardrail"):
            parts.append(f"{_lower_first(m['name'])} capped because {m['guardrail']}")
    if missing:
        parts.append(f"not measured: {', '.join(_lower_first(n) for n in missing)} (no "
                     "points, lower confidence)")
    text = "; ".join(parts)
    return text[:1].upper() + text[1:] + "." if parts else explanation


def metric_table(metrics: list[dict[str, Any]]) -> pd.DataFrame:
    def fmt_value(m):
        v = m.get("value")
        if v is None:
            return "N/A"
        return f"{v:,.1f}%" if m.get("unit") == "%" else f"{v:,.2f}x"

    return pd.DataFrame([{
        "Metric": m["name"],
        "Value": fmt_value(m),
        "Peer percentile": "-" if m.get("peer_percentile") is None
        else f"{m['peer_percentile']:.0f}",
        "Points": ("not applicable" if m.get("status") == "NOT_APPLICABLE" else
                   "no data" if m.get("points") is None else
                   f"{m['points']:.1f} / {m['weight']:.1f}"),
        "Note": m.get("guardrail") and f"capped: {m['guardrail']}" or m.get("note") or "",
    } for m in metrics])


def render(analysis) -> None:
    latest = {c.category: c for c in analysis.latest_scores()}
    overall = latest.get("OVERALL")
    meta = _component(overall) if overall is not None else {}
    if not meta.get("framework"):
        return                        # analysis stored before score 3.0
    score = None if overall.score is None else float(overall.score)
    rows = pillar_rows(latest)
    head = "N/A" if score is None else f"{score:.0f}/100"
    st.markdown(f"#### Financial Health: {head}")
    st.caption(f"{meta.get('framework_label')} framework · confidence "
               f"{meta.get('confidence_label')} ({meta.get('confidence', 0):.0%} of the "
               "framework's weight measured) · peers = same industry; guardrails apply "
               "absolute minimum standards. Describes historical financial position - not a "
               "prediction or a recommendation.")
    cards = []
    for r in rows:
        if r["status"] == "NOT_ASSESSED":
            cards.append((r["pillar"].title(), "Not assessed", "data not available for this "
                          "industry", None, r["explanation"]))
        elif r["points"] is None:
            cards.append((r["pillar"].title(), f"N/A / {r['max']:.0f}", "insufficient data",
                          None, r["explanation"]))
        else:
            cards.append((r["pillar"].title(), f"{r['points']:.0f} / {r['max']:.0f}",
                          f"{100 * r['points'] / r['max']:.0f}% of the pillar",
                          ui.score_color(100 * r["points"] / r["max"]), r["reason"]))
    st.markdown(ui.cards(cards), unsafe_allow_html=True)
    if score is not None and meta.get("assessed_points") != meta.get("total_points"):
        st.caption(f"Only {meta['assessed_points']:.0f} of {meta['total_points']:.0f} points "
                   "could be assessed, so the overall score is rescaled to 100 from the "
                   "assessable pillars (and the confidence is lower).")
    with st.expander("Why each pillar scored what it did (metric contributions)"):
        for r in rows:
            label = r["pillar"].title()
            if r["status"] == "NOT_ASSESSED":
                st.markdown(f"**{label} - not assessed.** {r['explanation']}")
                continue
            pts = "N/A" if r["points"] is None else f"{r['points']:.1f} / {r['max']:.0f}"
            st.markdown(f"**{label}: {pts}.** {r['reason']}")
            if r["metrics"]:
                st.dataframe(metric_table(r["metrics"]), hide_index=True, width="stretch")
