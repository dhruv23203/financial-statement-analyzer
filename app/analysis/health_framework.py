"""Industry-specific Financial Health Score (methodology 3.0.0).

The score has five pillars with declared point budgets (non-financial companies:
Profitability 25, Liquidity 20, Solvency 20, Efficiency 15, Growth 20). Each pillar uses a
small set of metrics chosen to avoid double counting (see the correlation check in
``app/calibration/framework_calibration.py``):

1. **Framework** - the company's industry decides which metrics apply and which absolute
   guardrails hold (``config/health_framework.json``). Banks, insurers and other financial
   companies use capital, ROA/ROE and growth; their liquidity and efficiency cannot be
   measured from the provider's data and are reported as *not assessed* (never guessed).
2. **Metric points** - the metric's percentile among industry peers (the same sector, or
   the same financial framework; universe fallback below ``min_peer_observations``),
   mapped piecewise-linearly to a 0-1 fraction. Lower-is-better metrics are mirrored.
3. **Guardrails** - absolute thresholds can only *lower* the fraction (e.g. interest
   coverage below 1.5x scores at most 10 %, whatever the peers look like).
4. **Pillar points** = budget x weighted mean of the available metric fractions, provided
   at least ``min_pillar_coverage`` of the pillar's metric weight is available. Missing
   metrics never receive points.
5. **Overall** = 100 x sum(pillar points) / sum(assessed pillar budgets), withheld when
   less than ``min_assessed_share`` of the framework's budget or fewer than ``min_pillars``
   pillars could be assessed. **Confidence** = share of the framework's weight that was
   actually measured.

Every number carries an explanation (value, peer percentile, peer group, guardrail).
The score describes relative financial position; it is not a validated predictor of
default or of returns.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import date
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.analysis.health_score import OVERALL, CategoryScore
from app.analysis.ratios import FinancialDataset, RatioOutcome
from app.database.models import SCORE_SCORED, SCORE_UNAVAILABLE
from config.settings import CONFIG_DIR

FRAMEWORK_FILE = CONFIG_DIR / "health_framework.json"
NOT_ASSESSED = "NOT_ASSESSED"
FINANCIAL_SECTORS = ("Financial Services",)

RATIO_METRICS = {"debt_ratio": "debt_ratio", "roce": "return_on_capital_employed", "operating_margin": "operating_profit_margin",
                 "fcf_margin": "free_cash_flow_margin", "current_ratio": "current_ratio",
                 "interest_coverage": "interest_coverage", "asset_turnover": "asset_turnover",
                 "inventory_turnover": "inventory_turnover",
                 "receivables_turnover": "receivables_turnover", "roa": "return_on_assets",
                 "roe": "return_on_equity", "net_margin": "net_profit_margin"}
GROWTH_METRICS = {"revenue_cagr": "total_revenue", "operating_income_cagr": "operating_income",
                  "net_income_cagr": "net_income"}


@lru_cache(maxsize=4)
def load_framework(path: str | None = None) -> dict[str, Any]:
    return json.loads(Path(path or FRAMEWORK_FILE).read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------- classification
def classify(sector: str | None, industry: str | None, config: dict[str, Any]) -> str:
    """Framework key for a company. Financial Services is split by industry keyword
    (banks, insurers, others); other sectors map through each framework's sector list."""
    frameworks = config["frameworks"]
    if sector in FINANCIAL_SECTORS:
        text = (industry or "").lower()
        for key, fw in frameworks.items():
            if fw.get("financial") and any(k in text for k in fw.get("industry_keywords", [])):
                return key
        return "financial_services"
    for key, fw in frameworks.items():
        if sector in fw.get("sectors", []):
            return key
    return config.get("default_framework", "general")


def peer_group(framework: str, sector: str | None, config: dict[str, Any]) -> str:
    """Benchmark group: the framework for financial companies (their sector mixes banks,
    insurers and brokers), otherwise the company's own sector."""
    if config["frameworks"][framework].get("financial"):
        return f"FW:{framework}"
    return f"SECTOR:{sector}" if sector else f"FW:{framework}"


# ----------------------------------------------------------------------------- metric values
@dataclass
class MetricValue:
    value: float | None
    status: str                      # OK | MISSING | NOT_APPLICABLE | FLOOR
    note: str = ""


def _f(v) -> float | None:
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _cagr(ds: FinancialDataset, period: date, code: str, window: tuple[float, float],
          loss_floor: bool) -> MetricValue:
    """CAGR (%) of ``code`` ending at ``period`` over the longest available window of
    window[0]-window[1] years. Only data up to ``period`` is used (no look-ahead)."""
    last = _f(ds.get(period, code))
    if last is None:
        return MetricValue(None, "MISSING", f"{code} not reported for {period}")
    starts = [p for p in ds.periods if window[0] <= (period - p).days / 365.25 <= window[1]
              and _f(ds.get(p, code)) is not None]
    if not starts:
        return MetricValue(None, "MISSING", "no earlier year within the growth window")
    start = starts[0]
    first = _f(ds.get(start, code))
    years = (period - start).days / 365.25
    if last <= 0 and loss_floor:
        return MetricValue(None, "FLOOR", f"{code.replace('_', ' ')} is negative in the latest "
                                          "year (a loss, so growth earns no points)")
    if first <= 0 or last <= 0:
        return MetricValue(None, "MISSING", "growth rate undefined across a sign change "
                                            f"({first:,.0f} -> {last:,.0f})")
    return MetricValue(((last / first) ** (1 / years) - 1) * 100, "OK",
                       f"{start.isoformat()} to {period.isoformat()} ({years:.1f} years)")


def metric_values(ds: FinancialDataset, period: date,
                  ratios: dict[tuple[str, date], RatioOutcome],
                  config: dict[str, Any]) -> dict[str, MetricValue]:
    """Every framework metric for one fiscal year (whether or not a framework uses it)."""
    out: dict[str, MetricValue] = {}
    g = lambda code: _f(ds.get(period, code))  # noqa: E731
    for key, ratio_code in RATIO_METRICS.items():
        o = ratios.get((ratio_code, period))
        if o is not None and o.ok:
            out[key] = MetricValue(float(o.value), "OK")
        elif o is not None and o.reason_code == "IMMATERIAL_INVENTORY":
            out[key] = MetricValue(None, "NOT_APPLICABLE", "inventory is immaterial")
        elif o is not None and o.reason_code == "NO_INTEREST_EXPENSE":
            out[key] = MetricValue(None, "BEST", "no interest expense (nothing to cover)")
        elif o is not None and o.reason_code == "NON_POSITIVE_EQUITY" and key == "roe":
            out[key] = MetricValue(None, "FLOOR", "shareholders' equity is not positive")
        else:
            out[key] = MetricValue(None, "MISSING", (o.explanation or "") if o else
                                   "not calculated")
    # interest coverage: a company with no debt and no interest line has nothing to cover
    if out["interest_coverage"].status == "MISSING" and g("total_debt") == 0:
        out["interest_coverage"] = MetricValue(None, "BEST", "no debt")

    ocf, cl = g("operating_cash_flow"), g("current_liabilities")
    out["ocf_to_current_liabilities"] = (
        MetricValue(ocf / cl, "OK") if ocf is not None and cl is not None and cl > 0
        else MetricValue(None, "MISSING", "operating cash flow or current liabilities missing"))

    debt = g("total_debt")
    cash = g("cash_and_short_term_investments")
    if cash is None:
        cash = g("cash_and_equivalents")
    op, dep = g("operating_income"), g("reconciled_depreciation")
    if dep is None:
        dep = g("depreciation_amortization_cf")
    if None in (debt, cash, op, dep):
        out["net_debt_to_ebitda"] = MetricValue(None, "MISSING",
                                                "debt, cash, operating income or D&A missing")
    else:
        net_debt, ebitda = debt - cash, op + abs(dep)
        if net_debt <= 0:
            out["net_debt_to_ebitda"] = MetricValue(min(net_debt / ebitda, 0.0)
                                                    if ebitda > 0 else 0.0, "OK",
                                                    "net cash (more cash than debt)")
        elif ebitda <= 0:
            out["net_debt_to_ebitda"] = MetricValue(None, "FLOOR", "net debt with negative "
                                                    "operating EBITDA: debt cannot be serviced "
                                                    "from operations")
        else:
            out["net_debt_to_ebitda"] = MetricValue(net_debt / ebitda, "OK")

    equity, assets = g("stockholders_equity"), g("total_assets")
    if equity is None or assets is None or assets <= 0:
        out["equity_to_assets"] = MetricValue(None, "MISSING", "equity or total assets missing")
    elif equity <= 0:
        out["equity_to_assets"] = MetricValue(None, "FLOOR", "shareholders' equity is not "
                                                             "positive")
    else:
        out["equity_to_assets"] = MetricValue(equity / assets * 100, "OK")

    window = tuple(config.get("growth_window_years", [0.9, 3.2]))
    for key, code in GROWTH_METRICS.items():
        out[key] = _cagr(ds, period, code, window, loss_floor=(key != "revenue_cagr"))
    return out


# ----------------------------------------------------------------------------- scoring
def percentile_fraction(value: float, benchmark: dict[str, float], higher_is_better: bool,
                        mapping: list[list[float]]) -> tuple[float, float]:
    """(fraction 0-1, approximate peer percentile) by piecewise-linear interpolation of the
    benchmark percentiles. Ties in the benchmark are collapsed so the curve is monotonic."""
    points: dict[float, list[float]] = {}
    for pct, _ in mapping:
        x = benchmark.get(f"p{int(pct)}")
        if x is not None:
            points.setdefault(float(x), []).append(float(pct))
    xs = sorted(points)
    ps = [sum(points[x]) / len(points[x]) for x in xs]
    if not xs:
        return 0.5, 50.0
    if value <= xs[0]:
        pct = ps[0]
    elif value >= xs[-1]:
        pct = ps[-1]
    else:
        pct = ps[-1]
        for (x0, p0), (x1, p1) in zip(zip(xs, ps), zip(xs[1:], ps[1:])):
            if x0 <= value <= x1:
                pct = p0 + (p1 - p0) * (value - x0) / (x1 - x0)
                break
    if not higher_is_better:
        pct = 100 - pct
    # map percentile to points with the configured curve (identity by default)
    mx = [m[0] for m in mapping]
    my = [m[1] for m in mapping]
    if pct <= mx[0]:
        frac = my[0]
    elif pct >= mx[-1]:
        frac = my[-1]
    else:
        frac = my[-1]
        for (a0, b0), (a1, b1) in zip(zip(mx, my), zip(mx[1:], my[1:])):
            if a0 <= pct <= a1:
                frac = b0 + (b1 - b0) * (pct - a0) / (a1 - a0)
                break
    return max(0.0, min(1.0, frac)), pct


@dataclass
class MetricScore:
    metric: str
    name: str
    weight: float                 # weight within the pillar (points if fully earned)
    value: float | None
    unit: str
    status: str                   # SCORED | MISSING | NOT_APPLICABLE
    fraction: float | None = None
    points: float | None = None
    peer_percentile: float | None = None
    peer_group: str | None = None
    guardrail: str | None = None
    note: str = ""


@dataclass
class PillarScore:
    pillar: str
    max_points: float
    points: float | None
    status: str                   # SCORED | UNAVAILABLE | NOT_ASSESSED
    coverage: float
    explanation: str
    metrics: list[MetricScore] = field(default_factory=list)

    @property
    def percent(self) -> float | None:
        return None if self.points is None or not self.max_points else \
            100 * self.points / self.max_points


@dataclass
class FrameworkScore:
    framework: str
    framework_label: str
    period: date
    score: float | None
    status: str
    confidence: float
    confidence_label: str
    assessed_points: float
    total_points: float
    pillars: dict[str, PillarScore]
    explanation: str
    interpretation: str | None = None


def _benchmark(config: dict[str, Any], metric: str, group: str,
               framework: str) -> tuple[dict[str, float] | None, str]:
    benchmarks = config.get("benchmarks") or {}
    fin = bool(config["frameworks"][framework].get("financial"))
    for key in (group, f"FW:{framework}", "ALL:financial" if fin else "ALL:non_financial"):
        b = (benchmarks.get(key) or {}).get(metric)
        if b and b.get("n", 0) >= int(config.get("min_peer_observations", 30)):
            return b, key
    return None, ""


def _group_label(key: str) -> str:
    kind, _, name = key.partition(":")
    return {"SECTOR": f"{name} sector peers", "FW": f"{name.replace('_', ' ')} peers",
            "ALL": f"all {name.replace('_', '-')} companies"}.get(kind, key)


def score_metric(metric: str, weight: float, mv: MetricValue, framework: str, group: str,
                 config: dict[str, Any]) -> MetricScore:
    spec = config["metrics"][metric]
    ms = MetricScore(metric, spec["name"], weight, mv.value, spec["unit"], "MISSING",
                     note=mv.note)
    if mv.status == "NOT_APPLICABLE":
        ms.status, ms.weight = "NOT_APPLICABLE", 0.0
        return ms
    if mv.status == "MISSING":
        return ms
    if mv.status == "FLOOR":
        ms.status, ms.fraction, ms.points = "SCORED", 0.0, 0.0
        ms.guardrail = mv.note
        return ms
    if mv.status == "BEST":
        ms.status, ms.fraction, ms.points = "SCORED", 1.0, weight
        return ms
    bench, used = _benchmark(config, metric, group, framework)
    if bench is None:
        ms.note = "no peer benchmark with enough observations"
        return ms
    frac, pct = percentile_fraction(mv.value, bench, spec["higher_is_better"],
                                    config["percentile_points"])
    ms.peer_percentile, ms.peer_group = round(pct, 1), _group_label(used)
    for rule in config["frameworks"][framework].get("guardrails", []):
        if rule["metric"] != metric:
            continue
        hit = ("below" in rule and mv.value < rule["below"]) or \
              ("above" in rule and mv.value > rule["above"])
        if hit:
            # always disclosed when the condition holds; it can only lower the points
            frac = min(frac, rule["cap"])
            ms.guardrail = ms.guardrail or rule["reason"]
    ms.status, ms.fraction, ms.points = "SCORED", frac, weight * frac
    return ms


def _band(value: float, bands: list[dict[str, Any]]) -> str | None:
    for band in bands:
        if value >= float(band["min"]):
            return str(band["label"])
    return None


def score_company(ds: FinancialDataset, period: date,
                  ratios: dict[tuple[str, date], RatioOutcome], sector: str | None,
                  industry: str | None, config: dict[str, Any] | None = None) -> FrameworkScore:
    config = config or load_framework()
    framework = classify(sector, industry, config)
    fw = config["frameworks"][framework]
    group = peer_group(framework, sector, config)
    values = metric_values(ds, period, ratios, config)
    min_cov = float(config["min_pillar_coverage"])
    pillars: dict[str, PillarScore] = {}
    for name in config["pillars"]:
        spec = fw["pillars"][name]
        if not spec.get("max"):
            pillars[name] = PillarScore(name, 0.0, None, NOT_ASSESSED, 0.0,
                                        spec.get("not_assessed", "not assessed"))
            continue
        weights = spec["metrics"]
        total_w = sum(weights.values())
        metrics = [score_metric(m, w * spec["max"] / total_w, values[m], framework, group,
                                config) for m, w in weights.items()]
        applicable = sum(m.weight for m in metrics)          # not-applicable weight removed
        scored = [m for m in metrics if m.status == "SCORED"]
        scored_w = sum(m.weight for m in scored)
        coverage = scored_w / applicable if applicable else 0.0
        if not scored or coverage < min_cov:
            pillars[name] = PillarScore(
                name, float(spec["max"]), None, SCORE_UNAVAILABLE, round(coverage, 4),
                f"Insufficient data: {coverage:.0%} of the pillar's metric weight available "
                f"(minimum {min_cov:.0%}).", metrics)
            continue
        points = spec["max"] * sum(m.points for m in scored) / scored_w
        pillars[name] = PillarScore(
            name, float(spec["max"]), round(points, 2), SCORE_SCORED, round(coverage, 4),
            "; ".join(_metric_text(m) for m in metrics), metrics)

    total = sum(float(fw["pillars"][p].get("max") or 0) for p in config["pillars"])
    assessed = [p for p in pillars.values() if p.status == SCORE_SCORED]
    assessed_pts = sum(p.max_points for p in assessed)
    measured = sum(p.max_points * p.coverage for p in assessed)
    confidence = measured / total if total else 0.0
    conf_label = _band(confidence, config["confidence_bands"]) or "Low"
    if fw.get("financial") and conf_label == "High":
        conf_label = "Medium"       # liquidity and asset quality cannot be measured
    enough = assessed_pts >= float(config["min_assessed_share"]) * total and \
        len(assessed) >= int(config["min_pillars"])
    if not enough:
        return FrameworkScore(framework, fw["label"], period, None, SCORE_UNAVAILABLE,
                              round(confidence, 4), conf_label, assessed_pts, total, pillars,
                              f"Withheld: only {len(assessed)} pillar(s) worth "
                              f"{assessed_pts:.0f} of {total:.0f} points could be assessed.")
    score = 100 * sum(p.points for p in assessed) / assessed_pts
    parts = ", ".join(f"{p.pillar.title()} {p.points:.1f}/{p.max_points:.0f}" for p in assessed)
    note = "" if assessed_pts == total else \
        f" Rescaled to 100 from the {assessed_pts:.0f} assessable points."
    result = FrameworkScore(framework, fw["label"], period, round(score, 2), SCORE_SCORED,
                            round(confidence, 4), conf_label, assessed_pts, total, pillars,
                            f"{fw['label']} framework: {parts}.{note}")
    result.interpretation = _band(score, config.get("interpretation_bands") or [])
    return result


def _metric_text(m: MetricScore) -> str:
    if m.status == "NOT_APPLICABLE":
        return f"{m.name}: not applicable ({m.note}), weight removed"
    if m.status == "MISSING":
        return f"{m.name}: unavailable ({m.note or 'missing'}), no points"
    value = "n/a" if m.value is None else f"{m.value:,.2f}{'%' if m.unit == '%' else 'x'}"
    peer = f", {m.peer_percentile:.0f}th percentile of {m.peer_group}" \
        if m.peer_percentile is not None else ""
    guard = f" - capped: {m.guardrail}" if m.guardrail else ""
    return f"{m.name} {value}{peer} -> {m.points:.1f}/{m.weight:.1f}{guard}"


# ----------------------------------------------------------------------------- storage shape
def to_category_scores(result: FrameworkScore) -> list[CategoryScore]:
    """Pillars as stored rows: score = pillar percentage (0-100) so every existing reader
    keeps working; points, budget and metric detail live in ``components``."""
    rows = []
    for p in result.pillars.values():
        pct = p.percent
        comp = {"max_points": p.max_points, "points": p.points, "status": p.status,
                "metrics": [asdict(m) for m in p.metrics]}
        status = SCORE_SCORED if pct is not None else SCORE_UNAVAILABLE
        rows.append(CategoryScore(p.pillar, result.period,
                                  None if pct is None else Decimal(str(round(pct, 2))),
                                  status, p.coverage,
                                  (f"Not assessed: {p.explanation}" if p.status == NOT_ASSESSED
                                   else p.explanation), [comp]))
    overall_comp = {"framework": result.framework, "framework_label": result.framework_label,
                    "confidence": result.confidence, "confidence_label": result.confidence_label,
                    "assessed_points": result.assessed_points, "total_points": result.total_points,
                    "pillars": {k: {"points": v.points, "max": v.max_points, "status": v.status}
                                for k, v in result.pillars.items()}}
    overall = CategoryScore(OVERALL, result.period,
                            None if result.score is None else Decimal(str(result.score)),
                            result.status, result.confidence, result.explanation, [overall_comp],
                            result.interpretation)
    return rows + [overall]


def score_all(ds: FinancialDataset, ratios: dict[tuple[str, date], RatioOutcome],
              sector: str | None, industry: str | None,
              config: dict[str, Any] | None = None) -> dict[date, list[CategoryScore]]:
    config = config or load_framework()
    return {p: to_category_scores(score_company(ds, p, ratios, sector, industry, config))
            for p in ds.periods}
