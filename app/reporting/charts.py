"""PNG charts with Matplotlib (non-interactive 'Agg' backend, safe on servers/CI).

Amount charts are always labelled with the reporting currency and are never drawn
for a mix of currencies.
"""

from __future__ import annotations

import logging
from datetime import date
from decimal import Decimal
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be selected first)

from app.analysis.health_score import OVERALL, CategoryScore  # noqa: E402
from app.analysis.ratios import FinancialDataset, RatioOutcome  # noqa: E402

logger = logging.getLogger(__name__)


def _scale(currency: str | None) -> tuple[float, str]:
    if currency == "INR":
        return 1e7, "INR crore"
    return 1e6, f"{currency or 'currency'} millions"


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def financial_trend_chart(ds: FinancialDataset, path: Path) -> Path | None:
    """Grouped bars: revenue, net income, operating cash flow, total debt by year."""
    periods = ds.periods
    if not periods:
        return None
    factor, unit = _scale(ds.currency)
    metrics = [("total_revenue", "Revenue"), ("net_income", "Net income"),
               ("operating_cash_flow", "Operating cash flow"), ("total_debt", "Total debt")]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.8 / len(metrics)
    for i, (code, label) in enumerate(metrics):
        xs, ys = [], []
        for j, period in enumerate(periods):
            value = ds.get(period, code)
            if value is not None:  # missing values are simply not drawn (never as 0)
                xs.append(j + i * width)
                ys.append(float(value) / factor)
        if xs:
            ax.bar(xs, ys, width=width, label=label)
    ax.set_xticks([j + 0.4 - width / 2 for j in range(len(periods))])
    ax.set_xticklabels([str(p) for p in periods])
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_ylabel(unit)
    ax.set_title(f"{ds.ticker}: key financials by fiscal year (missing values not plotted)")
    ax.legend(fontsize=8)
    return _save(fig, path)


def ratio_trend_chart(outcomes: list[RatioOutcome], codes: list[str], title: str,
                      path: Path) -> Path | None:
    fig, ax = plt.subplots(figsize=(8, 4))
    drawn = False
    for code in codes:
        points = sorted((o.period, float(o.value)) for o in outcomes
                        if o.code == code and o.ok)
        if points:
            drawn = True
            ax.plot([str(p) for p, _ in points], [v for _, v in points], marker="o",
                    label=next(o.definition.name for o in outcomes if o.code == code))
    if not drawn:
        plt.close(fig)
        return None
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return _save(fig, path)


def health_score_chart(ticker: str, categories: list[CategoryScore], path: Path) -> Path | None:
    items = [(c.category.title().replace("_", " "), float(c.score))
             for c in categories if c.score is not None and c.category != OVERALL]
    overall = next((c for c in categories if c.category == OVERALL), None)
    if not items:
        return None
    fig, ax = plt.subplots(figsize=(7, 3.8))
    names = [n for n, _ in items]
    values = [v for _, v in items]
    bars = ax.barh(names, values, color="#3b6fb6")
    for bar, value in zip(bars, values):
        ax.text(value + 1, bar.get_y() + bar.get_height() / 2, f"{value:.0f}", va="center")
    ax.set_xlim(0, 110)
    ax.invert_yaxis()
    overall_text = "N/A" if overall is None or overall.score is None else \
        f"{float(overall.score):.1f}"
    period = categories[0].period if categories else ""
    ax.set_title(f"{ticker} financial health FY ending {period}: overall {overall_text}/100")
    return _save(fig, path)


def comparison_ratio_chart(table: dict[str, dict[str, Decimal | None]], codes: list[str],
                           labels: dict[str, str], title: str, path: Path) -> Path | None:
    """Grouped bars: one group per ratio, one bar per company (missing = no bar)."""
    tickers = list(table)
    if not tickers:
        return None
    fig, ax = plt.subplots(figsize=(10, 4.5))
    width = 0.8 / len(tickers)
    drawn = False
    for i, ticker in enumerate(tickers):
        xs, ys = [], []
        for j, code in enumerate(codes):
            value = table[ticker].get(code)
            if value is not None:
                xs.append(j + i * width)
                ys.append(float(value))
        if xs:
            drawn = True
            ax.bar(xs, ys, width=width, label=ticker)
    if not drawn:
        plt.close(fig)
        return None
    ax.set_xticks([j + 0.4 - width / 2 for j in range(len(codes))])
    ax.set_xticklabels([labels[c] for c in codes], rotation=20, ha="right", fontsize=8)
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_title(title)
    ax.legend(fontsize=8)
    return _save(fig, path)


def comparison_score_chart(scores: dict[str, Decimal | None], periods: dict[str, date | None],
                           path: Path) -> Path | None:
    items = [(t, float(s)) for t, s in scores.items() if s is not None]
    if not items:
        return None
    fig, ax = plt.subplots(figsize=(7, 3.5))
    labels = [f"{t}\n(FY {periods.get(t)})" for t, _ in items]
    bars = ax.bar(labels, [v for _, v in items], color="#5aa469")
    for bar, (_, value) in zip(bars, items):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.1f}", ha="center")
    ax.set_ylim(0, 110)
    ax.set_ylabel("Overall health score (0-100)")
    ax.set_title("Financial health score comparison")
    return _save(fig, path)
