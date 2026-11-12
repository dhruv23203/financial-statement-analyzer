"""Interactive Plotly figures for the dashboard (hover, zoom, range buttons).

The static PNG charts for exported reports stay in :mod:`app.reporting.charts`
(Matplotlib). These builders only take plain Python / pandas data, so they have no
database or Streamlit dependency and are easy to test.
"""

from __future__ import annotations

from typing import Mapping

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.reporting.ui import ACCENT, DOWN, MUTED, UP, score_bands, score_color

PALETTE = ["#2962ff", "#26a69a", "#f9a825", "#ab47bc", "#ef5350", "#29b6f6", "#8d6e63",
           "#ec407a"]
# red -> amber -> teal, centred on the median score of 50
SCORE_SCALE = [[0.0, "#c62828"], [0.3, "#ef5350"], [0.45, "#f9a825"], [0.55, "#9ccc65"],
               [0.7, "#26a69a"], [1.0, "#00796b"]]
GRID = "rgba(139,151,168,0.15)"


def _style(fig: go.Figure, height: int, legend: bool = True) -> go.Figure:
    fig.update_layout(
        template="plotly_dark", height=height, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=30, b=10),
        font=dict(family="Inter, -apple-system, Segoe UI, sans-serif", size=12, color="#c9d1d9"),
        hoverlabel=dict(bgcolor="#172033", bordercolor="#2962ff", font_size=12,
                        font_family="Inter, sans-serif"),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
        colorway=PALETTE)
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def score_gauge(score: float | None, label: str = "") -> go.Figure:
    """Speedometer for the 0-100 health score. The coloured zones are the configured
    interpretation bands (universe quartiles), so colour and label always agree.

    A missing score is drawn as an empty, greyed-out dial with the text
    "N/A - Insufficient applicable financial data" - never as a needle at 0, which
    would look like a genuine score of zero."""
    bands = sorted(score_bands(), key=lambda b: b[0])  # lowest band first
    edges = [b[0] for b in bands] + [100.0]
    missing = score is None
    steps = [dict(range=[edges[i], edges[i + 1]],
                  color="rgba(139,151,168,.12)" if missing else _alpha(bands[i][2], 0.55))
             for i in range(len(bands))]
    fig = go.Figure(go.Indicator(
        mode="gauge" if missing else "gauge+number",
        value=0 if missing else score,
        number=dict(font=dict(size=44), valueformat=".1f"),
        title=dict(text=label, font=dict(size=14, color=MUTED)),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor=MUTED,
                      tickvals=[0] + [round(e, 1) for e in edges[1:-1]] + [100]),
            # no bar at all when the score is missing
            bar=dict(color="rgba(0,0,0,0)" if missing else "#e6edf3",
                     thickness=0 if missing else 0.22),
            bgcolor="rgba(0,0,0,0)", borderwidth=0, steps=steps)))
    if missing:
        fig.add_annotation(text="N/A<br><span style='font-size:12px'>Insufficient applicable"
                                "<br>financial data</span>",
                           x=0.5, y=0.12, xref="paper", yref="paper", showarrow=False,
                           font=dict(size=26, color=MUTED))
    return _style(fig, 250, legend=False).update_layout(margin=dict(l=25, r=25, t=40, b=0))


def _alpha(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def category_radar(series: Mapping[str, Mapping[str, float | None]],
                   height: int = 340) -> go.Figure:
    """'Snowflake' of category scores (0-100); one filled shape per company."""
    fig = go.Figure()
    for i, (name, scores) in enumerate(series.items()):
        labels = [k.replace("_", " ").title() for k in scores]
        values = [0 if v is None else v for v in scores.values()]
        color = PALETTE[i % len(PALETTE)]
        fig.add_trace(go.Scatterpolar(
            r=values + values[:1], theta=labels + labels[:1], name=name, fill="toself",
            line=dict(color=color, width=2), opacity=0.75 if len(series) > 1 else 0.9,
            hovertemplate="%{theta}: %{r:.1f}<extra>" + name + "</extra>"))
    fig.update_layout(polar=dict(
        bgcolor="rgba(0,0,0,0)",
        radialaxis=dict(range=[0, 100], gridcolor=GRID, tickfont=dict(size=9, color=MUTED),
                        tickvals=[25, 50, 75]),
        angularaxis=dict(gridcolor=GRID, tickfont=dict(size=12))))
    return _style(fig, height, legend=len(series) > 1).update_layout(
        margin=dict(l=70, r=70, t=40, b=30))


def grouped_bars(data: Mapping[str, Mapping[str, float | None]], unit: str,
                 height: int = 320) -> go.Figure:
    """Bars per period for several metrics (e.g. revenue, profit, cash flow, debt)."""
    fig = go.Figure()
    for name, points in data.items():
        periods = sorted(points)
        fig.add_trace(go.Bar(x=periods, y=[points[p] for p in periods], name=name,
                             hovertemplate="%{x}: %{y:,.1f} " + unit + "<extra>" + name +
                                           "</extra>"))
    fig.update_layout(barmode="group", bargap=0.25, yaxis_title=unit)
    fig.update_xaxes(type="category")
    return _style(fig, height)


def line_chart(data: Mapping[str, Mapping[str, float | None]], y_title: str = "",
               height: int = 300) -> go.Figure:
    """Lines with markers over periods (ratio trends)."""
    fig = go.Figure()
    for name, points in data.items():
        periods = sorted(points)
        fig.add_trace(go.Scatter(x=periods, y=[points[p] for p in periods], name=name,
                                 mode="lines+markers", line=dict(width=2.5),
                                 marker=dict(size=7)))
    fig.update_layout(hovermode="x unified", yaxis_title=y_title)
    fig.update_xaxes(type="category")
    return _style(fig, height)


def price_chart(prices: pd.DataFrame, style: str = "Candles", height: int = 430) -> go.Figure:
    """TradingView-style price chart: candles or area line, with volume underneath."""
    has_volume = "Volume" in prices and prices["Volume"].fillna(0).sum() > 0
    fig = make_subplots(rows=2 if has_volume else 1, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03, row_heights=[0.78, 0.22] if has_volume else [1])
    ohlc = all(c in prices for c in ("Open", "High", "Low", "Close"))
    if style == "Candles" and ohlc:
        fig.add_trace(go.Candlestick(
            x=prices.index, open=prices["Open"], high=prices["High"], low=prices["Low"],
            close=prices["Close"], name="Price", increasing_line_color=UP,
            decreasing_line_color=DOWN, increasing_fillcolor=UP, decreasing_fillcolor=DOWN),
            row=1, col=1)
    else:
        close = prices["Close"]
        rising = len(close) < 2 or close.iloc[-1] >= close.iloc[0]
        color = UP if rising else DOWN
        fill = "rgba(38,166,154,.15)" if rising else "rgba(239,83,80,.15)"
        fig.add_trace(go.Scatter(x=prices.index, y=close, name="Close", mode="lines",
                                 line=dict(color=color, width=2), fill="tozeroy",
                                 fillcolor=fill), row=1, col=1)
        low, high = float(close.min()), float(close.max())
        pad = (high - low) * 0.08 or high * 0.05
        fig.update_yaxes(range=[low - pad, high + pad], row=1, col=1)
    if has_volume:
        up = prices["Close"] >= prices.get("Open", prices["Close"])
        fig.add_trace(go.Bar(x=prices.index, y=prices["Volume"], name="Volume",
                             marker_color=[UP if u else DOWN for u in up], opacity=0.5),
                      row=2, col=1)
    fig.update_layout(xaxis_rangeslider_visible=False, hovermode="x unified")
    fig.update_yaxes(side="right")
    return _style(fig, height, legend=False)


def market_treemap(frame: pd.DataFrame, size: str = "market_cap", color: str = "score",
                   height: int = 560) -> go.Figure:
    """Finviz-style market map: sector -> company boxes, sized by ``size`` and
    coloured by ``color`` (0-100) in the same quartile bands as the score gauge; companies
    without a score (e.g. banks) are grey, never a made-up middle colour."""
    data = frame.dropna(subset=[size]).copy()
    data["sector"] = data["sector"].fillna("Other")
    data["label"] = data["ticker"].str.replace(r"\.(NS|BO)$", "", regex=True)
    labels, parents, values, colors, text, ids = [], [], [], [], [], []
    for sector, group in data.groupby("sector"):
        ids.append(f"sector::{sector}")
        labels.append(sector)
        parents.append("")
        values.append(float(group[size].sum()))
        scored = group[color].dropna()
        colors.append(score_color(float((scored * group.loc[scored.index, size]).sum() /
                                        group.loc[scored.index, size].sum()))
                      if len(scored) else score_color(None))
        text.append(f"{len(group)} companies")
    for _, row in data.iterrows():
        ids.append(f"co::{row['ticker']}")
        labels.append(row["label"])
        parents.append(f"sector::{row['sector']}")
        values.append(float(row[size]))
        value = row.get(color)
        colors.append(score_color(None if pd.isna(value) else float(value)))
        text.append(f"{row.get('name') or ''}<br>{color}: "
                    f"{'N/A' if pd.isna(value) else f'{value:.1f}'}")
    fig = go.Figure(go.Treemap(
        ids=ids, labels=labels, parents=parents, values=values, branchvalues="total",
        text=text, textinfo="label", hovertemplate="<b>%{label}</b><br>%{text}<extra></extra>",
        marker=dict(colors=colors, line=dict(width=1, color="#0b0f17")),
        tiling=dict(pad=2), pathbar=dict(visible=True)))
    return _style(fig, height, legend=False).update_layout(margin=dict(l=0, r=0, t=30, b=0))


def scatter_map(frame: pd.DataFrame, x: str, y: str, x_title: str, y_title: str,
                color: str = "score", height: int = 480) -> go.Figure:
    """Each company as a dot (e.g. quality vs valuation), coloured by score."""
    data = frame.dropna(subset=[x, y])
    # trim extreme outliers so one company does not flatten the chart (large sets only)
    if len(data) >= 50:
        for axis in (x, y):
            low, high = data[axis].quantile([0.02, 0.98])
            data = data[(data[axis] >= low) & (data[axis] <= high)]
    fig = go.Figure(go.Scatter(
        x=data[x], y=data[y], mode="markers", text=data["ticker"],
        customdata=data[["name", "sector", "ticker"]].fillna("").values,
        marker=dict(size=9, color=data[color], colorscale=SCORE_SCALE, cmin=0, cmax=100,
                    line=dict(width=0.5, color="#0b0f17"),
                    colorbar=dict(title=color, thickness=12), opacity=0.85),
        hovertemplate=("<b>%{text}</b> %{customdata[0]}<br>%{customdata[1]}<br>" + x_title +
                       ": %{x:.2f}<br>" + y_title + ": %{y:.2f}<extra></extra>")))
    fig.update_layout(xaxis_title=x_title, yaxis_title=y_title)
    return _style(fig, height, legend=False)


def ranked_bars(values: Mapping[str, float], title: str = "", height: int = 300,
                score_colors: bool = False) -> go.Figure:
    """Horizontal bars, best at the top (comparison charts)."""
    items = sorted(values.items(), key=lambda kv: kv[1])
    colors = [score_color(v) if score_colors else ACCENT for _, v in items]
    fig = go.Figure(go.Bar(x=[v for _, v in items], y=[k for k, _ in items], orientation="h",
                           marker_color=colors, text=[f"{v:,.2f}" for _, v in items],
                           textposition="auto", hovertemplate="%{y}: %{x:,.2f}<extra></extra>"))
    fig.update_layout(title=dict(text=title, font=dict(size=13)))
    return _style(fig, height, legend=False)
