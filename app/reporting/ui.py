"""HTML building blocks for the dashboard's trading-platform look.

Pure functions that return HTML strings (rendered with ``st.markdown(...,
unsafe_allow_html=True)``). Every piece of data is HTML-escaped, so company names or
provider text can never inject markup. Colours follow the common trading-terminal
convention: teal = good / up, red = bad / down, blue = accent.
"""

from __future__ import annotations

from functools import lru_cache
from html import escape
from typing import Iterable

UP, DOWN, ACCENT, MUTED = "#26a69a", "#ef5350", "#2962ff", "#8b97a8"
GRADE_COLORS = {"A": "#26a69a", "B": "#66bb6a", "C": "#f9a825", "D": "#fb8c00", "F": "#ef5350"}
SEVERITY_COLORS = {"HIGH": "#ef5350", "MEDIUM": "#fb8c00", "LOW": "#42a5f5"}
# Colours for the four interpretation bands, best first (TOP QUARTILE ... BOTTOM QUARTILE).
QUARTILE_COLORS = ("#26a69a", "#9ccc65", "#f9a825", "#ef5350")
NA_TEXT = "N/A - Insufficient applicable financial data"


@lru_cache(maxsize=1)
def score_bands() -> tuple[tuple[float, str, str], ...]:
    """(minimum score, label, colour) from the active scoring config, best band first.

    Colours therefore always match the interpretation labels (the calibrated universe
    quartiles), instead of arbitrary display cut-offs."""
    from app.analysis.config_loader import load_scoring_config
    from config.settings import get_settings

    bands = load_scoring_config(get_settings().scoring_config_file)["interpretation_bands"]
    ordered = sorted(bands, key=lambda b: -float(b["min"]))
    colors = QUARTILE_COLORS if len(ordered) == len(QUARTILE_COLORS) else \
        tuple(QUARTILE_COLORS[min(i, 3)] for i in range(len(ordered)))
    return tuple((float(b["min"]), str(b["label"]), c) for b, c in zip(ordered, colors))

CSS = """
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1400px;}
[data-testid="stSidebar"] {border-right: 1px solid rgba(139,151,168,.15);}
.fsa-brand {font-size: 1.25rem; font-weight: 800; letter-spacing: -.02em; margin-bottom: .1rem;}
.fsa-brand span {color: #2962ff;}
.fsa-sub {color: #8b97a8; font-size: .8rem; margin-bottom: .6rem;}
.fsa-hero {display: flex; flex-wrap: wrap; justify-content: space-between; gap: 1rem;
  align-items: flex-end; padding: 1.1rem 1.3rem; border-radius: 14px;
  background: linear-gradient(135deg, rgba(41,98,255,.16), rgba(38,166,154,.07));
  border: 1px solid rgba(139,151,168,.18); margin-bottom: 1rem;}
.fsa-hero h1 {font-size: 1.9rem; margin: 0; padding: 0; line-height: 1.15;}
.fsa-hero .meta {margin-top: .45rem; display: flex; flex-wrap: wrap; gap: .35rem;}
.fsa-price {text-align: right;}
.fsa-price .px {font-size: 2rem; font-weight: 700; font-variant-numeric: tabular-nums;}
.fsa-price .cur {color: #8b97a8; font-size: .95rem; margin-left: .25rem;}
.fsa-chip {display: inline-block; padding: .15rem .6rem; border-radius: 999px;
  font-size: .76rem; background: rgba(139,151,168,.14); color: inherit;
  border: 1px solid rgba(139,151,168,.2); white-space: nowrap;}
.fsa-chip.accent {background: rgba(41,98,255,.18); border-color: rgba(41,98,255,.45);}
.fsa-pill {display: inline-block; padding: .12rem .55rem; border-radius: 6px;
  font-weight: 700; font-size: .85rem; font-variant-numeric: tabular-nums;}
.fsa-grid {display: grid; gap: .7rem; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  margin: .2rem 0 1rem 0;}
.fsa-card {padding: .8rem 1rem; border-radius: 12px; background: rgba(139,151,168,.07);
  border: 1px solid rgba(139,151,168,.16);}
.fsa-card .lbl {color: #8b97a8; font-size: .74rem; text-transform: uppercase;
  letter-spacing: .05em;}
.fsa-card .val {font-size: 1.45rem; font-weight: 700; margin-top: .15rem;
  font-variant-numeric: tabular-nums; white-space: nowrap; overflow: hidden;
  text-overflow: ellipsis;}
.fsa-card .sub {font-size: .78rem; color: #8b97a8; margin-top: .1rem;}
.fsa-info {color: #8b97a8; font-size: .85rem; margin-left: .15rem;}
.fsa-card.has-tip {position: relative; cursor: pointer; outline: none;}
.fsa-card.has-tip:hover, .fsa-card.has-tip:focus {border-color: rgba(41,98,255,.6);}
.fsa-card.has-tip:hover .fsa-info, .fsa-card.has-tip:focus .fsa-info {color: #2962ff;}
.fsa-tiptext {display: none; position: absolute; left: 0; right: 0; top: calc(100% + 6px);
  z-index: 50; padding: .6rem .75rem; border-radius: 10px; font-size: .8rem;
  line-height: 1.4; background: #141b2a; color: #e6edf3; text-transform: none;
  letter-spacing: normal; border: 1px solid #2962ff; box-shadow: 0 8px 24px rgba(0,0,0,.45);}
.fsa-card.has-tip:hover .fsa-tiptext, .fsa-card.has-tip:focus .fsa-tiptext,
.fsa-card.has-tip:focus-within .fsa-tiptext {display: block;}
.fsa-status {display: grid; gap: .6rem; margin: .4rem 0 .8rem 0;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));}
.fsa-grades {display: flex; flex-wrap: wrap; gap: .5rem; margin: .3rem 0 1rem 0;}
.fsa-grade {display: flex; align-items: center; gap: .5rem; padding: .35rem .7rem .35rem .35rem;
  border-radius: 10px; background: rgba(139,151,168,.07); border: 1px solid rgba(139,151,168,.16);
  font-size: .85rem;}
.fsa-grade b {display: inline-flex; width: 1.8rem; height: 1.8rem; border-radius: 8px;
  align-items: center; justify-content: center; color: #fff; font-size: .95rem;}
.fsa-box {padding: .9rem 1.1rem; border-radius: 12px; height: 100%;
  border: 1px solid rgba(139,151,168,.16); background: rgba(139,151,168,.05);}
.fsa-box h4 {margin: 0 0 .5rem 0; font-size: .95rem;}
.fsa-box ul {margin: 0; padding-left: 1.1rem;}
.fsa-box li {margin-bottom: .3rem; font-size: .9rem;}
.fsa-box.pros {border-left: 4px solid #26a69a;}
.fsa-box.cons {border-left: 4px solid #ef5350;}
.fsa-flag {padding: .7rem 1rem; border-radius: 10px; margin-bottom: .55rem;
  background: rgba(139,151,168,.06); border: 1px solid rgba(139,151,168,.14);}
.fsa-flag .head {display: flex; gap: .5rem; align-items: center; flex-wrap: wrap;
  font-weight: 600;}
.fsa-flag .body {color: #b8c2cf; font-size: .86rem; margin-top: .3rem;}
.fsa-section {font-size: 1.15rem; font-weight: 700; margin: 1.4rem 0 .5rem 0;
  padding-left: .6rem; border-left: 3px solid #2962ff;}
.stTabs [data-baseweb="tab-list"] {gap: .3rem;}
.stTabs [data-baseweb="tab"] {padding: .4rem .9rem; border-radius: 8px 8px 0 0;}
div[data-testid="stMetric"] {background: rgba(139,151,168,.07); border-radius: 12px;
  padding: .7rem .9rem; border: 1px solid rgba(139,151,168,.16);}
</style>
"""


def score_color(score: float | None) -> str:
    """Colour of the interpretation band the score falls in (grey when missing)."""
    if score is None:
        return MUTED
    return next((color for floor, _, color in score_bands() if score >= floor),
                score_bands()[-1][2])


def band_legend() -> str:
    """Plain-text legend of the score colours, e.g. for chart captions."""
    names = {"#26a69a": "teal", "#9ccc65": "light green", "#f9a825": "amber",
             "#ef5350": "red"}
    return ", ".join(f"{names.get(c, c)} = {label.lower()} (>= {floor:g})" if floor else
                     f"{names.get(c, c)} = {label.lower()}" for floor, label, c in score_bands())


def chip(text: str, accent: bool = False) -> str:
    return f'<span class="fsa-chip{" accent" if accent else ""}">{escape(str(text))}</span>'


def pill(text: str, color: str) -> str:
    return (f'<span class="fsa-pill" style="background:{color}22;color:{color};'
            f'border:1px solid {color}66">{escape(str(text))}</span>')


def change_pill(change_pct: float | None) -> str:
    """Green/red percentage change badge, as on quote pages."""
    if change_pct is None:
        return ""
    color = UP if change_pct >= 0 else DOWN
    arrow = "▲" if change_pct >= 0 else "▼"
    return pill(f"{arrow} {abs(change_pct):.2f}%", color)


def hero(name: str, chips: Iterable[str], price: str | None = None,
         currency: str | None = None, change_html: str = "", note: str = "") -> str:
    """Quote-page header: company name and tags on the left, price on the right."""
    meta = "".join(chips)
    price_html = ""
    if price:
        price_html = (f'<div class="fsa-price"><div><span class="px">{escape(price)}</span>'
                      f'<span class="cur">{escape(currency or "")}</span></div>'
                      f'<div>{change_html} <span class="fsa-sub">{escape(note)}</span></div></div>')
    return (f'<div class="fsa-hero"><div><h1>{escape(name)}</h1>'
            f'<div class="meta">{meta}</div></div>{price_html}</div>')


def cards(items: Iterable[tuple]) -> str:
    """Grid of KPI cards; each item is (label, value, sub-text, value colour[, tooltip]).

    The tooltip (shown on hover, and marked with an "i") explains the metric."""
    parts = []
    for item in items:
        label, value, sub, color = item[:4]
        tip = item[4] if len(item) > 4 else None
        style = f' style="color:{color}"' if color else ""
        sub_html = f'<div class="sub">{escape(sub)}</div>' if sub else ""
        # CSS-only tooltip: shown on hover, and on click / tap / keyboard focus because
        # the card is focusable (st.markdown strips scripts, and native title tooltips
        # never appear on touch screens or on click).
        focus = ' tabindex="0" role="button" aria-label="' + escape(f"{label}: {tip}") + '"' \
            if tip else ""
        info = ' <span class="fsa-info" aria-hidden="true">ⓘ</span>' if tip else ""
        tip_html = f'<div class="fsa-tiptext">{escape(tip)}</div>' if tip else ""
        parts.append(f'<div class="fsa-card{" has-tip" if tip else ""}"{focus}>'
                     f'<div class="lbl">{escape(label)}{info}</div>'
                     f'<div class="val"{style}>{escape(value)}</div>{sub_html}{tip_html}</div>')
    return f'<div class="fsa-grid">{"".join(parts)}</div>'


def grade_badges(grades: dict[str, str]) -> str:
    """Letter-grade tiles (Tickertape / Morningstar style)."""
    parts = []
    for category, grade in grades.items():
        color = GRADE_COLORS.get(grade, MUTED)
        label = category.replace("_", " ").title()
        parts.append(f'<div class="fsa-grade"><b style="background:{color}">{escape(grade)}'
                     f'</b>{escape(label)}</div>')
    return f'<div class="fsa-grades">{"".join(parts)}</div>'


def bullet_box(title: str, items: Iterable[str], kind: str) -> str:
    """Pros (kind='pros') or cons (kind='cons') panel, as on Screener.in."""
    rows = "".join(f"<li>{escape(item)}</li>" for item in items) or \
        "<li>None stood out.</li>"
    return f'<div class="fsa-box {kind}"><h4>{escape(title)}</h4><ul>{rows}</ul></div>'


def flag_card(period: str, severity: str, title: str, condition: str,
              description: str) -> str:
    color = SEVERITY_COLORS.get(severity, MUTED)
    return (f'<div class="fsa-flag" style="border-left:4px solid {color}">'
            f'<div class="head">{pill(severity, color)} {chip(period)} {escape(title)}</div>'
            f'<div class="body">{escape(condition)}<br>{escape(description)}</div></div>')


def section(title: str) -> str:
    return f'<div class="fsa-section">{escape(title)}</div>'


def compact_number(value: float | None, currency: str | None = None) -> str:
    """1.23T / 45.6B / 789M style, or INR lakh crore / crore for Indian rupees."""
    if value is None:
        return "N/A"
    sign = "-" if value < 0 else ""
    value = abs(value)
    if currency == "INR":
        if value >= 1e12:
            return f"{sign}₹{value / 1e12:,.2f} L Cr"
        return f"{sign}₹{value / 1e7:,.0f} Cr"
    prefix = {"USD": "$", "EUR": "€", "GBP": "£"}.get(currency or "", "")
    for size, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if value >= size:
            return f"{sign}{prefix}{value / size:,.2f}{suffix}"
    return f"{sign}{prefix}{value:,.0f}"
