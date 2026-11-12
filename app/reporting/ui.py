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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap');

/* ---------- design tokens (deep-navy fintech palette) ---------- */
:root {
  --fsa-bg: #0a0e17; --fsa-surface: #111827; --fsa-surface-2: #161f30;
  --fsa-border: rgba(148,163,184,.14); --fsa-border-strong: rgba(148,163,184,.26);
  --fsa-text: #e6edf3; --fsa-muted: #8b97a8; --fsa-accent: #2962ff;
  --fsa-up: #26a69a; --fsa-down: #ef5350; --fsa-radius: 12px;
  /* type system: display (headings, prices, key figures), body (text, labels),
     mono (ticker-style numbers) */
  --fsa-font-display: 'Plus Jakarta Sans', 'Inter', -apple-system, 'Segoe UI', sans-serif;
  --fsa-font-body: 'Inter', -apple-system, 'Segoe UI', Roboto, sans-serif;
  --fsa-font-mono: 'JetBrains Mono', 'SFMono-Regular', Consolas, monospace;
  --fsa-shadow: 0 1px 2px rgba(0,0,0,.35), 0 8px 24px rgba(0,0,0,.18);
}
html, body, [class*="css"], .stApp, .stMarkdown, button, input, textarea, select {
  font-family: var(--fsa-font-body) !important;
}
.stApp {background: radial-gradient(1200px 600px at 85% -10%, rgba(41,98,255,.10), transparent 60%),
  var(--fsa-bg);}
[data-testid="stDecoration"] {display: none;}
h1, h2, h3, h4, h5, [data-testid="stHeading"] * {font-family: var(--fsa-font-display) !important;
  letter-spacing: -.025em; font-weight: 700;}
/* Streamlit styles <p>, <li>, labels and table cells directly, so set the body face there
   too (icon spans use their own Material Symbols font and are not matched). */
.stApp :is(p, li, label, input, textarea, td, th, summary, small) {
  font-family: var(--fsa-font-body) !important;}
.stApp p, .stApp li {line-height: 1.6;}
[data-testid="stCaptionContainer"], .stCaption {color: var(--fsa-muted) !important; font-size: .8rem;
  line-height: 1.55;}
.stApp label p {font-weight: 600; font-size: .82rem; color: #b8c2cf;}
/* Streamlit's fixed top bar is ~3.75rem tall: start content below it so the ticker tape
   (first element on the page) is never hidden behind the bar. */
.block-container, .stMainBlockContainer {padding-top: 4.4rem !important; padding-bottom: 2rem;
  max-width: 1360px;}

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] {background: linear-gradient(180deg, #0d1422 0%, #0a0e17 100%);
  border-right: 1px solid var(--fsa-border);}
[data-testid="stSidebar"] [data-testid="stForm"] {background: var(--fsa-surface);
  border: 1px solid var(--fsa-border); border-radius: var(--fsa-radius);}
.fsa-brand {display: flex; align-items: center; gap: .6rem; margin-bottom: .2rem;}
.fsa-brand .mark {width: 34px; height: 34px; border-radius: 9px; flex: none;
  display: inline-flex; align-items: center; justify-content: center;
  background: linear-gradient(135deg, #2962ff, #26a69a); box-shadow: 0 4px 14px rgba(41,98,255,.35);}
.fsa-brand .name {font-size: 1.02rem; font-weight: 800; line-height: 1.15; letter-spacing: -.01em;}
.fsa-brand .name span {color: #5b8cff;}
.fsa-sub {color: var(--fsa-muted); font-size: .76rem; margin: .1rem 0 .8rem 0;}
.fsa-side-label {color: var(--fsa-muted); font-size: .68rem; font-weight: 700;
  letter-spacing: .09em; text-transform: uppercase; margin: 1rem 0 .3rem 0;}

/* ---------- top navigation (tabs as a segmented bar; works with both the React-Aria
   tabs of Streamlit >= 1.5x (role/data-testid) and the older BaseWeb markup) ---------- */
.stTabs [role="tablist"], .stTabs [data-baseweb="tab-list"] {gap: .25rem; padding: .3rem;
  border-radius: 12px; background: rgba(17,24,39,.92); border: 1px solid var(--fsa-border) !important;
  box-shadow: none !important; overflow-x: auto; scrollbar-width: none;}
.stMainBlockContainer > .stVerticalBlock > .stTabs > div > [role="tablist"] {
  position: sticky; top: 3.9rem; z-index: 30; backdrop-filter: blur(10px);}
.stTabs [data-testid="stTab"], .stTabs [data-baseweb="tab"] {height: 2.3rem; padding: 0 .95rem;
  border-radius: 9px; border: none !important; box-shadow: none !important; font-weight: 600;
  font-size: .88rem; color: var(--fsa-muted); background: transparent; white-space: nowrap;
  display: inline-flex; align-items: center; transition: background .15s ease, color .15s ease;}
.stTabs [data-testid="stTab"] p, .stTabs [data-baseweb="tab"] p {font-weight: 600; font-size: .88rem;}
.stTabs [data-testid="stTab"]:hover, .stTabs [data-baseweb="tab"]:hover {color: var(--fsa-text);
  background: rgba(148,163,184,.08);}
.stTabs [data-testid="stTab"][aria-selected="true"], .stTabs [data-baseweb="tab"][aria-selected="true"] {
  color: #fff !important; background: rgba(41,98,255,.22) !important;
  box-shadow: inset 0 0 0 1px rgba(41,98,255,.55) !important;}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {display: none;}
.stTabs [role="tabpanel"], .stTabs [data-baseweb="tab-panel"] {padding-top: 1rem;}

/* ---------- inputs, buttons, expanders, tables ---------- */
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {
  border-radius: 9px; font-weight: 600; border: 1px solid var(--fsa-border-strong);
  transition: transform .08s ease, box-shadow .15s ease;}
.stButton > button:hover, .stDownloadButton > button:hover {transform: translateY(-1px);
  box-shadow: 0 6px 16px rgba(0,0,0,.3);}
[data-testid="stExpander"] details {border-radius: var(--fsa-radius); background: var(--fsa-surface);
  border: 1px solid var(--fsa-border);}
[data-testid="stDataFrame"] {border-radius: var(--fsa-radius); overflow: hidden;
  border: 1px solid var(--fsa-border);}
[data-testid="stAlert"] {border-radius: var(--fsa-radius);}
[data-testid="stPlotlyChart"] {background: var(--fsa-surface); border: 1px solid var(--fsa-border);
  border-radius: var(--fsa-radius); padding: .35rem .4rem;}

/* ---------- ticker tape (scrolling strip, pauses on hover) ---------- */
.fsa-tape {overflow: hidden; white-space: nowrap; border: 1px solid var(--fsa-border);
  border-radius: 10px; background: var(--fsa-surface); margin-bottom: .9rem;
  mask-image: linear-gradient(90deg, transparent, #000 4%, #000 96%, transparent);}
.fsa-tape .track {display: inline-block; padding: .45rem 0; animation: fsa-scroll 70s linear infinite;}
.fsa-tape:hover .track {animation-play-state: paused;}
.fsa-tape .item {display: inline-flex; align-items: center; gap: .4rem; margin: 0 1.1rem;
  font-size: .8rem; font-variant-numeric: tabular-nums;}
.fsa-tape .item b {font-weight: 700;}
.fsa-tape .dot {width: 7px; height: 7px; border-radius: 50%; display: inline-block;}
.fsa-tape .lead {color: var(--fsa-muted); font-weight: 700; font-size: .7rem; letter-spacing: .08em;
  text-transform: uppercase; margin: 0 1rem 0 .9rem;}
@keyframes fsa-scroll {from {transform: translateX(0);} to {transform: translateX(-50%);}}

/* ---------- quote header ---------- */
.fsa-hero {display: flex; flex-wrap: wrap; justify-content: space-between; gap: 1rem;
  align-items: center; padding: 1.15rem 1.35rem; border-radius: 16px;
  background: linear-gradient(135deg, rgba(41,98,255,.14), rgba(38,166,154,.05) 60%), var(--fsa-surface);
  border: 1px solid var(--fsa-border-strong); box-shadow: var(--fsa-shadow); margin-bottom: .8rem;}
.fsa-hero .ident {display: flex; gap: .9rem; align-items: center; min-width: 0;}
.fsa-logo {width: 52px; height: 52px; border-radius: 14px; flex: none; display: inline-flex;
  align-items: center; justify-content: center; font-weight: 800; font-size: 1.05rem; color: #fff;
  letter-spacing: -.02em; box-shadow: inset 0 0 0 1px rgba(255,255,255,.15);}
.fsa-hero h1 {font-size: 1.75rem; font-weight: 800; margin: 0; padding: 0; line-height: 1.15;}
.fsa-hero .meta {margin-top: .4rem; display: flex; flex-wrap: wrap; gap: .35rem;}
.fsa-price {text-align: right;}
.fsa-price .px {font-size: 2.1rem; font-weight: 800; font-variant-numeric: tabular-nums;
  letter-spacing: -.02em;}
.fsa-price .cur {color: var(--fsa-muted); font-size: .9rem; margin-left: .3rem; font-weight: 600;}
.fsa-chip {display: inline-block; padding: .14rem .6rem; border-radius: 999px; font-size: .74rem;
  font-weight: 500; background: rgba(148,163,184,.10); color: inherit;
  border: 1px solid var(--fsa-border); white-space: nowrap;}
.fsa-chip.accent {background: rgba(41,98,255,.18); border-color: rgba(41,98,255,.5); font-weight: 700;}
.fsa-pill {display: inline-block; padding: .14rem .55rem; border-radius: 7px; font-weight: 700;
  font-size: .82rem; font-variant-numeric: tabular-nums;}

/* ---------- quote statistics strip and 52-week range ---------- */
/* each cell draws its right and bottom divider as a 1px shadow; the container clips the
   ones on its outer edge, so dividers stay correct however the cells wrap */
.fsa-stats {display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 0;
  border: 1px solid var(--fsa-border); border-radius: var(--fsa-radius); background: var(--fsa-surface);
  margin-bottom: .9rem; overflow: hidden;}
.fsa-stats .stat {padding: .65rem .95rem;
  box-shadow: 1px 0 0 var(--fsa-border), 0 1px 0 var(--fsa-border);}
.fsa-stats .k {color: var(--fsa-muted); font-size: .7rem; font-weight: 600; letter-spacing: .06em;
  text-transform: uppercase;}
.fsa-stats .v {font-size: .98rem; font-weight: 700; margin-top: .15rem; font-variant-numeric: tabular-nums;}
.fsa-range {position: relative; height: 6px; border-radius: 999px; margin: .55rem 0 .25rem 0;
  background: linear-gradient(90deg, #ef5350, #f9a825, #26a69a);}
.fsa-range .mk {position: absolute; top: -4px; width: 4px; height: 14px; border-radius: 2px;
  background: #fff; box-shadow: 0 0 0 2px rgba(10,14,23,.9); transform: translateX(-50%);}
.fsa-range-ends {display: flex; justify-content: space-between; font-size: .72rem;
  color: var(--fsa-muted); font-variant-numeric: tabular-nums;}

/* ---------- at a glance ---------- */
.fsa-glance {display: flex; flex-wrap: wrap; gap: .5rem; margin: 0 0 1rem 0;}
.fsa-glance .g {display: inline-flex; align-items: center; gap: .45rem; padding: .4rem .75rem;
  border-radius: 10px; background: var(--fsa-surface); border: 1px solid var(--fsa-border);
  font-size: .84rem;}
.fsa-glance .g .k {color: var(--fsa-muted);}
.fsa-glance .g b {font-weight: 700;}

/* ---------- cards ---------- */
.fsa-grid {display: grid; gap: .7rem; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  margin: .2rem 0 1rem 0;}
.fsa-card {padding: .85rem 1rem; border-radius: var(--fsa-radius); background: var(--fsa-surface);
  border: 1px solid var(--fsa-border); transition: border-color .15s ease, transform .15s ease;}
.fsa-card .lbl {color: var(--fsa-muted); font-size: .7rem; font-weight: 600; text-transform: uppercase;
  letter-spacing: .07em;}
.fsa-card .val {font-size: 1.42rem; font-weight: 800; margin-top: .2rem; letter-spacing: -.01em;
  font-variant-numeric: tabular-nums; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}
.fsa-card .sub {font-size: .76rem; color: var(--fsa-muted); margin-top: .1rem;}
.fsa-info {color: var(--fsa-muted); font-size: .85rem; margin-left: .15rem;}
.fsa-card.has-tip {position: relative; cursor: pointer; outline: none;}
/* The hover lift (transform) creates a stacking context, so the tooltip's own z-index only
   counts inside its card: raise the whole active card above the cards of the next row. */
.fsa-card.has-tip:hover, .fsa-card.has-tip:focus, .fsa-card.has-tip:focus-within {
  border-color: rgba(41,98,255,.6); transform: translateY(-1px); z-index: 60;}
.fsa-card.has-tip:hover .fsa-info, .fsa-card.has-tip:focus .fsa-info {color: #5b8cff;}
.fsa-tiptext {display: none; position: absolute; left: 0; right: 0; top: calc(100% + 6px);
  z-index: 50; padding: .6rem .75rem; border-radius: 10px; font-size: .8rem; font-weight: 400;
  line-height: 1.45; background: #172033; color: var(--fsa-text); text-transform: none;
  letter-spacing: normal; border: 1px solid rgba(41,98,255,.7); box-shadow: 0 12px 28px rgba(0,0,0,.5);}
.fsa-card.has-tip:hover .fsa-tiptext, .fsa-card.has-tip:focus .fsa-tiptext,
.fsa-card.has-tip:focus-within .fsa-tiptext {display: block;}
.fsa-grades {display: flex; flex-wrap: wrap; gap: .45rem; margin: .3rem 0 .9rem 0;}
.fsa-grade {display: flex; align-items: center; gap: .5rem; padding: .3rem .7rem .3rem .3rem;
  border-radius: 10px; background: var(--fsa-surface); border: 1px solid var(--fsa-border);
  font-size: .83rem; font-weight: 500;}
.fsa-grade b {display: inline-flex; width: 1.75rem; height: 1.75rem; border-radius: 8px;
  align-items: center; justify-content: center; color: #fff; font-size: .92rem; font-weight: 800;}

/* ---------- pros / cons, flags, sections ---------- */
.fsa-box {padding: .95rem 1.15rem; border-radius: var(--fsa-radius); height: 100%;
  border: 1px solid var(--fsa-border); background: var(--fsa-surface);}
.fsa-box h4 {margin: 0 0 .55rem 0; font-size: .92rem; font-weight: 700;}
.fsa-box ul {margin: 0; padding-left: 1.1rem;}
.fsa-box li {margin-bottom: .35rem; font-size: .88rem; line-height: 1.45;}
.fsa-box.pros {box-shadow: inset 3px 0 0 #26a69a;}
.fsa-box.cons {box-shadow: inset 3px 0 0 #ef5350;}
.fsa-flag {padding: .75rem 1rem; border-radius: var(--fsa-radius); margin-bottom: .55rem;
  background: var(--fsa-surface); border: 1px solid var(--fsa-border);}
.fsa-flag .head {display: flex; gap: .5rem; align-items: center; flex-wrap: wrap; font-weight: 600;}
.fsa-flag .body {color: #b8c2cf; font-size: .85rem; margin-top: .3rem; line-height: 1.45;}
.fsa-section {display: flex; align-items: baseline; gap: .6rem; flex-wrap: wrap;
  font-size: 1.12rem; font-weight: 800; margin: 1.5rem 0 .55rem 0; letter-spacing: -.01em;}
.fsa-section::before {content: ""; width: 4px; height: 1.05rem; border-radius: 2px; align-self: center;
  background: linear-gradient(180deg, #2962ff, #26a69a);}
.fsa-section .hint {font-size: .8rem; font-weight: 400; color: var(--fsa-muted); letter-spacing: 0;}
.fsa-status {display: grid; gap: .6rem; margin: .4rem 0 .8rem 0;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));}
div[data-testid="stMetric"] {background: var(--fsa-surface); border-radius: var(--fsa-radius);
  padding: .75rem .95rem; border: 1px solid var(--fsa-border);}

/* ---------- empty state and footer ---------- */
.fsa-empty {text-align: center; padding: 2.2rem 1rem; border: 1px dashed var(--fsa-border-strong);
  border-radius: var(--fsa-radius); color: var(--fsa-muted);}
.fsa-empty .icon {font-size: 1.8rem; margin-bottom: .4rem;}
.fsa-empty b {color: var(--fsa-text); display: block; margin-bottom: .25rem; font-size: 1rem;}
.fsa-footer {margin-top: 2.2rem; padding: 1rem 0 .4rem 0; border-top: 1px solid var(--fsa-border);
  display: flex; flex-wrap: wrap; justify-content: space-between; gap: .6rem;
  color: var(--fsa-muted); font-size: .76rem;}

/* ---------- typography refinements ---------- */
.fsa-brand .name, .fsa-hero h1, .fsa-section, .fsa-price .px, .fsa-card .val, .fsa-stats .v,
.fsa-grade b, .fsa-glance .g b, .fsa-empty b, .fsa-box h4,
.stTabs [data-testid="stTab"] p, .stTabs [data-baseweb="tab"] p,
div[data-testid="stMetricValue"] {font-family: var(--fsa-font-display) !important;}
.fsa-hero h1 {font-size: 1.85rem; font-weight: 800; letter-spacing: -.035em;
  background: linear-gradient(90deg, #ffffff 0%, #c7d7ff 70%, #9fb9ff 100%);
  -webkit-background-clip: text; background-clip: text; color: transparent;}
.fsa-brand .name {letter-spacing: -.02em;}
.fsa-brand .name span {background: linear-gradient(90deg, #5b8cff, #26a69a);
  -webkit-background-clip: text; background-clip: text; color: transparent;}
.fsa-price .px {letter-spacing: -.035em;}
.fsa-card .val {font-weight: 700; letter-spacing: -.025em;}
.fsa-section {font-weight: 700; letter-spacing: -.025em; font-size: 1.18rem;}
.fsa-section .hint {font-family: var(--fsa-font-body);}
.fsa-card .lbl, .fsa-stats .k, .fsa-side-label, .fsa-tape .lead {font-family: var(--fsa-font-body);
  font-weight: 600; letter-spacing: .085em;}
.fsa-tape .item, .fsa-range-ends, .fsa-pill, .fsa-chip.accent {font-family: var(--fsa-font-mono) !important;
  letter-spacing: -.01em;}
.fsa-tape .item b {font-weight: 600;}
.fsa-tiptext {font-family: var(--fsa-font-body); font-size: .82rem;}
.fsa-box li, .fsa-flag .body {color: #c4cdd8;}
@media (max-width: 640px) {
  .fsa-hero h1 {font-size: 1.35rem;} .fsa-price {text-align: left;} .fsa-price .px {font-size: 1.6rem;}
}
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


def logo(name: str, seed: str = "") -> str:
    """Monogram logo (two initials on a colour derived from the ticker), as used by
    brokerage apps when no company logo is available."""
    words = [w for w in name.replace("&", " ").split() if w[:1].isalnum()]
    initials = "".join(w[0] for w in words[:2]).upper() or "?"
    palette = ("#2962ff", "#7c4dff", "#00897b", "#f57c00", "#c2185b", "#0288d1", "#5e35b1",
               "#2e7d32")
    color = palette[sum(map(ord, seed or name)) % len(palette)]
    return (f'<div class="fsa-logo" style="background:linear-gradient(135deg,{color},'
            f'{color}aa)">{escape(initials)}</div>')


def hero(name: str, chips: Iterable[str], price: str | None = None,
         currency: str | None = None, change_html: str = "", note: str = "",
         ticker: str = "") -> str:
    """Quote-page header: logo, company name and tags on the left, price on the right."""
    meta = "".join(chips)
    price_html = ""
    if price:
        price_html = (f'<div class="fsa-price"><div><span class="px">{escape(price)}</span>'
                      f'<span class="cur">{escape(currency or "")}</span></div>'
                      f'<div>{change_html} <span class="fsa-sub">{escape(note)}</span></div></div>')
    return (f'<div class="fsa-hero"><div class="ident">{logo(name, ticker)}<div>'
            f'<h1>{escape(name)}</h1><div class="meta">{meta}</div></div></div>'
            f'{price_html}</div>')


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


def section(title: str, hint: str | None = None) -> str:
    """Section heading with an accent bar and an optional muted explanation."""
    extra = f'<span class="hint">{escape(hint)}</span>' if hint else ""
    return f'<div class="fsa-section">{escape(title)}{extra}</div>'


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


def brand() -> str:
    """Sidebar brand: gradient mark with a small chart glyph."""
    glyph = ('<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#fff" '
             'stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round">'
             '<polyline points="3 17 9 11 13 15 21 7"/><polyline points="15 7 21 7 21 13"/>'
             '</svg>')
    return (f'<div class="fsa-brand"><span class="mark">{glyph}</span><div class="name">'
            'Financial Statement<br><span>Analyzer</span></div></div>'
            '<div class="fsa-sub">NIFTY 500 + S&amp;P 500 · historical financial health</div>')


def side_label(text: str) -> str:
    return f'<div class="fsa-side-label">{escape(text)}</div>'


def ticker_tape(items: Iterable[tuple[str, float, str]], lead: str = "Top health scores") -> str:
    """TradingView-style scrolling strip of (ticker, score, colour). The list is
    repeated once so the CSS animation can loop seamlessly; it pauses on hover."""
    cells = "".join(f'<span class="item"><span class="dot" style="background:{c}"></span>'
                    f'<b>{escape(t)}</b><span style="color:{c}">{v:.1f}</span></span>'
                    for t, v, c in items)
    if not cells:
        return ""
    return (f'<div class="fsa-tape" aria-label="{escape(lead)}"><div class="track">'
            f'<span class="lead">{escape(lead)}</span>{cells}'
            f'<span class="lead">{escape(lead)}</span>{cells}</div></div>')


def range_bar(low: float, high: float, current: float) -> str:
    """52-week range with a marker at the current price (Google Finance style)."""
    span = high - low
    pos = 50.0 if span <= 0 else max(0.0, min(100.0, (current - low) / span * 100))
    return (f'<div class="fsa-range"><span class="mk" style="left:{pos:.1f}%"></span></div>'
            f'<div class="fsa-range-ends"><span>{low:,.2f}</span><span>{high:,.2f}</span></div>')


def stats_strip(items: Iterable[tuple[str, str]], range_html: str = "") -> str:
    """Quote statistics row: (label, value) cells, optionally led by a range cell."""
    cells = "".join(f'<div class="stat"><div class="k">{escape(k)}</div>'
                    f'<div class="v">{escape(v)}</div></div>' for k, v in items)
    lead = (f'<div class="stat" style="grid-column: span 2; min-width: 240px">'
            f'<div class="k">52-week range</div>{range_html}</div>') if range_html else ""
    return f'<div class="fsa-stats">{lead}{cells}</div>'


def glance(items: Iterable[tuple[str, str, str | None]]) -> str:
    """'At a glance' summary chips: (label, value, value colour)."""
    parts = []
    for label, value, color in items:
        style = f' style="color:{color}"' if color else ""
        parts.append(f'<span class="g"><span class="k">{escape(label)}</span>'
                     f'<b{style}>{escape(value)}</b></span>')
    return f'<div class="fsa-glance">{"".join(parts)}</div>'


def empty_state(icon: str, title: str, text: str) -> str:
    return (f'<div class="fsa-empty"><div class="icon">{escape(icon)}</div>'
            f'<b>{escape(title)}</b>{escape(text)}</div>')


def footer(version: str) -> str:
    return ('<div class="fsa-footer"><span>Data: Yahoo Finance (statements, prices) · SEC '
            'EDGAR for validation · Educational project - not a credit rating or investment '
            f'advice.</span><span>Scoring {escape(version)} · Methodology: docs/'
            'SCORING_METHODOLOGY.md</span></div>')
