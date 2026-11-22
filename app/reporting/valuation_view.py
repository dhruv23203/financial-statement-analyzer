"""Professional presentation of one DCF valuation: dates, headline results, every
assumption with its reason, warnings for fragile assumptions and the cash-flow history.

Only displays numbers computed by ``app/valuation`` - nothing is recalculated here except
formatting."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.reporting import plots, ui
from app.valuation.company import CompanyValuation, history_flags


def _html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def _amount(value: float | None, currency: str | None) -> str:
    return "N/A" if value is None else ui.compact_number(value, currency)


def dates_line(v: CompanyValuation) -> str:
    val = v.valuation_date.isoformat() if v.valuation_date else "n/a"
    if v.price is None:
        return (f"Valuation date (balance sheet): {val} · no market price in the statement "
                "currency, so no upside is shown.")
    roll = (f" · value rolled forward {v.roll_years:.2f} years to the price date at the "
            f"{v.discount_rate:.2%} discount rate" if v.roll_years else "")
    return (f"Valuation date (balance sheet): {val} · market price date: "
            f"{v.price_date or 'n/a'}{roll}.")


def headline_cards(v: CompanyValuation, sym: str, currency: str | None) -> list[tuple]:
    share = v.terminal_share
    up = v.upside
    return [
        ("Intrinsic value / share", f"{sym}{v.value_per_share:,.2f}",
         f"at {v.price_date or v.valuation_date}", None,
         "Equity value divided by the share count, rolled to the price date."),
        ("Market price", f"{sym}{v.price:,.2f}" if v.price is not None else "N/A",
         f"stored snapshot {v.price_date or ''}", None, None),
        ("Implied upside / downside", f"{up:+.1%}" if up is not None else "N/A",
         "intrinsic value / price - 1", (ui.UP if up and up > 0 else ui.DOWN)
         if up is not None else None,
         "Positive = the model value is above the price. A model output under the stated "
         "assumptions, not a forecast or a recommendation."),
        ("Terminal value / EV", f"{share:.0%}" if share is not None else "N/A",
         "share of value beyond year 10", ui.DOWN if share and share > 0.75 else None,
         "Present value of the terminal value divided by enterprise value. Above 75 % the "
         "result depends mostly on the perpetuity assumptions."),
        ("Enterprise value", _amount(v.dcf.enterprise_value, currency),
         "PV of forecast FCFF + PV of terminal value", None, None),
        ("Net debt", _amount(v.net_debt, currency), "total debt - cash & short-term "
                                                     "investments", None, None),
        ("Equity value", _amount(v.dcf.equity_value, currency),
         "EV + cash - debt - minority interest", None, None),
    ]


def assumptions_table(v: CompanyValuation, market: dict[str, Any], sym: str,
                      currency: str | None) -> pd.DataFrame:
    c = v.fcff_components or {}
    sc = c.get("share_count") or {}
    b = v.dcf.bridge or {}
    rows = [
        ("Valuation date", str(v.valuation_date), "Balance-sheet date of the latest annual "
         "statements; cash, debt and the base cash flow are measured here."),
        ("Base FCFF", _amount(v.base_fcff, currency),
         f"{c.get('method', 'FCFF')}; tax at the statutory {c.get('tax_rate', 0):.2%}. "
         "Cash available to all capital providers in the latest year."),
        ("Growth, years 1-5", f"{v.growth:+.1%}", f"{v.growth_source}. Assumes FCFF grows with "
         "revenue (a constant FCFF margin); bounded to the configured range."),
        ("Growth, years 6-10", f"fades to {v.terminal_growth:.1%}", "Linear fade from the "
         "explicit growth to terminal growth, so high growth is not extended forever."),
        ("Terminal growth", f"{v.terminal_growth:.1%}",
         f"{market.get('terminal_growth_note', '')} Kept below the risk-free rate "
         f"({market['risk_free_rate']:.2%}) and the discount rate."),
        ("Discount rate", f"{v.discount_rate:.2%}", v.discount_note or ""),
        ("Discounting", "year-end", "Each year's FCFF is discounted from the end of that year; "
         "the terminal value is discounted with year 10."),
        ("Cash & short-term investments", _amount(b.get("plus_cash"), currency),
         "Added to enterprise value (not needed for operations is assumed)."),
        ("Total debt (incl. leases)", _amount(b.get("minus_debt"), currency),
         "Subtracted: claims of lenders and lessors."),
        ("Minority interest", _amount(b.get("minus_minority_interest"), currency),
         "Subtracted: the part of subsidiaries owned by others."),
        ("Share count", f"{sc.get('shares') or 0:,.0f}",
         f"{sc.get('classification') or ''}; {v.shares_source or ''}"),
    ]
    return pd.DataFrame(rows, columns=["Assumption", "Value", "Why / source"])


def history_figure(history: list[dict[str, Any]], scale: float, unit: str) -> go.Figure:
    years = [h["period"][:4] for h in history]
    def series(key):
        return [None if h.get(key) is None else h[key] / scale for h in history]
    fig = go.Figure()
    fig.add_bar(x=years, y=series("operating_cash_flow"), name="Operating cash flow")
    fig.add_bar(x=years, y=series("capital_expenditure"), name="Capital expenditure")
    fig.add_scatter(x=years, y=series("fcff"), name="FCFF", mode="lines+markers",
                    line=dict(width=3))
    fig.update_layout(barmode="relative", yaxis_title=unit, hovermode="x unified")
    return plots._style(fig, 320)


def history_table(history: list[dict[str, Any]], scale: float) -> pd.DataFrame:
    def f(x):
        return "N/A" if x is None else f"{x / scale:,.0f}"
    return pd.DataFrame([{
        "Fiscal year end": h["period"], "Revenue": f(h.get("revenue")),
        "Operating margin": "N/A" if h.get("operating_margin") is None
        else f"{h['operating_margin']:.1f} %",
        "Operating cash flow": f(h.get("operating_cash_flow")),
        "Capex": f(h.get("capital_expenditure")), "FCFF": f(h.get("fcff")),
        "Effective tax": "N/A" if h.get("effective_tax_rate") is None
        else f"{h['effective_tax_rate']:.1f} %",
        "FCFF method / note": h.get("method") or h.get("reason") or ""} for h in history])


def render(v: CompanyValuation, market: dict[str, Any], sym: str, currency: str | None,
           history: list[dict[str, Any]] | None, scale: tuple[float, str]) -> None:
    """Dates, headline results, warnings, assumptions and cash-flow history."""
    st.caption(dates_line(v))
    _html(ui.cards(headline_cards(v, sym, currency)))
    flags = list(v.assumption_flags)
    if history:
        flags += history_flags(history, v.base_fcff, market["statutory_tax_rate"])
    for flag in flags:
        st.warning(flag, icon="⚠️")
    for warning in v.warnings:
        st.caption(f"Note: {warning}")
    with st.expander("Assumptions and why they were chosen", expanded=False):
        st.dataframe(assumptions_table(v, market, sym, currency), hide_index=True,
                     width="stretch")
    if history:
        factor, unit = scale
        st.markdown(f"**Historical cash flows** ({unit}) - the base year is the last bar")
        st.plotly_chart(history_figure(history, factor, unit), width="stretch",
                        config={"displaylogo": False, "scrollZoom": False,
                                "doubleClick": False},
                        key=f"fcff_history_{v.method}")
        st.dataframe(history_table(history, factor), hide_index=True, width="stretch")
        st.caption("FCFF of every year uses the same method as the DCF base year. Years "
                   "with missing inputs show N/A with the reason - nothing is estimated.")


def render_history_only(history: list[dict[str, Any]], scale: tuple[float, str]) -> None:
    factor, unit = scale
    st.markdown(f"**Historical cash flows** ({unit})")
    st.plotly_chart(history_figure(history, factor, unit), width="stretch",
                    config={"displaylogo": False, "scrollZoom": False, "doubleClick": False},
                    key="fcff_history_only")
    st.dataframe(history_table(history, factor), hide_index=True, width="stretch")
