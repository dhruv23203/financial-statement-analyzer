"""Established third-party scores used by investment platforms (category BENCHMARK).

They are calculated alongside our own ratios to (a) show users the metrics that
platforms such as Screener.in, GuruFocus and Simply Wall St display, and (b) validate
our Financial Health Score against independent, published methods. They do not feed
into our health score.

Piotroski F-Score (0-9)
    Piotroski, J. D. (2000). Value investing: The use of historical financial statement
    information to separate winners from losers. Journal of Accounting Research, 38
    (Supplement), 1-41. Nine binary signals; ROA and CFO are scaled by beginning-of-year
    total assets. Leverage = long-term debt / average total assets.
Altman Z''-Score (book-value version for non-manufacturers)
    Altman, E. I., Hartzell, J. & Peck, M. (1995). Emerging markets corporate bonds: a
    scoring system. Salomon Brothers; see Altman & Hotchkiss (2006), Corporate Financial
    Distress and Bankruptcy. Z'' = 6.56 X1 + 3.26 X2 + 6.72 X3 + 1.05 X4; zones: > 2.60
    safe, 1.10-2.60 grey, < 1.10 distress. Book equity is used because this system has
    no market prices (the original 1968 Z needs market capitalisation).
Beneish M-Score (8-variable model)
    Beneish, M. D. (1999). The detection of earnings manipulation. Financial Analysts
    Journal, 55(5), 24-36. M = -4.84 + 0.920 DSRI + 0.528 GMI + 0.404 AQI + 0.892 SGI +
    0.115 DEPI - 0.172 SGAI + 4.679 TATA - 0.327 LVGI; M > -1.78 signals a higher
    probability of manipulation (a screening signal with many false positives).
Applicability (see MODEL_SECTOR_EXCLUSIONS)
    Altman's Z models and Beneish's M-Score were estimated on non-financial companies
    (Altman 1968 used manufacturers; Beneish 1999 excluded financial firms). Their
    balance-sheet inputs (working capital, receivables, asset quality) have a different
    meaning for banks and insurers, so for Financial Services they are reported as
    NOT_MEANINGFUL (reason MODEL_NOT_APPLICABLE) instead of producing false distress or
    manipulation signals. Simply Wall St uses a separate model for banks, so its
    non-financial health checks are excluded too. The Piotroski F-Score is still shown
    for financials but its leverage/liquidity signals are weak for banks.
Simply Wall St financial-health checks (0-6)
    Simply Wall St Company Analysis Model (MODEL.markdown, CC BY-NC-SA 4.0),
    github.com/SimplyWallSt/Company-Analysis-Model. Adaptations: debt-to-equity trend
    compares with the earliest available year (Yahoo provides ~4 years, the model uses
    5); the cash-runway variants for loss-making companies are not implemented.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any

from app.analysis.ratios import FinancialDataset, RatioOutcome, ok, unavailable

D = Decimal


def _missing(code: str, period: date, missing: list[str], inputs: dict[str, Any],
             detail: str = "") -> RatioOutcome:
    return unavailable(code, period, "MISSING_INPUT",
                       f"Cannot calculate for FY ending {period}: missing {', '.join(missing)}. "
                       f"{detail}".strip(), inputs)


def _previous(ds: FinancialDataset, period: date) -> date | None:
    return ds.adjacent_previous(period)


# --------------------------------------------------------------------------- Piotroski
def piotroski_f_score(ds: FinancialDataset, period: date) -> RatioOutcome:
    code = "piotroski_f_score"
    p1 = _previous(ds, period)
    p2 = _previous(ds, p1) if p1 else None
    g = ds.get
    need = {
        "net_income": g(period, "net_income"), "net_income_prev": g(p1, "net_income"),
        "operating_cash_flow": g(period, "operating_cash_flow"),
        "total_assets": g(period, "total_assets"), "total_assets_prev": g(p1, "total_assets"),
        "total_assets_prev2": g(p2, "total_assets"),
        "current_assets": g(period, "current_assets"),
        "current_liabilities": g(period, "current_liabilities"),
        "current_assets_prev": g(p1, "current_assets"),
        "current_liabilities_prev": g(p1, "current_liabilities"),
        "total_revenue": g(period, "total_revenue"), "total_revenue_prev": g(p1, "total_revenue"),
        "gross_profit": g(period, "gross_profit"), "gross_profit_prev": g(p1, "gross_profit"),
    }
    ltd = g(period, "long_term_debt") if g(period, "long_term_debt") is not None else \
        g(period, "long_term_debt_and_lease")
    ltd_prev = g(p1, "long_term_debt") if g(p1, "long_term_debt") is not None else \
        g(p1, "long_term_debt_and_lease")
    need.update(long_term_debt=ltd, long_term_debt_prev=ltd_prev)
    shares, shares_prev = g(period, "ordinary_shares_number"), g(p1, "ordinary_shares_number")
    issuance = g(period, "issuance_of_capital_stock")
    missing = [k for k, v in need.items() if v is None]
    if shares is None and issuance is None:
        missing.append("ordinary_shares_number or issuance_of_capital_stock")
    if p1 is None or p2 is None:
        missing.append("two prior consecutive fiscal years")
    if missing:
        return _missing(code, period, missing, {"available_signals": None},
                        "The F-Score needs this year and the two previous years.")
    if min(need["total_assets_prev"], need["total_assets_prev2"], need["total_assets"]) <= 0 \
            or need["current_liabilities"] <= 0 or need["current_liabilities_prev"] <= 0 \
            or need["total_revenue"] <= 0 or need["total_revenue_prev"] <= 0:
        return unavailable(code, period, "ZERO_DENOMINATOR",
                           "A denominator (assets, current liabilities or revenue) is not "
                           "positive.", need)
    roa = need["net_income"] / need["total_assets_prev"]
    roa_prev = need["net_income_prev"] / need["total_assets_prev2"]
    cfo = need["operating_cash_flow"] / need["total_assets_prev"]
    lever = ltd / ((need["total_assets"] + need["total_assets_prev"]) / 2)
    lever_prev = ltd_prev / ((need["total_assets_prev"] + need["total_assets_prev2"]) / 2)
    liquid = need["current_assets"] / need["current_liabilities"]
    liquid_prev = need["current_assets_prev"] / need["current_liabilities_prev"]
    margin = need["gross_profit"] / need["total_revenue"]
    margin_prev = need["gross_profit_prev"] / need["total_revenue_prev"]
    turn = need["total_revenue"] / need["total_assets_prev"]
    turn_prev = need["total_revenue_prev"] / need["total_assets_prev2"]
    if shares is not None and shares_prev is not None:
        no_new_equity = shares <= shares_prev
        equity_basis = "shares outstanding did not increase"
    else:
        no_new_equity = (issuance or 0) <= 0
        equity_basis = "no cash from issuance of capital stock"
    signals = {
        "F_ROA (ROA > 0)": roa > 0, "F_CFO (CFO > 0)": cfo > 0,
        "F_dROA (ROA improved)": roa > roa_prev, "F_ACCRUAL (CFO > ROA)": cfo > roa,
        "F_dLEVER (leverage fell)": lever < lever_prev,
        "F_dLIQUID (current ratio rose)": liquid > liquid_prev,
        f"F_EQ_OFFER ({equity_basis})": bool(no_new_equity),
        "F_dMARGIN (gross margin rose)": margin > margin_prev,
        "F_dTURN (asset turnover rose)": turn > turn_prev,
    }
    score = sum(signals.values())
    strength = "strong (8-9)" if score >= 8 else "weak (0-2)" if score <= 2 else "average (3-7)"
    return ok(code, period, D(score), {"signals": {k: int(v) for k, v in signals.items()},
                                       "roa": roa, "cfo_to_assets": cfo},
              "PIOTROSKI_2000", f"F-Score {score}/9 - {strength}.")


# --------------------------------------------------------------------------- Altman Z''
ALTMAN_SAFE, ALTMAN_DISTRESS = D("2.60"), D("1.10")


def altman_z_double_prime(ds: FinancialDataset, period: date) -> RatioOutcome:
    code = "altman_z_score"
    g = ds.get
    ta, tl = g(period, "total_assets"), g(period, "total_liabilities")
    wc = g(period, "working_capital")
    if wc is None and g(period, "current_assets") is not None and \
            g(period, "current_liabilities") is not None:
        wc = g(period, "current_assets") - g(period, "current_liabilities")
    ebit = g(period, "ebit") if g(period, "ebit") is not None else g(period, "operating_income")
    inputs = {"working_capital": wc, "retained_earnings": g(period, "retained_earnings"),
              "ebit_or_operating_income": ebit, "book_equity": g(period, "stockholders_equity"),
              "total_assets": ta, "total_liabilities": tl}
    missing = [k for k, v in inputs.items() if v is None]
    if missing:
        return _missing(code, period, missing, inputs)
    if ta <= 0 or tl <= 0:
        return unavailable(code, period, "ZERO_DENOMINATOR",
                           "Total assets or total liabilities not positive.", inputs)
    x1, x2, x3 = wc / ta, inputs["retained_earnings"] / ta, ebit / ta
    x4 = inputs["book_equity"] / tl
    z = D("6.56") * x1 + D("3.26") * x2 + D("6.72") * x3 + D("1.05") * x4
    zone = "safe" if z > ALTMAN_SAFE else "distress" if z < ALTMAN_DISTRESS else "grey"
    return ok(code, period, z, {**inputs, "X1": x1, "X2": x2, "X3": x3, "X4": x4, "zone": zone},
              "ALTMAN_Z_DOUBLE_PRIME", f"Z'' = {float(z):.2f}: {zone} zone (> 2.60 safe, "
                                       "< 1.10 distress).")


# --------------------------------------------------------------------------- Beneish M
BENEISH_THRESHOLD = D("-1.78")


def beneish_m_score(ds: FinancialDataset, period: date) -> RatioOutcome:
    code = "beneish_m_score"
    p1 = _previous(ds, period)
    g = ds.get

    def dep(p: date | None) -> Decimal | None:
        value = g(p, "reconciled_depreciation")
        return value if value is not None else g(p, "depreciation_amortization_cf")

    def ltd(p: date | None) -> Decimal | None:
        value = g(p, "long_term_debt")
        return value if value is not None else g(p, "long_term_debt_and_lease")

    values = {}
    for suffix, p in (("", period), ("_prev", p1)):
        values.update({
            f"receivables{suffix}": g(p, "accounts_receivable"),
            f"revenue{suffix}": g(p, "total_revenue"),
            f"cost_of_revenue{suffix}": g(p, "cost_of_revenue"),
            f"current_assets{suffix}": g(p, "current_assets"),
            f"net_ppe{suffix}": g(p, "net_ppe"), f"total_assets{suffix}": g(p, "total_assets"),
            f"depreciation{suffix}": dep(p),
            f"sga{suffix}": g(p, "selling_general_administrative"),
            f"current_liabilities{suffix}": g(p, "current_liabilities"),
            f"long_term_debt{suffix}": ltd(p),
        })
    income = g(period, "net_income_continuing")
    values["income_continuing_ops"] = income if income is not None else g(period, "net_income")
    values["operating_cash_flow"] = g(period, "operating_cash_flow")
    missing = [k for k, v in values.items() if v is None]
    if p1 is None:
        missing.append("previous fiscal year")
    if missing:
        return _missing(code, period, missing, {})
    v = values
    try:
        dsri = (v["receivables"] / v["revenue"]) / (v["receivables_prev"] / v["revenue_prev"])
        gm = (v["revenue"] - v["cost_of_revenue"]) / v["revenue"]
        gm_prev = (v["revenue_prev"] - v["cost_of_revenue_prev"]) / v["revenue_prev"]
        gmi = gm_prev / gm
        aqi = (1 - (v["current_assets"] + v["net_ppe"]) / v["total_assets"]) / \
            (1 - (v["current_assets_prev"] + v["net_ppe_prev"]) / v["total_assets_prev"])
        sgi = v["revenue"] / v["revenue_prev"]
        depi = (v["depreciation_prev"] / (v["depreciation_prev"] + v["net_ppe_prev"])) / \
            (v["depreciation"] / (v["depreciation"] + v["net_ppe"]))
        sgai = (v["sga"] / v["revenue"]) / (v["sga_prev"] / v["revenue_prev"])
        lvgi = ((v["current_liabilities"] + v["long_term_debt"]) / v["total_assets"]) / \
            ((v["current_liabilities_prev"] + v["long_term_debt_prev"]) / v["total_assets_prev"])
        tata = (v["income_continuing_ops"] - v["operating_cash_flow"]) / v["total_assets"]
    except (ArithmeticError, ZeroDivisionError):
        return unavailable(code, period, "ZERO_DENOMINATOR",
                           "An index denominator is zero (e.g. no receivables, SG&A or gross "
                           "margin in one year).", {})
    m = (D("-4.84") + D("0.920") * dsri + D("0.528") * gmi + D("0.404") * aqi
         + D("0.892") * sgi + D("0.115") * depi - D("0.172") * sgai + D("4.679") * tata
         - D("0.327") * lvgi)
    flag = m > BENEISH_THRESHOLD
    indices = {"DSRI": dsri, "GMI": gmi, "AQI": aqi, "SGI": sgi, "DEPI": depi, "SGAI": sgai,
               "LVGI": lvgi, "TATA": tata}
    return ok(code, period, m, {"indices": indices, "threshold": BENEISH_THRESHOLD},
              "BENEISH_1999", f"M = {float(m):.2f}: "
              + ("ABOVE -1.78 - profile resembles earnings manipulators; investigate "
                 "(screening signal, many false positives)." if flag else
                 "below -1.78 - no manipulation signal."))


# --------------------------------------------------------------------------- Simply Wall St
def sws_health_checks(ds: FinancialDataset, period: date) -> RatioOutcome:
    code = "sws_health_checks"
    g = ds.get
    ca, cl = g(period, "current_assets"), g(period, "current_liabilities")
    tl, debt = g(period, "total_liabilities"), g(period, "total_debt")
    equity, ocf = g(period, "stockholders_equity"), g(period, "operating_cash_flow")
    ebit = g(period, "ebit") if g(period, "ebit") is not None else g(period, "operating_income")
    interest = g(period, "interest_expense")
    earliest = next((p for p in ds.periods if p < period and (period - p).days >= 3 * 330), None)
    inputs = {"current_assets": ca, "current_liabilities": cl, "total_liabilities": tl,
              "total_debt": debt, "stockholders_equity": equity, "operating_cash_flow": ocf}
    missing = [k for k, v in inputs.items() if v is None]
    if missing:
        return _missing(code, period, missing, inputs)
    if equity <= 0:
        return unavailable(code, period, "NON_POSITIVE_EQUITY",
                           "Debt-to-equity checks need positive equity.", inputs,
                           not_meaningful=True)
    de = debt / equity
    checks: dict[str, bool | None] = {
        "1 short-term assets > short-term liabilities": ca > cl,
        "2 short-term assets > long-term liabilities": ca > (tl - cl),
        "4 debt-to-equity < 40%": de < D("0.40"),
        "5 operating cash flow > 20% of debt": debt == 0 or ocf > D("0.20") * debt,
        "6 EBIT > 5x interest (if interest paid)":
            True if not interest else (ebit is not None and ebit > 5 * abs(interest)),
    }
    if earliest is not None and g(earliest, "total_debt") is not None and \
            g(earliest, "stockholders_equity") and g(earliest, "stockholders_equity") > 0:
        checks["3 debt-to-equity not higher than earliest year (model: 5 years)"] = \
            de <= g(earliest, "total_debt") / g(earliest, "stockholders_equity")
    else:
        return _missing(code, period, ["debt-to-equity at least 3 years earlier"], inputs)
    passed = sum(bool(v) for v in checks.values())
    return ok(code, period, D(passed), {"checks": {k: int(bool(v)) for k, v in checks.items()}},
              "SIMPLY_WALL_ST_HEALTH", f"{passed}/6 Simply Wall St health checks passed.")


MODEL_SECTOR_EXCLUSIONS: dict[str, tuple[str, ...]] = {
    "Financial Services": ("altman_z_score", "altman_z_original", "beneish_m_score",
                           "sws_health_checks"),
}


def apply_model_applicability(outcomes: list[RatioOutcome],
                              sector: str | None) -> list[RatioOutcome]:
    """Replace benchmark results that the published model does not support for this
    sector with a NOT_MEANINGFUL outcome that explains why (no number is shown)."""
    from app.analysis.ratios import MODEL_NOT_APPLICABLE

    excluded = MODEL_SECTOR_EXCLUSIONS.get(sector or "", ())
    if not excluded:
        return outcomes
    return [unavailable(o.code, o.period, MODEL_NOT_APPLICABLE,
                        f"{o.definition.name} is not applicable to {sector} companies: the "
                        "model was estimated on non-financial firms and its balance-sheet "
                        "inputs mean something different for banks and insurers.",
                        not_meaningful=True) if o.code in excluded else o
            for o in outcomes]


def calculate_benchmark_scores(ds: FinancialDataset, period: date) -> list[RatioOutcome]:
    return [piotroski_f_score(ds, period), altman_z_double_prime(ds, period),
            beneish_m_score(ds, period), sws_health_checks(ds, period)]
