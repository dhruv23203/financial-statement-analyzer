"""Plain-language explanations of the financial terms used in the app.

One source for the Overview guide, the Methodology glossary and inline tooltips, so a
term is explained the same way everywhere. Definitions describe how *this app* uses the
term (see docs/FINANCIAL_FORMULAS.md for the exact formulas).
"""

from __future__ import annotations

# term -> (area, explanation)
TERMS: dict[str, tuple[str, str]] = {
    # statements
    "Revenue": ("Statements", "Money earned from selling goods and services in the year, "
                "before any costs."),
    "Operating income": ("Statements", "Profit from the core business after operating costs, "
                         "before interest, tax and one-off or investment income."),
    "Net income": ("Statements", "Profit left for shareholders after every cost, interest and "
                   "tax."),
    "Operating cash flow (OCF)": ("Statements", "Cash the business actually generated from "
                                  "operations in the year (profit adjusted for non-cash items "
                                  "and working capital)."),
    "Capital expenditure (capex)": ("Statements", "Cash spent on long-term assets such as "
                                    "plants, equipment and software."),
    "Free cash flow (FCF)": ("Statements", "Operating cash flow minus capex: cash left after "
                             "keeping and growing the asset base."),
    "Working capital": ("Statements", "Current assets minus current liabilities - the money "
                        "tied up in day-to-day operations (stock, receivables, payables)."),
    "Shareholders' equity": ("Statements", "Assets minus liabilities: the owners' book stake "
                             "in the company."),
    "Fiscal year (FY)": ("Statements", "The company's accounting year. Indian companies usually "
                         "end in March, US companies often in December - compare with care."),
    "TTM": ("Statements", "Trailing twelve months: the sum of the last four reported quarters."),
    # profitability
    "Net profit margin": ("Profitability", "Net income as a share of revenue: how much of each "
                          "rupee or dollar of sales becomes profit."),
    "Operating margin": ("Profitability", "Operating income as a share of revenue: profit from "
                         "the core business per unit of sales."),
    "ROE": ("Profitability", "Return on equity: net income relative to shareholders' equity. "
            "Leverage raises it, so read it with debt."),
    "ROA": ("Profitability", "Return on assets: net income relative to total assets - how hard "
            "the assets work."),
    "ROCE": ("Profitability", "Return on capital employed: EBIT relative to capital employed "
             "(total assets minus current liabilities)."),
    # liquidity / solvency / efficiency
    "Current ratio": ("Liquidity", "Current assets / current liabilities: can short-term "
                      "assets cover bills due within a year? Below 1 is a warning sign for "
                      "most industries."),
    "Quick ratio": ("Liquidity", "Like the current ratio but excluding inventory - the most "
                    "liquid assets only."),
    "Debt-to-equity": ("Solvency", "Total debt / shareholders' equity: how much the company "
                       "borrows for each unit of owners' money."),
    "Debt ratio": ("Solvency", "Total liabilities / total assets: the share of assets financed "
                   "by others. Above 1 means negative equity."),
    "Interest coverage": ("Solvency", "EBIT / interest expense: how many times earnings cover "
                          "interest. Below 1.5x is weak."),
    "Asset turnover": ("Efficiency", "Revenue / total assets: sales generated per unit of "
                       "assets."),
    "Inventory turnover": ("Efficiency", "Cost of sales / inventory: how many times stock is "
                           "sold and replaced in a year."),
    "Receivables turnover": ("Efficiency", "Revenue / receivables: how quickly customers pay."),
    # growth
    "CAGR": ("Growth", "Compound annual growth rate: the steady yearly growth that would turn "
             "the first value into the last."),
    "YoY": ("Growth", "Year over year: the change from the previous fiscal year."),
    # risk / scores
    "Financial Health Score": ("Health", "This app's rule-based 0-100 summary of profitability, "
                               "liquidity, solvency, efficiency and growth, scored against "
                               "industry peers with absolute guardrails. Describes the past, "
                               "not a prediction."),
    "Peer percentile": ("Health", "Where a value ranks among companies in the same industry: "
                        "the 80th percentile beats 80 % of peers."),
    "Confidence": ("Health", "Share of the scoring framework that could actually be measured "
                   "from the available data."),
    "Red flag": ("Risk", "A rule-based warning (e.g. profit rising while cash flow falls) that "
                 "deserves a closer look. Not proof of fraud or distress."),
    "Piotroski F-Score": ("Risk", "A published 0-9 checklist of profitability, leverage and "
                          "efficiency improvements; 8-9 is strong, 0-2 weak."),
    "Altman Z-Score": ("Risk", "A published bankruptcy-risk formula from balance-sheet ratios. "
                       "Built for non-financial companies."),
    "Beneish M-Score": ("Risk", "A published screen for earnings-manipulation patterns. "
                        "Above -1.78 = worth investigating; many false positives."),
    "Distress onset (ML)": ("Risk", "In the ML tab: a currently profitable, cash-generating "
                            "company reporting a loss or negative operating cash flow next "
                            "year. A proxy, not default."),
    # valuation
    "P/E": ("Valuation", "Price / earnings: market value per unit of annual profit. Not "
            "meaningful for loss-makers."),
    "P/B": ("Valuation", "Price / book: market value relative to shareholders' equity."),
    "EV/EBITDA": ("Valuation", "Enterprise value / operating earnings before depreciation - a "
                  "multiple that ignores capital structure."),
    "PEG": ("Valuation", "P/E divided by profit growth (%): adjusts P/E for growth."),
    "DCF": ("Valuation", "Discounted cash flow: today's value of the cash a business is "
            "expected to generate, discounted for time and risk."),
    "FCFF": ("Valuation", "Free cash flow to the firm: cash available to all lenders and "
             "shareholders after tax, reinvestment and working capital."),
    "WACC": ("Valuation", "Weighted average cost of capital: the blended return lenders and "
             "shareholders require; the DCF discount rate."),
    "CAPM": ("Valuation", "Capital asset pricing model: cost of equity = risk-free rate + beta "
             "x equity risk premium."),
    "Beta": ("Valuation", "How strongly the share price moved with the market index (1 = same "
             "as the market)."),
    "Terminal value": ("Valuation", "The value of all cash flows after the forecast years, "
                       "assuming steady growth forever. Often the largest part of a DCF."),
    "Enterprise value (EV)": ("Valuation", "Value of the operating business to all capital "
                              "providers; equity value = EV + cash - debt - minorities."),
    "Intrinsic value": ("Valuation", "The DCF value per share under the stated assumptions - a "
                        "model output, not a target price."),
    "Margin of safety": ("Valuation", "How far the price is below the model value, as a share "
                         "of the model value."),
    "Reverse DCF": ("Valuation", "The growth rate the current price implies under the model."),
    "Sensitivity analysis": ("Valuation", "How the value changes when the discount rate or "
                             "terminal growth is moved."),
}

AREAS = ("Statements", "Profitability", "Liquidity", "Solvency", "Efficiency", "Growth",
         "Health", "Risk", "Valuation")


def explain(term: str) -> str:
    """Short explanation of a term ('' if unknown)."""
    return TERMS.get(term, ("", ""))[1]


def search(query: str) -> list[tuple[str, str, str]]:
    """(term, area, explanation) whose term or explanation contains *query* (case-blind)."""
    q = (query or "").strip().lower()
    rows = [(t, a, e) for t, (a, e) in TERMS.items()]
    if not q:
        return rows
    return [r for r in rows if q in r[0].lower() or q in r[2].lower()]
