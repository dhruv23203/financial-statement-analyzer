# Automated Financial Statement Analyzer — hosted dashboard

B.Tech project: financial-health analysis of **NIFTY 500 and S&P 500** companies from their
historical financial statements. Includes ratios, trends, red flags, a sector-relative
health score with letter grades, Piotroski / Altman / Beneish / Simply Wall St benchmark
scores, valuation multiples, a stock screener and peer comparison.

This repository is the **deployment build** of the project: the Streamlit dashboard, the
analysis engine and a slim database with each company's latest stored analysis. Raw
statement data, run history, tests and documentation live in the full project.

## Quality & Value screen and Portfolio Doctor
* **💡 Quality & Value** - puts the health score next to a valuation check (P/E and P/B
  against sector medians of the same market, PEG, free-cash-flow yield and a 10-year DCF
  model value) and describes the combination, e.g. "Strong financials · price above model
  value" or "Weak financials · low valuation (value-trap profile)". It shows a
  margin-of-safety reference price (model value less 20 %), a Quality-vs-Value map of the
  whole universe and a filterable screen.
* **🩺 Portfolio** - upload a broker holdings export (symbol, quantity, average cost; Zerodha /
  Upstox / Groww / ICICI columns are recognised; CSV or Excel). Each holding gets a
  descriptive observation (financial strength, valuation, position size) and the portfolio
  gets a health score plus alerts for concentration and weak or highly valued holdings.
  Nothing is stored.

**These are descriptive screening results, not buy/sell recommendations.** The project is
educational; giving personalised investment advice to the public requires regulatory
registration (e.g. SEBI Investment Adviser / Research Analyst in India).

Both tabs only read what the pipeline already stored (`app/services/advisor.py`,
`app/services/portfolio_doctor.py`, `app/reporting/advisor_tabs.py`); the existing scoring,
database and tabs are unchanged. Tests: `python -m pytest tests`.

## Run on Streamlit Community Cloud
1. Sign in at https://share.streamlit.io with GitHub.
2. *Create app* → *Deploy a public app from GitHub*. Pick this repository, branch `main`,
   main file **`dashboard.py`**.
3. *Advanced settings* → Python **3.13**. Deploy.

## Run locally
```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run dashboard.py
```

## Notes
* Data: Yahoo Finance via `yfinance`, retrieved November 2026. "Fetch & analyse" in the
  sidebar downloads a company live. On Streamlit Cloud, additions are lost when the app
  restarts.
* The database file is about 84 MB. GitHub accepts it (limit 100 MB per file) but shows
  a size warning.

**Educational project — not a credit rating and not investment advice.**
