# Automated Financial Statement Analyzer — hosted dashboard

B.Tech project: financial-health analysis of **NIFTY 500 and S&P 500** companies from their
historical financial statements. Includes ratios, trends, red flags, a sector-relative
health score with letter grades, Piotroski / Altman / Beneish / Simply Wall St benchmark
scores, valuation multiples, a stock screener and peer comparison.

This repository is the **deployment build** of the project: the Streamlit dashboard, the
analysis engine and a slim database with each company's latest stored analysis. Raw
statement data, run history, tests and documentation live in the full project.

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
