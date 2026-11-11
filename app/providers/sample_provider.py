"""Offline provider that reads synthetic statements from ``sample_data/<TICKER>/``.

Used for demonstrations without internet access and for reproducible tests. The
files use exactly the Yahoo layout (rows = metric names, columns = period ends):

    sample_data/DEMOGOOD/income_statement.csv
    sample_data/DEMOGOOD/balance_sheet.csv
    sample_data/DEMOGOOD/cash_flow.csv
    sample_data/DEMOGOOD/metadata.json

A missing CSV simulates a statement the provider does not have. The numbers are
synthetic and do not describe any real company.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from app.providers.base import FinancialDataProvider
from app.utils.exceptions import ProviderError
from app.utils.helpers import safe_child_path


class SampleDataProvider(FinancialDataProvider):
    name = "sample"

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir)

    def _company_dir(self, ticker: str) -> Path:
        return safe_child_path(self.base_dir, ticker)

    def fetch_statement(self, ticker: str, statement_type: str, frequency: str) -> pd.DataFrame:
        if frequency != "annual":
            return pd.DataFrame()  # the synthetic data set is annual only
        path = self._company_dir(ticker) / f"{statement_type}.csv"
        if not path.is_file():
            return pd.DataFrame()
        try:
            frame = pd.read_csv(path, index_col=0)
        except (OSError, ValueError) as exc:
            raise ProviderError(f"Cannot read sample file {path.name}: {exc}") from exc
        frame.columns = pd.to_datetime(frame.columns, errors="coerce")
        return frame

    def fetch_market_snapshot(self, ticker: str) -> dict[str, Any]:
        return dict(self.fetch_metadata(ticker).get("market") or {})

    def fetch_metadata(self, ticker: str) -> dict[str, Any]:
        path = self._company_dir(ticker) / "metadata.json"
        if not path.is_file():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProviderError(f"Cannot read sample metadata: {exc}") from exc
