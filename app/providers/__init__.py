"""Financial data providers (Yahoo Finance and an offline sample-data provider)."""

from __future__ import annotations

from pathlib import Path

from app.providers.base import FinancialDataProvider


def create_provider(name: str, sample_data_dir: Path | None = None) -> FinancialDataProvider:
    """Factory used by the CLI/services to build a provider by name."""
    if name == "yfinance":
        from app.providers.yfinance_provider import YFinanceProvider

        return YFinanceProvider()
    if name == "sample":
        from app.providers.sample_provider import SampleDataProvider
        from config.settings import SAMPLE_DATA_DIR

        return SampleDataProvider(sample_data_dir or SAMPLE_DATA_DIR)
    raise ValueError(f"Unknown provider {name!r}; expected 'yfinance' or 'sample'")
