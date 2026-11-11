"""Provider abstraction.

A provider returns statements as ``pandas.DataFrame`` objects in the Yahoo layout:
rows = metric names, columns = fiscal period end dates, values = amounts in the
statement currency. Adding another data source (e.g. a filings API) only requires
a new subclass; ingestion, cleaning and analysis are provider-independent.

Providers signal failures with the typed exceptions in ``app.utils.exceptions``:

* ``ProviderNetworkError``   - connection problems / timeouts (retried)
* ``ProviderRateLimitError`` - HTTP 429 / provider throttling (retried with back-off)
* ``ProviderError``          - anything else (not retried)

An *empty* DataFrame means "the provider has no such statement" and is not an error.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

METADATA_FIELDS = ("name", "exchange", "sector", "industry", "country", "currency")


class FinancialDataProvider(ABC):
    """Interface every data source must implement."""

    name: str = "abstract"

    @abstractmethod
    def fetch_statement(self, ticker: str, statement_type: str, frequency: str) -> pd.DataFrame:
        """Return one statement (possibly empty) for ``annual`` or ``quarterly`` frequency."""

    @abstractmethod
    def fetch_metadata(self, ticker: str) -> dict[str, Any]:
        """Return company metadata with keys from :data:`METADATA_FIELDS` (best effort)."""

    def fetch_market_snapshot(self, ticker: str) -> dict[str, Any]:
        """Current market data: ``price``, ``market_cap``, ``shares_outstanding``,
        ``currency`` (trading currency) and optional ``provider_values`` (the provider's
        own multiples, kept only for cross-checking). Empty dict = not available."""
        return {}
