"""Yahoo Finance provider built on ``yfinance``.

Uses ``Ticker.get_income_stmt / get_balance_sheet / get_cashflow`` with
``pretty=True`` (human-readable row names such as "Total Revenue") and
``freq="yearly"`` or ``"quarterly"``.

Statement currency: Yahoo's ``financialCurrency`` is the currency the statements
are reported in; it can differ from the trading ``currency`` (e.g. ADRs), so it is
preferred for financial amounts.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import pandas as pd

from app.providers.base import FinancialDataProvider
from app.utils.exceptions import ProviderError, ProviderNetworkError, ProviderRateLimitError

logger = logging.getLogger(__name__)

_FREQ = {"annual": "yearly", "quarterly": "quarterly"}
_METHODS = {
    "income_statement": "get_income_stmt",
    "balance_sheet": "get_balance_sheet",
    "cash_flow": "get_cashflow",
}
_NETWORK_HINTS = ("timeout", "timed out", "connection", "resolve host", "network",
                  "ssl", "curl", "unreachable", "max retries")
_RATE_LIMIT_HINTS = ("too many requests", "rate limit", "429")


def translate_exception(exc: Exception) -> ProviderError:
    """Map yfinance / HTTP-client exceptions to the provider exception hierarchy."""
    if isinstance(exc, ProviderError):
        return exc
    class_name = type(exc).__name__.lower()
    message = str(exc)
    lowered = message.lower()
    if "ratelimit" in class_name or any(h in lowered for h in _RATE_LIMIT_HINTS):
        return ProviderRateLimitError(f"Yahoo Finance rate limit reached: {message}")
    if (isinstance(exc, (ConnectionError, TimeoutError))
            or "timeout" in class_name or "connection" in class_name
            or any(h in lowered for h in _NETWORK_HINTS)):
        return ProviderNetworkError(f"Network problem contacting Yahoo Finance: {message}")
    return ProviderError(f"Yahoo Finance error ({type(exc).__name__}): {message}")


class YFinanceProvider(FinancialDataProvider):
    """Fetch statements and metadata from Yahoo Finance."""

    name = "yfinance"

    def __init__(self, ticker_factory: Callable[[str], Any] | None = None) -> None:
        if ticker_factory is None:
            import yfinance as yf

            ticker_factory = yf.Ticker
        self._ticker_factory = ticker_factory
        self._cache: dict[str, Any] = {}
        self._info_cache: dict[str, dict[str, Any]] = {}

    def _ticker(self, ticker: str) -> Any:
        if ticker not in self._cache:
            self._cache[ticker] = self._ticker_factory(ticker)
        return self._cache[ticker]

    def fetch_statement(self, ticker: str, statement_type: str, frequency: str) -> pd.DataFrame:
        if statement_type not in _METHODS:
            raise ValueError(f"Unknown statement type {statement_type!r}")
        if frequency not in _FREQ:
            raise ValueError(f"Unknown frequency {frequency!r}")
        method = getattr(self._ticker(ticker), _METHODS[statement_type])
        try:
            frame = method(pretty=True, freq=_FREQ[frequency])
        except Exception as exc:  # translated into a typed, logged provider error
            logger.warning("yfinance %s failed for %s", statement_type, ticker,
                           extra={"ticker": ticker, "error": repr(exc)})
            raise translate_exception(exc) from exc
        if frame is None or not isinstance(frame, pd.DataFrame):
            return pd.DataFrame()
        return frame

    def _info(self, ticker: str) -> dict[str, Any]:
        """``Ticker.info`` fetched once per ticker (metadata and market data share it)."""
        if ticker not in self._info_cache:
            try:
                info = self._ticker(ticker).info or {}
            except Exception as exc:  # best effort; caller records the error
                logger.warning("yfinance info failed for %s", ticker,
                               extra={"ticker": ticker, "error": repr(exc)})
                raise translate_exception(exc) from exc
            self._info_cache[ticker] = info if isinstance(info, dict) else {}
        return self._info_cache[ticker]

    def fetch_price_history(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        interval = "1wk" if period in ("5y", "10y", "max") else "1d"
        try:
            frame = self._ticker(ticker).history(period=period, interval=interval,
                                                 auto_adjust=False)
        except Exception as exc:  # display-only data: log and show nothing
            logger.warning("yfinance price history failed for %s", ticker,
                           extra={"ticker": ticker, "error": repr(exc)})
            return pd.DataFrame()
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            return pd.DataFrame()
        columns = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in frame]
        return frame[columns].dropna(subset=["Close"]) if "Close" in columns else pd.DataFrame()

    def fetch_market_snapshot(self, ticker: str) -> dict[str, Any]:
        info = self._info(ticker)
        if not info.get("marketCap"):
            return {}
        return {
            "price": info.get("currentPrice") or info.get("regularMarketPrice"),
            "market_cap": info.get("marketCap"),
            "shares_outstanding": info.get("sharesOutstanding"),
            "currency": info.get("currency"),
            "provider_values": {k: info.get(k) for k in (
                "trailingPE", "priceToBook", "enterpriseValue", "enterpriseToEbitda",
                "dividendYield", "trailingEps", "bookValue")},
        }

    def fetch_metadata(self, ticker: str) -> dict[str, Any]:
        info = self._info(ticker)
        if not info:
            return {}
        return {
            "name": info.get("longName") or info.get("shortName"),
            "exchange": info.get("fullExchangeName") or info.get("exchange"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "country": info.get("country"),
            "currency": info.get("financialCurrency") or info.get("currency"),
            "trading_currency": info.get("currency"),
            "quote_type": info.get("quoteType"),
            # trailing-12-month revenue in the statement currency; used only for the
            # currency plausibility check during ingestion
            "ttm_total_revenue": info.get("totalRevenue"),
        }
