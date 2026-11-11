"""Application exception hierarchy.

Every error the CLI reports to the user derives from :class:`AnalyzerError`, so the
command layer can show a clear message without hiding unexpected programming errors.
"""

from __future__ import annotations


class AnalyzerError(Exception):
    """Base class for expected, user-reportable errors."""


class ConfigError(AnalyzerError):
    """A configuration file is missing or invalid."""


class InvalidTickerError(AnalyzerError):
    """The ticker is blank, malformed, unsupported or unknown to the provider."""


class ProviderError(AnalyzerError):
    """The data provider failed to return data."""


class ProviderNetworkError(ProviderError):
    """Network failure or timeout while contacting the provider (retryable)."""


class ProviderRateLimitError(ProviderError):
    """The provider rejected the request because of rate limiting (retryable)."""


class DataNotFoundError(AnalyzerError):
    """Required data has not been ingested / cleaned / analysed yet."""


class DatabaseError(AnalyzerError):
    """Database is unavailable, locked, not initialised or failed integrity checks."""


class ComparisonError(AnalyzerError):
    """A comparison request is invalid (e.g. wrong number of companies)."""
