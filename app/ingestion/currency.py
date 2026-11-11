"""Reporting-currency resolution and plausibility check (Day 10 / Day 20 support).

Yahoo's ``financialCurrency`` label is occasionally inconsistent with the values in
the statements (e.g. HCLTECH.NS: label USD, values in INR). The system therefore:

1. applies a user-verified label override from ``config/currency_overrides.json``
   when one exists (values are never converted - only the label is corrected);
2. otherwise uses the provider label and runs a plausibility check: the latest
   annual statement revenue divided by the provider's trailing-twelve-month
   revenue (reported in the claimed currency) must be within the configured
   range. A failure is recorded and shown as a warning everywhere the company's
   amounts are used.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd

from app.utils.exceptions import ConfigError
from app.utils.helpers import parse_period_end, to_decimal
from config.metric_aliases import lookup_metric
from config.settings import CURRENCY_CHECK_MAX_RATIO, CURRENCY_CHECK_MIN_RATIO

PASSED, FAILED, NOT_CHECKED, OVERRIDDEN = "PASSED", "FAILED", "NOT_CHECKED", "OVERRIDDEN"


def load_currency_overrides(path: Path) -> dict[str, dict[str, str]]:
    if not Path(path).is_file():
        return {}
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON in {path}: {exc}") from exc
    overrides = data.get("overrides", {})
    if not isinstance(overrides, dict):
        raise ConfigError(f"{path}: 'overrides' must be an object")
    for ticker, entry in overrides.items():
        if not isinstance(entry, dict) or not str(entry.get("currency", "")).strip() or \
                not str(entry.get("reason", "")).strip():
            raise ConfigError(f"{path}: override for {ticker} needs 'currency' and 'reason'")
    return {ticker.upper(): entry for ticker, entry in overrides.items()}


def latest_statement_revenue(income_statement: pd.DataFrame | None) -> Decimal | None:
    """Total revenue of the most recent period in a Yahoo-layout income statement."""
    if income_statement is None or income_statement.empty:
        return None
    for position, name in enumerate(income_statement.index):
        match = lookup_metric("income_statement", str(name))
        if match and match[0].code == "total_revenue":
            values = income_statement.iloc[position].tolist()
            dated = [(period, to_decimal(value)) for period, value in
                     zip((parse_period_end(c) for c in income_statement.columns), values)]
            dated = sorted(((p, v) for p, v in dated if p is not None and v is not None),
                           reverse=True)
            if dated:
                return dated[0][1]
    return None


def check_currency(statement_revenue: Decimal | None, metadata: dict[str, Any],
                   currency: str | None, override: dict[str, str] | None) -> dict[str, Any]:
    """Return a JSON-friendly description of the currency check."""
    if override is not None:
        return {"status": OVERRIDDEN, "currency": currency,
                "provider_label": metadata.get("currency"),
                "message": f"Currency label set to {currency} by config/currency_overrides.json: "
                           f"{override['reason']}"}
    ttm = to_decimal(metadata.get("ttm_total_revenue"))
    if statement_revenue is None or ttm is None or ttm <= 0 or statement_revenue <= 0:
        return {"status": NOT_CHECKED, "currency": currency,
                "message": "Plausibility check not possible (revenue or provider TTM revenue "
                           "unavailable)."}
    ratio = statement_revenue / ttm
    result = {"currency": currency, "statement_revenue": str(statement_revenue),
              "provider_ttm_revenue": str(ttm), "ratio": f"{float(ratio):.3f}",
              "trading_currency": metadata.get("trading_currency")}
    if Decimal(str(CURRENCY_CHECK_MIN_RATIO)) <= ratio <= Decimal(str(CURRENCY_CHECK_MAX_RATIO)):
        result.update(status=PASSED, message="Statement revenue is consistent with the "
                                             "provider's revenue in the labelled currency.")
    else:
        result.update(
            status=FAILED,
            message=(f"Currency label {currency} looks unreliable: latest statement revenue is "
                     f"{float(ratio):,.1f}x the provider's TTM revenue in {currency}. The "
                     "values may be in a different currency (e.g. "
                     f"{metadata.get('trading_currency')}). Verify against the company's "
                     "filings and, if confirmed, add an entry to "
                     "config/currency_overrides.json. Absolute amounts are not comparable "
                     "until then."))
    return result
