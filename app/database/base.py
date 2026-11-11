"""Declarative base and custom column types.

``FinancialNumber``
    Financial amounts and ratios. Python code always receives ``Decimal``.
    NaN and +/-infinity are rejected at bind time, so they can never be stored as
    valid numbers. Integral values are stored as SQLite INTEGER (exact up to
    9.2e18); fractional values as REAL. The provider (Yahoo Finance) delivers
    IEEE-754 doubles, so no precision present in the source is lost.

``UTCDateTime``
    Every timestamp is stored as UTC and returned as a timezone-aware UTC
    ``datetime``, regardless of the machine's local timezone.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from sqlalchemy import DateTime, MetaData, Numeric
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.types import TypeDecorator

from app.utils.helpers import to_decimal

# Deterministic constraint names are required for Alembic batch migrations on SQLite.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

_INT64_LIMIT = Decimal(2) ** 63


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class FinancialNumber(TypeDecorator):
    """Finite numeric value stored in SQLite, exposed to Python as ``Decimal``."""

    impl = Numeric(28, 6, asdecimal=False)
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Any) -> int | float | None:
        if value is None:
            return None
        decimal_value = to_decimal(value)
        if decimal_value is None:
            raise ValueError(f"Refusing to store non-finite or non-numeric value {value!r}")
        if decimal_value == decimal_value.to_integral_value() and abs(decimal_value) < _INT64_LIMIT:
            return int(decimal_value)
        return float(decimal_value)

    def process_result_value(self, value: Any, dialect: Any) -> Decimal | None:
        if value is None:
            return None
        return to_decimal(value)


class UTCDateTime(TypeDecorator):
    """Timezone-aware UTC datetime stored as naive UTC."""

    impl = DateTime()
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Any) -> datetime | None:
        if value is None:
            return None
        if not isinstance(value, datetime):
            raise TypeError(f"Expected datetime, got {type(value).__name__}")
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).replace(tzinfo=None)

    def process_result_value(self, value: Any, dialect: Any) -> datetime | None:
        if value is None:
            return None
        return value.replace(tzinfo=timezone.utc)
