"""SQLAlchemy 2.x ORM models (the relational schema).

Design principles
-----------------
* Raw provider data (``raw_financial_facts``) and cleaned data
  (``standardized_financial_facts``) live in separate tables; cleaning never
  modifies raw rows.
* Every fetch, cleaning, analysis and comparison is a *run* row, so history is
  preserved and every result can be traced to the exact data snapshot it used.
* Unique constraints prevent duplicate facts/results; CHECK constraints restrict
  status and enum columns; indexes cover the common look-ups (company + period).
* Amounts use :class:`FinancialNumber` (finite Decimal), timestamps use
  :class:`UTCDateTime`.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (JSON, Boolean, CheckConstraint, Date, Float, ForeignKey, Index,
                        Integer, String, Text, UniqueConstraint)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, FinancialNumber, UTCDateTime
from app.utils.helpers import utcnow

# --------------------------------------------------------------------------- enumerations
RUN_RUNNING = "RUNNING"
RUN_SUCCESS = "SUCCESS"
RUN_PARTIAL = "PARTIAL_SUCCESS"
RUN_FAILED = "FAILED"
RUN_STATUSES = (RUN_RUNNING, RUN_SUCCESS, RUN_PARTIAL, RUN_FAILED)

FREQ_ANNUAL = "annual"
FREQ_QUARTERLY = "quarterly"
FREQUENCIES = (FREQ_ANNUAL, FREQ_QUARTERLY)

STATEMENT_TYPES = ("income_statement", "balance_sheet", "cash_flow")

MAPPING_MAPPED = "MAPPED"
MAPPING_UNMAPPED = "UNMAPPED"
MAPPING_CONFLICT = "CONFLICT_RESOLVED"
MAPPING_STATUSES = (MAPPING_MAPPED, MAPPING_UNMAPPED, MAPPING_CONFLICT)

VALIDATION_VALID = "VALID"
VALIDATION_WARNING = "WARNING"
VALIDATION_MISSING = "MISSING"
VALIDATION_INVALID = "INVALID"
VALIDATION_STATUSES = (VALIDATION_VALID, VALIDATION_WARNING, VALIDATION_MISSING,
                       VALIDATION_INVALID)

CALC_OK = "OK"
CALC_UNAVAILABLE = "UNAVAILABLE"
CALC_NOT_MEANINGFUL = "NOT_MEANINGFUL"
CALC_STATUSES = (CALC_OK, CALC_UNAVAILABLE, CALC_NOT_MEANINGFUL)

TREND_OK = "OK"
TREND_ABSOLUTE_ONLY = "ABSOLUTE_ONLY"
TREND_UNAVAILABLE = "UNAVAILABLE"
TREND_STATUSES = (TREND_OK, TREND_ABSOLUTE_ONLY, TREND_UNAVAILABLE)

SEVERITIES = ("LOW", "MEDIUM", "HIGH")

SCORE_SCORED = "SCORED"
SCORE_UNAVAILABLE = "UNAVAILABLE"
SCORE_STATUSES = (SCORE_SCORED, SCORE_UNAVAILABLE)


def _in(column: str, values: tuple[str, ...]) -> str:
    quoted = ", ".join(f"'{v}'" for v in values)
    return f"{column} IN ({quoted})"


# --------------------------------------------------------------------------- A. companies
class Company(Base):
    __tablename__ = "companies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker: Mapped[str] = mapped_column(String(20), nullable=False, unique=True, index=True)
    name: Mapped[str | None] = mapped_column(String(255))
    exchange: Mapped[str | None] = mapped_column(String(50))
    sector: Mapped[str | None] = mapped_column(String(100))
    industry: Mapped[str | None] = mapped_column(String(150))
    country: Mapped[str | None] = mapped_column(String(100))
    currency: Mapped[str | None] = mapped_column(String(10))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False, default=utcnow,
                                                 onupdate=utcnow)

    fetch_runs: Mapped[list[FetchRun]] = relationship(back_populates="company")
    analysis_runs: Mapped[list[AnalysisRun]] = relationship(back_populates="company")

    __table_args__ = (CheckConstraint("ticker = UPPER(TRIM(ticker))", name="ticker_normalized"),)

    def __repr__(self) -> str:
        return f"<Company {self.ticker}>"


# --------------------------------------------------------------------------- B. fetch runs
class FetchRun(Base):
    __tablename__ = "fetch_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id", ondelete="RESTRICT"),
                                            nullable=False)
    provider: Mapped[str] = mapped_column(String(50), nullable=False)
    frequency: Mapped[str] = mapped_column(String(10), nullable=False, default=FREQ_ANNUAL)
    started_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False, default=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default=RUN_RUNNING)
    error_details: Mapped[str | None] = mapped_column(Text)
    statements_retrieved: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    raw_file_refs: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    fetch_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSON)

    company: Mapped[Company] = relationship(back_populates="fetch_runs")
    raw_facts: Mapped[list[RawFinancialFact]] = relationship(back_populates="fetch_run")

    __table_args__ = (
        CheckConstraint(_in("status", RUN_STATUSES), name="status_valid"),
        CheckConstraint(_in("frequency", FREQUENCIES), name="frequency_valid"),
        CheckConstraint("statements_retrieved BETWEEN 0 AND 3", name="statements_range"),
        Index("ix_fetch_runs_company_started", "company_id", "started_at"),
    )


# --------------------------------------------------------------------------- C. raw facts
class RawFinancialFact(Base):
    """One cell of a provider statement, exactly as received (value NULL = provider gave NaN)."""

    __tablename__ = "raw_financial_facts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    fetch_run_id: Mapped[int] = mapped_column(ForeignKey("fetch_runs.id", ondelete="CASCADE"),
                                              nullable=False)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id", ondelete="RESTRICT"),
                                            nullable=False)
    statement_type: Mapped[str] = mapped_column(String(20), nullable=False)
    original_metric_name: Mapped[str] = mapped_column(String(200), nullable=False)
    fiscal_period_end: Mapped[date] = mapped_column(Date, nullable=False)
    period_type: Mapped[str] = mapped_column(String(10), nullable=False)
    original_value: Mapped[Decimal | None] = mapped_column(FinancialNumber)
    currency: Mapped[str | None] = mapped_column(String(10))
    unit: Mapped[str] = mapped_column(String(20), nullable=False, default="units")
    retrieved_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False, default=utcnow)

    fetch_run: Mapped[FetchRun] = relationship(back_populates="raw_facts")

    __table_args__ = (
        UniqueConstraint("fetch_run_id", "statement_type", "original_metric_name",
                         "fiscal_period_end", "period_type"),
        CheckConstraint(_in("statement_type", STATEMENT_TYPES), name="statement_valid"),
        CheckConstraint(_in("period_type", FREQUENCIES), name="period_type_valid"),
        Index("ix_raw_facts_company_period", "company_id", "fiscal_period_end"),
    )


# --------------------------------------------------------------------------- cleaning runs
class CleaningRun(Base):
    """Summary of one cleaning/standardisation pass over a fetch run."""

    __tablename__ = "cleaning_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id", ondelete="RESTRICT"),
                                            nullable=False)
    fetch_run_id: Mapped[int] = mapped_column(ForeignKey("fetch_runs.id", ondelete="CASCADE"),
                                              nullable=False)
    executed_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False, default=utcnow)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    raw_record_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    standardized_record_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    missing_value_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    unmapped_metric_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    conflict_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    duplicate_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    warnings: Mapped[list[str] | None] = mapped_column(JSON)
    unmapped_metrics: Mapped[list[str] | None] = mapped_column(JSON)
    error_details: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        CheckConstraint(_in("status", RUN_STATUSES), name="status_valid"),
        Index("ix_cleaning_runs_fetch", "fetch_run_id", "executed_at"),
    )


# --------------------------------------------------------------------------- D. standardized facts
class StandardizedFinancialFact(Base):
    __tablename__ = "standardized_financial_facts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id", ondelete="RESTRICT"),
                                            nullable=False)
    fetch_run_id: Mapped[int] = mapped_column(ForeignKey("fetch_runs.id", ondelete="CASCADE"),
                                              nullable=False)
    cleaning_run_id: Mapped[int | None] = mapped_column(
        ForeignKey("cleaning_runs.id", ondelete="SET NULL"))
    raw_fact_id: Mapped[int | None] = mapped_column(
        ForeignKey("raw_financial_facts.id", ondelete="SET NULL"))
    statement_type: Mapped[str] = mapped_column(String(20), nullable=False)
    metric_code: Mapped[str] = mapped_column(String(210), nullable=False)
    metric_name: Mapped[str] = mapped_column(String(200), nullable=False)
    fiscal_period_end: Mapped[date] = mapped_column(Date, nullable=False)
    period_type: Mapped[str] = mapped_column(String(10), nullable=False)
    value: Mapped[Decimal | None] = mapped_column(FinancialNumber)
    currency: Mapped[str | None] = mapped_column(String(10))
    source_metric_name: Mapped[str] = mapped_column(String(200), nullable=False)
    mapping_status: Mapped[str] = mapped_column(String(20), nullable=False)
    validation_status: Mapped[str] = mapped_column(String(10), nullable=False)
    validation_message: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        UniqueConstraint("company_id", "statement_type", "metric_code", "fiscal_period_end",
                         "period_type", "fetch_run_id"),
        CheckConstraint(_in("statement_type", STATEMENT_TYPES), name="statement_valid"),
        CheckConstraint(_in("mapping_status", MAPPING_STATUSES), name="mapping_valid"),
        CheckConstraint(_in("validation_status", VALIDATION_STATUSES), name="validation_valid"),
        Index("ix_std_facts_lookup", "company_id", "fetch_run_id", "metric_code",
              "fiscal_period_end"),
    )


# --------------------------------------------------------------------------- E. analysis runs
class AnalysisRun(Base):
    __tablename__ = "analysis_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id", ondelete="RESTRICT"),
                                            nullable=False)
    fetch_run_id: Mapped[int] = mapped_column(ForeignKey("fetch_runs.id", ondelete="RESTRICT"),
                                              nullable=False)
    executed_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False, default=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default=RUN_RUNNING)
    methodology_version: Mapped[str] = mapped_column(String(20), nullable=False)
    scoring_methodology_version: Mapped[str | None] = mapped_column(String(20))
    red_flag_config_version: Mapped[str | None] = mapped_column(String(20))
    currency: Mapped[str | None] = mapped_column(String(10))
    latest_fiscal_period: Mapped[date | None] = mapped_column(Date)
    summary: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    error_details: Mapped[str | None] = mapped_column(Text)

    company: Mapped[Company] = relationship(back_populates="analysis_runs")
    ratios: Mapped[list[RatioResult]] = relationship(back_populates="analysis_run",
                                                     cascade="all, delete-orphan")
    trends: Mapped[list[TrendResult]] = relationship(back_populates="analysis_run",
                                                     cascade="all, delete-orphan")
    red_flags: Mapped[list[RedFlagResult]] = relationship(back_populates="analysis_run",
                                                          cascade="all, delete-orphan")
    health_scores: Mapped[list[HealthScore]] = relationship(back_populates="analysis_run",
                                                            cascade="all, delete-orphan")

    __table_args__ = (
        CheckConstraint(_in("status", RUN_STATUSES), name="status_valid"),
        Index("ix_analysis_runs_company_executed", "company_id", "executed_at"),
    )


# --------------------------------------------------------------------------- F. ratio results
class RatioResult(Base):
    __tablename__ = "ratio_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    analysis_run_id: Mapped[int] = mapped_column(
        ForeignKey("analysis_runs.id", ondelete="CASCADE"), nullable=False)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id", ondelete="RESTRICT"),
                                            nullable=False)
    fiscal_period_end: Mapped[date] = mapped_column(Date, nullable=False)
    category: Mapped[str] = mapped_column(String(20), nullable=False)
    ratio_code: Mapped[str] = mapped_column(String(50), nullable=False)
    ratio_name: Mapped[str] = mapped_column(String(100), nullable=False)
    value: Mapped[Decimal | None] = mapped_column(FinancialNumber)
    unit: Mapped[str] = mapped_column(String(10), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    reason_code: Mapped[str | None] = mapped_column(String(40))
    method: Mapped[str | None] = mapped_column(String(40))
    formula: Mapped[str] = mapped_column(Text, nullable=False)
    input_values: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    explanation: Mapped[str | None] = mapped_column(Text)

    analysis_run: Mapped[AnalysisRun] = relationship(back_populates="ratios")

    __table_args__ = (
        UniqueConstraint("analysis_run_id", "ratio_code", "fiscal_period_end"),
        CheckConstraint(_in("status", CALC_STATUSES), name="status_valid"),
        CheckConstraint(f"(status = '{CALC_OK}') = (value IS NOT NULL)",
                        name="value_matches_status"),
        Index("ix_ratio_results_company_ratio", "company_id", "ratio_code", "fiscal_period_end"),
    )


# --------------------------------------------------------------------------- G. trend results
class TrendResult(Base):
    __tablename__ = "trend_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    analysis_run_id: Mapped[int] = mapped_column(
        ForeignKey("analysis_runs.id", ondelete="CASCADE"), nullable=False)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id", ondelete="RESTRICT"),
                                            nullable=False)
    metric_code: Mapped[str] = mapped_column(String(60), nullable=False)
    comparison_type: Mapped[str] = mapped_column(String(15), nullable=False)
    current_period: Mapped[date] = mapped_column(Date, nullable=False)
    previous_period: Mapped[date | None] = mapped_column(Date)
    current_value: Mapped[Decimal | None] = mapped_column(FinancialNumber)
    previous_value: Mapped[Decimal | None] = mapped_column(FinancialNumber)
    absolute_change: Mapped[Decimal | None] = mapped_column(FinancialNumber)
    percentage_change: Mapped[Decimal | None] = mapped_column(FinancialNumber)
    direction: Mapped[str] = mapped_column(String(15), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    explanation: Mapped[str | None] = mapped_column(Text)

    analysis_run: Mapped[AnalysisRun] = relationship(back_populates="trends")

    __table_args__ = (
        UniqueConstraint("analysis_run_id", "metric_code", "comparison_type", "current_period"),
        CheckConstraint(_in("status", TREND_STATUSES), name="status_valid"),
        CheckConstraint(_in("comparison_type", ("YOY", "MULTI_YEAR")), name="type_valid"),
        Index("ix_trend_results_company_metric", "company_id", "metric_code"),
    )


# --------------------------------------------------------------------------- H. red flags
class RedFlagResult(Base):
    __tablename__ = "red_flag_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    analysis_run_id: Mapped[int] = mapped_column(
        ForeignKey("analysis_runs.id", ondelete="CASCADE"), nullable=False)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id", ondelete="RESTRICT"),
                                            nullable=False)
    fiscal_period_end: Mapped[date] = mapped_column(Date, nullable=False)
    previous_period_end: Mapped[date | None] = mapped_column(Date)
    rule_code: Mapped[str] = mapped_column(String(50), nullable=False)
    rule_name: Mapped[str] = mapped_column(String(100), nullable=False)
    severity: Mapped[str] = mapped_column(String(10), nullable=False)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    triggered_condition: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    evidence: Mapped[dict[str, Any] | None] = mapped_column(JSON)

    analysis_run: Mapped[AnalysisRun] = relationship(back_populates="red_flags")

    __table_args__ = (
        UniqueConstraint("analysis_run_id", "rule_code", "fiscal_period_end"),
        CheckConstraint(_in("severity", SEVERITIES), name="severity_valid"),
        Index("ix_red_flags_company_period", "company_id", "fiscal_period_end"),
    )


# --------------------------------------------------------------------------- I. health scores
class HealthScore(Base):
    __tablename__ = "health_scores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    analysis_run_id: Mapped[int] = mapped_column(
        ForeignKey("analysis_runs.id", ondelete="CASCADE"), nullable=False)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id", ondelete="RESTRICT"),
                                            nullable=False)
    fiscal_period_end: Mapped[date] = mapped_column(Date, nullable=False)
    category: Mapped[str] = mapped_column(String(20), nullable=False)
    score: Mapped[Decimal | None] = mapped_column(FinancialNumber)
    status: Mapped[str] = mapped_column(String(15), nullable=False)
    data_coverage: Mapped[float] = mapped_column(Float, nullable=False)
    interpretation: Mapped[str | None] = mapped_column(String(20))
    methodology_version: Mapped[str] = mapped_column(String(20), nullable=False)
    explanation: Mapped[str] = mapped_column(Text, nullable=False)
    components: Mapped[dict[str, Any] | None] = mapped_column(JSON)

    analysis_run: Mapped[AnalysisRun] = relationship(back_populates="health_scores")

    __table_args__ = (
        UniqueConstraint("analysis_run_id", "fiscal_period_end", "category"),
        CheckConstraint(_in("status", SCORE_STATUSES), name="status_valid"),
        CheckConstraint("score IS NULL OR (score >= 0 AND score <= 100)", name="score_range"),
        CheckConstraint("data_coverage >= 0 AND data_coverage <= 1", name="coverage_range"),
        CheckConstraint(f"(status = '{SCORE_SCORED}') = (score IS NOT NULL)",
                        name="score_matches_status"),
        Index("ix_health_scores_company_period", "company_id", "fiscal_period_end"),
    )


# --------------------------------------------------------------------------- market snapshots
class MarketSnapshot(Base):
    """Point-in-time market data (price, market cap) used for descriptive valuation multiples.

    Kept separate from statement data: market values change daily, statements yearly.
    ``provider_values`` stores the provider's own pre-computed multiples for cross-checking.
    """

    __tablename__ = "market_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id", ondelete="RESTRICT"),
                                            nullable=False)
    fetched_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False, default=utcnow)
    provider: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    price: Mapped[Decimal | None] = mapped_column(FinancialNumber)
    market_cap: Mapped[Decimal | None] = mapped_column(FinancialNumber)
    shares_outstanding: Mapped[Decimal | None] = mapped_column(FinancialNumber)
    currency: Mapped[str | None] = mapped_column(String(10))
    provider_values: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    error_details: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        CheckConstraint(_in("status", (RUN_SUCCESS, RUN_FAILED)), name="status_valid"),
        CheckConstraint("market_cap IS NULL OR market_cap > 0", name="market_cap_positive"),
        Index("ix_market_snapshots_company_fetched", "company_id", "fetched_at"),
    )


# --------------------------------------------------------------------------- J. comparisons
class ComparisonRun(Base):
    __tablename__ = "comparison_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    executed_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False, default=utcnow)
    requested_tickers: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    selected_fiscal_period: Mapped[str] = mapped_column(String(20), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    currency_compatibility: Mapped[str] = mapped_column(String(20), nullable=False)
    period_alignment: Mapped[str] = mapped_column(String(20), nullable=False)
    configuration: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    warnings: Mapped[list[str] | None] = mapped_column(JSON)
    error_details: Mapped[str | None] = mapped_column(Text)

    participants: Mapped[list[ComparisonCompany]] = relationship(
        back_populates="comparison_run", cascade="all, delete-orphan")
    results: Mapped[list[ComparisonResult]] = relationship(
        back_populates="comparison_run", cascade="all, delete-orphan")

    __table_args__ = (
        CheckConstraint(_in("status", RUN_STATUSES), name="status_valid"),
        CheckConstraint(_in("currency_compatibility", ("COMPATIBLE", "MIXED", "UNKNOWN")),
                        name="currency_valid"),
        CheckConstraint(_in("period_alignment", ("ALIGNED", "MISALIGNED", "UNKNOWN")),
                        name="alignment_valid"),
    )


class ComparisonCompany(Base):
    """Association between a comparison run and each requested company."""

    __tablename__ = "comparison_companies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    comparison_run_id: Mapped[int] = mapped_column(
        ForeignKey("comparison_runs.id", ondelete="CASCADE"), nullable=False)
    company_id: Mapped[int | None] = mapped_column(
        ForeignKey("companies.id", ondelete="RESTRICT"))
    ticker: Mapped[str] = mapped_column(String(20), nullable=False)
    analysis_run_id: Mapped[int | None] = mapped_column(
        ForeignKey("analysis_runs.id", ondelete="SET NULL"))
    fiscal_period_end: Mapped[date | None] = mapped_column(Date)
    currency: Mapped[str | None] = mapped_column(String(10))
    included: Mapped[bool] = mapped_column(Boolean, nullable=False)
    exclusion_reason: Mapped[str | None] = mapped_column(Text)

    comparison_run: Mapped[ComparisonRun] = relationship(back_populates="participants")

    __table_args__ = (UniqueConstraint("comparison_run_id", "ticker"),)


class ComparisonResult(Base):
    __tablename__ = "comparison_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    comparison_run_id: Mapped[int] = mapped_column(
        ForeignKey("comparison_runs.id", ondelete="CASCADE"), nullable=False)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id", ondelete="RESTRICT"),
                                            nullable=False)
    metric_code: Mapped[str] = mapped_column(String(60), nullable=False)
    metric_kind: Mapped[str] = mapped_column(String(10), nullable=False)
    value: Mapped[Decimal | None] = mapped_column(FinancialNumber)
    text_value: Mapped[str | None] = mapped_column(String(50))
    unit: Mapped[str | None] = mapped_column(String(10))
    currency: Mapped[str | None] = mapped_column(String(10))
    fiscal_period_end: Mapped[date | None] = mapped_column(Date)
    rank: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    note: Mapped[str | None] = mapped_column(Text)

    comparison_run: Mapped[ComparisonRun] = relationship(back_populates="results")

    __table_args__ = (
        UniqueConstraint("comparison_run_id", "company_id", "metric_code"),
        CheckConstraint(_in("metric_kind", ("AMOUNT", "RATIO", "SCORE", "COUNT", "TREND")),
                        name="kind_valid"),
        Index("ix_comparison_results_metric", "comparison_run_id", "metric_code"),
    )
