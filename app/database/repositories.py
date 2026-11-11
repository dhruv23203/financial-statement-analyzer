"""Repository functions: every database query used by the services lives here.

All queries use the SQLAlchemy ORM / ``select()`` with bound parameters (no
string-built SQL from user input). Functions receive an open ``Session`` and never
commit; the caller's ``session_scope`` owns the transaction.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from sqlalchemy import delete, desc, func, select
from sqlalchemy.orm import Session

from app.database.models import (FREQ_ANNUAL, RUN_PARTIAL, RUN_SUCCESS, AnalysisRun,
                                 CleaningRun, Company, ComparisonCompany, ComparisonRun,
                                 FetchRun, HealthScore, MarketSnapshot, RatioResult,
                                 RawFinancialFact,
                                 RedFlagResult, StandardizedFinancialFact, TrendResult)

USABLE_RUN_STATUSES = (RUN_SUCCESS, RUN_PARTIAL)
_COMPANY_METADATA_FIELDS = ("name", "exchange", "sector", "industry", "country", "currency")


# --------------------------------------------------------------------------- companies
def get_company(session: Session, ticker: str) -> Company | None:
    return session.scalar(select(Company).where(Company.ticker == ticker))


def get_or_create_company(session: Session, ticker: str) -> tuple[Company, bool]:
    """Return ``(company, created)``; the ticker must already be normalised."""
    company = get_company(session, ticker)
    if company is not None:
        return company, False
    company = Company(ticker=ticker)
    session.add(company)
    session.flush()
    return company, True


def update_company_metadata(company: Company, metadata: dict[str, Any]) -> list[str]:
    """Copy non-empty metadata fields onto the company; returns the fields updated."""
    updated = []
    for field_name in _COMPANY_METADATA_FIELDS:
        value = metadata.get(field_name)
        if value is None or str(value).strip() == "":
            continue
        value = str(value).strip()
        if getattr(company, field_name) != value:
            setattr(company, field_name, value)
            updated.append(field_name)
    return updated


def list_companies(session: Session) -> Sequence[Company]:
    return session.scalars(select(Company).order_by(Company.ticker)).all()


# --------------------------------------------------------------------------- fetch runs
def latest_fetch_run(session: Session, company_id: int, frequency: str = FREQ_ANNUAL,
                     statuses: Iterable[str] = USABLE_RUN_STATUSES) -> FetchRun | None:
    return session.scalar(
        select(FetchRun)
        .where(FetchRun.company_id == company_id, FetchRun.frequency == frequency,
               FetchRun.status.in_(tuple(statuses)))
        .order_by(desc(FetchRun.started_at), desc(FetchRun.id))
        .limit(1))


def fetch_runs_for_company(session: Session, company_id: int) -> Sequence[FetchRun]:
    return session.scalars(select(FetchRun).where(FetchRun.company_id == company_id)
                           .order_by(desc(FetchRun.id))).all()


def raw_facts_for_run(session: Session, fetch_run_id: int) -> Sequence[RawFinancialFact]:
    return session.scalars(
        select(RawFinancialFact).where(RawFinancialFact.fetch_run_id == fetch_run_id)
        .order_by(RawFinancialFact.statement_type, RawFinancialFact.original_metric_name,
                  RawFinancialFact.fiscal_period_end)).all()


def count_raw_facts(session: Session, fetch_run_id: int) -> int:
    return session.scalar(select(func.count()).select_from(RawFinancialFact)
                          .where(RawFinancialFact.fetch_run_id == fetch_run_id)) or 0


# --------------------------------------------------------------------------- cleaning
def delete_standardized_for_run(session: Session, fetch_run_id: int) -> int:
    result = session.execute(delete(StandardizedFinancialFact)
                             .where(StandardizedFinancialFact.fetch_run_id == fetch_run_id))
    return result.rowcount or 0


def standardized_facts_for_run(session: Session, fetch_run_id: int,
                               period_type: str | None = None,
                               mapped_only: bool = False) -> Sequence[StandardizedFinancialFact]:
    query = select(StandardizedFinancialFact).where(
        StandardizedFinancialFact.fetch_run_id == fetch_run_id)
    if period_type:
        query = query.where(StandardizedFinancialFact.period_type == period_type)
    if mapped_only:
        query = query.where(~StandardizedFinancialFact.metric_code.like("UNMAPPED:%"))
    return session.scalars(query.order_by(StandardizedFinancialFact.statement_type,
                                          StandardizedFinancialFact.metric_code,
                                          StandardizedFinancialFact.fiscal_period_end)).all()


def latest_cleaning_run(session: Session, fetch_run_id: int,
                        statuses: Iterable[str] = USABLE_RUN_STATUSES) -> CleaningRun | None:
    return session.scalar(
        select(CleaningRun).where(CleaningRun.fetch_run_id == fetch_run_id,
                                  CleaningRun.status.in_(tuple(statuses)))
        .order_by(desc(CleaningRun.id)).limit(1))


def latest_cleaned_fetch_run(session: Session, company_id: int,
                             frequency: str = FREQ_ANNUAL) -> FetchRun | None:
    """Most recent usable fetch run that has a successful cleaning run."""
    return session.scalar(
        select(FetchRun)
        .join(CleaningRun, CleaningRun.fetch_run_id == FetchRun.id)
        .where(FetchRun.company_id == company_id, FetchRun.frequency == frequency,
               FetchRun.status.in_(USABLE_RUN_STATUSES),
               CleaningRun.status.in_(USABLE_RUN_STATUSES))
        .order_by(desc(FetchRun.started_at), desc(FetchRun.id))
        .limit(1))


def cleaning_runs_for_company(session: Session, company_id: int) -> Sequence[CleaningRun]:
    return session.scalars(select(CleaningRun).where(CleaningRun.company_id == company_id)
                           .order_by(desc(CleaningRun.id))).all()


# --------------------------------------------------------------------------- analysis
def latest_analysis_run(session: Session, company_id: int,
                        statuses: Iterable[str] = USABLE_RUN_STATUSES) -> AnalysisRun | None:
    return session.scalar(
        select(AnalysisRun).where(AnalysisRun.company_id == company_id,
                                  AnalysisRun.status.in_(tuple(statuses)))
        .order_by(desc(AnalysisRun.executed_at), desc(AnalysisRun.id)).limit(1))


def analysis_runs_for_company(session: Session, company_id: int) -> Sequence[AnalysisRun]:
    return session.scalars(select(AnalysisRun).where(AnalysisRun.company_id == company_id)
                           .order_by(desc(AnalysisRun.id))).all()


def ratios_for_run(session: Session, analysis_run_id: int) -> Sequence[RatioResult]:
    return session.scalars(select(RatioResult)
                           .where(RatioResult.analysis_run_id == analysis_run_id)
                           .order_by(RatioResult.category, RatioResult.ratio_code,
                                     RatioResult.fiscal_period_end)).all()


def trends_for_run(session: Session, analysis_run_id: int) -> Sequence[TrendResult]:
    return session.scalars(select(TrendResult)
                           .where(TrendResult.analysis_run_id == analysis_run_id)
                           .order_by(TrendResult.metric_code, TrendResult.comparison_type,
                                     TrendResult.current_period)).all()


def red_flags_for_run(session: Session, analysis_run_id: int) -> Sequence[RedFlagResult]:
    return session.scalars(select(RedFlagResult)
                           .where(RedFlagResult.analysis_run_id == analysis_run_id)
                           .order_by(desc(RedFlagResult.fiscal_period_end),
                                     RedFlagResult.rule_code)).all()


def health_scores_for_run(session: Session, analysis_run_id: int) -> Sequence[HealthScore]:
    return session.scalars(select(HealthScore)
                           .where(HealthScore.analysis_run_id == analysis_run_id)
                           .order_by(HealthScore.fiscal_period_end, HealthScore.category)).all()


# --------------------------------------------------------------------------- market data
def latest_market_snapshot(session: Session, company_id: int) -> MarketSnapshot | None:
    return session.scalar(select(MarketSnapshot).where(
        MarketSnapshot.company_id == company_id, MarketSnapshot.status == RUN_SUCCESS)
        .order_by(desc(MarketSnapshot.fetched_at), desc(MarketSnapshot.id)).limit(1))


# --------------------------------------------------------------------------- comparisons
def comparison_runs_for_company(session: Session, company_id: int) -> Sequence[ComparisonRun]:
    return session.scalars(
        select(ComparisonRun)
        .join(ComparisonCompany, ComparisonCompany.comparison_run_id == ComparisonRun.id)
        .where(ComparisonCompany.company_id == company_id)
        .order_by(desc(ComparisonRun.id))).unique().all()
