"""initial schema

Revision ID: 0001
Revises: 
Create Date: 2026-11-11 15:21:43.220309
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

import app.database.base


revision: str = '0001'
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Generated with alembic --autogenerate from app/database/models.py and reviewed.
    op.create_table('companies',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('ticker', sa.String(length=20), nullable=False),
    sa.Column('name', sa.String(length=255), nullable=True),
    sa.Column('exchange', sa.String(length=50), nullable=True),
    sa.Column('sector', sa.String(length=100), nullable=True),
    sa.Column('industry', sa.String(length=150), nullable=True),
    sa.Column('country', sa.String(length=100), nullable=True),
    sa.Column('currency', sa.String(length=10), nullable=True),
    sa.Column('created_at', app.database.base.UTCDateTime(), nullable=False),
    sa.Column('updated_at', app.database.base.UTCDateTime(), nullable=False),
    sa.CheckConstraint('ticker = UPPER(TRIM(ticker))', name=op.f('ck_companies_ticker_normalized')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_companies'))
    )
    with op.batch_alter_table('companies', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_companies_ticker'), ['ticker'], unique=True)

    op.create_table('comparison_runs',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('executed_at', app.database.base.UTCDateTime(), nullable=False),
    sa.Column('requested_tickers', sa.JSON(), nullable=False),
    sa.Column('selected_fiscal_period', sa.String(length=20), nullable=False),
    sa.Column('status', sa.String(length=20), nullable=False),
    sa.Column('currency_compatibility', sa.String(length=20), nullable=False),
    sa.Column('period_alignment', sa.String(length=20), nullable=False),
    sa.Column('configuration', sa.JSON(), nullable=True),
    sa.Column('warnings', sa.JSON(), nullable=True),
    sa.Column('error_details', sa.Text(), nullable=True),
    sa.CheckConstraint("currency_compatibility IN ('COMPATIBLE', 'MIXED', 'UNKNOWN')", name=op.f('ck_comparison_runs_currency_valid')),
    sa.CheckConstraint("period_alignment IN ('ALIGNED', 'MISALIGNED', 'UNKNOWN')", name=op.f('ck_comparison_runs_alignment_valid')),
    sa.CheckConstraint("status IN ('RUNNING', 'SUCCESS', 'PARTIAL_SUCCESS', 'FAILED')", name=op.f('ck_comparison_runs_status_valid')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_comparison_runs'))
    )
    op.create_table('comparison_results',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('comparison_run_id', sa.Integer(), nullable=False),
    sa.Column('company_id', sa.Integer(), nullable=False),
    sa.Column('metric_code', sa.String(length=60), nullable=False),
    sa.Column('metric_kind', sa.String(length=10), nullable=False),
    sa.Column('value', app.database.base.FinancialNumber(precision=28, scale=6, asdecimal=False), nullable=True),
    sa.Column('text_value', sa.String(length=50), nullable=True),
    sa.Column('unit', sa.String(length=10), nullable=True),
    sa.Column('currency', sa.String(length=10), nullable=True),
    sa.Column('fiscal_period_end', sa.Date(), nullable=True),
    sa.Column('rank', sa.Integer(), nullable=True),
    sa.Column('status', sa.String(length=20), nullable=False),
    sa.Column('note', sa.Text(), nullable=True),
    sa.CheckConstraint("metric_kind IN ('AMOUNT', 'RATIO', 'SCORE', 'COUNT', 'TREND')", name=op.f('ck_comparison_results_kind_valid')),
    sa.ForeignKeyConstraint(['company_id'], ['companies.id'], name=op.f('fk_comparison_results_company_id_companies'), ondelete='RESTRICT'),
    sa.ForeignKeyConstraint(['comparison_run_id'], ['comparison_runs.id'], name=op.f('fk_comparison_results_comparison_run_id_comparison_runs'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_comparison_results')),
    sa.UniqueConstraint('comparison_run_id', 'company_id', 'metric_code', name=op.f('uq_comparison_results_comparison_run_id_company_id_metric_code'))
    )
    with op.batch_alter_table('comparison_results', schema=None) as batch_op:
        batch_op.create_index('ix_comparison_results_metric', ['comparison_run_id', 'metric_code'], unique=False)

    op.create_table('fetch_runs',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('company_id', sa.Integer(), nullable=False),
    sa.Column('provider', sa.String(length=50), nullable=False),
    sa.Column('frequency', sa.String(length=10), nullable=False),
    sa.Column('started_at', app.database.base.UTCDateTime(), nullable=False),
    sa.Column('completed_at', app.database.base.UTCDateTime(), nullable=True),
    sa.Column('status', sa.String(length=20), nullable=False),
    sa.Column('error_details', sa.Text(), nullable=True),
    sa.Column('statements_retrieved', sa.Integer(), nullable=False),
    sa.Column('raw_file_refs', sa.JSON(), nullable=True),
    sa.Column('fetch_metadata', sa.JSON(), nullable=True),
    sa.CheckConstraint("frequency IN ('annual', 'quarterly')", name=op.f('ck_fetch_runs_frequency_valid')),
    sa.CheckConstraint("status IN ('RUNNING', 'SUCCESS', 'PARTIAL_SUCCESS', 'FAILED')", name=op.f('ck_fetch_runs_status_valid')),
    sa.CheckConstraint('statements_retrieved BETWEEN 0 AND 3', name=op.f('ck_fetch_runs_statements_range')),
    sa.ForeignKeyConstraint(['company_id'], ['companies.id'], name=op.f('fk_fetch_runs_company_id_companies'), ondelete='RESTRICT'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_fetch_runs'))
    )
    with op.batch_alter_table('fetch_runs', schema=None) as batch_op:
        batch_op.create_index('ix_fetch_runs_company_started', ['company_id', 'started_at'], unique=False)

    op.create_table('analysis_runs',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('company_id', sa.Integer(), nullable=False),
    sa.Column('fetch_run_id', sa.Integer(), nullable=False),
    sa.Column('executed_at', app.database.base.UTCDateTime(), nullable=False),
    sa.Column('completed_at', app.database.base.UTCDateTime(), nullable=True),
    sa.Column('status', sa.String(length=20), nullable=False),
    sa.Column('methodology_version', sa.String(length=20), nullable=False),
    sa.Column('scoring_methodology_version', sa.String(length=20), nullable=True),
    sa.Column('red_flag_config_version', sa.String(length=20), nullable=True),
    sa.Column('currency', sa.String(length=10), nullable=True),
    sa.Column('latest_fiscal_period', sa.Date(), nullable=True),
    sa.Column('summary', sa.JSON(), nullable=True),
    sa.Column('error_details', sa.Text(), nullable=True),
    sa.CheckConstraint("status IN ('RUNNING', 'SUCCESS', 'PARTIAL_SUCCESS', 'FAILED')", name=op.f('ck_analysis_runs_status_valid')),
    sa.ForeignKeyConstraint(['company_id'], ['companies.id'], name=op.f('fk_analysis_runs_company_id_companies'), ondelete='RESTRICT'),
    sa.ForeignKeyConstraint(['fetch_run_id'], ['fetch_runs.id'], name=op.f('fk_analysis_runs_fetch_run_id_fetch_runs'), ondelete='RESTRICT'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_analysis_runs'))
    )
    with op.batch_alter_table('analysis_runs', schema=None) as batch_op:
        batch_op.create_index('ix_analysis_runs_company_executed', ['company_id', 'executed_at'], unique=False)

    op.create_table('cleaning_runs',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('company_id', sa.Integer(), nullable=False),
    sa.Column('fetch_run_id', sa.Integer(), nullable=False),
    sa.Column('executed_at', app.database.base.UTCDateTime(), nullable=False),
    sa.Column('status', sa.String(length=20), nullable=False),
    sa.Column('raw_record_count', sa.Integer(), nullable=False),
    sa.Column('standardized_record_count', sa.Integer(), nullable=False),
    sa.Column('missing_value_count', sa.Integer(), nullable=False),
    sa.Column('unmapped_metric_count', sa.Integer(), nullable=False),
    sa.Column('conflict_count', sa.Integer(), nullable=False),
    sa.Column('duplicate_count', sa.Integer(), nullable=False),
    sa.Column('warnings', sa.JSON(), nullable=True),
    sa.Column('unmapped_metrics', sa.JSON(), nullable=True),
    sa.Column('error_details', sa.Text(), nullable=True),
    sa.CheckConstraint("status IN ('RUNNING', 'SUCCESS', 'PARTIAL_SUCCESS', 'FAILED')", name=op.f('ck_cleaning_runs_status_valid')),
    sa.ForeignKeyConstraint(['company_id'], ['companies.id'], name=op.f('fk_cleaning_runs_company_id_companies'), ondelete='RESTRICT'),
    sa.ForeignKeyConstraint(['fetch_run_id'], ['fetch_runs.id'], name=op.f('fk_cleaning_runs_fetch_run_id_fetch_runs'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_cleaning_runs'))
    )
    with op.batch_alter_table('cleaning_runs', schema=None) as batch_op:
        batch_op.create_index('ix_cleaning_runs_fetch', ['fetch_run_id', 'executed_at'], unique=False)

    op.create_table('raw_financial_facts',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('fetch_run_id', sa.Integer(), nullable=False),
    sa.Column('company_id', sa.Integer(), nullable=False),
    sa.Column('statement_type', sa.String(length=20), nullable=False),
    sa.Column('original_metric_name', sa.String(length=200), nullable=False),
    sa.Column('fiscal_period_end', sa.Date(), nullable=False),
    sa.Column('period_type', sa.String(length=10), nullable=False),
    sa.Column('original_value', app.database.base.FinancialNumber(precision=28, scale=6, asdecimal=False), nullable=True),
    sa.Column('currency', sa.String(length=10), nullable=True),
    sa.Column('unit', sa.String(length=20), nullable=False),
    sa.Column('retrieved_at', app.database.base.UTCDateTime(), nullable=False),
    sa.CheckConstraint("period_type IN ('annual', 'quarterly')", name=op.f('ck_raw_financial_facts_period_type_valid')),
    sa.CheckConstraint("statement_type IN ('income_statement', 'balance_sheet', 'cash_flow')", name=op.f('ck_raw_financial_facts_statement_valid')),
    sa.ForeignKeyConstraint(['company_id'], ['companies.id'], name=op.f('fk_raw_financial_facts_company_id_companies'), ondelete='RESTRICT'),
    sa.ForeignKeyConstraint(['fetch_run_id'], ['fetch_runs.id'], name=op.f('fk_raw_financial_facts_fetch_run_id_fetch_runs'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_raw_financial_facts')),
    sa.UniqueConstraint('fetch_run_id', 'statement_type', 'original_metric_name', 'fiscal_period_end', 'period_type', name=op.f('uq_raw_financial_facts_fetch_run_id_statement_type_original_metric_name_fiscal_period_end_period_type'))
    )
    with op.batch_alter_table('raw_financial_facts', schema=None) as batch_op:
        batch_op.create_index('ix_raw_facts_company_period', ['company_id', 'fiscal_period_end'], unique=False)

    op.create_table('comparison_companies',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('comparison_run_id', sa.Integer(), nullable=False),
    sa.Column('company_id', sa.Integer(), nullable=True),
    sa.Column('ticker', sa.String(length=20), nullable=False),
    sa.Column('analysis_run_id', sa.Integer(), nullable=True),
    sa.Column('fiscal_period_end', sa.Date(), nullable=True),
    sa.Column('currency', sa.String(length=10), nullable=True),
    sa.Column('included', sa.Boolean(), nullable=False),
    sa.Column('exclusion_reason', sa.Text(), nullable=True),
    sa.ForeignKeyConstraint(['analysis_run_id'], ['analysis_runs.id'], name=op.f('fk_comparison_companies_analysis_run_id_analysis_runs'), ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['company_id'], ['companies.id'], name=op.f('fk_comparison_companies_company_id_companies'), ondelete='RESTRICT'),
    sa.ForeignKeyConstraint(['comparison_run_id'], ['comparison_runs.id'], name=op.f('fk_comparison_companies_comparison_run_id_comparison_runs'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_comparison_companies')),
    sa.UniqueConstraint('comparison_run_id', 'ticker', name=op.f('uq_comparison_companies_comparison_run_id_ticker'))
    )
    op.create_table('health_scores',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('analysis_run_id', sa.Integer(), nullable=False),
    sa.Column('company_id', sa.Integer(), nullable=False),
    sa.Column('fiscal_period_end', sa.Date(), nullable=False),
    sa.Column('category', sa.String(length=20), nullable=False),
    sa.Column('score', app.database.base.FinancialNumber(precision=28, scale=6, asdecimal=False), nullable=True),
    sa.Column('status', sa.String(length=15), nullable=False),
    sa.Column('data_coverage', sa.Float(), nullable=False),
    sa.Column('interpretation', sa.String(length=20), nullable=True),
    sa.Column('methodology_version', sa.String(length=20), nullable=False),
    sa.Column('explanation', sa.Text(), nullable=False),
    sa.Column('components', sa.JSON(), nullable=True),
    sa.CheckConstraint("(status = 'SCORED') = (score IS NOT NULL)", name=op.f('ck_health_scores_score_matches_status')),
    sa.CheckConstraint("status IN ('SCORED', 'UNAVAILABLE')", name=op.f('ck_health_scores_status_valid')),
    sa.CheckConstraint('data_coverage >= 0 AND data_coverage <= 1', name=op.f('ck_health_scores_coverage_range')),
    sa.CheckConstraint('score IS NULL OR (score >= 0 AND score <= 100)', name=op.f('ck_health_scores_score_range')),
    sa.ForeignKeyConstraint(['analysis_run_id'], ['analysis_runs.id'], name=op.f('fk_health_scores_analysis_run_id_analysis_runs'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['company_id'], ['companies.id'], name=op.f('fk_health_scores_company_id_companies'), ondelete='RESTRICT'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_health_scores')),
    sa.UniqueConstraint('analysis_run_id', 'fiscal_period_end', 'category', name=op.f('uq_health_scores_analysis_run_id_fiscal_period_end_category'))
    )
    with op.batch_alter_table('health_scores', schema=None) as batch_op:
        batch_op.create_index('ix_health_scores_company_period', ['company_id', 'fiscal_period_end'], unique=False)

    op.create_table('ratio_results',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('analysis_run_id', sa.Integer(), nullable=False),
    sa.Column('company_id', sa.Integer(), nullable=False),
    sa.Column('fiscal_period_end', sa.Date(), nullable=False),
    sa.Column('category', sa.String(length=20), nullable=False),
    sa.Column('ratio_code', sa.String(length=50), nullable=False),
    sa.Column('ratio_name', sa.String(length=100), nullable=False),
    sa.Column('value', app.database.base.FinancialNumber(precision=28, scale=6, asdecimal=False), nullable=True),
    sa.Column('unit', sa.String(length=10), nullable=False),
    sa.Column('status', sa.String(length=20), nullable=False),
    sa.Column('reason_code', sa.String(length=40), nullable=True),
    sa.Column('method', sa.String(length=40), nullable=True),
    sa.Column('formula', sa.Text(), nullable=False),
    sa.Column('input_values', sa.JSON(), nullable=True),
    sa.Column('explanation', sa.Text(), nullable=True),
    sa.CheckConstraint("(status = 'OK') = (value IS NOT NULL)", name=op.f('ck_ratio_results_value_matches_status')),
    sa.CheckConstraint("status IN ('OK', 'UNAVAILABLE', 'NOT_MEANINGFUL')", name=op.f('ck_ratio_results_status_valid')),
    sa.ForeignKeyConstraint(['analysis_run_id'], ['analysis_runs.id'], name=op.f('fk_ratio_results_analysis_run_id_analysis_runs'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['company_id'], ['companies.id'], name=op.f('fk_ratio_results_company_id_companies'), ondelete='RESTRICT'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_ratio_results')),
    sa.UniqueConstraint('analysis_run_id', 'ratio_code', 'fiscal_period_end', name=op.f('uq_ratio_results_analysis_run_id_ratio_code_fiscal_period_end'))
    )
    with op.batch_alter_table('ratio_results', schema=None) as batch_op:
        batch_op.create_index('ix_ratio_results_company_ratio', ['company_id', 'ratio_code', 'fiscal_period_end'], unique=False)

    op.create_table('red_flag_results',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('analysis_run_id', sa.Integer(), nullable=False),
    sa.Column('company_id', sa.Integer(), nullable=False),
    sa.Column('fiscal_period_end', sa.Date(), nullable=False),
    sa.Column('previous_period_end', sa.Date(), nullable=True),
    sa.Column('rule_code', sa.String(length=50), nullable=False),
    sa.Column('rule_name', sa.String(length=100), nullable=False),
    sa.Column('severity', sa.String(length=10), nullable=False),
    sa.Column('title', sa.String(length=200), nullable=False),
    sa.Column('triggered_condition', sa.Text(), nullable=False),
    sa.Column('description', sa.Text(), nullable=False),
    sa.Column('evidence', sa.JSON(), nullable=True),
    sa.CheckConstraint("severity IN ('LOW', 'MEDIUM', 'HIGH')", name=op.f('ck_red_flag_results_severity_valid')),
    sa.ForeignKeyConstraint(['analysis_run_id'], ['analysis_runs.id'], name=op.f('fk_red_flag_results_analysis_run_id_analysis_runs'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['company_id'], ['companies.id'], name=op.f('fk_red_flag_results_company_id_companies'), ondelete='RESTRICT'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_red_flag_results')),
    sa.UniqueConstraint('analysis_run_id', 'rule_code', 'fiscal_period_end', name=op.f('uq_red_flag_results_analysis_run_id_rule_code_fiscal_period_end'))
    )
    with op.batch_alter_table('red_flag_results', schema=None) as batch_op:
        batch_op.create_index('ix_red_flags_company_period', ['company_id', 'fiscal_period_end'], unique=False)

    op.create_table('standardized_financial_facts',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('company_id', sa.Integer(), nullable=False),
    sa.Column('fetch_run_id', sa.Integer(), nullable=False),
    sa.Column('cleaning_run_id', sa.Integer(), nullable=True),
    sa.Column('raw_fact_id', sa.Integer(), nullable=True),
    sa.Column('statement_type', sa.String(length=20), nullable=False),
    sa.Column('metric_code', sa.String(length=210), nullable=False),
    sa.Column('metric_name', sa.String(length=200), nullable=False),
    sa.Column('fiscal_period_end', sa.Date(), nullable=False),
    sa.Column('period_type', sa.String(length=10), nullable=False),
    sa.Column('value', app.database.base.FinancialNumber(precision=28, scale=6, asdecimal=False), nullable=True),
    sa.Column('currency', sa.String(length=10), nullable=True),
    sa.Column('source_metric_name', sa.String(length=200), nullable=False),
    sa.Column('mapping_status', sa.String(length=20), nullable=False),
    sa.Column('validation_status', sa.String(length=10), nullable=False),
    sa.Column('validation_message', sa.Text(), nullable=True),
    sa.CheckConstraint("mapping_status IN ('MAPPED', 'UNMAPPED', 'CONFLICT_RESOLVED')", name=op.f('ck_standardized_financial_facts_mapping_valid')),
    sa.CheckConstraint("statement_type IN ('income_statement', 'balance_sheet', 'cash_flow')", name=op.f('ck_standardized_financial_facts_statement_valid')),
    sa.CheckConstraint("validation_status IN ('VALID', 'WARNING', 'MISSING', 'INVALID')", name=op.f('ck_standardized_financial_facts_validation_valid')),
    sa.ForeignKeyConstraint(['cleaning_run_id'], ['cleaning_runs.id'], name=op.f('fk_standardized_financial_facts_cleaning_run_id_cleaning_runs'), ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['company_id'], ['companies.id'], name=op.f('fk_standardized_financial_facts_company_id_companies'), ondelete='RESTRICT'),
    sa.ForeignKeyConstraint(['fetch_run_id'], ['fetch_runs.id'], name=op.f('fk_standardized_financial_facts_fetch_run_id_fetch_runs'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['raw_fact_id'], ['raw_financial_facts.id'], name=op.f('fk_standardized_financial_facts_raw_fact_id_raw_financial_facts'), ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_standardized_financial_facts')),
    sa.UniqueConstraint('company_id', 'statement_type', 'metric_code', 'fiscal_period_end', 'period_type', 'fetch_run_id', name=op.f('uq_standardized_financial_facts_company_id_statement_type_metric_code_fiscal_period_end_period_type_fetch_run_id'))
    )
    with op.batch_alter_table('standardized_financial_facts', schema=None) as batch_op:
        batch_op.create_index('ix_std_facts_lookup', ['company_id', 'fetch_run_id', 'metric_code', 'fiscal_period_end'], unique=False)

    op.create_table('trend_results',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('analysis_run_id', sa.Integer(), nullable=False),
    sa.Column('company_id', sa.Integer(), nullable=False),
    sa.Column('metric_code', sa.String(length=60), nullable=False),
    sa.Column('comparison_type', sa.String(length=15), nullable=False),
    sa.Column('current_period', sa.Date(), nullable=False),
    sa.Column('previous_period', sa.Date(), nullable=True),
    sa.Column('current_value', app.database.base.FinancialNumber(precision=28, scale=6, asdecimal=False), nullable=True),
    sa.Column('previous_value', app.database.base.FinancialNumber(precision=28, scale=6, asdecimal=False), nullable=True),
    sa.Column('absolute_change', app.database.base.FinancialNumber(precision=28, scale=6, asdecimal=False), nullable=True),
    sa.Column('percentage_change', app.database.base.FinancialNumber(precision=28, scale=6, asdecimal=False), nullable=True),
    sa.Column('direction', sa.String(length=15), nullable=False),
    sa.Column('status', sa.String(length=20), nullable=False),
    sa.Column('explanation', sa.Text(), nullable=True),
    sa.CheckConstraint("comparison_type IN ('YOY', 'MULTI_YEAR')", name=op.f('ck_trend_results_type_valid')),
    sa.CheckConstraint("status IN ('OK', 'ABSOLUTE_ONLY', 'UNAVAILABLE')", name=op.f('ck_trend_results_status_valid')),
    sa.ForeignKeyConstraint(['analysis_run_id'], ['analysis_runs.id'], name=op.f('fk_trend_results_analysis_run_id_analysis_runs'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['company_id'], ['companies.id'], name=op.f('fk_trend_results_company_id_companies'), ondelete='RESTRICT'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_trend_results')),
    sa.UniqueConstraint('analysis_run_id', 'metric_code', 'comparison_type', 'current_period', name=op.f('uq_trend_results_analysis_run_id_metric_code_comparison_type_current_period'))
    )
    with op.batch_alter_table('trend_results', schema=None) as batch_op:
        batch_op.create_index('ix_trend_results_company_metric', ['company_id', 'metric_code'], unique=False)



def downgrade() -> None:
    # Generated with alembic --autogenerate from app/database/models.py and reviewed.
    with op.batch_alter_table('trend_results', schema=None) as batch_op:
        batch_op.drop_index('ix_trend_results_company_metric')

    op.drop_table('trend_results')
    with op.batch_alter_table('standardized_financial_facts', schema=None) as batch_op:
        batch_op.drop_index('ix_std_facts_lookup')

    op.drop_table('standardized_financial_facts')
    with op.batch_alter_table('red_flag_results', schema=None) as batch_op:
        batch_op.drop_index('ix_red_flags_company_period')

    op.drop_table('red_flag_results')
    with op.batch_alter_table('ratio_results', schema=None) as batch_op:
        batch_op.drop_index('ix_ratio_results_company_ratio')

    op.drop_table('ratio_results')
    with op.batch_alter_table('health_scores', schema=None) as batch_op:
        batch_op.drop_index('ix_health_scores_company_period')

    op.drop_table('health_scores')
    op.drop_table('comparison_companies')
    with op.batch_alter_table('raw_financial_facts', schema=None) as batch_op:
        batch_op.drop_index('ix_raw_facts_company_period')

    op.drop_table('raw_financial_facts')
    with op.batch_alter_table('cleaning_runs', schema=None) as batch_op:
        batch_op.drop_index('ix_cleaning_runs_fetch')

    op.drop_table('cleaning_runs')
    with op.batch_alter_table('analysis_runs', schema=None) as batch_op:
        batch_op.drop_index('ix_analysis_runs_company_executed')

    op.drop_table('analysis_runs')
    with op.batch_alter_table('fetch_runs', schema=None) as batch_op:
        batch_op.drop_index('ix_fetch_runs_company_started')

    op.drop_table('fetch_runs')
    with op.batch_alter_table('comparison_results', schema=None) as batch_op:
        batch_op.drop_index('ix_comparison_results_metric')

    op.drop_table('comparison_results')
    op.drop_table('comparison_runs')
    with op.batch_alter_table('companies', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_companies_ticker'))

    op.drop_table('companies')
