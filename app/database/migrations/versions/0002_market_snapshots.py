"""market snapshots

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-11 17:19:14.787019
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

import app.database.base


revision: str = '0002'
down_revision: str | None = '0001'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Generated with alembic --autogenerate from app/database/models.py and reviewed.
    op.create_table('market_snapshots',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('company_id', sa.Integer(), nullable=False),
    sa.Column('fetched_at', app.database.base.UTCDateTime(), nullable=False),
    sa.Column('provider', sa.String(length=50), nullable=False),
    sa.Column('status', sa.String(length=20), nullable=False),
    sa.Column('price', app.database.base.FinancialNumber(precision=28, scale=6, asdecimal=False), nullable=True),
    sa.Column('market_cap', app.database.base.FinancialNumber(precision=28, scale=6, asdecimal=False), nullable=True),
    sa.Column('shares_outstanding', app.database.base.FinancialNumber(precision=28, scale=6, asdecimal=False), nullable=True),
    sa.Column('currency', sa.String(length=10), nullable=True),
    sa.Column('provider_values', sa.JSON(), nullable=True),
    sa.Column('error_details', sa.Text(), nullable=True),
    sa.CheckConstraint("status IN ('SUCCESS', 'FAILED')", name=op.f('ck_market_snapshots_status_valid')),
    sa.CheckConstraint('market_cap IS NULL OR market_cap > 0', name=op.f('ck_market_snapshots_market_cap_positive')),
    sa.ForeignKeyConstraint(['company_id'], ['companies.id'], name=op.f('fk_market_snapshots_company_id_companies'), ondelete='RESTRICT'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_market_snapshots'))
    )
    with op.batch_alter_table('market_snapshots', schema=None) as batch_op:
        batch_op.create_index('ix_market_snapshots_company_fetched', ['company_id', 'fetched_at'], unique=False)



def downgrade() -> None:
    # Generated with alembic --autogenerate from app/database/models.py and reviewed.
    with op.batch_alter_table('market_snapshots', schema=None) as batch_op:
        batch_op.drop_index('ix_market_snapshots_company_fetched')

    op.drop_table('market_snapshots')
