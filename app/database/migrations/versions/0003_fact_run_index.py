"""index standardised facts by fetch run

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-22

Valuation inputs, the screener and the health-score calibration read standardised facts
by ``fetch_run_id IN (...) AND metric_code IN (...)``. The existing composite index starts
with ``company_id``, so those queries scanned the whole table. This adds an index that
starts with the fetch run. Additive only: no data is changed, and downgrade drops it.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = '0003'
down_revision: str | None = '0002'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index("ix_std_facts_run_metric", "standardized_financial_facts",
                    ["fetch_run_id", "metric_code", "fiscal_period_end"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_std_facts_run_metric", table_name="standardized_financial_facts")
