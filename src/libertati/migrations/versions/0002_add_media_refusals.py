"""Add media_refusals.

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-07 00:02:50.963767
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Apply this revision."""
    op.create_table(
        "media_refusals",
        sa.Column("file_unique_id", sa.Text(), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column(
            "refused_at",
            sa.Text(),
            server_default=sa.text("(datetime('now'))"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("file_unique_id"),
    )


def downgrade() -> None:
    """Revert this revision."""
    op.drop_table("media_refusals")
