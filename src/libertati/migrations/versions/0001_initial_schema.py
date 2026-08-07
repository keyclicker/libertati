"""Initial schema.

Revision ID: 0001
Revises:
Create Date: 2026-08-06 14:51:28.951942
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Apply this revision."""
    op.create_table(
        "agent_turns",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("start_context_id", sa.Integer(), nullable=False),
        sa.Column("end_context_id", sa.Integer(), nullable=True),
        sa.Column(
            "status", sa.Text(), server_default=sa.text("'running'"), nullable=False
        ),
        sa.Column(
            "started_at",
            sa.Text(),
            server_default=sa.text("(datetime('now'))"),
            nullable=False,
        ),
        sa.Column("finished_at", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sqlite_autoincrement=True,
    )
    op.create_table(
        "api_usage",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("response_id", sa.Text(), nullable=True),
        sa.Column("turn_id", sa.Integer(), nullable=True),
        sa.Column("dream_id", sa.Integer(), nullable=True),
        sa.Column(
            "input_context_id",
            sa.Integer(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=False),
        sa.Column("cached_tokens", sa.Integer(), nullable=False),
        sa.Column("cache_write_tokens", sa.Integer(), nullable=False),
        sa.Column("output_tokens", sa.Integer(), nullable=False),
        sa.Column("reasoning_tokens", sa.Integer(), nullable=False),
        sa.Column("total_tokens", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.Text(),
            server_default=sa.text("(datetime('now'))"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sqlite_autoincrement=True,
    )
    op.create_table(
        "chats",
        sa.Column("id", sa.Integer(), autoincrement=False, nullable=False),
        sa.Column("type", sa.Text(), nullable=False),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("username", sa.Text(), nullable=True),
        sa.Column("raw", sa.Text(), nullable=False),
        sa.Column(
            "first_seen_at",
            sa.Text(),
            server_default=sa.text("(datetime('now'))"),
            nullable=False,
        ),
        sa.Column(
            "last_seen_at",
            sa.Text(),
            server_default=sa.text("(datetime('now'))"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "context",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("item", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.Text(),
            server_default=sa.text("(datetime('now'))"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sqlite_autoincrement=True,
    )
    op.create_table(
        "dreams",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("trigger", sa.Text(), nullable=False),
        sa.Column(
            "status", sa.Text(), server_default=sa.text("'running'"), nullable=False
        ),
        sa.Column("steps", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column(
            "started_at",
            sa.Text(),
            server_default=sa.text("(datetime('now'))"),
            nullable=False,
        ),
        sa.Column("finished_at", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sqlite_autoincrement=True,
    )
    with op.batch_alter_table("dreams", schema=None) as batch_op:
        batch_op.create_index("idx_dreams_started", ["started_at"], unique=False)

    op.create_table(
        "media_notes",
        sa.Column("file_unique_id", sa.Text(), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("note", sa.Text(), nullable=False),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.Text(),
            server_default=sa.text("(datetime('now'))"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("file_unique_id"),
    )
    op.create_table(
        "steering",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("urgent", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("done", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column(
            "created_at",
            sa.Text(),
            server_default=sa.text("(datetime('now'))"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sqlite_autoincrement=True,
    )
    with op.batch_alter_table("steering", schema=None) as batch_op:
        batch_op.create_index("idx_steering_pending", ["done", "id"], unique=False)

    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), autoincrement=False, nullable=False),
        sa.Column("is_bot", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("username", sa.Text(), nullable=True),
        sa.Column("first_name", sa.Text(), nullable=True),
        sa.Column("last_name", sa.Text(), nullable=True),
        sa.Column("language_code", sa.Text(), nullable=True),
        sa.Column("raw", sa.Text(), nullable=False),
        sa.Column(
            "first_seen_at",
            sa.Text(),
            server_default=sa.text("(datetime('now'))"),
            nullable=False,
        ),
        sa.Column(
            "last_seen_at",
            sa.Text(),
            server_default=sa.text("(datetime('now'))"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "wakeups",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("due_at", sa.Text(), nullable=False),
        sa.Column("note", sa.Text(), nullable=False),
        sa.Column("done", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column(
            "created_at",
            sa.Text(),
            server_default=sa.text("(datetime('now'))"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sqlite_autoincrement=True,
    )
    with op.batch_alter_table("wakeups", schema=None) as batch_op:
        batch_op.create_index("idx_wakeups_due", ["done", "due_at"], unique=False)

    op.create_table(
        "dream_context",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("dream_id", sa.Integer(), nullable=False),
        sa.Column("item", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.Text(),
            server_default=sa.text("(datetime('now'))"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["dream_id"],
            ["dreams.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sqlite_autoincrement=True,
    )
    with op.batch_alter_table("dream_context", schema=None) as batch_op:
        batch_op.create_index(
            "idx_dream_context_dream", ["dream_id", "id"], unique=False
        )

    op.create_table(
        "message_read_cursors",
        sa.Column("chat_id", sa.Integer(), nullable=False),
        sa.Column(
            "message_thread_id",
            sa.Integer(),
            server_default=sa.text("0"),
            autoincrement=False,
            nullable=False,
        ),
        sa.Column("message_id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["chat_id"],
            ["chats.id"],
        ),
        sa.PrimaryKeyConstraint("chat_id", "message_thread_id"),
    )
    op.create_table(
        "messages",
        sa.Column("chat_id", sa.Integer(), nullable=False),
        sa.Column("message_id", sa.Integer(), autoincrement=False, nullable=False),
        sa.Column("from_user_id", sa.Integer(), nullable=True),
        sa.Column("date", sa.Text(), nullable=False),
        sa.Column("edit_date", sa.Text(), nullable=True),
        sa.Column("content_type", sa.Text(), nullable=False),
        sa.Column("text", sa.Text(), nullable=True),
        sa.Column("caption", sa.Text(), nullable=True),
        sa.Column("reply_to_message_id", sa.Integer(), nullable=True),
        sa.Column("message_thread_id", sa.Integer(), nullable=True),
        sa.Column("media_group_id", sa.Text(), nullable=True),
        sa.Column("media_uid", sa.Text(), nullable=True),
        sa.Column(
            "outgoing", sa.Integer(), server_default=sa.text("0"), nullable=False
        ),
        sa.Column("raw", sa.Text(), nullable=False),
        sa.Column(
            "saved_at",
            sa.Text(),
            server_default=sa.text("(datetime('now'))"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["chat_id"],
            ["chats.id"],
        ),
        sa.ForeignKeyConstraint(
            ["from_user_id"],
            ["users.id"],
        ),
        sa.PrimaryKeyConstraint("chat_id", "message_id"),
    )
    with op.batch_alter_table("messages", schema=None) as batch_op:
        batch_op.create_index(
            "idx_messages_chat_date", ["chat_id", "date"], unique=False
        )
        batch_op.create_index(
            "idx_messages_chat_thread",
            ["chat_id", "message_thread_id", "date"],
            unique=False,
        )
        batch_op.create_index("idx_messages_from_user", ["from_user_id"], unique=False)


def downgrade() -> None:
    """Revert this revision."""
    with op.batch_alter_table("messages", schema=None) as batch_op:
        batch_op.drop_index("idx_messages_from_user")
        batch_op.drop_index("idx_messages_chat_thread")
        batch_op.drop_index("idx_messages_chat_date")

    op.drop_table("messages")
    op.drop_table("message_read_cursors")
    with op.batch_alter_table("dream_context", schema=None) as batch_op:
        batch_op.drop_index("idx_dream_context_dream")

    op.drop_table("dream_context")
    with op.batch_alter_table("wakeups", schema=None) as batch_op:
        batch_op.drop_index("idx_wakeups_due")

    op.drop_table("wakeups")
    op.drop_table("users")
    with op.batch_alter_table("steering", schema=None) as batch_op:
        batch_op.drop_index("idx_steering_pending")

    op.drop_table("steering")
    op.drop_table("media_notes")
    with op.batch_alter_table("dreams", schema=None) as batch_op:
        batch_op.drop_index("idx_dreams_started")

    op.drop_table("dreams")
    op.drop_table("context")
    op.drop_table("chats")
    op.drop_table("api_usage")
    op.drop_table("agent_turns")
