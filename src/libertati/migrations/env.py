"""Alembic migration environment for the bot's SQLite database."""

from alembic import context
from sqlalchemy import create_engine

from libertati import schema

config = context.config
target_metadata = schema.metadata


def run_migrations_offline() -> None:
    """Emit migration SQL without connecting (``--sql`` mode)."""
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        render_as_batch=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live connection."""
    engine = create_engine(config.get_main_option("sqlalchemy.url") or "")
    with engine.connect() as connection:
        # SQLite cannot ALTER most things in place; batch mode rebuilds
        # the table instead, so future revisions can alter columns.
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,
        )
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
