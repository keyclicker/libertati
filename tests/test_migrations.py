"""Alembic migrations create exactly the schema the metadata declares."""

import stat
from pathlib import Path

from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import create_engine, inspect

from libertati import schema
from libertati.migrations import upgrade_to_head


def test_upgrade_creates_the_declared_schema(tmp_path: Path) -> None:
    """A fresh upgrade matches schema.metadata with no drift."""
    db_path = tmp_path / "fresh.db"
    upgrade_to_head(db_path)
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.connect() as conn:
        diff = compare_metadata(MigrationContext.configure(conn), schema.metadata)
    engine.dispose()
    assert diff == []


def test_upgrade_is_idempotent(tmp_path: Path) -> None:
    """Running the upgrade twice leaves the database untouched."""
    db_path = tmp_path / "twice.db"
    upgrade_to_head(db_path)
    upgrade_to_head(db_path)
    engine = create_engine(f"sqlite:///{db_path}")
    assert "messages" in inspect(engine).get_table_names()
    engine.dispose()


def test_upgraded_database_is_private_to_its_owner(tmp_path: Path) -> None:
    """The migrated file carries owner-only permissions."""
    db_path = tmp_path / "private.db"
    upgrade_to_head(db_path)
    assert stat.S_IMODE(db_path.stat().st_mode) == 0o600
