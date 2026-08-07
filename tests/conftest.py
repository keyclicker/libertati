"""Fixtures shared by the tests that need real persistence."""

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from sqlalchemy import text

from libertati import schema
from libertati.db import Database


@pytest.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    """Yield a connected database in a temporary directory.

    The schema comes straight from the metadata rather than from the
    Alembic migrations — tests stay fast, and test_migrations.py pins
    the two to be identical.
    """
    database = Database(tmp_path / "test.db")
    await database.connect()
    async with database.engine.begin() as conn:
        await conn.run_sync(schema.metadata.create_all)
    yield database
    await database.close()


async def fetch_rows(db: Database, sql: str, params: dict | None = None) -> list[dict]:
    """Run one raw SELECT against a database, returning dict rows."""
    async with db.engine.connect() as conn:
        result = await conn.execute(text(sql), params or {})
        return [dict(row) for row in result.mappings()]


async def run_sql(db: Database, sql: str, params: dict | None = None) -> None:
    """Run one raw statement against a database and commit it."""
    async with db.engine.begin() as conn:
        await conn.execute(text(sql), params or {})
