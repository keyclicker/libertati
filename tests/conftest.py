"""Fixtures shared by the tests that need real persistence."""

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from libertati.db import Database


@pytest.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    """Yield a connected database in a temporary directory."""
    database = Database(tmp_path / "test.db")
    await database.connect()
    yield database
    await database.close()
