"""Tests for the interactive context viewer."""

import json
import sqlite3

import pytest
from textual.widgets import RichLog

from libertati.context_tail import ContextApp


def make_context_db(rows: int = 30) -> sqlite3.Connection:
    """Create an in-memory context database with enough rows to scroll."""
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """CREATE TABLE context (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            item TEXT NOT NULL
        )"""
    )
    conn.executemany(
        "INSERT INTO context (item) VALUES (?)",
        [
            (json.dumps({"role": "user", "content": f"event {i}"}),)
            for i in range(rows)
        ],
    )
    return conn


def append_event(conn: sqlite3.Connection, text: str) -> None:
    """Append one event to a test context database."""
    conn.execute(
        "INSERT INTO context (item) VALUES (?)",
        (json.dumps({"role": "user", "content": text}),),
    )


@pytest.mark.asyncio
async def test_vim_keys_scroll_to_both_ends() -> None:
    """The full-screen viewer supports k/g and j/G navigation."""
    conn = make_context_db()
    app = ContextApp(conn, full=False, last_id=0)

    async with app.run_test(size=(80, 10)) as pilot:
        await pilot.pause()
        log = app.query_one("#context", RichLog)
        assert log.is_vertical_scroll_end

        await pilot.press("k")
        await pilot.pause()
        assert log.scroll_y < log.max_scroll_y

        await pilot.press("j")
        await pilot.pause()
        assert log.is_vertical_scroll_end

        await pilot.press("g")
        await pilot.pause()
        assert log.scroll_y == 0

        await pilot.press("G")
        await pilot.pause()
        assert log.is_vertical_scroll_end

    conn.close()


@pytest.mark.asyncio
async def test_new_events_preserve_scrolled_viewport() -> None:
    """Polling follows at the bottom but preserves manual scroll position."""
    conn = make_context_db()
    app = ContextApp(conn, full=False, last_id=0)

    async with app.run_test(size=(80, 10)) as pilot:
        await pilot.pause()
        log = app.query_one("#context", RichLog)

        await pilot.press("pageup")
        await pilot.pause()
        scrolled_y = log.scroll_y
        assert not log.is_vertical_scroll_end

        append_event(conn, "new while reading history")
        app._poll_context()
        await pilot.pause()
        assert app.last_id == 31
        assert log.scroll_y == scrolled_y

        await pilot.press("G")
        await pilot.pause()
        append_event(conn, "new while following")
        app._poll_context()
        await pilot.pause()
        assert app.last_id == 32
        assert log.is_vertical_scroll_end

    conn.close()


@pytest.mark.asyncio
async def test_viewer_pages_back_to_database_start() -> None:
    """Reaching the loaded top pages backward; g loads the true start."""
    conn = make_context_db()
    app = ContextApp(conn, full=False, last_id=25, page_size=5)

    async with app.run_test(size=(80, 10)) as pilot:
        await pilot.pause()
        log = app.query_one("#context", RichLog)
        assert app.rows[0][0] == 26

        old_max_y = log.max_scroll_y
        log.scroll_home(animate=False, immediate=True)
        app._load_older_if_at_top()
        await pilot.pause()
        assert app.rows[0][0] == 21
        assert log.scroll_y == log.max_scroll_y - old_max_y

        await pilot.press("g")
        await pilot.pause()
        assert app.rows[0][0] == 1
        assert log.scroll_y == 0
        assert not app.has_older

    conn.close()
