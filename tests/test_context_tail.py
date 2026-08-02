"""Tests for the interactive context viewer."""

import json
import sqlite3

import pytest
from textual.widgets import RichLog

from libertati.context_tail import (
    ContextApp,
    UsageSnapshot,
    build_block,
    build_status,
    fetch_active_turn_start,
    fetch_effective_window,
    fetch_latest_usage,
)


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
    conn.execute(
        """CREATE TABLE agent_turns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            start_context_id INTEGER NOT NULL,
            status TEXT NOT NULL
        )"""
    )
    conn.execute(
        """CREATE TABLE api_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            input_tokens INTEGER NOT NULL,
            cached_tokens INTEGER NOT NULL,
            cache_write_tokens INTEGER NOT NULL,
            output_tokens INTEGER NOT NULL,
            reasoning_tokens INTEGER NOT NULL,
            input_context_id INTEGER NOT NULL
        )"""
    )
    conn.executemany(
        "INSERT INTO context (item) VALUES (?)",
        [(json.dumps({"role": "user", "content": f"event {i}"}),) for i in range(rows)],
    )
    return conn


def append_event(conn: sqlite3.Connection, text: str) -> None:
    """Append one event to a test context database."""
    conn.execute(
        "INSERT INTO context (item) VALUES (?)",
        (json.dumps({"role": "user", "content": text}),),
    )


def append_item(conn: sqlite3.Connection, item: dict) -> int:
    """Append arbitrary context and return its row id."""
    cursor = conn.execute(
        "INSERT INTO context (item) VALUES (?)",
        (json.dumps(item),),
    )
    return cursor.lastrowid or 0


def test_empty_final_output_hidden_by_default() -> None:
    """Empty API final-answer envelopes stay out of the useful viewer."""
    item = {
        "type": "message",
        "role": "assistant",
        "phase": "final_answer",
        "content": [{"type": "output_text", "text": ""}],
    }
    row = (1, "2026-08-02 10:00:00", json.dumps(item))

    assert build_block(row, full=False) is None
    assert build_block(row, full=False, show_empty_final=True) is not None


def test_nonempty_message_uses_final_output_label() -> None:
    """Viewer names assistant output without implying hidden reasoning."""
    item = {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "private note"}],
    }
    row = (1, "2026-08-02 10:00:00", json.dumps(item))

    block = build_block(row, full=False)

    assert block is not None
    assert "FINAL OUTPUT" in block.plain


def test_effective_window_keeps_only_active_current_turn_reasoning() -> None:
    """Current-turn estimate drops settled reasoning, not active reasoning."""
    conn = make_context_db(0)
    append_event(conn, "old event")
    append_item(conn, {"type": "reasoning", "encrypted_content": "old"})
    append_item(conn, {"type": "message", "role": "assistant", "content": []})
    append_event(conn, "active event")
    active_start = 4
    conn.execute(
        "INSERT INTO agent_turns (start_context_id, status) VALUES (?, 'running')",
        (active_start,),
    )
    append_item(conn, {"type": "reasoning", "encrypted_content": "current"})

    rows = fetch_effective_window(conn, 20, "current_turn", False, active_start)
    items = [json.loads(row[2]) for row in rows]

    assert [item.get("encrypted_content") for item in items] == [
        None,
        None,
        None,
        "current",
    ]
    assert fetch_active_turn_start(conn) == active_start
    conn.close()


def test_pruned_window_drops_completed_output_envelopes() -> None:
    """Pruned estimate retains active ephemeral output until turn closes."""
    conn = make_context_db(0)
    append_event(conn, "event")
    append_item(conn, {"type": "reasoning", "encrypted_content": "current"})
    append_item(conn, {"type": "message", "role": "assistant", "content": []})

    active = fetch_effective_window(conn, 20, "current_turn", True, 1)
    settled = fetch_effective_window(conn, 20, "current_turn", True, None)

    assert len(active) == 3
    assert len(settled) == 1
    conn.close()


def test_latest_usage_and_status_use_authoritative_counts() -> None:
    """Status distinguishes exact API counters from next-context estimate."""
    conn = make_context_db(0)
    conn.execute(
        """INSERT INTO api_usage (
            input_tokens, cached_tokens, cache_write_tokens,
            output_tokens, reasoning_tokens, input_context_id
        ) VALUES (18400, 12800, 1000, 1200, 900, 42)"""
    )

    usage = fetch_latest_usage(conn)
    status = build_status(50, None, window_tokens=19100, usage=usage)

    assert usage == UsageSnapshot(18400, 12800, 1000, 1200, 900, 42)
    assert "last in 18.4k" in status.plain
    assert "12.8k cached" in status.plain
    assert "next ctx ~19.1k" in status.plain
    assert status.plain.count("\n") == 1
    conn.close()


def test_status_explains_missing_api_usage() -> None:
    """Viewer distinguishes missing usage data from zero-token usage."""
    status = build_status(50, None, window_tokens=19100, usage=None)

    first_line, second_line = status.plain.splitlines()
    assert "last API usage unavailable" in first_line
    assert "next ctx ~19.1k" in second_line


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
