"""Tests for the context spy."""

import json
import sqlite3

import pytest

from libertati.spy import (
    TRUNCATE_AT,
    SpyApp,
    Usage,
    build_block,
    build_status,
    estimate_tokens,
    fetch_usage,
    predict_context,
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


def append_usage(conn: sqlite3.Connection, tokens: int, context_id: int) -> None:
    """Record one authoritative API usage snapshot."""
    conn.execute(
        """INSERT INTO api_usage (
            input_tokens, cached_tokens, cache_write_tokens,
            output_tokens, reasoning_tokens, input_context_id
        ) VALUES (?, ?, 100, 1200, 900, ?)""",
        (tokens, tokens // 2, context_id),
    )


def test_empty_final_output_is_hidden() -> None:
    """Empty API final-answer envelopes stay out of the viewer."""
    item = {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": ""}],
    }

    assert build_block((1, "2026-08-02 10:00:00", json.dumps(item))) is None


def test_block_header_carries_kind_and_estimate() -> None:
    """Each block names its kind and its estimated token cost."""
    item = {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "private note"}],
    }
    row = (7, "2026-08-02 10:00:00", json.dumps(item))

    block = build_block(row)

    assert block is not None
    assert "#7" in block.plain
    assert "FINAL OUTPUT" in block.plain
    assert "tok" in block.plain


def test_estimate_scales_with_script_and_shape() -> None:
    """Cyrillic costs more per byte than JSON; base64 reasoning far less."""
    ascii_event = json.dumps({"role": "user", "content": "hello there friend"})
    cyrillic = json.dumps(
        {"role": "user", "content": "привіт, як справи"}, ensure_ascii=False
    )
    reasoning = json.dumps({"type": "reasoning", "encrypted_content": "A" * 400})

    per_byte = [
        estimate_tokens(ascii_event, "event") / len(ascii_event.encode()),
        estimate_tokens(cyrillic, "event") / len(cyrillic.encode()),
        estimate_tokens(reasoning, "reasoning") / len(reasoning.encode()),
    ]

    assert per_byte[1] < per_byte[0] < per_byte[2]


def test_prediction_anchors_on_last_authoritative_usage() -> None:
    """The next context is the last measured input plus what came after."""
    conn = make_context_db(0)
    append_event(conn, "already sent")
    append_usage(conn, 18400, context_id=1)
    append_event(conn, "queued since the last call")

    usage = fetch_usage(conn)
    pending = conn.execute("SELECT item FROM context WHERE id > 1").fetchall()
    predicted = predict_context(conn, usage, 300)

    assert usage == Usage(18400, 9200, 100, 1200, 900, 1)
    assert predicted == 18400 + estimate_tokens(pending[0][0], "")
    conn.close()


def test_prediction_falls_back_to_the_window_without_usage() -> None:
    """A database with no API usage yet still gets an estimate."""
    conn = make_context_db(5)

    predicted = predict_context(conn, None, 300)

    assert predicted is not None and predicted > 0
    conn.close()


def test_status_shows_usage_and_prediction_together() -> None:
    """Token figures share one line; position and follow state the other."""
    usage = Usage(18400, 12800, 1000, 1200, 900, 42)

    status = build_status(50, None, True, usage, 19100)
    position, tokens = status.plain.splitlines()

    assert "#50" in position and "FOLLOW" in position
    assert "in 18.4k" in tokens
    assert "cached 12.8k 70%" in tokens
    assert "next ~19.1k" in tokens


def test_status_explains_missing_api_usage() -> None:
    """The viewer distinguishes missing usage data from zero-token usage."""
    status = build_status(50, None, next_tokens=19100)

    assert "no API usage yet" in status.plain
    assert "next ~19.1k" in status.plain


@pytest.mark.asyncio
async def test_vim_keys_scroll_to_both_ends() -> None:
    """Vim motions reach both ends of the buffer."""
    conn = make_context_db()
    app = SpyApp(conn, last_id=0)

    async with app.run_test(size=(80, 10)) as pilot:
        await pilot.pause()
        view = app.view
        assert view.is_vertical_scroll_end

        await pilot.press("ctrl+u")
        await pilot.pause()
        assert not view.is_vertical_scroll_end

        await pilot.press("ctrl+d")
        await pilot.pause()
        assert view.is_vertical_scroll_end

        await pilot.press("k")
        await pilot.pause()
        assert not view.is_vertical_scroll_end

        await pilot.press("j")
        await pilot.pause()
        assert view.is_vertical_scroll_end

        await pilot.press("g")
        await pilot.pause()
        assert view.scroll_y == 0

        await pilot.press("G")
        await pilot.pause()
        assert view.is_vertical_scroll_end

    conn.close()


@pytest.mark.asyncio
async def test_new_events_preserve_scrolled_viewport() -> None:
    """Polling follows at the bottom but preserves manual scroll position."""
    conn = make_context_db()
    app = SpyApp(conn, last_id=0)

    async with app.run_test(size=(80, 10)) as pilot:
        await pilot.pause()
        view = app.view

        await pilot.press("ctrl+b")
        await pilot.pause()
        scrolled_y = view.scroll_y
        assert not view.is_vertical_scroll_end

        append_event(conn, "new while reading history")
        app.poll()
        await pilot.pause()
        assert app.last_id == 31
        assert view.scroll_y == scrolled_y

        await pilot.press("G")
        await pilot.pause()
        append_event(conn, "new while following")
        app.poll()
        await pilot.pause()
        assert app.last_id == 32
        assert view.is_vertical_scroll_end

    conn.close()


@pytest.mark.asyncio
async def test_reaching_the_top_pages_history_in() -> None:
    """Scrolling to the loaded top pulls older rows without losing place."""
    conn = make_context_db()
    app = SpyApp(conn, last_id=25, page_size=5)

    async with app.run_test(size=(80, 10)) as pilot:
        await pilot.pause()
        view = app.view
        # The tail alone does not fill the screen, so one page is already in.
        assert view.blocks[0].row[0] == 21
        assert view.blocks[-1].row[0] == 30

        for _ in range(10):
            await pilot.press("ctrl+b")
            await pilot.pause()

        assert view.blocks[0].row[0] == 1
        assert not view.has_older
        assert view.blocks[0].start == 0
        assert [block.row[0] for block in view.blocks] == list(range(1, 31))
    conn.close()


@pytest.mark.asyncio
async def test_search_jumps_and_reports_misses() -> None:
    """`/` finds a match; a miss says so instead of moving the viewport."""
    conn = make_context_db(0)
    for index in range(40):
        append_event(conn, f"needle {index}" if index == 3 else f"filler {index}")
    app = SpyApp(conn, last_id=0)

    async with app.run_test(size=(80, 10)) as pilot:
        await pilot.pause()
        view = app.view

        await pilot.press("slash")
        await pilot.press(*"Needle")
        await pilot.press("enter")
        await pilot.pause()
        # Smartcase: an upper-case pattern is matched case-sensitively.
        assert app.note == "pattern not found"

        await pilot.press("slash")
        await pilot.press(*"needle")
        await pilot.press("enter")
        await pilot.pause()

        assert view.pattern is not None
        assert app.note == ""
        assert view.scroll_y == view.blocks[3].start

        await pilot.press("escape")
        await pilot.pause()
        assert view.pattern is None

    conn.close()


@pytest.mark.asyncio
async def test_search_prompt_takes_keys_that_are_bound_elsewhere() -> None:
    """Typing `q` or `f` into the prompt searches, and does not quit."""
    conn = make_context_db(0)
    append_event(conn, "quirk")
    app = SpyApp(conn, last_id=0)

    async with app.run_test(size=(80, 10)) as pilot:
        await pilot.pause()
        await pilot.press("slash")
        await pilot.press(*"qui")
        await pilot.pause()
        assert app.is_running
        assert not app.view.full

        await pilot.press("enter")
        await pilot.pause()
        assert app.view.pattern is not None
        assert app.note == ""

    conn.close()


@pytest.mark.asyncio
async def test_full_toggle_untruncates_bodies() -> None:
    """`f` swaps the truncated body for the whole thing."""
    conn = make_context_db(0)
    append_event(conn, "x" * (TRUNCATE_AT + 200))
    app = SpyApp(conn, last_id=0)

    async with app.run_test(size=(80, 10)) as pilot:
        await pilot.pause()
        assert "200 chars]" in app.view.blocks[0].plain

        await pilot.press("f")
        await pilot.pause()
        assert app.view.full
        assert "chars]" not in app.view.blocks[0].plain

    conn.close()
