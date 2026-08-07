"""Tests for the context spy."""

import json
from pathlib import Path

import pytest
from sqlalchemy import Connection, create_engine, insert, select, text
from sqlalchemy.pool import StaticPool
from textual.widgets import Static

from libertati import schema
from libertati.render import TRUNCATE_AT
from libertati.spy import (
    STEERING_MAX_CHARS,
    Match,
    SpyApp,
    SteerInput,
    Usage,
    build_block,
    build_status,
    estimate_tokens,
    fetch_after,
    fetch_before,
    fetch_usage,
    latest_recorded_dream,
    predict_context,
    tail_anchor,
)


def make_context_db(rows: int = 30, path: Path | None = None) -> Connection:
    """Create a context database with enough rows to scroll.

    In memory unless ``path`` is given; a file is what the tests that
    post instructions need, since those open a connection of their own.
    The real schema, straight from the metadata — a hand-written subset
    would drift. Autocommit, like the viewer's own connection: each
    write lands at once and each poll reads a fresh snapshot.
    """
    engine = create_engine(
        f"sqlite:///{path}" if path else "sqlite://",
        poolclass=StaticPool,
        isolation_level="AUTOCOMMIT",
    )
    schema.metadata.create_all(engine)
    conn = engine.connect()
    for i in range(rows):
        append_event(conn, f"event {i}")
    return conn


def append_event(conn: Connection, body: str) -> None:
    """Append one event to a test context database."""
    conn.execute(
        insert(schema.context).values(
            item=json.dumps({"role": "user", "content": body})
        )
    )


def start_dream(conn: Connection, status: str = "running") -> int:
    """Open a dream ledger row and return its id."""
    result = conn.execute(
        insert(schema.dreams)
        .values(trigger="idle", status=status)
        .returning(schema.dreams.c.id)
    )
    return int(result.scalar_one())


def append_dream_event(conn: Connection, dream_id: int, body: str) -> None:
    """Append one event to a dream's recorded context."""
    conn.execute(
        insert(schema.dream_context).values(
            dream_id=dream_id, item=json.dumps({"role": "user", "content": body})
        )
    )


def append_usage(
    conn: Connection,
    tokens: int,
    context_id: int,
    dream_id: int | None = None,
) -> None:
    """Record one authoritative API usage snapshot."""
    conn.execute(
        insert(schema.api_usage).values(
            dream_id=dream_id,
            model="m",
            input_tokens=tokens,
            cached_tokens=tokens // 2,
            cache_write_tokens=100,
            output_tokens=1200,
            reasoning_tokens=900,
            total_tokens=tokens + 1200,
            input_context_id=context_id,
        )
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


def test_internal_input_message_renders_canonical_content() -> None:
    """Typed user dialogue displays separately from external events."""
    item = {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "delivery correction"}],
    }

    block = build_block((8, "2026-08-02 10:00:00", json.dumps(item)))

    assert block is not None
    assert "INTERNAL INPUT" in block.plain
    assert "delivery correction" in block.plain


def test_internal_input_message_renders_legacy_string_content() -> None:
    """Spy remains compatible with string-form rows already persisted."""
    item = {
        "type": "message",
        "role": "user",
        "content": "[delivery correction] legacy",
    }

    block = build_block((9, "2026-08-02 10:00:00", json.dumps(item)))

    assert block is not None
    assert "INTERNAL INPUT" in block.plain
    # The tag becomes the first line; the text it prefixed follows it.
    assert block.plain.endswith("delivery correction\nlegacy")


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
    pending = conn.execute(
        select(schema.context.c.item).where(schema.context.c.id > 1)
    ).fetchall()
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


def test_status_counts_the_search_position() -> None:
    """An active search says which of how many hits is under the cursor."""
    counted = build_status(50, None, matches=(2, 7))
    pending = build_status(50, None, matches=(0, 7))

    assert "match 2/7" in counted.plain
    assert "7 matches" in pending.plain


def test_status_explains_missing_api_usage() -> None:
    """The viewer distinguishes missing usage data from zero-token usage."""
    status = build_status(50, None, next_tokens=19100)

    assert "no API usage yet" in status.plain
    assert "next ~19.1k" in status.plain


def test_status_names_the_dream_being_viewed() -> None:
    """Dreaming mode is unmistakable in the status chip."""
    waking = build_status(50, None)
    dreaming = build_status(50, None, dream_id=3)

    assert waking.plain.startswith("spy")
    assert dreaming.plain.startswith("DREAM #3")


def test_fetching_is_scoped_to_the_viewed_context() -> None:
    """Waking and dreaming rows never appear in each other's view."""
    conn = make_context_db(0)
    append_event(conn, "awake")
    first = start_dream(conn)
    second = start_dream(conn)
    append_dream_event(conn, first, "wandering")
    append_dream_event(conn, second, "wandering elsewhere")

    assert [row[2] for row in fetch_after(conn, 0)] == [
        json.dumps({"role": "user", "content": "awake"})
    ]
    assert len(fetch_after(conn, 0, first)) == 1
    assert json.loads(fetch_after(conn, 0, first)[0][2])["content"] == "wandering"
    assert fetch_before(conn, 99, 10, second)[0][0] == 2
    conn.close()


def test_waking_usage_ignores_dreaming_rows() -> None:
    """A dream's cost never lands in the waking status line."""
    conn = make_context_db(0)
    append_usage(conn, 18400, context_id=1)
    append_usage(conn, 90000, context_id=4, dream_id=1)

    assert fetch_usage(conn) == Usage(18400, 9200, 100, 1200, 900, 1)
    assert fetch_usage(conn, 1) == Usage(90000, 45000, 100, 1200, 900, 4)
    assert fetch_usage(conn, 2) is None
    conn.close()


def test_usage_ignores_calls_that_never_read_the_context() -> None:
    """A memory extraction booked after a round does not hide the round.

    `recall` bills its own call to the same turn with no context id, and
    it lands last — so the newest row is not the one that measured the
    window the viewer is showing.
    """
    conn = make_context_db(0)
    append_usage(conn, 18400, context_id=1)
    append_usage(conn, 900, context_id=0)

    usage = fetch_usage(conn)

    assert usage == Usage(18400, 9200, 100, 1200, 900, 1)
    conn.close()


def test_latest_recorded_dream_ignores_dreams_with_no_context() -> None:
    """A dream that never got to think has nothing to show."""
    conn = make_context_db(0)
    start_dream(conn, "woke")

    assert latest_recorded_dream(conn) is None

    second = start_dream(conn, "woke")
    append_dream_event(conn, second, "wandering")

    assert latest_recorded_dream(conn) == second
    conn.close()


def test_tail_anchor_measures_the_context_being_viewed() -> None:
    """A dream anchors on its own rows, not on the waking history's ids."""
    conn = make_context_db(0)
    for index in range(200):
        append_event(conn, f"awake {index}")
    dream = start_dream(conn)
    append_dream_event(conn, dream, "wandering")

    assert tail_anchor(conn, 50) == 150
    # The dream has one row; a waking anchor would hide it entirely.
    assert tail_anchor(conn, 50, dream) == 0
    conn.close()


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
        assert app.matches == []

        await pilot.press("slash")
        await pilot.press(*"needle")
        await pilot.press("enter")
        await pilot.pause()

        assert view.pattern is not None
        assert view.cursor == Match(4, 0)
        # The only match is behind the tail, so the search wraps to it.
        assert app.note == "search hit BOTTOM, continuing at TOP"
        assert app.match_at == 0
        assert app.matches == [Match(4, 0)]
        assert view.scroll_y <= view.blocks[3].start

        await pilot.press("escape")
        await pilot.pause()
        assert view.pattern is None
        assert view.cursor is None
        assert app.matches == []

    conn.close()


@pytest.mark.asyncio
async def test_search_reaches_matches_older_than_the_loaded_tail() -> None:
    """A hit thousands of rows back is indexed from the database and shown."""
    conn = make_context_db(0)
    append_event(conn, "the needle, right at the start")
    for index in range(300):
        append_event(conn, f"filler {index}")
    app = SpyApp(conn, last_id=tail_anchor(conn, 20), page_size=20)

    async with app.run_test(size=(80, 10)) as pilot:
        await pilot.pause()
        view = app.view
        assert view.oldest_id is not None and view.oldest_id > 1

        await pilot.press("slash")
        await pilot.press(*"needle")
        await pilot.press("enter")
        await pilot.pause()

        assert app.matches == [Match(1, 0)]
        # History paged in until the match was loadable, then landed on it.
        assert view.oldest_id == 1
        assert view.cursor == Match(1, 0)
        assert "needle" in view.blocks[0].plain

    conn.close()


@pytest.mark.asyncio
async def test_repeat_search_steps_through_every_occurrence() -> None:
    """`n` and `N` walk occurrence by occurrence, not block by block."""
    conn = make_context_db(0)
    append_event(conn, "needle and needle again")
    append_event(conn, "one more needle")
    app = SpyApp(conn, last_id=0)

    async with app.run_test(size=(80, 10)) as pilot:
        await pilot.pause()
        await pilot.press("slash")
        await pilot.press(*"needle")
        await pilot.press("enter")
        await pilot.pause()

        assert app.matches == [Match(1, 0), Match(1, 1), Match(2, 0)]
        assert app.view.cursor == Match(1, 0)

        await pilot.press("n")
        await pilot.pause()
        assert app.view.cursor == Match(1, 1)
        assert app.match_at == 1

        await pilot.press("N")
        await pilot.pause()
        assert app.view.cursor == Match(1, 0)

        await pilot.press("N")
        await pilot.pause()
        assert app.view.cursor == Match(2, 0)
        assert app.note == "search hit TOP, continuing at BOTTOM"

    conn.close()


@pytest.mark.asyncio
async def test_rows_arriving_during_a_search_are_indexed() -> None:
    """A match written while the pattern is active is reachable at once."""
    conn = make_context_db(0)
    append_event(conn, "needle one")
    app = SpyApp(conn, last_id=0)

    async with app.run_test(size=(80, 10)) as pilot:
        await pilot.pause()
        await pilot.press("slash")
        await pilot.press(*"needle")
        await pilot.press("enter")
        await pilot.pause()
        assert app.matches == [Match(1, 0)]

        append_event(conn, "needle two")
        app.poll()
        await pilot.pause()

        assert app.matches == [Match(1, 0), Match(2, 0)]

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


@pytest.mark.asyncio
async def test_dream_key_switches_the_viewed_context_both_ways() -> None:
    """`d` swaps the loaded blocks for a dream's, and back again."""
    conn = make_context_db(0)
    append_event(conn, "awake")
    dream = start_dream(conn, "woke")
    append_dream_event(conn, dream, "dreaming about Alice")
    app = SpyApp(conn, last_id=0)

    async with app.run_test(size=(80, 10)) as pilot:
        await pilot.pause()
        assert "awake" in app.view.blocks[0].plain

        await pilot.press("d")
        app.poll()
        await pilot.pause()
        assert app.dream_id == dream
        assert not app.auto
        assert [block.plain for block in app.view.blocks] == [
            block.plain for block in app.view.blocks if "Alice" in block.plain
        ]
        await pilot.press("d")
        app.poll()
        await pilot.pause()
        assert app.dream_id is None
        assert "awake" in app.view.blocks[0].plain

    conn.close()


@pytest.mark.asyncio
async def test_dream_key_says_so_when_there_is_nothing_to_show() -> None:
    """Pressing `d` on a database with no dreams explains itself."""
    conn = make_context_db(0)
    append_event(conn, "awake")
    app = SpyApp(conn, last_id=0)

    async with app.run_test(size=(80, 10)) as pilot:
        await pilot.pause()
        await pilot.press("d")
        await pilot.pause()

        assert app.dream_id is None
        assert app.hint == "no dream recorded yet"

    conn.close()


@pytest.mark.asyncio
async def test_a_running_dream_is_followed_and_let_go_on_waking() -> None:
    """While following, the viewer rides along with a dream by itself."""
    conn = make_context_db(0)
    append_event(conn, "awake")
    app = SpyApp(conn, last_id=0)

    async with app.run_test(size=(80, 10)) as pilot:
        await pilot.pause()
        assert app.dream_id is None

        dream = start_dream(conn)
        append_dream_event(conn, dream, "wandering")
        app.poll()
        await pilot.pause()
        assert app.dream_id == dream
        assert "wandering" in app.view.blocks[0].plain

        conn.execute(
            text("UPDATE dreams SET status = 'woke' WHERE id = :id"), {"id": dream}
        )
        append_event(conn, "[dream #1 ended] say hi to Bob")
        app.poll()
        await pilot.pause()
        assert app.dream_id is None
        # Back on the waking tail, with what happened while asleep.
        assert "awake" in app.view.blocks[0].plain
        assert "say hi to Bob" in app.view.blocks[-1].plain

    conn.close()


@pytest.mark.asyncio
async def test_a_scrolled_viewer_is_not_dragged_into_a_dream() -> None:
    """Reading history beats following: the view stays put, and says why."""
    conn = make_context_db(30)
    app = SpyApp(conn, last_id=0)

    async with app.run_test(size=(80, 10)) as pilot:
        await pilot.pause()
        await pilot.press("ctrl+b")
        await pilot.pause()
        scrolled_y = app.view.scroll_y

        dream = start_dream(conn)
        append_dream_event(conn, dream, "wandering")
        app.poll()
        await pilot.pause()

        assert app.dream_id is None
        assert app.view.scroll_y == scrolled_y
        assert app.hint == "dream #1 running (d)"

    conn.close()


@pytest.mark.asyncio
async def test_opening_a_dream_directly_pins_the_view() -> None:
    """`--dream` starts in a dream's context and stays there."""
    conn = make_context_db(0)
    append_event(conn, "awake")
    dream = start_dream(conn, "woke")
    append_dream_event(conn, dream, "dreaming about Alice")
    running = start_dream(conn)
    append_dream_event(conn, running, "wandering right now")
    app = SpyApp(conn, last_id=0, dream_id=dream)

    async with app.run_test(size=(80, 10)) as pilot:
        await pilot.pause()
        assert app.dream_id == dream
        assert not app.auto
        assert "Alice" in app.view.blocks[0].plain

        app.poll()
        await pilot.pause()
        # A dream running right now must not steal a pinned view.
        assert app.dream_id == dream

    conn.close()


@pytest.mark.asyncio
async def test_instruction_prompt_posts_to_the_database(tmp_path: Path) -> None:
    """`i` writes what was typed where the bot process picks it up."""
    path = tmp_path / "context.db"
    conn = make_context_db(0, path)
    app = SpyApp(conn, last_id=0, db_path=path)

    async with app.run_test(size=(80, 10)) as pilot:
        await pilot.pause()
        await pilot.press("i")
        await pilot.press(*"stop replying to bob")
        await pilot.press("enter")
        await pilot.pause()

        assert conn.execute(
            text("SELECT text, urgent, done FROM steering")
        ).fetchall() == [("stop replying to bob", 0, 0)]
        assert app.note == "instruction #1 queued"

    conn.close()


@pytest.mark.asyncio
async def test_instruction_urgency_starts_from_the_key_and_toggles(
    tmp_path: Path,
) -> None:
    """`I` interrupts the running turn; ctrl+t changes its mind."""
    path = tmp_path / "context.db"
    conn = make_context_db(0, path)
    app = SpyApp(conn, last_id=0, db_path=path)

    async with app.run_test(size=(80, 10)) as pilot:
        await pilot.pause()
        await pilot.press("I")
        assert app.steer_urgent is True
        await pilot.press("ctrl+t")
        assert app.steer_urgent is False
        await pilot.press("ctrl+t")
        await pilot.press(*"drop it")
        await pilot.press("enter")
        await pilot.pause()

        assert conn.execute(text("SELECT text, urgent FROM steering")).fetchall() == [
            ("drop it", 1)
        ]
        assert app.note == "instruction #1 interrupting"

    conn.close()


@pytest.mark.asyncio
async def test_instruction_during_a_dream_promises_no_interruption(
    tmp_path: Path,
) -> None:
    """A dream has no round boundary to cut into, and the note says so."""
    path = tmp_path / "context.db"
    conn = make_context_db(0, path)
    dream = start_dream(conn)
    app = SpyApp(conn, last_id=0, db_path=path)

    async with app.run_test(size=(80, 10)) as pilot:
        await pilot.pause()
        await pilot.press("I")
        await pilot.press(*"drop it")
        await pilot.press("enter")
        await pilot.pause()

        assert conn.execute(text("SELECT urgent FROM steering")).fetchall() == [(1,)]
        assert app.note == f"instruction #1 queued (dream #{dream} first)"

    conn.close()


@pytest.mark.asyncio
async def test_instruction_prompt_caps_what_can_be_typed(tmp_path: Path) -> None:
    """Keystrokes past the cap are refused rather than elided later."""
    path = tmp_path / "context.db"
    conn = make_context_db(0, path)
    app = SpyApp(conn, last_id=0, db_path=path)

    async with app.run_test(size=(80, 10)) as pilot:
        await pilot.pause()
        await pilot.press("i")
        assert app.query_one("#steer", SteerInput).max_length == STEERING_MAX_CHARS

    conn.close()


@pytest.mark.asyncio
async def test_empty_instruction_writes_nothing(tmp_path: Path) -> None:
    """Opening the prompt and thinking better of it costs nothing."""
    path = tmp_path / "context.db"
    conn = make_context_db(0, path)
    app = SpyApp(conn, last_id=0, db_path=path)

    async with app.run_test(size=(80, 10)) as pilot:
        await pilot.pause()
        await pilot.press("i")
        await pilot.press("space", "enter")
        await pilot.press("i")
        await pilot.press("escape")
        await pilot.pause()

        assert conn.execute(text("SELECT COUNT(*) FROM steering")).fetchone() == (0,)
        assert app.note == ""

    conn.close()


@pytest.mark.asyncio
async def test_instruction_without_a_database_path_says_so() -> None:
    """A viewer opened on a connection alone cannot post; it admits it."""
    conn = make_context_db(0)
    app = SpyApp(conn, last_id=0)

    async with app.run_test(size=(80, 10)) as pilot:
        await pilot.pause()
        await pilot.press("i")
        await pilot.press(*"hello")
        await pilot.press("enter")
        await pilot.pause()

        assert app.note == "no database path: instructions unavailable"

    conn.close()


@pytest.mark.asyncio
async def test_instruction_reports_a_database_that_refuses_it(
    tmp_path: Path,
) -> None:
    """A database without the table (an old bot) fails visibly, not silently."""
    path = tmp_path / "context.db"
    conn = make_context_db(0, path)
    conn.execute(text("DROP TABLE steering"))
    app = SpyApp(conn, last_id=0, db_path=path)

    async with app.run_test(size=(80, 10)) as pilot:
        await pilot.pause()
        await pilot.press("i")
        await pilot.press(*"hello")
        await pilot.press("enter")
        await pilot.pause()

        assert app.note.startswith("instruction failed:")
        assert app.is_running

    conn.close()


@pytest.mark.asyncio
async def test_prompts_paint_over_the_status_not_under_it() -> None:
    """An open prompt owns the bottom row; the status never covers it.

    The bars live on a layer above the status because both dock to the
    bottom edge; before they did, the status was painted last and the
    prompts were typed into blind.
    """
    conn = make_context_db(5)
    app = SpyApp(conn, last_id=0)

    async with app.run_test(size=(80, 12)) as pilot:
        await pilot.pause()
        widget, _ = app.screen.get_widget_at(5, 11)
        assert widget.id == "status"

        await pilot.press("slash")
        await pilot.pause()
        widget, _ = app.screen.get_widget_at(5, 11)
        assert widget.id == "search"

        await pilot.press("escape")
        await pilot.press("i")
        await pilot.pause()
        widget, _ = app.screen.get_widget_at(20, 11)
        assert widget.id == "steer"
        widget, _ = app.screen.get_widget_at(1, 11)
        assert widget.id == "steer-prefix"

    conn.close()


@pytest.mark.asyncio
async def test_steer_prefix_names_the_mode_while_typing() -> None:
    """ctrl+t mid-typing changes the visible mode, not just the flag."""
    conn = make_context_db(1)
    app = SpyApp(conn, last_id=0)

    async with app.run_test(size=(80, 12)) as pilot:
        await pilot.pause()
        await pilot.press("I")
        await pilot.press(*"wait")
        prefix = app.query_one("#steer-prefix", Static)
        assert "instruct now!" in str(prefix.render())

        await pilot.press("ctrl+t")
        assert "instruct:" in str(prefix.render())

    conn.close()


@pytest.mark.asyncio
async def test_wrap_narrows_the_laid_out_text() -> None:
    """Block text wraps at the configured width, not the terminal's."""
    conn = make_context_db(0)
    append_event(conn, "x" * 100)
    wide = SpyApp(conn, last_id=0)
    async with wide.run_test(size=(80, 12)) as pilot:
        await pilot.pause()
        wide_height = wide.view.blocks[0].height

    narrow = SpyApp(conn, last_id=0, wrap=20)
    async with narrow.run_test(size=(80, 12)) as pilot:
        await pilot.pause()
        # 100 characters at 20 columns are five lines; at the default
        # width they fit in two.
        assert narrow.view.blocks[0].height >= wide_height + 3

    conn.close()
