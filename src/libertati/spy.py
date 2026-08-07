"""Dev TUI: live spy on the agent's context (``libertati-spy``).

Tails the ``context`` table in a full-screen viewer: external events,
hidden reasoning, tool calls/results and private final output, each laid
out by kind (:mod:`libertati.render`). Navigation is vim-like —
``j``/``k``, ``ctrl+e``/``ctrl+y``, ``ctrl+d``/``ctrl+u``, ``ctrl+f``/
``ctrl+b``, ``g``/``G`` — with ``/``, ``?``, ``n``, ``N`` search and
``f`` to un-truncate bodies.

Search runs over the whole stored history, not the part that happens to
be on screen: a pattern is indexed occurrence by occurrence out of the
database, ``n``/``N`` step through those occurrences in order (paging
history in as they go), the one under the cursor is marked apart from
the rest, and the status line counts them.

``i`` and ``I`` post an instruction to the agent: not a chat message but
steering, delivered as an event that says it came from this console.
``i`` waits for the turn in flight to end, ``I`` interrupts it between
rounds — except during a dream, which has no round boundary to cut into.
``ctrl+t`` switches between the two while typing. The instruction is
written to the ``steering`` table — the only thing this viewer writes —
and a loop in the bot process picks it up from there.

``d`` switches to a dream's context (``dream_context``) and back; while
nothing is pinned and the view is following, a starting dream is picked
up on its own and dropped again on waking. ``--dream ID`` opens a past
dream directly.

Per-item token figures are estimates from UTF-8 byte length, scaled per
content shape (dense JSON framing, multi-byte prose, base64 reasoning
blobs). The projected next context is anchored on the newest API usage
row that measured the context window plus the estimate of everything
appended since it, so the fixed instructions/tool overhead comes from
real numbers instead of a guess. Requires the ``textual`` and ``rich`` dev
dependencies; run via ``uv run libertati-spy``.
"""

import argparse
import asyncio
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, cast

from pydantic import ValidationError
from rich.segment import Segment
from rich.text import Text
from sqlalchemy import Connection, Table, create_engine, func, insert, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import NullPool
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.geometry import Size
from textual.scroll_view import ScrollView
from textual.strip import Strip
from textual.widgets import Input, Static

from libertati.config import Settings
from libertati.render import (
    Row,
    build_block,
    call_name,
    decode,
    estimate_tokens,
    searchable,
)
from libertati.schema import api_usage, context, dream_context, dreams, steering

#: Seconds between polls for newly appended context rows.
POLL_SECONDS = 0.5

#: History pages one jump may page in before giving up. Generous on
#: purpose: the cap exists so a pathological database cannot hang the
#: viewer, not to bound how far back a search may reach.
SEARCH_PAGES = 500

#: Rows read at a time while indexing a pattern.
SCAN_CHUNK = 500

#: Seconds an instruction waits for the bot's writer before giving up.
#: The bot writes on every message it stores, so a busy database is
#: normal and worth waiting out rather than reporting as a failure.
STEERING_TIMEOUT = 5.0

#: Longest instruction the prompt accepts. Mirrors ``STEERING_TEXT_LIMIT``
#: on the bot side, which is what actually bounds the event; stopping the
#: keystrokes here just means a long paste is visibly refused instead of
#: silently elided later.
STEERING_MAX_CHARS = 1000


@dataclass(frozen=True)
class Usage:
    """Authoritative counters for the latest completed API request."""

    input_tokens: int
    cached_tokens: int
    cache_write_tokens: int
    output_tokens: int
    reasoning_tokens: int
    context_id: int


@dataclass(frozen=True)
class Match:
    """One occurrence of the active pattern: its row, and where in it."""

    row_id: int
    index: int


# ===== Timestamps =====


def parse_stamp(created_at: str) -> datetime | None:
    """Parse a UTC ``datetime('now')`` column, or ``None`` if malformed."""
    try:
        return datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
    except ValueError:
        return None


def local_clock(created_at: str) -> str:
    """Render a UTC ``datetime('now')`` timestamp as local wall time.

    Converted per row rather than through one offset captured at import:
    the TUI is left running for hours, and a DST change would otherwise
    shift every timestamp it shows by an hour.
    """
    stamp = parse_stamp(created_at)
    if stamp is None:
        return "??:??:??"
    return stamp.astimezone().strftime("%H:%M:%S")


def age_text(created_at: str) -> str:
    """Format how long ago a UTC ``datetime('now')`` timestamp was."""
    then = parse_stamp(created_at)
    if then is None:
        return "?"
    seconds = max(0, int((datetime.now(UTC) - then).total_seconds()))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    return f"{seconds // 3600}h {seconds % 3600 // 60}m"


def render_row(
    row: Row,
    full: bool = False,
    pattern: re.Pattern[str] | None = None,
    name: str | None = None,
    cursor: int | None = None,
) -> Text | None:
    """Render one stored row, stamped with its local arrival time."""
    return build_block(row, full, pattern, local_clock(row[1]), name, cursor)


# ===== Queries =====


def source(dream_id: int | None) -> tuple[Table, tuple[Any, ...]]:
    """Return the table and ``WHERE`` conditions for one view mode.

    The conditions come as a tuple so every caller can add its own to
    the same ``where()``.
    """
    if dream_id is None:
        return context, ()
    return dream_context, (dream_context.c.dream_id == dream_id,)


def fetch_after(
    conn: Connection,
    last_id: int,
    dream_id: int | None = None,
) -> list[Row]:
    """Return all rows of the viewed context with id greater than ``last_id``."""
    table, conditions = source(dream_id)
    stmt = (
        select(table.c.id, table.c.created_at, table.c.item)
        .where(*conditions, table.c.id > last_id)
        .order_by(table.c.id)
    )
    return [(row[0], row[1], row[2]) for row in conn.execute(stmt)]


def fetch_before(
    conn: Connection,
    first_id: int,
    limit: int,
    dream_id: int | None = None,
) -> list[Row]:
    """Return up to ``limit`` rows before ``first_id``, oldest first."""
    table, conditions = source(dream_id)
    stmt = (
        select(table.c.id, table.c.created_at, table.c.item)
        .where(*conditions, table.c.id < first_id)
        .order_by(table.c.id.desc())
        .limit(limit)
    )
    rows = [(row[0], row[1], row[2]) for row in conn.execute(stmt)]
    return list(reversed(rows))


def scan_matches(
    conn: Connection,
    pattern: re.Pattern[str],
    dream_id: int | None = None,
    after: int = 0,
) -> list[Match]:
    """Index every occurrence of a pattern in the stored context.

    The search space is the database, not the loaded window: the viewer
    holds a tail of history, and a match older than that tail has to be
    findable all the same. Occurrences are counted in the very body the
    block renders, so counts and highlights cannot disagree.
    """
    table, conditions = source(dream_id)
    stmt = (
        select(table.c.id, table.c.item)
        .where(*conditions, table.c.id > after)
        .order_by(table.c.id)
    )
    result = conn.execute(stmt)
    matches: list[Match] = []
    while chunk := result.fetchmany(SCAN_CHUNK):
        for row_id, raw in chunk:
            matches.extend(
                Match(row_id, index)
                for index, _ in enumerate(pattern.finditer(searchable(raw)))
            )
    return matches


def _usage_select():
    """Usage columns the status line needs, in :class:`Usage` field order."""
    return select(
        api_usage.c.input_tokens,
        api_usage.c.cached_tokens,
        api_usage.c.cache_write_tokens,
        api_usage.c.output_tokens,
        api_usage.c.reasoning_tokens,
        api_usage.c.input_context_id,
    )


def fetch_usage(
    conn: Connection,
    dream_id: int | None = None,
) -> Usage | None:
    """Return exact usage from the newest API response of one mode.

    Waking usage excludes dreaming rows explicitly, so a dream's cost
    never shows up under the waking context. Calls that answer from
    their own input rather than from the context window — a memory
    extraction, which books itself as ``input_context_id = 0`` — are
    skipped too: they say nothing about the window this viewer shows,
    and one of them lands after the round that made it.
    """
    scope = (
        api_usage.c.dream_id.is_(None)
        if dream_id is None
        else api_usage.c.dream_id == dream_id
    )
    stmt = (
        _usage_select()
        .where(scope, api_usage.c.input_context_id > 0)
        .order_by(api_usage.c.id.desc())
        .limit(1)
    )
    row = conn.execute(stmt).first()
    return Usage(*row) if row else None


def post_steering(db_path: Path, text: str, urgent: bool) -> int:
    """Queue one operator instruction for the bot; return its row id.

    The one thing this viewer writes, and it opens its own engine to do
    it: the guarantee that watching the agent cannot disturb it is worth
    keeping for the connection everything else goes through.
    """
    engine = create_engine(
        f"sqlite:///{db_path}",
        poolclass=NullPool,
        connect_args={"timeout": STEERING_TIMEOUT},
    )
    try:
        with engine.begin() as conn:
            stmt = (
                insert(steering)
                .values(text=text, urgent=int(urgent))
                .returning(steering.c.id)
            )
            return int(conn.execute(stmt).scalar_one())
    finally:
        engine.dispose()


def newest_id(conn: Connection, dream_id: int | None = None) -> int:
    """Return the newest row id of one view mode, or zero when it is empty."""
    table, conditions = source(dream_id)
    stmt = select(func.coalesce(func.max(table.c.id), 0)).where(*conditions)
    return int(conn.execute(stmt).scalar_one())


def tail_anchor(conn: Connection, tail: int, dream_id: int | None = None) -> int:
    """Row id a view opens just after, to start on its newest ``tail`` rows.

    Always measured against the table actually being viewed: dream ids
    run independently of the waking ones, so anchoring a dream on the
    waking history would skip past everything the dream recorded.
    """
    return max(0, newest_id(conn, dream_id) - tail)


def latest_dream(conn: Connection) -> tuple[int, str] | None:
    """Return the newest dream's id and status, if the ledger has one."""
    stmt = select(dreams.c.id, dreams.c.status).order_by(dreams.c.id.desc()).limit(1)
    row = conn.execute(stmt).first()
    return (row[0], row[1]) if row else None


def latest_recorded_dream(conn: Connection) -> int | None:
    """Return the newest dream that actually has context rows."""
    newest = conn.execute(select(func.max(dream_context.c.dream_id))).scalar_one()
    return newest if newest is not None else None


def _estimate_rows(rows: list[tuple[str, str]]) -> int:
    """Sum the estimate over ``(item, type)`` pairs."""
    return sum(estimate_tokens(item, kind) for item, kind in rows)


def predict_context(
    conn: Connection,
    usage: Usage | None,
    max_items: int,
    dream_id: int | None = None,
) -> int | None:
    """Estimate the input size of the agent's next API call.

    Anchored on the last authoritative count: everything the API already
    charged for is taken as measured, and only the items appended since
    that request are estimated. Falls back to estimating the whole live
    window when there is no usage row yet, or when so much has piled up
    since one that the window would have been trimmed anyway.
    """
    table, conditions = source(dream_id)
    columns = select(
        table.c.item, func.coalesce(func.json_extract(table.c.item, "$.type"), "")
    )
    if usage is not None:
        pending_stmt = (
            columns.where(*conditions, table.c.id > usage.context_id)
            .order_by(table.c.id)
            .limit(max_items)
        )
        pending = [(row[0], row[1]) for row in conn.execute(pending_stmt)]
        if len(pending) < max_items:
            return usage.input_tokens + _estimate_rows(pending)
    window_stmt = (
        columns.where(*conditions).order_by(table.c.id.desc()).limit(max_items)
    )
    window = [(row[0], row[1]) for row in conn.execute(window_stmt)]
    return _estimate_rows(window) or None


# ===== Status =====


def build_status(
    last_id: int,
    last_activity: str | None,
    following: bool = True,
    usage: Usage | None = None,
    next_tokens: int | None = None,
    note: str = "",
    dream_id: int | None = None,
    matches: tuple[int, int] | None = None,
) -> Text:
    """Build the two-line status: position on top, token figures below."""
    status = Text()
    if dream_id is None:
        status.append("spy", style="bold reverse")
    else:
        status.append(f"DREAM #{dream_id}", style="bold reverse magenta")
    status.append(f"  #{last_id}", style="bold")
    if last_activity is not None:
        status.append(f"  {age_text(last_activity)}", style="cyan")
    status.append(
        "  FOLLOW" if following else "  SCROLLED",
        style="green" if following else "yellow",
    )
    if matches is not None:
        at, total = matches
        status.append(
            f"  match {at}/{total}" if at else f"  {total} matches",
            style="bold blue",
        )
    if note:
        status.append(f"  {note}", style="bright_black")
    status.append("\n")
    if usage is not None:
        share = usage.cached_tokens / usage.input_tokens if usage.input_tokens else 0.0
        status.append(f"in {usage.input_tokens / 1000:.1f}k", style="green")
        status.append(
            f"  cached {usage.cached_tokens / 1000:.1f}k {share:.0%}",
            style="cyan",
        )
        status.append(f"  write {usage.cache_write_tokens / 1000:.1f}k", style="yellow")
    else:
        status.append("no API usage yet", style="yellow")
    if next_tokens is not None:
        status.append(f"  next ~{next_tokens / 1000:.1f}k", style="bold magenta")
    if usage is not None:
        status.append(
            f"  out {usage.output_tokens / 1000:.1f}k"
            f" ({usage.reasoning_tokens / 1000:.1f}k reason)",
            style="bright_black",
        )
    return status


# ===== View =====


@dataclass
class Block:
    """One rendered context row: its lines and where they live."""

    row: Row
    plain: str
    start: int
    lines: list[Strip] = field(repr=False, default_factory=list)
    cursor_line: int | None = None

    @property
    def height(self) -> int:
        """How many lines the block occupies."""
        return len(self.lines)


def cursor_line(rendered: list[list[Segment]]) -> int | None:
    """Find which rendered line carries the current-match mark.

    The mark rides along as style metadata, so the line is read back off
    the finished layout instead of being recomputed from character
    offsets that word wrapping has already invalidated.
    """
    for index, line in enumerate(rendered):
        for segment in line:
            if segment.style is not None and segment.style.meta.get("spy_cursor"):
                return index
    return None


class ContextView(ScrollView):
    """Scrollable, searchable view over rendered context blocks.

    Renders each row to strips once and keeps a flat line list, so
    scrolling is O(visible lines) and paging older history in is
    O(page) instead of re-rendering everything that is already loaded.
    """

    can_focus = True

    DEFAULT_CSS = """
    ContextView {
        background: transparent;
        overflow-x: hidden;
        scrollbar-gutter: stable;
        scrollbar-size-vertical: 1;
        scrollbar-background: transparent;
        scrollbar-background-hover: transparent;
        scrollbar-background-active: transparent;
        scrollbar-color: ansi_bright_black;
        scrollbar-color-hover: ansi_white;
        scrollbar-color-active: ansi_white;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("j,down,ctrl+e", "line_down", "Down", show=False),
        Binding("k,up,ctrl+y", "line_up", "Up", show=False),
        Binding("ctrl+d", "half_down", "Half page down", show=False),
        Binding("ctrl+u", "half_up", "Half page up", show=False),
        Binding("ctrl+f,pagedown", "page_down", "Page down", show=False),
        Binding("ctrl+b,pageup", "page_up", "Page up", show=False),
        Binding("g,home", "scroll_home", "Oldest loaded", show=False),
        Binding("G,end", "scroll_end", "Latest", show=False),
    ]

    def __init__(
        self,
        conn: Connection,
        page_size: int,
        dream_id: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Create a view over an open read-only database connection."""
        super().__init__(**kwargs)
        self.conn = conn
        self.page_size = max(1, page_size)
        self.dream_id = dream_id
        self.full = False
        self.pattern: re.Pattern[str] | None = None
        self.cursor: Match | None = None
        self.blocks: list[Block] = []
        self.lines: list[Strip] = []
        self.oldest_id: int | None = None
        self.has_older = True
        # A tool result names no tool; the call that opened it does.
        self.call_names: dict[str, str] = {}
        self._paging = False

    def switch(self, dream_id: int | None) -> None:
        """Point the view at another context and empty it.

        Nothing is fetched here: the next poll refills the view from the
        new source, through the same append path as any other row.
        """
        self.dream_id = dream_id
        self.blocks = []
        self.lines = []
        self.oldest_id = None
        self.has_older = True
        self.cursor = None
        self.call_names = {}
        self._resize_virtual()
        self.refresh()

    @property
    def top_row_id(self) -> int | None:
        """Row id of the block the viewport currently starts on."""
        origin = self.scroll_offset.y
        block = next((b for b in self.blocks if b.start + b.height > origin), None)
        return block.row[0] if block else None

    def append(self, rows: list[Row]) -> None:
        """Append newly arrived rows, keeping the follow position."""
        if not rows:
            return
        following = self.is_vertical_scroll_end
        if self.oldest_id is None:
            self.oldest_id = rows[0][0]
        for block in self._build(rows, start=len(self.lines)):
            self.blocks.append(block)
            self.lines.extend(block.lines)
        self._resize_virtual()
        if following:
            self.scroll_end(animate=False, immediate=True)
        self.refresh()

    def load_older(self) -> int:
        """Page one screenful of history in; return lines prepended."""
        if self._paging or not self.has_older or self.oldest_id is None:
            return 0
        self._paging = True
        try:
            rows = fetch_before(
                self.conn, self.oldest_id, self.page_size, self.dream_id
            )
            if len(rows) < self.page_size:
                self.has_older = False
            if not rows:
                return 0
            self.oldest_id = rows[0][0]
            blocks = self._build(rows, start=0)
            added = sum(block.height for block in blocks)
            lines = [line for block in blocks for line in block.lines]
            for block in self.blocks:
                block.start += added
            self.blocks[:0] = blocks
            self.lines[:0] = lines
            self._resize_virtual()
            self.scroll_to(
                y=self.scroll_offset.y + added, animate=False, immediate=True
            )
            self.refresh()
            return added
        finally:
            self._paging = False

    def set_full(self, full: bool) -> None:
        """Toggle body truncation and re-render the loaded blocks."""
        self.full = full
        self._rebuild()

    def set_pattern(self, pattern: re.Pattern[str] | None) -> None:
        """Set (or clear) the highlighted search pattern."""
        self.pattern = pattern
        self.cursor = None
        self._rebuild()

    # ----- Search -----

    def reveal(self, match: Match) -> bool:
        """Put one occurrence on screen and mark it as the current hit.

        History pages in until the row is loaded, so a match the index
        found in the database is reachable even when it sits thousands
        of rows behind the tail the viewer opened on.
        """
        if not self._ensure_loaded(match.row_id):
            return False
        previous = self.cursor
        self.cursor = match
        self._restyle({match.row_id} | ({previous.row_id} if previous else set()))
        block = next((b for b in self.blocks if b.row[0] == match.row_id), None)
        if block is None:
            return False
        # A third of a screen of lead-in, so the hit arrives together
        # with the header that says which item it is in.
        margin = self.scrollable_content_region.height // 3
        line = block.start + (block.cursor_line or 0)
        self.scroll_to(y=max(0, line - margin), animate=False, immediate=True)
        return True

    def _ensure_loaded(self, row_id: int) -> bool:
        """Page history in until ``row_id`` is among the loaded blocks."""
        for _ in range(SEARCH_PAGES):
            if self.oldest_id is not None and row_id >= self.oldest_id:
                return True
            if not self.has_older or not self.load_older():
                break
        return self.oldest_id is not None and row_id >= self.oldest_id

    def _restyle(self, row_ids: set[int]) -> None:
        """Re-render the given blocks in place, keeping the layout.

        Only the cursor mark changes, so heights hold; should one move
        anyway, the whole buffer is rebuilt rather than left torn.
        """
        for block in self.blocks:
            if block.row[0] not in row_ids:
                continue
            rebuilt = self._build([block.row], start=block.start)
            if len(rebuilt) != 1 or rebuilt[0].height != block.height:
                self._rebuild()
                return
            block.lines = rebuilt[0].lines
            block.cursor_line = rebuilt[0].cursor_line
            self.lines[block.start : block.start + block.height] = block.lines
        self.refresh()

    # ----- Rendering -----

    def _build(self, rows: list[Row], start: int) -> list[Block]:
        """Render rows to strips, laid out from line ``start``."""
        width = max(1, self.scrollable_content_region.width)
        options = self.app.console.options.update(
            width=width, height=None, no_wrap=False, overflow="fold"
        )
        blocks: list[Block] = []
        line = start
        for row in rows:
            item = decode(row[2])
            named = call_name(item)
            if named is not None:
                self.call_names[named[0]] = named[1]
            marked = self.cursor is not None and self.cursor.row_id == row[0]
            text = render_row(
                row,
                self.full,
                self.pattern,
                self.call_names.get(str(item.get("call_id", ""))),
                self.cursor.index if marked and self.cursor else None,
            )
            if text is None:
                continue
            rendered = self.app.console.render_lines(text, options, pad=False)
            lines = [Strip(segments).adjust_cell_length(width) for segments in rendered]
            lines.append(Strip.blank(width))  # one blank line between blocks
            blocks.append(
                Block(
                    row,
                    text.plain,
                    line,
                    lines,
                    cursor_line(rendered) if marked else None,
                )
            )
            line += len(lines)
        return blocks

    def _rebuild(self) -> None:
        """Re-render every loaded block, keeping the topmost one in view."""
        if not self.blocks:
            return
        following = self.is_vertical_scroll_end
        origin = self.scroll_offset.y
        # Blocks keep their order and count, so the topmost visible one
        # can be found again by index after re-rendering.
        anchor = next(
            (i for i, b in enumerate(self.blocks) if b.start + b.height > origin), 0
        )
        offset = origin - self.blocks[anchor].start
        self.blocks = self._build([block.row for block in self.blocks], start=0)
        self.lines = [line for block in self.blocks for line in block.lines]
        self._resize_virtual()
        if following:
            self.scroll_end(animate=False, immediate=True)
        elif anchor < len(self.blocks):
            y = self.blocks[anchor].start + offset
            self.scroll_to(y=y, animate=False, immediate=True)
        self.refresh()

    def _resize_virtual(self) -> None:
        """Match the virtual size to the current line count.

        Scrollbar state has to be refreshed with it: until it is, the
        widget still believes it cannot scroll and rejects any jump to
        the end of freshly appended content.
        """
        self.virtual_size = Size(self.scrollable_content_region.width, len(self.lines))
        self._refresh_scrollbars()

    def on_resize(self, event: events.Resize) -> None:
        """Re-wrap the loaded blocks when the viewport width changes."""
        self.call_after_refresh(self._rebuild)

    def render_line(self, y: int) -> Strip:
        """Return one visible line from the flat strip list."""
        index = int(self.scroll_offset.y) + y
        if 0 <= index < len(self.lines):
            return self.lines[index]
        return Strip.blank(self.scrollable_content_region.width)

    def watch_scroll_y(self, old_value: float, new_value: float) -> None:
        """Page history in as the viewport approaches the loaded top."""
        super().watch_scroll_y(old_value, new_value)
        if new_value < self.scrollable_content_region.height:
            self.load_older()

    def action_line_down(self) -> None:
        """Scroll one line down."""
        self.scroll_relative(y=1, animate=False)

    def action_line_up(self) -> None:
        """Scroll one line up."""
        self.scroll_relative(y=-1, animate=False)

    def action_half_down(self) -> None:
        """Scroll half a screen down."""
        self.scroll_relative(
            y=self.scrollable_content_region.height // 2, animate=False
        )

    def action_half_up(self) -> None:
        """Scroll half a screen up."""
        self.scroll_relative(
            y=-(self.scrollable_content_region.height // 2), animate=False
        )

    def action_page_down(self) -> None:
        """Scroll one screen down."""
        self.scroll_relative(y=self.scrollable_content_region.height, animate=False)

    def action_page_up(self) -> None:
        """Scroll one screen up."""
        self.scroll_relative(y=-self.scrollable_content_region.height, animate=False)

    def action_scroll_home(self) -> None:
        """Jump to the oldest loaded item."""
        self.scroll_to(y=0, animate=False, immediate=True)

    def action_scroll_end(self) -> None:
        """Jump to the latest item and resume following."""
        self.scroll_end(animate=False, immediate=True)


# ===== App =====


class SearchInput(Input):
    """One-line search prompt that closes on escape."""

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("escape", "close", "Cancel", show=False),
    ]

    def action_close(self) -> None:
        """Abandon the search and hand focus back to the context."""
        cast(SpyApp, self.app).close_search()


class SteerInput(Input):
    """One-line prompt for an instruction to post to the agent."""

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("escape", "close", "Cancel", show=False),
        Binding("ctrl+t", "toggle_urgent", "Urgent", show=False),
    ]

    def action_close(self) -> None:
        """Abandon the instruction and hand focus back to the context."""
        cast(SpyApp, self.app).close_steering()

    def action_toggle_urgent(self) -> None:
        """Switch between interrupting the running turn and awaiting it."""
        app = cast(SpyApp, self.app)
        app.set_steer_urgent(not app.steer_urgent)


class SpyApp(App[None]):
    """Full-screen live view of the agent's context."""

    TITLE = "libertati spy"
    # Paint on the terminal's own colors: a transparent terminal stays
    # transparent, and every style follows its palette.
    THEME = "ansi-dark"

    # The prompt bars live on their own layer: docked to the same edge
    # as the status they would otherwise fight it for the bottom rows,
    # and the loser is painted over. On a layer they overlay the second
    # status line only while open, vim-style, and the context view never
    # reflows (a reflow re-wraps every loaded block).
    CSS = """
    Screen {
        background: transparent;
        layout: vertical;
        layers: base prompt;
    }

    ContextView {
        height: 1fr;
    }

    #status {
        dock: bottom;
        height: 2;
        padding: 0 1;
        background: transparent;
    }

    #searchbar, #steerbar {
        layer: prompt;
        dock: bottom;
        display: none;
        height: 1;
        background: transparent;
    }

    #search-prefix {
        width: auto;
        padding: 0 0 0 1;
        background: transparent;
    }

    #steer-prefix {
        width: auto;
        padding: 0 1;
        background: transparent;
    }

    #search, #steer {
        width: 1fr;
        height: 1;
        border: none;
        padding: 0 1 0 0;
        background: transparent;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("q", "quit", "Quit", show=False),
        Binding("f", "toggle_full", "Full bodies", show=False),
        Binding("d", "toggle_dream", "Dream context", show=False),
        Binding("i", "steer(False)", "Instruct", show=False),
        Binding("I", "steer(True)", "Instruct now", show=False),
        Binding("slash", "search('forward')", "Search", show=False),
        Binding("question_mark", "search('backward')", "Search back", show=False),
        Binding("n", "repeat_search(False)", "Next match", show=False),
        Binding("N", "repeat_search(True)", "Previous match", show=False),
        Binding("escape", "clear_search", "Clear search", show=False),
    ]

    def __init__(
        self,
        conn: Connection,
        last_id: int,
        page_size: int = 50,
        max_items: int = 300,
        dream_id: int | None = None,
        db_path: Path | None = None,
    ) -> None:
        """Create a viewer over an open read-only database connection.

        ``db_path`` is what instructions are posted through; without one
        the viewer is read-only in every sense.
        """
        super().__init__()
        self.conn = conn
        self.db_path = db_path
        self.page_size = page_size
        self.max_items = max_items
        self.dream_id = dream_id
        # Newest row already on screen for the mode being viewed;
        # re-anchored to a tail on every switch.
        self.last_id = last_id
        # Opening straight into a dream is a deliberate choice; don't
        # then drag the view somewhere else.
        self.auto = dream_id is None
        self.last_activity: str | None = None
        self.usage: Usage | None = None
        self.next_tokens: int | None = None
        self.note = ""
        self.hint = ""
        self.search_backward = False
        # Whether the instruction being typed interrupts the turn in
        # flight; the prompt says which, and ctrl+t flips it.
        self.steer_urgent = False
        # The dream in flight, as of the last poll: what an instruction
        # posted now would have to wait out. Set before the first key
        # can reach the prompt, but named here so nothing reads it unset.
        self.running_dream: int | None = None
        # Every occurrence of the active pattern in the stored history,
        # and where in that list the cursor sits.
        self.matches: list[Match] = []
        self.match_at: int | None = None
        self.scanned_id = 0

    @property
    def view(self) -> ContextView:
        """The context view widget."""
        return self.query_one(ContextView)

    def compose(self) -> ComposeResult:
        """Create the context view, both prompt bars and the status."""
        yield ContextView(self.conn, self.page_size, self.dream_id, id="context")
        yield Static(id="status")
        with Horizontal(id="searchbar"):
            yield Static(id="search-prefix")
            yield SearchInput(id="search", placeholder="pattern")
        with Horizontal(id="steerbar"):
            yield Static(id="steer-prefix")
            yield SteerInput(id="steer", max_length=STEERING_MAX_CHARS)

    def on_mount(self) -> None:
        """Load the initial tail and start polling for new rows."""
        self.theme = self.THEME
        self.view.focus()
        # After the first refresh, so the tail renders at the real width
        # instead of being laid out twice.
        self.call_after_refresh(self.poll)
        self.set_interval(POLL_SECONDS, self.poll)

    def poll(self) -> None:
        """Append rows added since the last poll and refresh the status."""
        self.follow_dream()
        rows = fetch_after(self.conn, self.last_id, self.dream_id)
        if rows:
            self.last_id = rows[-1][0]
            self.last_activity = rows[-1][1]
            self.view.append(rows)
            self.extend_matches()
        usage = fetch_usage(self.conn, self.dream_id)
        if rows or usage != self.usage:
            self.usage = usage
            self.next_tokens = predict_context(
                self.conn, usage, self.max_items, self.dream_id
            )
        self.update_status()

    def follow_dream(self) -> None:
        """Track a dream that starts or ends, unless something says not to.

        Two gates: a manual ``d`` pins the mode, and a scrolled-back
        viewport is left alone — being yanked to another context while
        reading history is worse than missing the switch.
        """
        dream = latest_dream(self.conn)
        self.running_dream = dream[0] if dream and dream[1] == "running" else None
        if self.running_dream is not None and self.dream_id != self.running_dream:
            self.hint = f"dream #{self.running_dream} running (d)"
        else:
            self.hint = ""
        if not self.auto or not self.view.is_vertical_scroll_end:
            return
        if self.running_dream is not None:
            self.open_dream(self.running_dream)
        elif self.dream_id is not None:
            self.open_dream(None)

    def open_dream(self, dream_id: int | None) -> None:
        """Point the viewer at a dream's context, or back at the waking one.

        The new mode opens on its tail — the same anchor the viewer
        starts at — rather than wherever it was last left, so switching
        never lands on an empty screen. Older rows page in on scroll.
        """
        if dream_id == self.dream_id:
            return
        self.dream_id = dream_id
        self.last_id = tail_anchor(self.conn, self.page_size, dream_id)
        self.hint = ""
        self.last_activity = None
        self.usage = None
        self.next_tokens = None
        self.view.switch(dream_id)
        # An index belongs to the context it was built from.
        self.matches = []
        self.match_at = None
        self.scanned_id = 0
        if self.view.pattern is not None:
            self.index_matches(self.view.pattern)

    def update_status(self) -> None:
        """Redraw the status line."""
        counts = None
        if self.view.pattern is not None:
            at = 0 if self.match_at is None else self.match_at + 1
            counts = (at, len(self.matches))
        self.query_one("#status", Static).update(
            build_status(
                self.last_id,
                self.last_activity,
                self.view.is_vertical_scroll_end,
                self.usage,
                self.next_tokens,
                self.note or self.hint,
                self.dream_id,
                counts,
            )
        )

    def action_toggle_full(self) -> None:
        """Show full bodies instead of truncated ones."""
        self.view.set_full(not self.view.full)

    def action_toggle_dream(self) -> None:
        """Switch between the waking context and a dream's, pinning the mode."""
        self.auto = False
        if self.dream_id is not None:
            self.open_dream(None)
        else:
            target = self.running_dream or latest_recorded_dream(self.conn)
            if target is None:
                self.hint = "no dream recorded yet"
            else:
                self.open_dream(target)
        self.update_status()

    # ----- Search -----

    def action_search(self, direction: str) -> None:
        """Open the search prompt, prefixed with its direction."""
        self.search_backward = direction == "backward"
        self.query_one("#search-prefix", Static).update(
            "?" if self.search_backward else "/"
        )
        prompt = self.query_one("#search", SearchInput)
        prompt.value = ""
        self.query_one("#searchbar", Horizontal).display = True
        prompt.focus()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Run whichever prompt was submitted."""
        if event.input.id == "steer":
            await self.submit_steering(event.value)
        else:
            self.submit_search(event.value)

    def submit_search(self, pattern: str) -> None:
        """Index the typed search, close the prompt and jump to a hit."""
        self.close_search()
        if not pattern:
            return
        flags = 0 if any(c.isupper() for c in pattern) else re.IGNORECASE
        try:
            compiled = re.compile(pattern, flags)
        except re.error as error:
            self.note = f"bad pattern: {error}"
            self.update_status()
            return
        self.view.set_pattern(compiled)
        self.index_matches(compiled)
        self.jump_match(self.search_backward, first=True)

    def index_matches(self, pattern: re.Pattern[str]) -> None:
        """Index a pattern over the whole stored context."""
        self.matches = scan_matches(self.conn, pattern, self.dream_id)
        self.match_at = None
        self.scanned_id = newest_id(self.conn, self.dream_id)

    def extend_matches(self) -> None:
        """Index the rows appended since the last scan, if searching."""
        pattern = self.view.pattern
        if pattern is None:
            return
        self.matches.extend(
            scan_matches(self.conn, pattern, self.dream_id, self.scanned_id)
        )
        self.scanned_id = newest_id(self.conn, self.dream_id)

    def jump_match(self, backward: bool, first: bool = False) -> None:
        """Move the cursor to the next occurrence and put it on screen.

        The first jump of a search starts from the row on screen; later
        ones step through the index, wrapping the way vim does — and
        saying so when they do.
        """
        if not self.matches:
            self.note = "pattern not found"
            self.update_status()
            return
        if first or self.match_at is None:
            found = self._first_target(backward)
            wrapped = found is None
            target = (len(self.matches) - 1 if backward else 0) if wrapped else found
        else:
            target = self.match_at + (-1 if backward else 1)
            wrapped = not 0 <= target < len(self.matches)
            target %= len(self.matches)
        self.match_at = target
        if not self.view.reveal(self.matches[self.match_at]):
            self.note = "match is older than the loaded history"
        elif wrapped:
            self.note = (
                "search hit TOP, continuing at BOTTOM"
                if backward
                else "search hit BOTTOM, continuing at TOP"
            )
        else:
            self.note = ""
        self.update_status()

    def _first_target(self, backward: bool) -> int | None:
        """Index of the first occurrence from the row the viewport is on.

        Inclusive of that row: a hit inside the item already on screen
        is the nearest one there is, and skipping it to land further
        away reads as the search having missed it.
        """
        top = self.view.top_row_id
        if top is None:
            return 0
        if backward:
            return next(
                (
                    i
                    for i in reversed(range(len(self.matches)))
                    if self.matches[i].row_id <= top
                ),
                None,
            )
        return next((i for i, m in enumerate(self.matches) if m.row_id >= top), None)

    def action_repeat_search(self, backward: bool) -> None:
        """Jump to the next (or previous) match of the active pattern."""
        if self.view.pattern is None:
            return
        self.jump_match(backward)

    def action_clear_search(self) -> None:
        """Drop the search highlight and its index."""
        self.view.set_pattern(None)
        self.matches = []
        self.match_at = None
        self.note = ""
        self.update_status()

    def close_search(self) -> None:
        """Hide the search prompt and focus the context again."""
        self.query_one("#searchbar", Horizontal).display = False
        self.view.focus()

    # ----- Steering -----

    def action_steer(self, urgent: bool) -> None:
        """Open the prompt for an instruction to the agent."""
        prompt = self.query_one("#steer", SteerInput)
        prompt.value = ""
        self.query_one("#steerbar", Horizontal).display = True
        self.set_steer_urgent(urgent)
        prompt.focus()

    def set_steer_urgent(self, urgent: bool) -> None:
        """Choose when the instruction lands, and say so in the prompt.

        The two differ enough to be worth naming — one waits for
        whatever the agent is doing, the other cuts into it at the next
        round — and the mode lives in the prefix label, not the
        placeholder: a placeholder vanishes under the first keystroke,
        and ctrl+t is pressed mid-sentence more often than not.
        """
        self.steer_urgent = urgent
        prefix = self.query_one("#steer-prefix", Static)
        if urgent:
            prefix.update(Text("instruct now!", style="bold red"))
            hint = "interrupts the turn (ctrl+t: queue)"
        else:
            prefix.update(Text("instruct:", style="bold"))
            hint = "queued for the next turn (ctrl+t: interrupt)"
        self.query_one("#steer", SteerInput).placeholder = hint

    async def submit_steering(self, text: str) -> None:
        """Post the typed instruction for the bot process to deliver.

        Nothing here waits for the agent to read it: the event shows up
        in this very view once the bot picks it up, which is the honest
        confirmation. The write itself goes to a thread — the bot writes
        on every message it stores, and a busy database is worth waiting
        out rather than freezing the viewer for the wait.
        """
        urgent = self.steer_urgent
        self.close_steering()
        text = text.strip()
        if not text:
            return
        if self.db_path is None:
            self.note = "no database path: instructions unavailable"
            self.update_status()
            return
        try:
            steering_id = await asyncio.to_thread(
                post_steering, self.db_path, text, urgent
            )
        except SQLAlchemyError as error:
            self.note = f"instruction failed: {error}"
        else:
            self.note = f"instruction #{steering_id} {self._steering_fate(urgent)}"
        self.update_status()

    def _steering_fate(self, urgent: bool) -> str:
        """Say what the instruction just posted is actually waiting for.

        A dream holds the turn lock with no round boundary to interrupt,
        so an urgent instruction posted during one waits it out like any
        other. Promising an interruption that cannot happen for another
        half hour is worse than saying nothing.
        """
        if self.running_dream is not None:
            return f"queued (dream #{self.running_dream} first)"
        return "interrupting" if urgent else "queued"

    def close_steering(self) -> None:
        """Hide the instruction prompt and focus the context again."""
        self.query_one("#steerbar", Horizontal).display = False
        self.view.focus()


def load_settings() -> Settings | None:
    """Load settings, tolerating a machine without secrets configured."""
    try:
        return Settings()
    except ValidationError:
        return None


def main() -> None:
    """Parse arguments and open the viewer."""
    parser = argparse.ArgumentParser(
        prog="libertati-spy",
        description="Live full-screen view of the agent's context history.",
        epilog=(
            "keys: j/k ctrl+e/ctrl+y line, ctrl+d/ctrl+u half page, "
            "ctrl+f/ctrl+b page, g/G ends, / ? n N search, f full bodies, "
            "d dream context, i/I instruct the agent, q quit"
        ),
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="database path (default: db_path from settings)",
    )
    parser.add_argument(
        "--dream",
        type=int,
        default=None,
        metavar="ID",
        help="open a dream's context instead of the waking one",
    )
    parser.add_argument(
        "-n",
        "--tail",
        type=int,
        default=50,
        help="initial items and history page size (default 50)",
    )
    args = parser.parse_args()

    settings = load_settings()
    max_items = (
        settings.context_max_items
        if settings
        else Settings.model_fields["context_max_items"].default
    )
    db_path = args.db or (settings.db_path if settings else None)
    if db_path is None:
        parser.error("no --db given and settings could not be loaded")
    if not db_path.exists():
        parser.error(f"database not found: {db_path}")

    # Read-only by URI: watching the agent must not be able to disturb
    # it, so the viewer's one long-lived connection cannot write at all.
    # Autocommit keeps every poll on a fresh snapshot — a transaction
    # left open across polls would pin the first one and go blind to
    # everything the bot writes after it.
    engine = create_engine(
        f"sqlite:///file:{db_path}?mode=ro&uri=true",
        poolclass=NullPool,
        isolation_level="AUTOCOMMIT",
    )
    conn = engine.connect()
    if args.dream is not None and (
        conn.execute(select(dreams.c.id).where(dreams.c.id == args.dream)).first()
        is None
    ):
        conn.close()
        engine.dispose()
        parser.error(f"no dream #{args.dream} in {db_path}")
    anchor = tail_anchor(conn, args.tail, args.dream)
    try:
        SpyApp(conn, anchor, args.tail, max_items, args.dream, db_path).run()
    except KeyboardInterrupt:
        pass
    finally:
        conn.close()
        engine.dispose()
