"""Dev TUI: live spy on the agent's context (``libertati-spy``).

Tails the ``context`` table in a full-screen viewer: external events,
hidden reasoning, tool calls/results and private final output, each
styled by kind. Navigation is vim-like — ``j``/``k``, ``ctrl+e``/
``ctrl+y``, ``ctrl+d``/``ctrl+u``, ``ctrl+f``/``ctrl+b``, ``g``/``G`` —
with ``/``, ``?``, ``n``, ``N`` search and ``f`` to un-truncate bodies.

Per-item token figures are estimates from UTF-8 byte length, scaled per
content shape (dense JSON framing, multi-byte prose, base64 reasoning
blobs). The projected next context is anchored on the newest
authoritative API usage row plus the estimate of everything appended
since it, so the fixed instructions/tool overhead comes from real
numbers instead of a guess. Requires the ``textual`` and ``rich`` dev
dependencies; run via ``uv run libertati-spy``.
"""

import argparse
import json
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, cast

from pydantic import ValidationError
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.geometry import Size
from textual.scroll_view import ScrollView
from textual.strip import Strip
from textual.widgets import Input, Static

from libertati.config import Settings

#: Header style and label per item kind.
KINDS = {
    "event": ("cyan", "EVENT"),
    "reasoning": ("bright_black", "REASONING"),
    "message": ("yellow", "FINAL OUTPUT"),
    "function_call": ("magenta", "TOOL CALL"),
    "function_call_output": ("blue", "TOOL RESULT"),
    "web_search_call": ("green", "WEB SEARCH"),
    "other": ("white", "OTHER"),
}

#: Body truncation limit until ``f`` toggles full bodies.
TRUNCATE_AT = 600

#: Seconds between polls for newly appended context rows.
POLL_SECONDS = 0.5

#: History pages a single search may pull in before giving up.
SEARCH_PAGES = 20

#: UTF-8 bytes per o200k token, by content shape. ASCII in the context
#: is mostly dense JSON framing; multi-byte text is mostly Cyrillic and
#: emoji; reasoning items are base64 blobs, which pack far more tokens
#: per byte than anything else here.
ASCII_BYTES_PER_TOKEN = 2.8
WIDE_BYTES_PER_TOKEN = 5.0
BASE64_BYTES_PER_TOKEN = 1.5

_NON_ASCII = re.compile(r"[^\x00-\x7f]")

#: Offset used to show UTC row timestamps in local time.
_LOCAL_OFFSET = int(
    (datetime.now().astimezone().utcoffset() or UTC.utcoffset(None)).total_seconds()
)

Row = tuple[int, str, str]


@dataclass(frozen=True)
class Usage:
    """Authoritative counters for the latest completed API request."""

    input_tokens: int
    cached_tokens: int
    cache_write_tokens: int
    output_tokens: int
    reasoning_tokens: int
    context_id: int


def estimate_tokens(raw: str, kind: str) -> int:
    """Estimate what one raw context item costs as API input.

    Byte-length based rather than tokenizer based: loading a real BPE
    table costs seconds of startup for numbers that are approximate
    anyway (encrypted reasoning bills as its hidden original, not as the
    base64 that is actually sent).
    """
    size = len(raw.encode())
    if kind == "reasoning":
        return round(size / BASE64_BYTES_PER_TOKEN)
    if raw.isascii():
        return round(size / ASCII_BYTES_PER_TOKEN)
    narrow = len(_NON_ASCII.sub("", raw))
    return round(
        narrow / ASCII_BYTES_PER_TOKEN + (size - narrow) / WIDE_BYTES_PER_TOKEN
    )


def classify(item: dict[str, Any]) -> str:
    """Map a raw context item to one of the ``KINDS``."""
    if item.get("role") == "user" and "type" not in item:
        return "event"
    kind = item.get("type", "other")
    return kind if kind in KINDS else "other"


def body_text(kind: str, item: dict[str, Any]) -> str:
    """Extract the human-interesting body of an item, by kind."""
    if kind == "event":
        return str(item.get("content", ""))
    if kind == "reasoning":
        parts = [s.get("text", "") for s in item.get("summary", [])]
        return "\n".join(p for p in parts if p) or "(hidden)"
    if kind == "message":
        parts = item.get("content", [])
        return "\n".join(
            p.get("text", "") or p.get("refusal", "")
            for p in parts
            if p.get("type") in {"output_text", "refusal"}
        )
    if kind == "function_call":
        args = item.get("arguments") or "{}"
        try:
            args = json.dumps(json.loads(args), ensure_ascii=False)
        except json.JSONDecodeError:
            pass
        return f"{item.get('name', '?')} {args}"
    if kind == "function_call_output":
        return str(item.get("output", ""))
    if kind == "web_search_call":
        return json.dumps(item.get("action", {}), ensure_ascii=False)
    return json.dumps(item, ensure_ascii=False)


def local_clock(created_at: str) -> str:
    """Render a UTC ``datetime('now')`` timestamp as local wall time."""
    try:
        seconds = (
            int(created_at[11:13]) * 3600
            + int(created_at[14:16]) * 60
            + int(created_at[17:19])
            + _LOCAL_OFFSET
        ) % 86400
    except ValueError:
        return "??:??:??"
    return f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def age_text(created_at: str) -> str:
    """Format how long ago a UTC ``datetime('now')`` timestamp was."""
    try:
        then = datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
    except ValueError:
        return "?"
    seconds = max(0, int((datetime.now(UTC) - then).total_seconds()))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    return f"{seconds // 3600}h {seconds % 3600 // 60}m"


def build_block(
    row: Row,
    full: bool = False,
    pattern: re.Pattern[str] | None = None,
) -> Text | None:
    """Render one context row, or ``None`` for empty output envelopes."""
    row_id, created_at, raw = row
    try:
        item = json.loads(raw)
    except json.JSONDecodeError:
        item = {"type": "other", "unparsed": raw}
    kind = classify(item)
    color, label = KINDS[kind]
    body = body_text(kind, item)
    if kind == "message" and not body:
        return None
    if not full and len(body) > TRUNCATE_AT:
        body = f"{body[:TRUNCATE_AT]} […{len(body) - TRUNCATE_AT} chars]"
    block = Text()
    block.append(f"#{row_id} ", style="bold bright_black")
    block.append(label, style=f"bold {color}")
    block.append(
        f"  {local_clock(created_at)}  ~{estimate_tokens(raw, kind)} tok\n",
        style="bright_black",
    )
    block.append(body, style="default" if kind == "event" else color)
    if pattern is not None:
        block.highlight_regex(pattern, style="reverse")
    return block


def fetch_after(conn: sqlite3.Connection, last_id: int) -> list[Row]:
    """Return all context rows with id greater than ``last_id``."""
    return conn.execute(
        "SELECT id, created_at, item FROM context WHERE id > ? ORDER BY id",
        (last_id,),
    ).fetchall()


def fetch_before(conn: sqlite3.Connection, first_id: int, limit: int) -> list[Row]:
    """Return up to ``limit`` rows before ``first_id``, oldest first."""
    rows = conn.execute(
        "SELECT id, created_at, item FROM context WHERE id < ? ORDER BY id DESC LIMIT ?",
        (first_id, limit),
    ).fetchall()
    return list(reversed(rows))


def fetch_usage(conn: sqlite3.Connection) -> Usage | None:
    """Return exact usage from the newest API response when available."""
    try:
        row = conn.execute(
            """
            SELECT input_tokens, cached_tokens, cache_write_tokens,
                   output_tokens, reasoning_tokens, input_context_id
            FROM api_usage ORDER BY id DESC LIMIT 1
            """
        ).fetchone()
    except sqlite3.OperationalError:
        return None
    return Usage(*row) if row else None


def _estimate_rows(rows: list[tuple[str, str]]) -> int:
    """Sum the estimate over ``(item, type)`` pairs."""
    return sum(estimate_tokens(item, kind) for item, kind in rows)


def predict_context(
    conn: sqlite3.Connection,
    usage: Usage | None,
    max_items: int,
) -> int | None:
    """Estimate the input size of the agent's next API call.

    Anchored on the last authoritative count: everything the API already
    charged for is taken as measured, and only the items appended since
    that request are estimated. Falls back to estimating the whole live
    window when there is no usage row yet, or when so much has piled up
    since one that the window would have been trimmed anyway.
    """
    select = "SELECT item, COALESCE(json_extract(item, '$.type'), '') FROM context"
    if usage is not None:
        pending = conn.execute(
            f"{select} WHERE id > ? ORDER BY id LIMIT ?",
            (usage.context_id, max_items),
        ).fetchall()
        if len(pending) < max_items:
            return usage.input_tokens + _estimate_rows(pending)
    window = conn.execute(f"{select} ORDER BY id DESC LIMIT ?", (max_items,)).fetchall()
    return _estimate_rows(window) or None


def build_status(
    last_id: int,
    last_activity: str | None,
    following: bool = True,
    usage: Usage | None = None,
    next_tokens: int | None = None,
    note: str = "",
) -> Text:
    """Build the two-line status: position on top, token figures below."""
    status = Text()
    status.append("spy", style="bold reverse")
    status.append(f"  #{last_id}", style="bold")
    if last_activity is not None:
        status.append(f"  {age_text(last_activity)}", style="cyan")
    status.append(
        "  FOLLOW" if following else "  SCROLLED",
        style="green" if following else "yellow",
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


@dataclass
class Block:
    """One rendered context row: its lines and where they live."""

    row: Row
    plain: str
    start: int
    lines: list[Strip] = field(repr=False, default_factory=list)

    @property
    def height(self) -> int:
        """How many lines the block occupies."""
        return len(self.lines)


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

    def __init__(self, conn: sqlite3.Connection, page_size: int, **kwargs: Any) -> None:
        """Create a view over an open read-only database connection."""
        super().__init__(**kwargs)
        self.conn = conn
        self.page_size = max(1, page_size)
        self.full = False
        self.pattern: re.Pattern[str] | None = None
        self.blocks: list[Block] = []
        self.lines: list[Strip] = []
        self.oldest_id: int | None = None
        self.has_older = True
        self._paging = False

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
            rows = fetch_before(self.conn, self.oldest_id, self.page_size)
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
        self._rebuild()

    def search(self, backward: bool) -> bool:
        """Jump to the next block matching the pattern; vim semantics."""
        if self.pattern is None or not self.blocks:
            return False
        origin = self.scroll_offset.y
        if backward:
            target = self._search_backward(origin)
        else:
            target = next(
                (b for b in self.blocks if b.start > origin and self._matches(b)), None
            )
        if target is None:  # wrap around the loaded buffer, like vim
            matches = [b for b in self.blocks if self._matches(b)]
            target = (matches[-1] if backward else matches[0]) if matches else None
        if target is None:
            return False
        self.scroll_to(y=target.start, animate=False, immediate=True)
        return True

    def _search_backward(self, origin: int) -> Block | None:
        """Look backwards, pulling in history until a match or the start."""
        for _ in range(SEARCH_PAGES):
            match = next(
                (
                    b
                    for b in reversed(self.blocks)
                    if b.start < origin and self._matches(b)
                ),
                None,
            )
            if match is not None or not self.has_older:
                return match
            origin += self.load_older()
        return None

    def _matches(self, block: Block) -> bool:
        """Whether a block's rendered text contains the search pattern."""
        return self.pattern is not None and self.pattern.search(block.plain) is not None

    def _build(self, rows: list[Row], start: int) -> list[Block]:
        """Render rows to strips, laid out from line ``start``."""
        width = max(1, self.scrollable_content_region.width)
        options = self.app.console.options.update(
            width=width, height=None, no_wrap=False, overflow="fold"
        )
        blocks: list[Block] = []
        line = start
        for row in rows:
            text = build_block(row, self.full, self.pattern)
            if text is None:
                continue
            rendered = self.app.console.render_lines(text, options, pad=False)
            lines = [Strip(segments).adjust_cell_length(width) for segments in rendered]
            lines.append(Strip.blank(width))  # one blank line between blocks
            blocks.append(Block(row, text.plain, line, lines))
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


class SearchInput(Input):
    """One-line search prompt that closes on escape."""

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("escape", "close", "Cancel", show=False),
    ]

    def action_close(self) -> None:
        """Abandon the search and hand focus back to the context."""
        cast(SpyApp, self.app).close_search()


class SpyApp(App[None]):
    """Full-screen live view of the agent's context."""

    TITLE = "libertati spy"
    # Paint on the terminal's own colors: a transparent terminal stays
    # transparent, and every style follows its palette.
    THEME = "ansi-dark"

    CSS = """
    Screen {
        background: transparent;
        layout: vertical;
    }

    ContextView {
        height: 1fr;
    }

    #search {
        dock: bottom;
        display: none;
        height: 1;
        border: none;
        padding: 0 1;
        background: transparent;
    }

    #status {
        dock: bottom;
        height: 2;
        padding: 0 1;
        background: transparent;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("q", "quit", "Quit", show=False),
        Binding("f", "toggle_full", "Full bodies", show=False),
        Binding("slash", "search('forward')", "Search", show=False),
        Binding("question_mark", "search('backward')", "Search back", show=False),
        Binding("n", "repeat_search(False)", "Next match", show=False),
        Binding("N", "repeat_search(True)", "Previous match", show=False),
        Binding("escape", "clear_search", "Clear search", show=False),
    ]

    def __init__(
        self,
        conn: sqlite3.Connection,
        last_id: int,
        page_size: int = 50,
        max_items: int = 300,
    ) -> None:
        """Create a viewer over an open read-only database connection."""
        super().__init__()
        self.conn = conn
        self.last_id = last_id
        self.page_size = page_size
        self.max_items = max_items
        self.last_activity: str | None = None
        self.usage: Usage | None = None
        self.next_tokens: int | None = None
        self.note = ""
        self.search_backward = False

    @property
    def view(self) -> ContextView:
        """The context view widget."""
        return self.query_one(ContextView)

    def compose(self) -> ComposeResult:
        """Create the context view, the search prompt and the status."""
        yield ContextView(self.conn, self.page_size, id="context")
        yield SearchInput(id="search")
        yield Static(id="status")

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
        rows = fetch_after(self.conn, self.last_id)
        if rows:
            self.last_id = rows[-1][0]
            self.last_activity = rows[-1][1]
            self.view.append(rows)
        usage = fetch_usage(self.conn)
        if rows or usage != self.usage:
            self.usage = usage
            self.next_tokens = predict_context(self.conn, usage, self.max_items)
        self.update_status()

    def update_status(self) -> None:
        """Redraw the status line."""
        self.query_one("#status", Static).update(
            build_status(
                self.last_id,
                self.last_activity,
                self.view.is_vertical_scroll_end,
                self.usage,
                self.next_tokens,
                self.note,
            )
        )

    def action_toggle_full(self) -> None:
        """Show full bodies instead of truncated ones."""
        self.view.set_full(not self.view.full)

    def action_search(self, direction: str) -> None:
        """Open the search prompt."""
        self.search_backward = direction == "backward"
        prompt = self.query_one("#search", SearchInput)
        prompt.value = ""
        prompt.placeholder = "?pattern" if self.search_backward else "/pattern"
        prompt.display = True
        prompt.focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Run the typed search and close the prompt."""
        pattern = event.value
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
        self.action_repeat_search(self.search_backward)

    def action_repeat_search(self, backward: bool) -> None:
        """Jump to the next (or previous) match of the active pattern."""
        if self.view.pattern is None:
            return
        found = self.view.search(backward)
        self.note = "" if found else "pattern not found"
        self.update_status()

    def action_clear_search(self) -> None:
        """Drop the search highlight."""
        self.view.set_pattern(None)
        self.note = ""
        self.update_status()

    def close_search(self) -> None:
        """Hide the search prompt and focus the context again."""
        prompt = self.query_one("#search", SearchInput)
        prompt.display = False
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
            "ctrl+f/ctrl+b page, g/G ends, / ? n N search, f full bodies, q quit"
        ),
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="database path (default: db_path from settings)",
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

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    row = conn.execute("SELECT COALESCE(MAX(id), 0) FROM context").fetchone()
    try:
        SpyApp(conn, max(0, row[0] - args.tail), args.tail, max_items).run()
    except KeyboardInterrupt:
        pass
    finally:
        conn.close()
