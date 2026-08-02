"""Dev CLI: live, colored view of the agent's context history.

``libertati-ctx`` tails the ``context`` table like ``tail -f`` for the
agent's brain: external events, hidden reasoning, tool calls/results and
private final output, each styled by kind, with a rough per-item token
count (o200k_base). Default mode streams new items to stdout; ``--live``
opens a scrollable full-screen viewer whose status line adds the
latest authoritative API input/cache usage and a separate approximate
next-context total. Requires the ``rich``, ``textual`` and ``tiktoken``
dev dependencies; run via ``uv run libertati-ctx``.
"""

import argparse
import json
import sqlite3
import time
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, ClassVar

import tiktoken
from pydantic import ValidationError
from rich.console import Console
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import RichLog, Static

from libertati.config import Settings

#: Header style and label per item kind.
STYLES = {
    "event": ("cyan", "EVENT"),
    "reasoning": ("bright_black", "REASONING"),
    "message": ("yellow", "FINAL OUTPUT"),
    "function_call": ("magenta", "TOOL CALL"),
    "function_call_output": ("blue", "TOOL RESULT"),
    "web_search_call": ("green", "WEB SEARCH"),
    "other": ("white", "OTHER"),
}

#: Body truncation limit without ``--full``.
TRUNCATE_AT = 600

POLL_SECONDS = 0.5

Row = tuple[int, str, str]


@dataclass(frozen=True)
class UsageSnapshot:
    """Authoritative counters for the latest completed API request."""

    input_tokens: int
    cached_tokens: int
    cache_write_tokens: int
    output_tokens: int
    reasoning_tokens: int
    input_context_id: int


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    """Load the o200k_base encoding once (cached on disk after first use)."""
    return tiktoken.get_encoding("o200k_base")


@lru_cache(maxsize=2048)
def token_count(raw: str) -> int:
    """Rough token count of a raw context item (o200k_base on its JSON).

    An approximation of what the item costs as API input: JSON framing
    is counted verbatim, and encrypted reasoning is counted as its
    base64 text although it bills as the original hidden tokens.
    """
    return len(_encoding().encode(raw, disallowed_special=()))


def classify(item: dict[str, Any]) -> str:
    """Map a raw context item to one of the ``STYLES`` kinds."""
    if item.get("role") == "user" and "type" not in item:
        return "event"
    kind = item.get("type", "other")
    return kind if kind in STYLES else "other"


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


def build_block(row: Row, full: bool, show_empty_final: bool = False) -> Text | None:
    """Render one context row as a styled two-part text block."""
    row_id, created_at, raw = row
    try:
        item = json.loads(raw)
    except json.JSONDecodeError:
        item = {"type": "other", "unparsed": raw}
    kind = classify(item)
    color, label = STYLES[kind]
    body = body_text(kind, item)
    if kind == "message" and not body and not show_empty_final:
        return None
    if not full and len(body) > TRUNCATE_AT:
        body = body[:TRUNCATE_AT] + f" […{len(body) - TRUNCATE_AT} chars]"
    block = Text()
    block.append(f"#{row_id} {label} ", style=f"bold {color}")
    block.append(created_at, style="bright_black")
    block.append(f" · {token_count(raw)} tok", style="bright_black")
    block.append("\n")
    block.append(body, style="default" if kind == "event" else color)
    return block


def fetch_after(conn: sqlite3.Connection, last_id: int) -> list[Row]:
    """Return all context rows with id greater than ``last_id``."""
    cursor = conn.execute(
        "SELECT id, created_at, item FROM context WHERE id > ? ORDER BY id",
        (last_id,),
    )
    return cursor.fetchall()


def fetch_before(
    conn: sqlite3.Connection,
    first_id: int,
    limit: int | None = None,
    exclude_types: tuple[str, ...] = (),
) -> list[Row]:
    """Return context rows before ``first_id``, oldest first."""
    exclusion = ""
    params: list[object] = [first_id]
    if exclude_types:
        placeholders = ", ".join("?" for _ in exclude_types)
        exclusion = (
            f" AND COALESCE(json_extract(item, '$.type'), '') NOT IN ({placeholders})"
        )
        params.extend(exclude_types)
    if limit is None:
        cursor = conn.execute(
            "SELECT id, created_at, item FROM context WHERE id < ?"
            + exclusion
            + " ORDER BY id",
            params,
        )
        return cursor.fetchall()
    params.append(limit)
    cursor = conn.execute(
        "SELECT id, created_at, item FROM context WHERE id < ?"
        + exclusion
        + " ORDER BY id DESC LIMIT ?",
        params,
    )
    return list(reversed(cursor.fetchall()))


def fetch_latest_usage(conn: sqlite3.Connection) -> UsageSnapshot | None:
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
    return UsageSnapshot(*row) if row else None


def fetch_active_turn_start(conn: sqlite3.Connection) -> int | None:
    """Return context boundary for the currently running turn, if any."""
    try:
        row = conn.execute(
            """
            SELECT start_context_id FROM agent_turns
            WHERE status = 'running' ORDER BY id DESC LIMIT 1
            """
        ).fetchone()
    except sqlite3.OperationalError:
        return None
    return int(row[0]) if row else None


def fetch_effective_window(
    conn: sqlite3.Connection,
    limit: int,
    reasoning_context: str,
    prune_completed_reasoning: bool,
    active_turn_start: int | None,
) -> list[Row]:
    """Return items expected to render into the next model context."""
    query = "SELECT id, created_at, item FROM context"
    params: list[object] = []
    excluded_types: tuple[str, ...] = ()
    if prune_completed_reasoning:
        excluded_types = ("reasoning", "message")
    elif reasoning_context == "current_turn":
        excluded_types = ("reasoning",)

    if excluded_types:
        placeholders = ", ".join("?" for _ in excluded_types)
        type_expression = "COALESCE(json_extract(item, '$.type'), '')"
        if active_turn_start is None:
            query += f" WHERE {type_expression} NOT IN ({placeholders})"
            params.extend(excluded_types)
        else:
            query += f" WHERE NOT (id <= ? AND {type_expression} IN ({placeholders}))"
            params.append(active_turn_start)
            params.extend(excluded_types)

    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    rows = conn.execute(query, params).fetchall()
    return list(reversed(rows))


def age_text(created_at: str) -> str:
    """Format how long ago a UTC ``datetime('now')`` timestamp was."""
    try:
        then = datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
    except ValueError:
        return "?"
    seconds = max(0, int((datetime.now(UTC) - then).total_seconds()))
    if seconds < 60:
        return f"{seconds}s ago"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    return f"{seconds // 3600}h {seconds % 3600 // 60}m ago"


def build_status(
    total: int,
    last_activity: str | None,
    following: bool = True,
    window_tokens: int | None = None,
    usage: UsageSnapshot | None = None,
) -> Text:
    """Build usage and projected-context lines below the context log."""
    status = Text()
    status.append(" libertati-ctx ", style="bold reverse")
    status.append(f"  #{total}", style="bold")
    if usage is not None:
        status.append(f"  ·  last in {usage.input_tokens / 1000:.1f}k", style="green")
        status.append(f" ({usage.cached_tokens / 1000:.1f}k cached", style="cyan")
        status.append(
            f", {usage.cache_write_tokens / 1000:.1f}k write)",
            style="yellow",
        )
        status.append(
            f"  ·  reason {usage.reasoning_tokens / 1000:.1f}k",
            style="bright_black",
        )
    else:
        status.append("  ·  last API usage unavailable", style="yellow")
    status.append("\n")
    if window_tokens is not None:
        status.append(f" next ctx ~{window_tokens / 1000:.1f}k", style="magenta")
    if last_activity is not None:
        status.append(f"  ·  {age_text(last_activity)}", style="cyan")
    status.append(
        "  ·  FOLLOW" if following else "  ·  SCROLLED",
        style="green" if following else "yellow",
    )
    status.append("  ·  j/k  PgUp/PgDn  g/G  q", style="bright_black")
    return status


def stream(conn: sqlite3.Connection, console: Console, args: Any, last_id: int) -> None:
    """Print items as they appear (default mode); ``--once`` dumps once."""
    while True:
        for row in fetch_after(conn, last_id):
            block = build_block(row, args.full, args.show_empty_final)
            if block is not None:
                console.print(block)
                console.print()
            last_id = row[0]
        if args.once:
            return
        time.sleep(POLL_SECONDS)


class ContextApp(App[None]):
    """Scrollable full-screen context viewer."""

    TITLE = "libertati context"
    CSS = """
    Screen {
        layout: vertical;
    }

    #context {
        height: 1fr;
        scrollbar-gutter: stable;
    }

    #status {
        dock: bottom;
        height: 2;
        padding: 0 1;
        background: $surface;
    }
    """
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("q", "quit", "Quit", priority=True),
        Binding("j,down", "context_down", "Down", show=False),
        Binding("k,up", "context_up", "Up", show=False),
        Binding("pagedown", "context_page_down", "Page down", show=False),
        Binding("pageup", "context_page_up", "Page up", show=False),
        Binding("g,home", "context_home", "First item", show=False),
        Binding("G,end", "context_end", "Latest item", show=False),
    ]

    def __init__(
        self,
        conn: sqlite3.Connection,
        full: bool,
        last_id: int,
        page_size: int = 20,
        window_items: int = 300,
        show_empty_final: bool = False,
        prune_completed_reasoning: bool = False,
        reasoning_context: str = "auto",
    ) -> None:
        """Create a viewer over an open read-only database connection."""
        super().__init__()
        self.conn = conn
        self.full = full
        self.last_id = last_id
        self.page_size = max(1, page_size)
        self.show_empty_final = show_empty_final
        self.prune_completed_reasoning = prune_completed_reasoning
        self.reasoning_context = reasoning_context
        self.window_items = window_items
        self.last_activity: str | None = None
        self.rows: list[Row] = []
        self.has_older = True
        self.paging_ready = False
        self.active_turn_start: int | None = None
        self.usage: UsageSnapshot | None = None
        self.window_tokens: deque[int] = deque(maxlen=window_items)
        self._refresh_metrics(force=True)

    def _refresh_metrics(self, force: bool = False) -> None:
        """Refresh exact usage and effective next-context estimate."""
        active_turn_start = fetch_active_turn_start(self.conn)
        usage = fetch_latest_usage(self.conn)
        state_changed = (
            active_turn_start != self.active_turn_start or usage != self.usage
        )
        self.active_turn_start = active_turn_start
        self.usage = usage
        if not force and not state_changed:
            return
        rows = fetch_effective_window(
            self.conn,
            self.window_items,
            self.reasoning_context,
            self.prune_completed_reasoning,
            active_turn_start,
        )
        self.window_tokens = deque(
            (token_count(row[2]) for row in rows),
            maxlen=self.window_items,
        )

    def _window_total(self) -> int | None:
        """Sum the rolling window token counts (None while empty)."""
        return sum(self.window_tokens) if self.window_tokens else None

    def compose(self) -> ComposeResult:
        """Create the scrollable log and fixed status line."""
        yield RichLog(id="context", min_width=1, wrap=True, auto_scroll=False)
        yield Static(
            build_status(
                self.last_id,
                None,
                window_tokens=self._window_total(),
                usage=self.usage,
            ),
            id="status",
        )

    def on_mount(self) -> None:
        """Load initial rows and start polling for new ones."""
        self._poll_context()
        self.call_after_refresh(self._enable_paging)
        self.set_interval(POLL_SECONDS, self._poll_context)
        self.set_interval(POLL_SECONDS, self._load_older_if_at_top)

    def _enable_paging(self) -> None:
        """Finish initial bottom positioning before watching the top edge."""
        self.query_one("#context", RichLog).scroll_end(animate=False, immediate=True)
        self.paging_ready = True
        self._update_status()

    def _poll_context(self) -> None:
        """Append new rows without moving a manually scrolled viewport."""
        log = self.query_one("#context", RichLog)
        following = log.is_vertical_scroll_end
        new = fetch_after(self.conn, self.last_id)
        for row in new:
            block = build_block(row, self.full, self.show_empty_final)
            if block is None:
                continue
            block.append("\n")
            log.write(block, scroll_end=following, animate=False)
        if new:
            self.rows.extend(new)
            self.last_id = new[-1][0]
            self.last_activity = new[-1][1]
        self._refresh_metrics(force=bool(new))
        self._update_status()

    def _render_rows(self) -> None:
        """Rebuild the log after older rows are prepended."""
        log = self.query_one("#context", RichLog)
        log.clear()
        for row in self.rows:
            block = build_block(row, self.full, self.show_empty_final)
            if block is None:
                continue
            block.append("\n")
            log.write(block, scroll_end=False, animate=False)

    def _load_older_if_at_top(self) -> None:
        """Load one older page when the viewport reaches its top."""
        log = self.query_one("#context", RichLog)
        if (
            not self.paging_ready
            or not self.has_older
            or not self.rows
            or log.scroll_y > 0
        ):
            return

        older = fetch_before(self.conn, self.rows[0][0], self.page_size)
        if not older:
            self.has_older = False
            self._update_status()
            return

        old_y = log.scroll_y
        old_max_y = log.max_scroll_y
        self.rows[:0] = older
        self._render_rows()
        added_height = log.max_scroll_y - old_max_y
        log.scroll_to(y=old_y + added_height, animate=False, immediate=True)
        if len(older) < self.page_size:
            self.has_older = False
        self.call_after_refresh(self._update_status)

    def _update_status(self) -> None:
        """Refresh activity age and follow state."""
        log = self.query_one("#context", RichLog)
        self.query_one("#status", Static).update(
            build_status(
                self.last_id,
                self.last_activity,
                log.is_vertical_scroll_end,
                self._window_total(),
                self.usage,
            )
        )

    def action_context_down(self) -> None:
        """Scroll down one line."""
        self.query_one("#context", RichLog).scroll_down(animate=False)
        self.call_after_refresh(self._update_status)

    def action_context_up(self) -> None:
        """Scroll up one line."""
        self.query_one("#context", RichLog).scroll_up(animate=False)
        self.call_after_refresh(self._after_upward_scroll)

    def action_context_page_down(self) -> None:
        """Scroll down one page."""
        self.query_one("#context", RichLog).scroll_page_down(animate=False)
        self.call_after_refresh(self._update_status)

    def action_context_page_up(self) -> None:
        """Scroll up one page."""
        self.query_one("#context", RichLog).scroll_page_up(animate=False)
        self.call_after_refresh(self._after_upward_scroll)

    def _after_upward_scroll(self) -> None:
        """Load history when an upward scroll reaches the current top."""
        self._load_older_if_at_top()
        self._update_status()

    def action_context_home(self) -> None:
        """Load all older rows and scroll to the first item."""
        log = self.query_one("#context", RichLog)
        if self.has_older and self.rows:
            older = fetch_before(self.conn, self.rows[0][0])
            if older:
                self.rows[:0] = older
                self._render_rows()
            self.has_older = False
        log.scroll_home(animate=False, immediate=True)
        self.call_after_refresh(self._update_status)

    def action_context_end(self) -> None:
        """Scroll to the latest item and resume following."""
        self.query_one("#context", RichLog).scroll_end(animate=False, immediate=True)
        self.call_after_refresh(self._update_status)


def run_live(
    conn: sqlite3.Connection,
    full: bool,
    last_id: int,
    page_size: int,
    window_items: int,
    show_empty_final: bool,
    prune_completed_reasoning: bool,
    reasoning_context: str,
) -> None:
    """Run the interactive full-screen context viewer."""
    ContextApp(
        conn,
        full,
        last_id,
        page_size,
        window_items,
        show_empty_final,
        prune_completed_reasoning,
        reasoning_context,
    ).run()


def main() -> None:
    """Parse arguments and tail the context table."""
    parser = argparse.ArgumentParser(
        prog="libertati-ctx",
        description="Live colored view of the agent's context history.",
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
        default=20,
        help="initial items and history page size (default 20)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="open the scrollable full-screen viewer",
    )
    parser.add_argument(
        "--once", action="store_true", help="dump and exit instead of following"
    )
    parser.add_argument(
        "--full", action="store_true", help="never truncate long bodies"
    )
    parser.add_argument(
        "--show-empty-final",
        action="store_true",
        help="show empty assistant final-output envelopes",
    )
    args = parser.parse_args()

    # Settings are optional when --db is given (e.g. inspecting a copied
    # database on a machine without .env); fall back to field defaults.
    try:
        settings: Settings | None = Settings()
    except ValidationError:
        settings = None
    window_items = (
        settings.context_max_items
        if settings
        else Settings.model_fields["context_max_items"].default
    )
    prune_completed_reasoning = (
        settings.prune_completed_reasoning
        if settings
        else Settings.model_fields["prune_completed_reasoning"].default
    )
    reasoning_context = (
        settings.reasoning_context
        if settings
        else Settings.model_fields["reasoning_context"].default
    )
    db_path = args.db or (settings.db_path if settings else None)
    if db_path is None:
        parser.error("no --db given and settings could not be loaded")
    if not db_path.exists():
        parser.error(f"database not found: {db_path}")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    row = conn.execute("SELECT COALESCE(MAX(id), 0) FROM context").fetchone()
    start_id = max(0, row[0] - args.tail)
    try:
        if args.live:
            run_live(
                conn,
                args.full,
                start_id,
                args.tail,
                window_items,
                args.show_empty_final,
                prune_completed_reasoning,
                reasoning_context,
            )
        else:
            stream(conn, Console(), args, start_id)
    except KeyboardInterrupt:
        pass
    finally:
        conn.close()
