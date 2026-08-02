"""Dev CLI: live, colored view of the agent's context history.

``libertati-ctx`` tails the ``context`` table like ``tail -f`` for the
agent's brain: external events, hidden reasoning, tool calls/results and
private monologue, each styled by kind. Default mode streams new items to
stdout; ``--live`` opens a scrollable full-screen viewer. Requires the
``rich`` and ``textual`` dev dependencies; run via
``uv run libertati-ctx``.
"""

import argparse
import json
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

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
    "message": ("yellow", "MONOLOGUE"),
    "function_call": ("magenta", "TOOL CALL"),
    "function_call_output": ("blue", "TOOL RESULT"),
    "other": ("white", "OTHER"),
}

#: Body truncation limit without ``--full``.
TRUNCATE_AT = 600

POLL_SECONDS = 0.5

Row = tuple[int, str, str]


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
            p.get("text", "") for p in parts if p.get("type") == "output_text"
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
    return json.dumps(item, ensure_ascii=False)


def build_block(row: Row, full: bool) -> Text:
    """Render one context row as a styled two-part text block."""
    row_id, created_at, raw = row
    try:
        item = json.loads(raw)
    except json.JSONDecodeError:
        item = {"type": "other", "unparsed": raw}
    kind = classify(item)
    color, label = STYLES[kind]
    body = body_text(kind, item)
    if not full and len(body) > TRUNCATE_AT:
        body = body[:TRUNCATE_AT] + f" […{len(body) - TRUNCATE_AT} chars]"
    block = Text()
    block.append(f"#{row_id} {label} ", style=f"bold {color}")
    block.append(created_at, style="bright_black")
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
    conn: sqlite3.Connection, first_id: int, limit: int | None = None
) -> list[Row]:
    """Return context rows before ``first_id``, oldest first."""
    if limit is None:
        cursor = conn.execute(
            "SELECT id, created_at, item FROM context WHERE id < ? ORDER BY id",
            (first_id,),
        )
        return cursor.fetchall()
    cursor = conn.execute(
        "SELECT id, created_at, item FROM context "
        "WHERE id < ? ORDER BY id DESC LIMIT ?",
        (first_id, limit),
    )
    return list(reversed(cursor.fetchall()))


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


def build_status(total: int, last_activity: str | None, following: bool = True) -> Text:
    """Build the status line shown below the full-screen context log."""
    status = Text()
    status.append(" libertati-ctx ", style="bold reverse")
    status.append(f"  #{total}", style="bold")
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
            console.print(build_block(row, args.full))
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
        height: 1;
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
    ) -> None:
        """Create a viewer over an open read-only database connection."""
        super().__init__()
        self.conn = conn
        self.full = full
        self.last_id = last_id
        self.page_size = max(1, page_size)
        self.last_activity: str | None = None
        self.rows: list[Row] = []
        self.has_older = True
        self.paging_ready = False

    def compose(self) -> ComposeResult:
        """Create the scrollable log and fixed status line."""
        yield RichLog(id="context", min_width=1, wrap=True, auto_scroll=False)
        yield Static(build_status(self.last_id, None), id="status")

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
            block = build_block(row, self.full)
            block.append("\n")
            log.write(block, scroll_end=following, animate=False)
        if new:
            self.rows.extend(new)
            self.last_id = new[-1][0]
            self.last_activity = new[-1][1]
        self._update_status()

    def _render_rows(self) -> None:
        """Rebuild the log after older rows are prepended."""
        log = self.query_one("#context", RichLog)
        log.clear()
        for row in self.rows:
            block = build_block(row, self.full)
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
            build_status(self.last_id, self.last_activity, log.is_vertical_scroll_end)
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
    conn: sqlite3.Connection, full: bool, last_id: int, page_size: int
) -> None:
    """Run the interactive full-screen context viewer."""
    ContextApp(conn, full, last_id, page_size).run()


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
    args = parser.parse_args()

    db_path = args.db or Settings().db_path
    if not db_path.exists():
        parser.error(f"database not found: {db_path}")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    row = conn.execute("SELECT COALESCE(MAX(id), 0) FROM context").fetchone()
    start_id = max(0, row[0] - args.tail)
    try:
        if args.live:
            run_live(conn, args.full, start_id, args.tail)
        else:
            stream(conn, Console(), args, start_id)
    except KeyboardInterrupt:
        pass
    finally:
        conn.close()
