"""Dev CLI: live, colored view of the agent's context history.

``libertati-ctx`` tails the ``context`` table like ``tail -f`` for the
agent's brain: external events, hidden reasoning, tool calls/results and
private monologue, each styled by kind. Default mode streams new items to
stdout; ``--live`` switches to a full-screen dashboard that redraws in
place. Requires the ``rich`` dev dependency; run via
``uv run libertati-ctx``.
"""

import argparse
import json
import math
import sqlite3
import time
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich.console import Console, Group
from rich.live import Live
from rich.text import Text

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

#: How many recent rows the live dashboard keeps around.
LIVE_BACKLOG = 300

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


def block_height(block: Text, width: int) -> int:
    """Estimate rendered line count of a block at the given width."""
    lines = 0
    for line in block.plain.splitlines() or [""]:
        lines += max(1, math.ceil(len(line) / max(1, width)))
    return lines + 1  # trailing blank separator


def build_view(console: Console, rows: deque[Row], total: int, full: bool) -> Group:
    """Compose the live dashboard: status header plus newest items.

    Items are picked newest-first until the terminal height is filled,
    then rendered oldest-to-newest so the feed reads downward.
    """
    header = Text()
    header.append(" libertati-ctx ", style="bold reverse")
    header.append(f"  items {total}", style="bold")
    if rows:
        header.append(f"  ·  last activity {age_text(rows[-1][1])}", style="cyan")
    header.append("  ·  ctrl-c to quit", style="bright_black")

    width = console.size.width
    budget = console.size.height - 3
    chosen: list[Text] = []
    for row in reversed(rows):
        block = build_block(row, full)
        height = block_height(block, width)
        if chosen and height > budget:
            break
        chosen.append(block)
        budget -= height
    chosen.reverse()

    parts: list[Any] = [header, Text()]
    for block in chosen:
        parts.append(block)
        parts.append(Text())
    return Group(*parts)


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


def run_dashboard(
    conn: sqlite3.Connection, console: Console, args: Any, last_id: int
) -> None:
    """Run the full-screen dashboard, redrawing on every poll."""
    rows: deque[Row] = deque(maxlen=LIVE_BACKLOG)
    total = 0
    with Live(console=console, screen=True, auto_refresh=False) as view:
        while True:
            new = fetch_after(conn, last_id)
            if new:
                rows.extend(new)
                last_id = new[-1][0]
                total = last_id
            view.update(build_view(console, rows, total, args.full), refresh=True)
            time.sleep(POLL_SECONDS)


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
        help="how many existing items to show first (default 20)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="full-screen dashboard instead of streaming output",
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
    console = Console()

    row = conn.execute("SELECT COALESCE(MAX(id), 0) FROM context").fetchone()
    start_id = max(0, row[0] - (LIVE_BACKLOG if args.live else args.tail))
    try:
        if args.live:
            run_dashboard(conn, console, args, start_id)
        else:
            stream(conn, console, args, start_id)
    except KeyboardInterrupt:
        pass
    finally:
        conn.close()
