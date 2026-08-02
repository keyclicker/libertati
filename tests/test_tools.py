"""Tests for the toolbox: dispatch, error handling and argument limits."""

import json
from typing import Any, cast
from zoneinfo import ZoneInfo

from aiogram import Bot

from libertati.db import Database
from libertati.tools import Toolbox

UTC_TZ = ZoneInfo("UTC")


class FakeDB:
    """Records calls made by tool handlers."""

    def __init__(self) -> None:
        """Start with empty call records."""
        self.recent_calls: list[tuple[int, int]] = []
        self.wakeups: list[tuple[str, str]] = []

    async def recent_messages(self, chat_id: int, limit: int) -> list[dict]:
        """Record the query and return no rows."""
        self.recent_calls.append((chat_id, limit))
        return []

    async def add_wakeup(self, due_at: str, note: str) -> int:
        """Record the wakeup and return a fixed id."""
        self.wakeups.append((due_at, note))
        return 7


class ExplodingBot:
    """A bot whose send always fails, to exercise error wrapping."""

    async def send_message(self, *args: Any, **kwargs: Any) -> Any:
        """Raise unconditionally."""
        raise RuntimeError("boom")


def make_toolbox(db: Any | None = None, bot: Any | None = None) -> Toolbox:
    """Build a Toolbox around fakes."""
    return Toolbox(
        cast(Database, db or FakeDB()), cast(Bot, bot or ExplodingBot()), UTC_TZ
    )


async def test_unknown_tool() -> None:
    """Unknown tool names come back as error strings."""
    result = await make_toolbox().run("rm_rf", "{}")
    assert result == "error: unknown tool 'rm_rf'"


async def test_invalid_arguments() -> None:
    """Malformed JSON arguments come back as error strings."""
    result = await make_toolbox().run("send_message", "{not json")
    assert result == "error: invalid tool arguments"


async def test_handler_exception_is_wrapped() -> None:
    """A raising handler is reported as an error, not propagated."""
    args = json.dumps({"chat_id": 1, "text": "hi", "reply_to_message_id": None})
    result = await make_toolbox().run("send_message", args)
    assert result == "error: boom"


async def test_schedule_wakeup_rejects_past() -> None:
    """Past times are refused with the current time in the message."""
    args = json.dumps({"when": "2000-01-01 00:00", "note": "n"})
    result = await make_toolbox().run("schedule_wakeup", args)
    assert result.startswith("error: 2000-01-01 00:00 is in the past")


async def test_schedule_wakeup_rejects_garbage() -> None:
    """Unparseable times are refused with the expected format."""
    args = json.dumps({"when": "next tuesday", "note": "n"})
    result = await make_toolbox().run("schedule_wakeup", args)
    assert result == "error: 'when' must be 'YYYY-MM-DD HH:MM'"


async def test_schedule_wakeup_stores_future() -> None:
    """A future wakeup is stored (as a UTC stamp) and confirmed."""
    db = FakeDB()
    args = json.dumps({"when": "2999-01-01 12:00", "note": "ping"})
    result = await make_toolbox(db=db).run("schedule_wakeup", args)
    assert result == "wakeup #7 scheduled for Tue 2999-01-01 12:00"
    assert db.wakeups == [("2999-01-01 12:00:00", "ping")]


async def test_get_recent_messages_clamps_limit() -> None:
    """The limit is clamped to [1, 50]; null falls back to 20."""
    db = FakeDB()
    toolbox = make_toolbox(db=db)
    for limit, expected in ((999, 50), (None, 20), (-5, 1)):
        await toolbox.run(
            "get_recent_messages", json.dumps({"chat_id": 1, "limit": limit})
        )
        assert db.recent_calls[-1] == (1, expected)
