"""Tests for the toolbox: dispatch, error handling and argument limits."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from zoneinfo import ZoneInfo

import pytest
from aiogram import Bot
from openai import AsyncOpenAI

from libertati.db import Database
from libertati.memory import Mind
from libertati.tools import (
    RECALL_PROMPT,
    SUMMARY_PROMPT,
    TOOLS,
    TYPING_MAX_SECONDS,
    TYPING_MIN_SECONDS,
    Toolbox,
    build_tools,
    typing_delay,
)

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

    #: ChatActionSender logs bot.id before anything else; without it the
    #: sender's worker dies pre-``try`` and its stop event never fires.
    id = 1

    async def send_chat_action(self, *args: Any, **kwargs: Any) -> bool:
        """Accept typing indicators silently."""
        return True

    async def send_message(self, *args: Any, **kwargs: Any) -> Any:
        """Raise unconditionally."""
        raise RuntimeError("boom")


class FakeClient:
    """Records recall extraction calls and returns a canned answer."""

    def __init__(self) -> None:
        """Expose a responses.create stub that logs its kwargs."""
        self.calls: list[dict[str, Any]] = []

        async def create(**kwargs: Any) -> Any:
            self.calls.append(kwargs)
            return SimpleNamespace(output_text="the cat is named Bober")

        self.responses = SimpleNamespace(create=create)


def make_toolbox(
    db: Any | None = None,
    bot: Any | None = None,
    client: Any | None = None,
    mind: Mind | None = None,
) -> Toolbox:
    """Build a Toolbox around fakes."""
    return Toolbox(
        cast(Database, db or FakeDB()),
        cast(Bot, bot or ExplodingBot()),
        UTC_TZ,
        cast(AsyncOpenAI, client or FakeClient()),
        "recall-model",
        cast(Mind, mind),
        15.0,
    )


def make_mind(tmp_path: Path) -> Mind:
    """Build an ensured Mind in a temporary directory."""
    mind = Mind(tmp_path / "mind")
    mind.ensure()
    return mind


async def test_unknown_tool() -> None:
    """Unknown tool names come back as error strings."""
    result = await make_toolbox().run("rm_rf", "{}")
    assert result == "error: unknown tool 'rm_rf'"


async def test_invalid_arguments() -> None:
    """Malformed JSON arguments come back as error strings."""
    result = await make_toolbox().run("send_message", "{not json")
    assert result == "error: invalid tool arguments"


async def test_handler_exception_is_wrapped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A raising handler is reported as an error, not propagated."""
    monkeypatch.setattr("libertati.tools.typing_delay", lambda text, cps: 0.0)
    args = json.dumps({"chat_id": 1, "text": "hi", "reply_to_message_id": None})
    result = await make_toolbox().run("send_message", args)
    assert result == "error: boom"


def test_typing_delay_bounds() -> None:
    """Typing time grows with length within the min/max bounds."""
    assert typing_delay("hi", 15) >= TYPING_MIN_SECONDS * 0.8
    assert typing_delay("hi", 15) <= (TYPING_MIN_SECONDS + 1) * 1.2
    assert typing_delay("x" * 10_000, 15) <= TYPING_MAX_SECONDS * 1.2


def test_typing_delay_disabled() -> None:
    """A non-positive speed turns the emulation off entirely."""
    assert typing_delay("some long message", 0) == 0.0


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


async def test_remember_appends_and_confirms(tmp_path: Path) -> None:
    """A remembered fact lands stamped in MEMORY.md and is confirmed."""
    mind = make_mind(tmp_path)
    result = await make_toolbox(mind=mind).run(
        "remember", json.dumps({"text": "  cat named Bober  "})
    )
    assert result == "remembered: cat named Bober"
    memory = mind.memory_path.read_text(encoding="utf-8")
    assert memory.startswith("- [")
    assert memory.endswith("] cat named Bober\n")


async def test_recall_short_circuits_on_empty_memory(tmp_path: Path) -> None:
    """Empty memory answers immediately without an API call."""
    client = FakeClient()
    toolbox = make_toolbox(client=client, mind=make_mind(tmp_path))
    result = await toolbox.run("recall", json.dumps({"query": "cat name?"}))
    assert result == "memory is empty"
    assert client.calls == []


async def test_recall_extracts_from_notes(tmp_path: Path) -> None:
    """Recall sends notes plus query to the recall model and relays the answer."""
    mind = make_mind(tmp_path)
    mind.append_memory("cat named Bober", "Sun 2026-08-02 12:00")
    client = FakeClient()
    toolbox = make_toolbox(client=client, mind=mind)
    result = await toolbox.run("recall", json.dumps({"query": "cat name?"}))
    assert result == "the cat is named Bober"
    (call,) = client.calls
    assert call["model"] == "recall-model"
    assert call["instructions"] == RECALL_PROMPT
    assert "cat named Bober" in call["input"]
    assert "cat name?" in call["input"]
    assert call["store"] is False


async def test_summarize_memory_short_circuits_on_empty(tmp_path: Path) -> None:
    """Empty memory answers immediately without an API call."""
    client = FakeClient()
    toolbox = make_toolbox(client=client, mind=make_mind(tmp_path))
    result = await toolbox.run("summarize_memory", "{}")
    assert result == "memory is empty"
    assert client.calls == []


async def test_summarize_memory_overviews_notes(tmp_path: Path) -> None:
    """The summary call sends all notes with the overview prompt, no query."""
    mind = make_mind(tmp_path)
    mind.append_memory("cat named Bober", "Sun 2026-08-02 12:00")
    client = FakeClient()
    result = await make_toolbox(client=client, mind=mind).run("summarize_memory", "{}")
    assert result == "the cat is named Bober"
    (call,) = client.calls
    assert call["instructions"] == SUMMARY_PROMPT
    assert "cat named Bober" in call["input"]


def test_build_tools_web_search_toggle() -> None:
    """Web search is appended only when enabled."""
    assert build_tools(False) == TOOLS
    assert build_tools(True) == [*TOOLS, {"type": "web_search"}]
