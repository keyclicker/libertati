"""Tests for the SQLite persistence layer against a temporary database."""

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from aiogram.types import Message

from libertati.db import Database

#: 2026-08-02 12:00:00 UTC as a Telegram unix timestamp.
STAMP = 1785672000


@pytest.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    """Yield a connected database in a temporary directory."""
    database = Database(tmp_path / "test.db")
    await database.connect()
    yield database
    await database.close()


def make_message(
    message_id: int, text: str, date: int = STAMP, **overrides: Any
) -> Message:
    """Build a minimal private-chat message for persistence tests."""
    data: dict[str, Any] = {
        "message_id": message_id,
        "date": date,
        "chat": {"id": 100, "type": "private", "first_name": "Alice"},
        "from": {"id": 7, "is_bot": False, "first_name": "Alice", "username": "alice"},
        "text": text,
    }
    data.update(overrides)
    return Message.model_validate(data)


async def test_save_and_recent_messages(db: Database) -> None:
    """Saved messages come back oldest first with sender info."""
    await db.save_message(make_message(1, "first", date=STAMP))
    await db.save_message(make_message(2, "second", date=STAMP + 60))
    rows = await db.recent_messages(100, 10)
    assert [row["text"] for row in rows] == ["first", "second"]
    assert rows[0]["username"] == "alice"
    assert rows[0]["outgoing"] == 0


async def test_resave_updates_in_place(db: Database) -> None:
    """Re-saving the same message id (an edit) updates, not duplicates."""
    await db.save_message(make_message(1, "typo"))
    await db.save_message(make_message(1, "fixed"))
    rows = await db.recent_messages(100, 10)
    assert [row["text"] for row in rows] == ["fixed"]


async def test_unanswered_chats(db: Database) -> None:
    """A chat is unanswered until the latest message is outgoing."""
    await db.save_message(make_message(1, "hi", date=STAMP))
    unanswered = await db.unanswered_chats()
    assert [row["chat_id"] for row in unanswered] == [100]
    await db.save_message(make_message(2, "hey", date=STAMP + 60), outgoing=True)
    assert await db.unanswered_chats() == []


async def test_wakeup_lifecycle(db: Database) -> None:
    """Wakeups appear in due/pending queries until completed."""
    wakeup_id = await db.add_wakeup("2026-08-02 10:00:00", "ping")
    assert await db.due_wakeups("2026-08-02 09:59:59") == []
    due = await db.due_wakeups("2026-08-02 10:00:00")
    assert [(row["id"], row["note"]) for row in due] == [(wakeup_id, "ping")]
    assert len(await db.pending_wakeups()) == 1
    await db.complete_wakeup(wakeup_id)
    assert await db.due_wakeups("2026-08-02 10:00:00") == []
    assert await db.pending_wakeups() == []


async def test_context_roundtrip(db: Database) -> None:
    """Context items persist and load newest-tail, oldest first."""
    for i in range(5):
        await db.append_context({"role": "user", "content": f"event {i}"})
    tail = await db.load_context(3)
    assert [item["content"] for item in tail] == ["event 2", "event 3", "event 4"]
