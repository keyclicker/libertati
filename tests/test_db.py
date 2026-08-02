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
    message_id: int, text: str | None, date: int = STAMP, **overrides: Any
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


async def test_delete_message(db: Database) -> None:
    """Deleting removes exactly the one row."""
    await db.save_message(make_message(1, "keep", date=STAMP))
    await db.save_message(make_message(2, "drop", date=STAMP + 60))
    await db.delete_message(100, 2)
    rows = await db.recent_messages(100, 10)
    assert [row["text"] for row in rows] == ["keep"]


async def test_recent_messages_pagination(db: Database) -> None:
    """before_message_id pages into the past, excluding the cursor itself."""
    for i in range(1, 6):
        await db.save_message(make_message(i, f"msg {i}", date=STAMP + i))
    page = await db.recent_messages(100, 2, before_message_id=4)
    assert [row["message_id"] for row in page] == [2, 3]


async def test_search_messages(db: Database) -> None:
    """Substring search is case-insensitive, newest first, wildcards literal."""
    await db.save_message(make_message(1, "my Cat is grumpy", date=STAMP))
    await db.save_message(make_message(2, "no pets here", date=STAMP + 60))
    await db.save_message(make_message(3, "cat again", date=STAMP + 120))
    rows = await db.search_messages(100, "cat", 10)
    assert [row["message_id"] for row in rows] == [3, 1]
    assert await db.search_messages(100, "100%", 10) == []


async def test_list_chats(db: Database) -> None:
    """Chats list with a display name (peer's name for private chats)."""
    peer = {"id": 100, "is_bot": False, "first_name": "Alice"}
    await db.save_message(make_message(1, "hi", date=STAMP, **{"from": peer}))
    (chat,) = await db.list_chats()
    assert chat["chat_id"] == 100
    assert chat["type"] == "private"
    assert chat["name"] == "Alice"
    assert chat["messages"] == 1
    assert chat["last_date"] is not None


async def test_chat_members(db: Database) -> None:
    """Members are the distinct senders seen, most recently active first."""
    bob = {"id": 8, "is_bot": False, "first_name": "Bob"}
    await db.save_message(make_message(1, "hi", date=STAMP))
    await db.save_message(make_message(2, "hey", date=STAMP + 60, **{"from": bob}))
    await db.save_message(make_message(3, "again", date=STAMP + 120, **{"from": bob}))
    members = await db.chat_members(100)
    assert [(m["user_id"], m["messages"]) for m in members] == [(8, 2), (7, 1)]
    assert await db.chat_members(999) == []


def make_sticker_message(
    message_id: int, file_id: str, unique_id: str, date: int
) -> Message:
    """Build a sticker message for the known-sticker tests."""
    sticker = {
        "file_id": file_id,
        "file_unique_id": unique_id,
        "type": "regular",
        "width": 512,
        "height": 512,
        "is_animated": False,
        "is_video": False,
        "emoji": "😀",
        "set_name": "pack",
    }
    return make_message(message_id, None, date=date, sticker=sticker)


async def test_known_stickers(db: Database) -> None:
    """Stickers dedupe on file_unique_id and list newest first."""
    await db.save_message(make_sticker_message(1, "AAA", "u1", STAMP))
    await db.save_message(make_sticker_message(2, "AAA2", "u1", STAMP + 60))
    await db.save_message(make_sticker_message(3, "BBB", "u2", STAMP + 120))
    await db.save_message(make_message(4, "not a sticker", date=STAMP + 180))
    stickers = await db.known_stickers(10)
    assert [row["file_id"] for row in stickers] == ["BBB", "AAA2"]
    assert stickers[0]["emoji"] == "😀"
    assert stickers[0]["set_name"] == "pack"


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


async def test_cancel_wakeup(db: Database) -> None:
    """Cancelling removes a pending wakeup; done/unknown ids report False."""
    wakeup_id = await db.add_wakeup("2026-08-02 10:00:00", "ping")
    assert await db.cancel_wakeup(wakeup_id) is True
    assert await db.pending_wakeups() == []
    assert await db.cancel_wakeup(wakeup_id) is False
    assert await db.cancel_wakeup(999) is False


async def test_context_roundtrip(db: Database) -> None:
    """Context items persist and load newest-tail, oldest first."""
    for i in range(5):
        await db.append_context({"role": "user", "content": f"event {i}"})
    tail = await db.load_context(3)
    assert [item["content"] for item in tail] == ["event 2", "event 3", "event 4"]
