"""Tests for the SQLite persistence layer against a temporary database."""

import sqlite3
from pathlib import Path
from typing import Any

from aiogram.types import Message

from libertati.db import Database

#: 2026-08-02 12:00:00 UTC as a Telegram unix timestamp.
STAMP = 1785672000


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


async def test_search_messages_folds_unicode_case(db: Database) -> None:
    """Case-insensitivity holds beyond ASCII (SQLite LIKE would not)."""
    await db.save_message(make_message(1, "Привіт з Києва", date=STAMP))
    rows = await db.search_messages(100, "привіт", 10)
    assert [row["message_id"] for row in rows] == [1]
    assert await db.search_messages(100, "КИЄВА", 10) != []


async def test_recent_messages_same_second_keeps_id_order(db: Database) -> None:
    """Messages sharing a timestamp are ordered by message id."""
    for i in (1, 2, 3):
        await db.save_message(make_message(i, f"m{i}", date=STAMP))
    rows = await db.recent_messages(100, 2)
    assert [row["message_id"] for row in rows] == [2, 3]


def make_reply(message_id: int, text: str, reply_to: int, date: int) -> Message:
    """Build a message replying to another message in the same chat."""
    parent = {
        "message_id": reply_to,
        "date": STAMP,
        "chat": {"id": 100, "type": "private", "first_name": "Alice"},
    }
    return make_message(message_id, text, date=date, reply_to_message=parent)


async def test_message_thread(db: Database) -> None:
    """The thread walks reply links both ways and skips unrelated messages."""
    await db.save_message(make_message(1, "root", date=STAMP))
    await db.save_message(make_reply(2, "reply", 1, STAMP + 60))
    await db.save_message(make_reply(3, "deeper", 2, STAMP + 120))
    await db.save_message(make_message(4, "unrelated", date=STAMP + 180))
    await db.save_message(make_reply(5, "branch", 1, STAMP + 240))
    rows = await db.message_thread(100, 3, 10)
    assert [row["message_id"] for row in rows] == [1, 2, 3, 5]
    assert rows[2]["reply_to_message_id"] == 2
    assert rows[0]["reply_to_message_id"] is None


async def test_message_thread_truncates_to_newest(db: Database) -> None:
    """Over-limit threads keep the newest messages, still oldest first."""
    await db.save_message(make_message(1, "root", date=STAMP))
    await db.save_message(make_reply(2, "reply", 1, STAMP + 60))
    await db.save_message(make_reply(3, "deeper", 2, STAMP + 120))
    rows = await db.message_thread(100, 1, 2)
    assert [row["message_id"] for row in rows] == [2, 3]


async def test_message_thread_unknown_message(db: Database) -> None:
    """A message id that was never stored yields no rows."""
    await db.save_message(make_message(1, "hi", date=STAMP))
    assert await db.message_thread(100, 999, 10) == []


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


async def test_unanswered_chats_same_second_reply_counts(db: Database) -> None:
    """An outgoing reply in the same second still marks the chat answered."""
    await db.save_message(make_message(1, "hi", date=STAMP))
    await db.save_message(make_message(2, "yo", date=STAMP), outgoing=True)
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


async def test_context_load_excludes_ephemeral_types_before_limit(
    db: Database,
) -> None:
    """Filtered restore returns requested count of retained item types."""
    await db.append_context({"role": "user", "content": "event 1"})
    await db.append_context({"type": "reasoning", "encrypted_content": "x"})
    await db.append_context({"type": "message", "role": "assistant", "content": []})
    await db.append_context({"role": "user", "content": "event 2"})

    items = await db.load_context(2, exclude_types=("reasoning", "message"))

    assert [item["content"] for item in items] == ["event 1", "event 2"]


async def test_api_usage_roundtrip(db: Database) -> None:
    """Exact response usage is retained for later cost analysis."""
    turn_id = await db.start_agent_turn(await db.latest_context_id())
    await db.append_api_usage(
        response_id="resp_1",
        turn_id=turn_id,
        input_context_id=42,
        model="gpt-test",
        input_tokens=100,
        cached_tokens=80,
        cache_write_tokens=20,
        output_tokens=30,
        reasoning_tokens=25,
        total_tokens=130,
    )

    row = await (await db.conn.execute("SELECT * FROM api_usage")).fetchone()
    assert row is not None
    assert row["response_id"] == "resp_1"
    assert row["turn_id"] == turn_id
    assert row["input_context_id"] == 42
    assert row["model"] == "gpt-test"
    assert row["cached_tokens"] == 80
    assert row["cache_write_tokens"] == 20
    assert row["reasoning_tokens"] == 25


async def test_agent_turn_lifecycle(db: Database) -> None:
    """Turn boundaries and outcomes persist for viewer context filtering."""
    await db.append_context({"role": "user", "content": "event"})
    start_context_id = await db.latest_context_id()
    turn_id = await db.start_agent_turn(start_context_id)
    await db.append_context({"type": "reasoning", "encrypted_content": "x"})
    end_context_id = await db.latest_context_id()
    await db.finish_agent_turn(turn_id, end_context_id, "completed")

    row = await (
        await db.conn.execute("SELECT * FROM agent_turns WHERE id = ?", (turn_id,))
    ).fetchone()
    assert row is not None
    assert row["start_context_id"] == start_context_id
    assert row["end_context_id"] == end_context_id
    assert row["status"] == "completed"
    assert row["finished_at"] is not None


async def test_usage_schema_migrates_existing_table(tmp_path: Path) -> None:
    """Connecting adds context linkage to pre-linkage usage tables."""
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.execute(
        """CREATE TABLE api_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            response_id TEXT,
            model TEXT NOT NULL,
            input_tokens INTEGER NOT NULL,
            cached_tokens INTEGER NOT NULL,
            cache_write_tokens INTEGER NOT NULL,
            output_tokens INTEGER NOT NULL,
            reasoning_tokens INTEGER NOT NULL,
            total_tokens INTEGER NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )"""
    )
    conn.close()

    database = Database(path)
    await database.connect()
    columns = {
        row[1]
        for row in await (
            await database.conn.execute("PRAGMA table_info(api_usage)")
        ).fetchall()
    }
    await database.close()

    assert {"turn_id", "input_context_id"} <= columns


async def test_dream_ledger_round_trip(db: Database) -> None:
    """A dream is opened running and closed with its outcome."""
    dream_id = await db.start_dream("idle")
    await db.finish_dream(dream_id, "woke", 14, "talk to Alice about the trip")

    async with db.conn.execute(
        "SELECT trigger, status, steps, summary, finished_at FROM dreams WHERE id = ?",
        (dream_id,),
    ) as cursor:
        row = await cursor.fetchone()

    assert row is not None
    assert row["trigger"] == "idle"
    assert row["status"] == "woke"
    assert row["steps"] == 14
    assert row["summary"] == "talk to Alice about the trip"
    assert row["finished_at"] is not None


async def test_start_dream_leaves_other_running_rows_alone(db: Database) -> None:
    """Unlike agent turns, opening a dream never sweeps older rows."""
    first = await db.start_dream("idle")
    await db.start_dream("requested")

    async with db.conn.execute(
        "SELECT status FROM dreams WHERE id = ?", (first,)
    ) as cursor:
        row = await cursor.fetchone()

    assert row is not None
    assert row["status"] == "running"


async def test_dreams_since_counts_from_a_stamp(db: Database) -> None:
    """Budget accounting counts every dream started in the window."""
    await db.start_dream("idle")
    await db.start_dream("requested")

    assert await db.dreams_since("1970-01-01 00:00:00") == 2
    assert await db.dreams_since("2999-01-01 00:00:00") == 0


async def test_dreams_since_counts_a_dream_that_never_finished(db: Database) -> None:
    """A crashed dream still spends its budget, so it cannot loop."""
    await db.start_dream("idle")
    assert await db.dreams_since("1970-01-01 00:00:00") == 1
    assert await db.last_dream_end() is None


async def test_last_dream_end_reports_the_newest_finish(db: Database) -> None:
    """The cooldown reads the most recent finished dream."""
    assert await db.last_dream_end() is None
    dream_id = await db.start_dream("idle")
    await db.finish_dream(dream_id, "woke", 3, "nothing much")
    assert await db.last_dream_end() is not None
