"""Tests for the SQLite persistence layer against a temporary database."""

import json
import sqlite3
import stat
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


async def test_database_is_private_to_its_owner(db: Database) -> None:
    """Stored chats and model context are not world-readable."""
    assert stat.S_IMODE(db.path.stat().st_mode) == 0o600


async def test_resave_updates_in_place(db: Database) -> None:
    """Re-saving the same message id (an edit) updates, not duplicates."""
    await db.save_message(make_message(1, "typo"))
    await db.save_message(make_message(1, "fixed"))
    rows = await db.recent_messages(100, 10)
    assert [row["text"] for row in rows] == ["fixed"]


async def test_message_is_outgoing(db: Database) -> None:
    """Only stored bot-sent messages count as outgoing; unknown ones don't."""
    await db.save_message(make_message(1, "theirs"))
    await db.save_message(make_message(2, "ours", date=STAMP + 60), outgoing=True)
    assert await db.message_is_outgoing(100, 1) is False
    assert await db.message_is_outgoing(100, 2) is True
    assert await db.message_is_outgoing(100, 3) is False


async def test_delete_message(db: Database) -> None:
    """Deleting removes exactly the one row."""
    await db.save_message(make_message(1, "keep", date=STAMP))
    await db.save_message(make_message(2, "drop", date=STAMP + 60))
    await db.delete_message(100, 2)
    rows = await db.recent_messages(100, 10)
    assert [row["text"] for row in rows] == ["keep"]


async def test_recent_messages_expose_reply_links(db: Database) -> None:
    """History rows carry the reply target, so transcripts can mark it."""
    await db.save_message(make_message(1, "first"))
    await db.save_message(
        make_message(
            2,
            "answer",
            date=STAMP + 60,
            reply_to_message={
                "message_id": 1,
                "date": STAMP,
                "chat": {"id": 100, "type": "private", "first_name": "Alice"},
            },
        )
    )
    rows = await db.recent_messages(100, 10)
    assert [row["reply_to_message_id"] for row in rows] == [None, 1]


async def test_message_row_returns_one_message_or_nothing(db: Database) -> None:
    """A single lookup answers what an event's reply target said."""
    await db.save_message(make_message(1, "first"))
    row = await db.message_row(100, 1)
    assert row is not None
    assert row["text"] == "first"
    assert await db.message_row(100, 2) is None


async def test_messages_since_read_stops_at_the_cursor(db: Database) -> None:
    """Only messages the agent has not been shown come back, oldest first."""
    for i in range(1, 5):
        await db.save_message(make_message(i, f"msg {i}", date=STAMP + i))
    await db.mark_messages_read(100, 2)
    rows = await db.messages_since_read(100, 10)
    assert [row["message_id"] for row in rows] == [3, 4]
    rows = await db.messages_since_read(100, 10, before_message_id=4)
    assert [row["message_id"] for row in rows] == [3]


async def test_messages_since_read_keeps_the_newest_and_own_replies(
    db: Database,
) -> None:
    """Over the limit the newest survive, and the agent's own words stay."""
    await db.save_message(make_message(1, "theirs"))
    await db.save_message(make_message(2, "mine", date=STAMP + 1), outgoing=True)
    await db.save_message(make_message(3, "theirs again", date=STAMP + 2))
    assert [row["message_id"] for row in await db.messages_since_read(100, 10)] == [
        1,
        2,
        3,
    ]
    assert [row["message_id"] for row in await db.messages_since_read(100, 2)] == [2, 3]


async def test_messages_since_read_honours_topic_scope(db: Database) -> None:
    """A topic read cursor governs that topic, the chat cursor governs all."""
    await db.save_message(make_topic_message(12, "one"))
    await db.save_message(make_topic_message(13, "two", date=STAMP + 1))
    await db.save_message(make_topic_message(14, "elsewhere", 34, date=STAMP + 2))
    await db.mark_messages_read(-1001, 12, 12)
    rows = await db.messages_since_read(-1001, 10, message_thread_id=12)
    assert [row["message_id"] for row in rows] == [13]
    rows = await db.messages_since_read(-1001, 10, message_thread_id=34)
    assert [row["message_id"] for row in rows] == [14]


async def test_recent_messages_pagination(db: Database) -> None:
    """before_message_id pages into the past, excluding the cursor itself."""
    for i in range(1, 6):
        await db.save_message(make_message(i, f"msg {i}", date=STAMP + i))
    page = await db.recent_messages(100, 2, before_message_id=4)
    assert [row["message_id"] for row in page] == [2, 3]


async def test_unread_count_tracks_whole_chat_and_topic_reads(db: Database) -> None:
    """Read cursors persist independently for a whole chat and each topic."""
    await db.save_message(make_topic_message(12, None))
    await db.save_message(make_topic_message(13, "one", date=STAMP + 1))
    await db.save_message(make_topic_message(14, "two", date=STAMP + 2))
    assert await db.unread_messages_count(-1001, 12) == 3
    await db.mark_messages_read(-1001, 14, 12)
    assert await db.unread_messages_count(-1001, 12) == 0
    assert await db.unread_messages_count(-1001) == 3
    await db.save_message(make_topic_message(15, "three", date=STAMP + 3))
    assert await db.unread_messages_count(-1001, 12) == 1


async def test_unread_count_ignores_the_agents_own_messages(db: Database) -> None:
    """Replies the agent sent itself are never reported back as unread."""
    await db.save_message(make_message(1, "theirs"))
    await db.save_message(make_message(2, "ours", date=STAMP + 60), outgoing=True)
    assert await db.unread_messages_count(100) == 1


async def test_whole_chat_read_clears_topic_unread_counts(db: Database) -> None:
    """A chat-wide history read also exposed each topic's older messages."""
    await db.save_message(make_topic_message(13, "one", date=STAMP + 1))
    await db.save_message(make_topic_message(14, "two", thread_id=99, date=STAMP + 2))
    await db.mark_messages_read(-1001, 14)
    assert await db.unread_messages_count(-1001, 12) == 0
    assert await db.unread_messages_count(-1001, 99) == 0
    await db.save_message(make_topic_message(15, "three", date=STAMP + 3))
    assert await db.unread_messages_count(-1001, 12) == 1


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


#: The forum supergroup used by topic tests.
FORUM_CHAT = {"id": -1001, "type": "supergroup", "title": "Hub", "is_forum": True}


def topic_creation(thread_id: int, name: str = "Ideas") -> dict[str, Any]:
    """Build the topic-creation service payload topic messages "reply" to."""
    return {
        "message_id": thread_id,
        "date": STAMP,
        "chat": FORUM_CHAT,
        "forum_topic_created": {"name": name, "icon_color": 0x6FB9F0},
    }


def make_topic_message(
    message_id: int,
    text: str | None,
    thread_id: int = 12,
    date: int = STAMP,
    **overrides: Any,
) -> Message:
    """Build a forum-topic message carrying Telegram's pseudo-reply."""
    data: dict[str, Any] = {
        "chat": FORUM_CHAT,
        "message_thread_id": thread_id,
        "is_topic_message": True,
        "reply_to_message": topic_creation(thread_id),
    }
    data.update(overrides)
    return make_message(message_id, text, date=date, **data)


def make_topic_service(
    message_id: int, thread_id: int, date: int = STAMP, **service: Any
) -> Message:
    """Build a forum topic service message (created/edited/closed/reopened)."""
    return make_topic_message(
        message_id, None, thread_id, date, reply_to_message=None, **service
    )


async def test_save_message_stores_topic_id(db: Database) -> None:
    """Topic messages store their topic id; plain messages store NULL."""
    await db.save_message(make_topic_message(43, "in topic"))
    await db.save_message(make_message(1, "plain", date=STAMP))
    async with db.conn.execute(
        "SELECT chat_id, message_id, message_thread_id FROM messages"
    ) as cursor:
        rows = {(r[0], r[1]): r[2] for r in await cursor.fetchall()}
    assert rows[(-1001, 43)] == 12
    assert rows[(100, 1)] is None


async def test_save_message_suppresses_topic_pseudo_reply(db: Database) -> None:
    """The pseudo-reply is dropped; a real in-topic reply is kept."""
    await db.save_message(make_topic_message(43, "not a reply"))
    real_parent = {"message_id": 43, "date": STAMP, "chat": FORUM_CHAT}
    await db.save_message(
        make_topic_message(44, "a real reply", reply_to_message=real_parent)
    )
    async with db.conn.execute(
        "SELECT message_id, reply_to_message_id FROM messages"
    ) as cursor:
        rows = {row[0]: row[1] for row in await cursor.fetchall()}
    assert rows[43] is None
    assert rows[44] == 43


async def test_message_thread_ignores_topic_pseudo_replies(db: Database) -> None:
    """A reply chain inside a topic does not sweep in the whole topic."""
    await db.save_message(make_topic_message(43, "root"))
    await db.save_message(make_topic_message(44, "unrelated", date=STAMP + 60))
    real_parent = {"message_id": 43, "date": STAMP, "chat": FORUM_CHAT}
    await db.save_message(
        make_topic_message(45, "reply", date=STAMP + 120, reply_to_message=real_parent)
    )
    rows = await db.message_thread(-1001, 45, 10)
    assert [row["message_id"] for row in rows] == [43, 45]


async def test_recent_messages_filters_by_topic(db: Database) -> None:
    """The topic filter narrows history; rows expose their topic id."""
    await db.save_message(make_topic_message(43, "in 12"))
    await db.save_message(
        make_topic_message(44, "in 13", thread_id=13, date=STAMP + 60)
    )
    all_rows = await db.recent_messages(-1001, 10)
    assert [row["message_id"] for row in all_rows] == [43, 44]
    assert [row["message_thread_id"] for row in all_rows] == [12, 13]
    topic_rows = await db.recent_messages(-1001, 10, message_thread_id=12)
    assert [row["message_id"] for row in topic_rows] == [43]


async def test_search_messages_filters_by_topic(db: Database) -> None:
    """Search restricted to a topic skips matches in other topics."""
    await db.save_message(make_topic_message(43, "cat here"))
    await db.save_message(
        make_topic_message(44, "cat there", thread_id=13, date=STAMP + 60)
    )
    rows = await db.search_messages(-1001, "cat", 10, message_thread_id=13)
    assert [row["message_id"] for row in rows] == [44]


async def test_topic_name_prefers_latest_rename(db: Database) -> None:
    """Renames beat the creation-time name pseudo-replies keep echoing."""
    await db.save_message(
        make_topic_service(
            12, 12, forum_topic_created={"name": "Ideas", "icon_color": 0x6FB9F0}
        )
    )
    await db.save_message(make_topic_message(43, "chat", date=STAMP + 60))
    assert await db.topic_name(-1001, 12) == "Ideas"
    await db.save_message(
        make_topic_service(
            44, 12, date=STAMP + 120, forum_topic_edited={"name": "Plans"}
        )
    )
    # An icon-only edit carries no name and must not win.
    await db.save_message(
        make_topic_service(
            45, 12, date=STAMP + 180, forum_topic_edited={"icon_custom_emoji_id": "x"}
        )
    )
    assert await db.topic_name(-1001, 12) == "Plans"


async def test_topic_name_falls_back_to_pseudo_reply(db: Database) -> None:
    """A topic created before the bot joined still gets its name."""
    await db.save_message(make_topic_message(43, "hello", thread_id=13))
    assert await db.topic_name(-1001, 13) == "Ideas"
    assert await db.topic_name(-1001, 999) is None


async def test_topic_observed(db: Database) -> None:
    """Only topics with stored messages count as observed."""
    await db.save_message(make_topic_message(43, "hello"))
    assert await db.topic_observed(-1001, 12) is True
    assert await db.topic_observed(-1001, 13) is False
    assert await db.topic_observed(100, 12) is False


async def test_list_topics(db: Database) -> None:
    """Topics list with name, count, activity order and closed flag."""
    await db.save_message(make_topic_message(43, "one"))
    await db.save_message(make_topic_message(44, "two", date=STAMP + 60))
    await db.save_message(
        make_topic_message(
            45,
            "newer",
            thread_id=13,
            date=STAMP + 120,
            reply_to_message=topic_creation(13, name="Chatter"),
        )
    )
    await db.save_message(
        make_topic_service(46, 13, date=STAMP + 180, forum_topic_closed={})
    )
    await db.save_message(make_message(1, "no topic", date=STAMP))
    topics = await db.list_topics(-1001)
    assert [t["topic_id"] for t in topics] == [13, 12]
    assert [t["name"] for t in topics] == ["Chatter", "Ideas"]
    assert [t["closed"] for t in topics] == [True, False]
    assert topics[1]["messages"] == 2
    assert await db.list_topics(100) == []


async def test_list_topics_reopened_clears_closed(db: Database) -> None:
    """The newest close/reopen service message decides the closed flag."""
    await db.save_message(make_topic_message(43, "hello"))
    await db.save_message(
        make_topic_service(44, 12, date=STAMP + 60, forum_topic_closed={})
    )
    await db.save_message(
        make_topic_service(45, 12, date=STAMP + 120, forum_topic_reopened={})
    )
    (topic,) = await db.list_topics(-1001)
    assert topic["closed"] is False


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
    message_id: int,
    file_id: str,
    unique_id: str,
    date: int,
    chat_id: int = 100,
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
    chat = {"id": chat_id, "type": "private", "first_name": "Alice"}
    return make_message(message_id, None, date=date, sticker=sticker, chat=chat)


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


async def test_known_stickers_can_be_scoped_to_chats(db: Database) -> None:
    """Sticker discovery excludes every chat outside its allowlist."""
    await db.save_message(make_sticker_message(1, "AAA", "u1", STAMP))
    await db.save_message(make_sticker_message(2, "BBB", "u2", STAMP + 60, chat_id=200))
    assert [row["file_id"] for row in await db.known_stickers(10, [100])] == ["AAA"]
    assert await db.known_stickers(10, []) == []
    assert await db.sticker_is_known("AAA", [100]) is True
    assert await db.sticker_is_known("BBB", [100]) is False


async def test_message_exists(db: Database) -> None:
    """Message presence is scoped by both chat and message id."""
    await db.save_message(make_message(1, "hi"))
    assert await db.message_exists(100, 1) is True
    assert await db.message_exists(100, 2) is False
    assert await db.message_exists(200, 1) is False


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


async def test_unanswered_chats_per_topic(db: Database) -> None:
    """A bot reply in one topic does not answer another topic."""
    await db.save_message(make_topic_message(43, "anyone?"))
    await db.save_message(
        make_topic_message(44, "sure", thread_id=13, date=STAMP + 60),
        outgoing=True,
    )
    (row,) = await db.unanswered_chats()
    assert (row["chat_id"], row["message_thread_id"]) == (-1001, 12)
    await db.save_message(
        make_topic_message(45, "done", date=STAMP + 120), outgoing=True
    )
    assert await db.unanswered_chats() == []


async def test_unanswered_chats_ignores_topic_service_messages(db: Database) -> None:
    """A freshly created, never-used topic does not nag forever."""
    await db.save_message(
        make_topic_service(
            12, 12, forum_topic_created={"name": "Ideas", "icon_color": 0x6FB9F0}
        )
    )
    assert await db.unanswered_chats() == []


async def test_topic_service_message_does_not_mask_unanswered_message(
    db: Database,
) -> None:
    """A later topic rename does not count as an answer to a message."""
    await db.save_message(make_topic_message(43, "anyone?"))
    await db.save_message(
        make_topic_service(
            44, 12, date=STAMP + 60, forum_topic_edited={"name": "Plans"}
        )
    )

    (row,) = await db.unanswered_chats()
    assert (row["chat_id"], row["message_thread_id"]) == (-1001, 12)


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


async def test_context_accepts_unpaired_provider_surrogate(db: Database) -> None:
    """Broken provider Unicode cannot wedge append-only persistence."""
    item = {"type": "function_call", "arguments": "\ud800"}
    await db.append_context(item)
    assert await db.load_context(1) == [item]


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

    assert {"turn_id", "input_context_id", "dream_id"} <= columns


async def test_messages_schema_migrates_and_backfills(tmp_path: Path) -> None:
    """Connecting adds the topic column, backfills it and strips pseudo-replies.

    Also pins the ordering the topic index depends on: it names a column
    the migration adds, so creating it from ``SCHEMA`` would fail to open
    every database written before that column existed.
    """
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE chats (
            id INTEGER PRIMARY KEY, type TEXT NOT NULL, title TEXT,
            username TEXT, raw TEXT NOT NULL,
            first_seen_at TEXT NOT NULL DEFAULT (datetime('now')),
            last_seen_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE messages (
            chat_id INTEGER NOT NULL REFERENCES chats(id),
            message_id INTEGER NOT NULL,
            from_user_id INTEGER,
            date TEXT NOT NULL,
            edit_date TEXT,
            content_type TEXT NOT NULL,
            text TEXT,
            caption TEXT,
            reply_to_message_id INTEGER,
            media_group_id TEXT,
            outgoing INTEGER NOT NULL DEFAULT 0,
            raw TEXT NOT NULL,
            saved_at TEXT NOT NULL DEFAULT (datetime('now')),
            PRIMARY KEY (chat_id, message_id)
        );
        """
    )
    conn.execute("INSERT INTO chats (id, type, raw) VALUES (-1001, 'supergroup', '{}')")
    topic_raw = json.dumps(
        {
            "message_id": 43,
            "is_topic_message": True,
            "message_thread_id": 12,
            "reply_to_message": {
                "message_id": 12,
                "forum_topic_created": {"name": "Ideas", "icon_color": 1},
            },
        }
    )
    conn.execute(
        "INSERT INTO messages (chat_id, message_id, date, content_type,"
        " reply_to_message_id, raw) VALUES (-1001, 43, 'd', 'text', 12, ?)",
        (topic_raw,),
    )
    conn.execute(
        "INSERT INTO messages (chat_id, message_id, date, content_type,"
        " reply_to_message_id, raw) VALUES (-1001, 44, 'd', 'text', 43,"
        " '{\"message_id\": 44}')"
    )
    conn.commit()
    conn.close()

    database = Database(path)
    await database.connect()
    async with database.conn.execute(
        "SELECT message_id, message_thread_id, reply_to_message_id FROM messages"
    ) as cursor:
        rows = {r[0]: (r[1], r[2]) for r in await cursor.fetchall()}
    async with database.conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'index'"
    ) as cursor:
        indexes = {r[0] for r in await cursor.fetchall()}
    await database.close()

    assert rows[43] == (12, None)
    assert rows[44] == (None, 43)
    assert "idx_messages_chat_thread" in indexes


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


async def test_dream_context_round_trip(db: Database) -> None:
    """Dreaming context is stored per dream and apart from the waking one."""
    first = await db.start_dream("idle")
    second = await db.start_dream("requested")
    await db.append_dream_context(first, {"role": "user", "content": "asleep"})
    await db.append_dream_context(second, {"type": "reasoning"})

    async with db.conn.execute(
        "SELECT dream_id, item FROM dream_context ORDER BY id"
    ) as cursor:
        rows = list(await cursor.fetchall())

    assert [row["dream_id"] for row in rows] == [first, second]
    assert json.loads(rows[0]["item"])["content"] == "asleep"
    assert await db.latest_dream_context_id() == 2
    assert await db.latest_context_id() == 0


async def test_latest_dream_context_id_is_zero_when_empty(db: Database) -> None:
    """An unslept database anchors dreaming usage at zero."""
    assert await db.latest_dream_context_id() == 0


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
