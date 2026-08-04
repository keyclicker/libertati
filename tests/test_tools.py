"""Tests for the toolbox: dispatch, error handling and argument limits."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from zoneinfo import ZoneInfo

import pytest
from aiogram import Bot
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest
from openai import AsyncOpenAI, Omit

from libertati.chats import ChatRegistry
from libertati.db import Database
from libertati.memory import DEFAULT_SOUL, MEMORY_MAX_CHARS, SOUL_MAX_CHARS, Mind
from libertati.tools import (
    DREAM_API_TOOLS,
    DREAM_TOOL_NAMES,
    GATED_CHAT_ARGS,
    MESSAGING_TOOLS,
    SLEEP_TOOLS,
    TOOLS,
    TYPING_MAX_SECONDS,
    TYPING_MIN_SECONDS,
    Toolbox,
    build_dream_tools,
    build_tools,
    function_names,
    strip_citation_artifacts,
    typing_delay,
)

UTC_TZ = ZoneInfo("UTC")
RECALL_PROMPT = "test recall prompt"
SUMMARY_PROMPT = "test summary prompt"

#: Registry with approval mode off — allows everything, touches no file.
OPEN_REGISTRY = ChatRegistry(Path("unused-chats.toml"), enabled=False)


class FakeDB:
    """Records calls made by tool handlers."""

    def __init__(self) -> None:
        """Start with empty call records."""
        self.recent_calls: list[tuple[int, int, int | None, int | None]] = []
        self.read_count = 0
        self.read_marks: list[tuple[int, int, int | None]] = []
        self.recent_rows: list[dict] = []
        self.search_calls: list[tuple[int, str, int, int | None]] = []
        self.thread_calls: list[tuple[int, int, int]] = []
        self.wakeups: list[tuple[str, str]] = []
        self.pending: list[dict] = []
        self.chats: list[dict] = []
        self.members: list[dict] = []
        self.stickers: list[dict] = []
        self.known_sticker_ids: set[str] = set()
        self.stored_rows: set[tuple[int, int]] = set()
        self.deleted: list[tuple[int, int]] = []
        self.outgoing_rows: set[tuple[int, int]] = set()
        self.topics: list[dict] = []
        self.observed_topics: set[tuple[int, int]] = set()

    async def recent_messages(
        self,
        chat_id: int,
        limit: int,
        before_message_id: int | None = None,
        message_thread_id: int | None = None,
    ) -> list[dict]:
        """Record the query and return no rows."""
        self.recent_calls.append((chat_id, limit, before_message_id, message_thread_id))
        return self.recent_rows

    async def unread_messages_count(
        self, chat_id: int, message_thread_id: int | None = None
    ) -> int:
        """Return canned unread count."""
        return self.read_count

    async def mark_messages_read(
        self,
        chat_id: int,
        message_id: int,
        message_thread_id: int | None = None,
    ) -> None:
        """Record history cursor advancement."""
        self.read_marks.append((chat_id, message_id, message_thread_id))

    async def search_messages(
        self,
        chat_id: int,
        needle: str,
        limit: int,
        message_thread_id: int | None = None,
    ) -> list[dict]:
        """Record the query and return no rows."""
        self.search_calls.append((chat_id, needle, limit, message_thread_id))
        return []

    async def list_topics(self, chat_id: int) -> list[dict]:
        """Return the canned topic list."""
        return self.topics

    async def topic_observed(self, chat_id: int, thread_id: int) -> bool:
        """Report whether a topic is in the canned observed set."""
        return (chat_id, thread_id) in self.observed_topics

    async def message_thread(
        self, chat_id: int, message_id: int, limit: int
    ) -> list[dict]:
        """Record the query and return no rows."""
        self.thread_calls.append((chat_id, message_id, limit))
        return []

    async def list_chats(self) -> list[dict]:
        """Return the canned chat list."""
        return self.chats

    async def chat_members(self, chat_id: int) -> list[dict]:
        """Return the canned member list."""
        return self.members

    async def known_stickers(
        self, limit: int, chat_ids: list[int] | None = None
    ) -> list[dict]:
        """Return the canned sticker list."""
        return self.stickers

    async def sticker_is_known(self, file_id: str, chat_ids: list[int]) -> bool:
        """Report whether a sticker is in the canned approved set."""
        return bool(chat_ids) and file_id in self.known_sticker_ids

    async def message_exists(self, chat_id: int, message_id: int) -> bool:
        """Report whether a message is in the canned stored set."""
        return (chat_id, message_id) in self.stored_rows

    async def save_message(self, message: Any, outgoing: bool = False) -> None:
        """Accept saved messages silently."""

    async def delete_message(self, chat_id: int, message_id: int) -> None:
        """Record the deletion."""
        self.deleted.append((chat_id, message_id))

    async def message_is_outgoing(self, chat_id: int, message_id: int) -> bool:
        """Report ownership from the canned outgoing set."""
        return (chat_id, message_id) in self.outgoing_rows

    async def add_wakeup(self, due_at: str, note: str) -> int:
        """Record the wakeup and return a fixed id."""
        self.wakeups.append((due_at, note))
        return 7

    async def pending_wakeups(self) -> list[dict]:
        """Return the canned pending-wakeup rows."""
        return self.pending

    async def cancel_wakeup(self, wakeup_id: int) -> bool:
        """Pretend only wakeup #7 exists."""
        return wakeup_id == 7


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


class MarkdownRejectingBot:
    """A bot that rejects markdown parse mode but accepts plain text."""

    id = 1

    def __init__(self) -> None:
        """Start with no sends recorded."""
        self.parse_modes: list[Any] = []

    async def send_chat_action(self, *args: Any, **kwargs: Any) -> bool:
        """Accept typing indicators silently."""
        return True

    async def send_message(
        self, chat_id: int, text: str, *, parse_mode: Any = None, **kwargs: Any
    ) -> Any:
        """Reject any parse mode as Telegram does for unbalanced markup."""
        self.parse_modes.append(parse_mode)
        if parse_mode is not None:
            raise TelegramBadRequest(
                method=cast(Any, None), message="can't parse entities"
            )
        return SimpleNamespace(message_id=5, chat=SimpleNamespace(id=chat_id))


class RecordingBot:
    """A bot that records message actions and succeeds."""

    id = 1

    def __init__(self) -> None:
        """Start with no calls recorded."""
        self.sent_messages: list[dict[str, Any]] = []
        self.reactions: list[tuple[int, int, list]] = []
        self.edits: list[dict[str, Any]] = []
        self.deletes: list[tuple[int, int]] = []
        self.sent_stickers: list[tuple[int, str, int | None]] = []
        self.forwards: list[tuple[int, int, int, int | None]] = []
        self.chat_info: Any = None
        self.member_count = 0
        self.admins: list[Any] = []

    async def send_chat_action(self, *args: Any, **kwargs: Any) -> bool:
        """Accept typing indicators silently."""
        return True

    async def send_message(self, chat_id: int, text: str, **kwargs: Any) -> Any:
        """Record a sent message and return its minimal Telegram shape."""
        self.sent_messages.append({"chat_id": chat_id, "text": text, **kwargs})
        return SimpleNamespace(message_id=5, chat=SimpleNamespace(id=chat_id))

    async def set_message_reaction(
        self, chat_id: int, message_id: int, reaction: list | None = None
    ) -> bool:
        """Record the reaction change."""
        self.reactions.append((chat_id, message_id, reaction or []))
        return True

    async def edit_message_text(self, **kwargs: Any) -> Any:
        """Record the edit; return a non-Message like inline edits do."""
        self.edits.append(kwargs)
        return True

    async def delete_message(self, chat_id: int, message_id: int) -> bool:
        """Record the deletion."""
        self.deletes.append((chat_id, message_id))
        return True

    async def send_sticker(
        self, chat_id: int, file_id: str, message_thread_id: int | None = None
    ) -> Any:
        """Record the sticker send and return a minimal sent message."""
        self.sent_stickers.append((chat_id, file_id, message_thread_id))
        return SimpleNamespace(message_id=6, chat=SimpleNamespace(id=chat_id))

    async def forward_message(
        self,
        chat_id: int,
        from_chat_id: int,
        message_id: int,
        message_thread_id: int | None = None,
    ) -> Any:
        """Record the forward and return a minimal sent message."""
        self.forwards.append((chat_id, from_chat_id, message_id, message_thread_id))
        return SimpleNamespace(message_id=9, chat=SimpleNamespace(id=chat_id))

    async def get_chat(self, chat_id: int) -> Any:
        """Return the canned chat profile."""
        return self.chat_info

    async def get_chat_member_count(self, chat_id: int) -> int:
        """Return the canned member count."""
        return self.member_count

    async def get_chat_administrators(self, chat_id: int) -> list[Any]:
        """Return the canned admin list."""
        return self.admins


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
    registry: ChatRegistry | None = None,
    **dream: Any,
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
        RECALL_PROMPT,
        SUMMARY_PROMPT,
        registry=registry or OPEN_REGISTRY,
        **dream,
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


@pytest.mark.parametrize(
    "arguments",
    [
        "null",
        "[]",
        json.dumps(
            {
                "chat_id": "@unguarded",
                "text": "hi",
                "reply_to_message_id": None,
                "message_thread_id": None,
            }
        ),
        json.dumps(
            {
                "chat_id": True,
                "text": "hi",
                "reply_to_message_id": None,
                "message_thread_id": None,
            }
        ),
        json.dumps({"chat_id": 1, "text": "hi"}),
        json.dumps(
            {
                "chat_id": 1,
                "text": "hi",
                "reply_to_message_id": None,
                "message_thread_id": None,
                "extra": "field",
            }
        ),
    ],
)
async def test_arguments_are_validated_locally(arguments: str) -> None:
    """Provider schema violations fail closed before reaching handlers."""
    bot = RecordingBot()
    result = await make_toolbox(bot=bot).run("send_message", arguments)
    assert result == "error: invalid tool arguments"
    assert bot.sent_messages == []


async def test_invalid_utf8_is_refused_before_file_writes(tmp_path: Path) -> None:
    """A lone JSON surrogate cannot truncate a model-managed file."""
    mind = make_mind(tmp_path)
    before = mind.soul()
    toolbox = make_dream_toolbox(mind)
    result = await toolbox.run("write_soul", '{"text": "\\ud800"}')
    assert result == "error: invalid tool arguments"
    assert mind.soul() == before


async def test_handler_exception_is_wrapped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A raising handler is reported as an error, not propagated."""
    monkeypatch.setattr("libertati.tools.typing_delay", lambda text, cps: 0.0)
    args = json.dumps(
        {
            "chat_id": 1,
            "text": "hi",
            "reply_to_message_id": None,
            "message_thread_id": None,
        }
    )
    result = await make_toolbox().run("send_message", args)
    assert result == "error: send_message failed"


async def test_send_message_falls_back_to_plain_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Markdown rejected by Telegram is resent as plain text, not lost."""
    monkeypatch.setattr("libertati.tools.typing_delay", lambda text, cps: 0.0)
    bot = MarkdownRejectingBot()
    args = json.dumps(
        {
            "chat_id": 1,
            "text": "a_b",
            "reply_to_message_id": None,
            "message_thread_id": None,
        }
    )
    result = await make_toolbox(bot=bot).run("send_message", args)
    assert result.startswith("sent message 5 to chat 1")
    assert bot.parse_modes == [ParseMode.MARKDOWN, None]


async def test_send_message_preserves_username_underscores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Markdown escaping keeps underscores visible in Telegram mentions."""
    monkeypatch.setattr("libertati.tools.typing_delay", lambda text, cps: 0.0)
    bot = RecordingBot()
    args = json.dumps(
        {
            "chat_id": 1,
            "text": "hi @hermes_keyclicker_bot",
            "reply_to_message_id": None,
            "message_thread_id": None,
        }
    )
    result = await make_toolbox(bot=bot).run("send_message", args)
    assert result.startswith("sent message 5 to chat 1")
    (sent,) = bot.sent_messages
    assert sent["text"] == r"hi @hermes\_keyclicker\_bot"
    assert sent["parse_mode"] == ParseMode.MARKDOWN


def test_strip_citation_artifacts_handles_search_marker_formats() -> None:
    """Internal search references never become visible Telegram text."""
    text = (
        "Ukraine <cite|turn0search1|turn0search5> and Crimea "
        "\ue200cite\ue202turn1search2\ue201."
    )
    assert strip_citation_artifacts(text) == "Ukraine and Crimea."


def test_strip_citation_artifacts_removes_closing_tags() -> None:
    """A closing </cite> is an artifact too; leaving it half-strips the text."""
    assert strip_citation_artifacts("Kyiv <cite|turn0search1> is warm </cite>") == (
        "Kyiv is warm"
    )


async def test_send_message_strips_citation_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Search markers are removed at the outward-message boundary."""
    monkeypatch.setattr("libertati.tools.typing_delay", lambda text, cps: 0.0)
    bot = RecordingBot()
    args = json.dumps(
        {
            "chat_id": 1,
            "text": "Claim.<cite|turn0search1|turn0search5>",
            "reply_to_message_id": None,
            "message_thread_id": None,
        }
    )
    await make_toolbox(bot=bot).run("send_message", args)
    assert bot.sent_messages[0]["text"] == "Claim."


async def test_send_message_refuses_unobserved_reply_target() -> None:
    """Replying cannot target a message hidden from local history."""
    bot = RecordingBot()
    args = {
        "chat_id": 1,
        "text": "hi",
        "reply_to_message_id": 99,
        "message_thread_id": None,
    }
    result = await make_toolbox(bot=bot).run("send_message", json.dumps(args))
    assert result == "error: message 99 in chat 1 was not observed"
    assert bot.sent_messages == []


async def test_react_sets_and_removes() -> None:
    """An emoji sets a reaction; null clears it."""
    bot = RecordingBot()
    db = FakeDB()
    db.stored_rows = {(1, 2)}
    toolbox = make_toolbox(db=db, bot=bot)
    args = {"chat_id": 1, "message_id": 2, "emoji": "👍"}
    result = await toolbox.run("react", json.dumps(args))
    assert result == "reacted 👍 to message 2 in chat 1"
    result = await toolbox.run("react", json.dumps({**args, "emoji": None}))
    assert result == "reaction removed from message 2 in chat 1"
    (set_call, clear_call) = bot.reactions
    assert set_call[2][0].emoji == "👍"
    assert clear_call[2] == []


async def test_react_refuses_unobserved_message() -> None:
    """A reaction cannot probe or target a message absent from history."""
    bot = RecordingBot()
    args = {"chat_id": 1, "message_id": 2, "emoji": "👍"}
    result = await make_toolbox(bot=bot).run("react", json.dumps(args))
    assert result == "error: message 2 in chat 1 was not observed"
    assert bot.reactions == []


async def test_edit_message() -> None:
    """Edits go out with markdown and are confirmed."""
    bot = RecordingBot()
    db = FakeDB()
    db.outgoing_rows = {(1, 2)}
    args = {
        "chat_id": 1,
        "message_id": 2,
        "text": "fixed<cite|turn0search1>",
    }
    result = await make_toolbox(db=db, bot=bot).run("edit_message", json.dumps(args))
    assert result == "edited message 2 in chat 1"
    (edit,) = bot.edits
    assert edit["text"] == "fixed"
    assert edit["parse_mode"] == ParseMode.MARKDOWN


async def test_edit_message_refuses_other_peoples_messages() -> None:
    """An edit must target a stored outgoing message."""
    bot = RecordingBot()
    args = {"chat_id": 1, "message_id": 2, "text": "hijack"}
    result = await make_toolbox(bot=bot).run("edit_message", json.dumps(args))
    assert result.startswith("error:")
    assert bot.edits == []


async def test_delete_message() -> None:
    """Deleting an own message hits Telegram and removes the stored row."""
    bot = RecordingBot()
    db = FakeDB()
    db.outgoing_rows = {(1, 2)}
    args = {"chat_id": 1, "message_id": 2}
    result = await make_toolbox(db=db, bot=bot).run("delete_message", json.dumps(args))
    assert result == "deleted message 2 in chat 1"
    assert bot.deletes == [(1, 2)]
    assert db.deleted == [(1, 2)]


async def test_delete_message_refuses_other_peoples_messages() -> None:
    """A message the bot didn't send is refused before reaching Telegram."""
    bot = RecordingBot()
    db = FakeDB()
    args = {"chat_id": 1, "message_id": 2}
    result = await make_toolbox(db=db, bot=bot).run("delete_message", json.dumps(args))
    assert result.startswith("error:")
    assert bot.deletes == []
    assert db.deleted == []


async def test_send_sticker() -> None:
    """Stickers go out by file_id and are confirmed with the message id."""
    bot = RecordingBot()
    db = FakeDB()
    db.chats = [{"chat_id": 1}]
    db.known_sticker_ids = {"AAA"}
    args = {"chat_id": 1, "file_id": "AAA", "message_thread_id": None}
    result = await make_toolbox(db=db, bot=bot).run("send_sticker", json.dumps(args))
    assert result == "sent sticker as message 6 to chat 1"
    assert bot.sent_stickers == [(1, "AAA", None)]


async def test_send_sticker_refuses_unknown_file_id() -> None:
    """Description-only sticker restriction is enforced locally."""
    bot = RecordingBot()
    db = FakeDB()
    db.chats = [{"chat_id": 1}]
    result = await make_toolbox(db=db, bot=bot).run(
        "send_sticker",
        json.dumps({"chat_id": 1, "file_id": "UNKNOWN", "message_thread_id": None}),
    )
    assert result == "error: sticker was not observed in an approved chat"
    assert bot.sent_stickers == []


async def test_list_stickers() -> None:
    """Known stickers come back as JSON; none seen yet says so."""
    db = FakeDB()
    db.chats = [{"chat_id": 1}]
    toolbox = make_toolbox(db=db)
    result = await toolbox.run("list_stickers", "{}")
    assert result.startswith("no stickers seen yet")
    db.stickers = [{"file_id": "AAA", "emoji": "😀", "set_name": "pack"}]
    result = await toolbox.run("list_stickers", "{}")
    assert json.loads(result) == db.stickers


async def test_forward_message() -> None:
    """Forwards reach the bot with the right chats and are confirmed."""
    bot = RecordingBot()
    db = FakeDB()
    db.stored_rows = {(1, 42)}
    args = {
        "to_chat_id": 2,
        "from_chat_id": 1,
        "message_id": 42,
        "message_thread_id": None,
    }
    result = await make_toolbox(db=db, bot=bot).run("forward_message", json.dumps(args))
    assert result == "forwarded message 42 from chat 1 to chat 2 as message 9"
    assert bot.forwards == [(2, 1, 42, None)]


async def test_forward_message_refuses_unobserved_source_message() -> None:
    """Forwarding cannot retrieve a guessed message id from Telegram."""
    bot = RecordingBot()
    args = {
        "to_chat_id": 2,
        "from_chat_id": 1,
        "message_id": 42,
        "message_thread_id": None,
    }
    result = await make_toolbox(bot=bot).run("forward_message", json.dumps(args))
    assert result == "error: message 42 in chat 1 was not observed"
    assert bot.forwards == []


async def test_send_message_passes_topic(monkeypatch: pytest.MonkeyPatch) -> None:
    """A topic id reaches the bot call and shows in the confirmation."""
    monkeypatch.setattr("libertati.tools.typing_delay", lambda text, cps: 0.0)
    bot = RecordingBot()
    db = FakeDB()
    db.observed_topics = {(1, 12)}
    args = {
        "chat_id": 1,
        "text": "hi",
        "reply_to_message_id": None,
        "message_thread_id": 12,
    }
    result = await make_toolbox(db=db, bot=bot).run("send_message", json.dumps(args))
    assert result.startswith("sent message 5 to chat 1 (topic 12)")
    (sent,) = bot.sent_messages
    assert sent["message_thread_id"] == 12


async def test_send_message_refuses_unobserved_topic() -> None:
    """Sending into a topic the bot never saw fails closed."""
    bot = RecordingBot()
    args = {
        "chat_id": 1,
        "text": "hi",
        "reply_to_message_id": None,
        "message_thread_id": 99,
    }
    result = await make_toolbox(bot=bot).run("send_message", json.dumps(args))
    assert result == "error: topic 99 in chat 1 was not observed"
    assert bot.sent_messages == []


async def test_send_sticker_passes_topic() -> None:
    """A sticker send carries the topic id to the bot call."""
    bot = RecordingBot()
    db = FakeDB()
    db.chats = [{"chat_id": 1}]
    db.known_sticker_ids = {"AAA"}
    db.observed_topics = {(1, 12)}
    args = {"chat_id": 1, "file_id": "AAA", "message_thread_id": 12}
    result = await make_toolbox(db=db, bot=bot).run("send_sticker", json.dumps(args))
    assert result == "sent sticker as message 6 to chat 1"
    assert bot.sent_stickers == [(1, "AAA", 12)]


async def test_send_sticker_refuses_unobserved_topic() -> None:
    """A sticker cannot go into a topic the bot never saw."""
    bot = RecordingBot()
    db = FakeDB()
    db.chats = [{"chat_id": 1}]
    db.known_sticker_ids = {"AAA"}
    args = {"chat_id": 1, "file_id": "AAA", "message_thread_id": 99}
    result = await make_toolbox(db=db, bot=bot).run("send_sticker", json.dumps(args))
    assert result == "error: topic 99 in chat 1 was not observed"
    assert bot.sent_stickers == []


async def test_forward_message_passes_topic() -> None:
    """A forward carries the destination-chat topic id to the bot call."""
    bot = RecordingBot()
    db = FakeDB()
    db.stored_rows = {(1, 42)}
    db.observed_topics = {(2, 12)}
    args = {
        "to_chat_id": 2,
        "from_chat_id": 1,
        "message_id": 42,
        "message_thread_id": 12,
    }
    result = await make_toolbox(db=db, bot=bot).run("forward_message", json.dumps(args))
    assert result == "forwarded message 42 from chat 1 to chat 2 as message 9"
    assert bot.forwards == [(2, 1, 42, 12)]


async def test_forward_message_refuses_unobserved_topic() -> None:
    """The topic pre-check runs against the destination chat."""
    bot = RecordingBot()
    db = FakeDB()
    db.stored_rows = {(1, 42)}
    # Topic 12 exists in the source chat only.
    db.observed_topics = {(1, 12)}
    args = {
        "to_chat_id": 2,
        "from_chat_id": 1,
        "message_id": 42,
        "message_thread_id": 12,
    }
    result = await make_toolbox(db=db, bot=bot).run("forward_message", json.dumps(args))
    assert result == "error: topic 12 in chat 2 was not observed"
    assert bot.forwards == []


async def test_get_recent_messages_passes_topic_filter() -> None:
    """The topic filter reaches the database query."""
    db = FakeDB()
    args = {
        "chat_id": 1,
        "limit": None,
        "before_message_id": None,
        "message_thread_id": 12,
    }
    await make_toolbox(db=db).run("get_recent_messages", json.dumps(args))
    assert db.recent_calls == [(1, 20, None, 12)]


async def test_search_messages_passes_topic_filter() -> None:
    """The topic filter reaches the search query."""
    db = FakeDB()
    args = {"chat_id": 1, "query": "cat", "limit": None, "message_thread_id": 12}
    await make_toolbox(db=db).run("search_messages", json.dumps(args))
    assert db.search_calls == [(1, "cat", 20, 12)]


async def test_list_topics() -> None:
    """Seen topics come back as JSON; a topicless chat says so."""
    db = FakeDB()
    toolbox = make_toolbox(db=db)
    result = await toolbox.run("list_topics", json.dumps({"chat_id": 1}))
    assert result.startswith("no topics seen in this chat")
    db.topics = [
        {
            "topic_id": 12,
            "messages": 3,
            "last_date": "2026-08-02",
            "closed": False,
            "name": "Ideas",
        }
    ]
    result = await toolbox.run("list_topics", json.dumps({"chat_id": 1}))
    assert json.loads(result) == db.topics


async def test_list_topics_is_gated(tmp_path: Path) -> None:
    """An unapproved chat's topics are unreachable."""
    db = FakeDB()
    db.topics = [{"topic_id": 12}]
    toolbox = make_toolbox(
        db=db, registry=make_approving_registry(tmp_path, approved=100)
    )
    result = await toolbox.run("list_topics", json.dumps({"chat_id": 200}))
    assert result == "error: chat 200 is not approved"


async def test_get_chat_info_surfaces_is_forum() -> None:
    """Forum supergroups report is_forum; other chats omit it."""
    bot = RecordingBot()
    bot.chat_info = SimpleNamespace(
        id=-1001,
        type="supergroup",
        title="Hub",
        first_name=None,
        last_name=None,
        username=None,
        bio=None,
        description=None,
        is_forum=True,
    )
    result = await make_toolbox(bot=bot).run(
        "get_chat_info", json.dumps({"chat_id": -1001})
    )
    assert json.loads(result)["is_forum"] is True


async def test_get_chat_info_private() -> None:
    """Private chats return profile fields only, with nulls dropped."""
    bot = RecordingBot()
    bot.chat_info = SimpleNamespace(
        id=100,
        type="private",
        title=None,
        first_name="Alice",
        last_name=None,
        username="alice",
        bio="just a cat person",
        description=None,
        is_forum=None,
    )
    result = await make_toolbox(bot=bot).run(
        "get_chat_info", json.dumps({"chat_id": 100})
    )
    assert json.loads(result) == {
        "chat_id": 100,
        "type": "private",
        "first_name": "Alice",
        "username": "alice",
        "bio": "just a cat person",
    }


async def test_get_chat_info_group_adds_members_and_admins() -> None:
    """Group chats include the member count and admin names."""
    bot = RecordingBot()
    bot.chat_info = SimpleNamespace(
        id=-500,
        type="group",
        title="friends",
        first_name=None,
        last_name=None,
        username=None,
        bio=None,
        description="the gang",
        is_forum=None,
    )
    bot.member_count = 5
    bot.admins = [
        SimpleNamespace(user=SimpleNamespace(full_name="Bob", username="bob"))
    ]
    result = await make_toolbox(bot=bot).run(
        "get_chat_info", json.dumps({"chat_id": -500})
    )
    info = json.loads(result)
    assert info["member_count"] == 5
    assert info["admins"] == ["Bob @bob"]
    assert info["description"] == "the gang"


async def test_list_chat_speakers() -> None:
    """Seen speakers come back as JSON; an empty chat says so."""
    db = FakeDB()
    toolbox = make_toolbox(db=db)
    result = await toolbox.run("list_chat_speakers", json.dumps({"chat_id": 1}))
    assert result == "no speakers observed in stored chat history yet"
    db.members = [{"user_id": 7, "first_name": "Alice", "messages": 3}]
    result = await toolbox.run("list_chat_speakers", json.dumps({"chat_id": 1}))
    assert json.loads(result) == db.members


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


async def test_list_wakeups() -> None:
    """Pending wakeups come back with local due times; empty says so."""
    db = FakeDB()
    toolbox = make_toolbox(db=db)
    assert await toolbox.run("list_wakeups", "{}") == "no pending wakeups"
    db.pending = [{"id": 7, "due_at": "2026-08-02 10:00:00", "note": "ping"}]
    result = await toolbox.run("list_wakeups", "{}")
    assert json.loads(result) == [
        {"id": 7, "due": "Sun 2026-08-02 10:00", "note": "ping"}
    ]


async def test_cancel_wakeup() -> None:
    """Cancelling confirms for a pending id and errors for unknown ones."""
    toolbox = make_toolbox()
    ok = await toolbox.run("cancel_wakeup", json.dumps({"wakeup_id": 7}))
    assert ok == "wakeup #7 cancelled"
    missing = await toolbox.run("cancel_wakeup", json.dumps({"wakeup_id": 8}))
    assert missing == "error: no pending wakeup #8"


async def test_get_recent_messages_clamps_limit() -> None:
    """The limit is clamped to [1, 50]; null falls back to 20."""
    db = FakeDB()
    toolbox = make_toolbox(db=db)
    for limit, expected in ((999, 50), (None, 20), (-5, 1)):
        args = {
            "chat_id": 1,
            "limit": limit,
            "before_message_id": None,
            "message_thread_id": None,
        }
        await toolbox.run("get_recent_messages", json.dumps(args))
        assert db.recent_calls[-1] == (1, expected, None, None)


async def test_get_recent_messages_passes_cursor() -> None:
    """The pagination cursor reaches the database query."""
    db = FakeDB()
    args = {
        "chat_id": 1,
        "limit": None,
        "before_message_id": 42,
        "message_thread_id": None,
    }
    await make_toolbox(db=db).run("get_recent_messages", json.dumps(args))
    assert db.recent_calls == [(1, 20, 42, None)]
    assert db.read_marks == []


async def test_unread_count_and_recent_read_cursor() -> None:
    """Unread count is exposed and newest-history reads advance its cursor."""
    db = FakeDB()
    db.read_count = 3
    db.recent_rows = [{"message_id": 9}]
    toolbox = make_toolbox(db=db)
    count_args = {"chat_id": 1, "message_thread_id": 12}
    assert await toolbox.run("get_unread_messages_count", json.dumps(count_args)) == "3"
    recent_args = {
        "chat_id": 1,
        "limit": 3,
        "before_message_id": None,
        "message_thread_id": 12,
    }
    await toolbox.run("get_recent_messages", json.dumps(recent_args))
    assert db.read_marks == [(1, 9, 12)]


async def test_dreaming_history_reads_leave_the_cursor_alone() -> None:
    """A dream browsing old chats must not zero the waking unread counts."""
    db = FakeDB()
    db.recent_rows = [{"message_id": 9}]
    toolbox = make_toolbox(db=db, track_reads=False)
    recent_args = {
        "chat_id": 1,
        "limit": 3,
        "before_message_id": None,
        "message_thread_id": 12,
    }
    await toolbox.run("get_recent_messages", json.dumps(recent_args))
    assert db.read_marks == []


async def test_get_message_thread_clamps_limit_and_reports_missing() -> None:
    """The limit clamps to [1, 50] and an unknown message says so."""
    db = FakeDB()
    toolbox = make_toolbox(db=db)
    for limit, expected in ((999, 50), (None, 20), (-5, 1)):
        args = {"chat_id": 1, "message_id": 7, "limit": limit}
        result = await toolbox.run("get_message_thread", json.dumps(args))
        assert result == "no such message stored"
        assert db.thread_calls[-1] == (1, 7, expected)


async def test_search_messages_reports_no_matches() -> None:
    """An empty result says so instead of returning bare JSON."""
    db = FakeDB()
    args = {"chat_id": 1, "query": "cat", "limit": None, "message_thread_id": None}
    result = await make_toolbox(db=db).run("search_messages", json.dumps(args))
    assert result == "no matches"
    assert db.search_calls == [(1, "cat", 20, None)]


async def test_list_chats() -> None:
    """Chats come back as JSON; an empty list says so."""
    db = FakeDB()
    assert await make_toolbox(db=db).run("list_chats", "{}") == "no chats yet"
    db.chats = [{"chat_id": 100, "name": "Alice"}]
    result = await make_toolbox(db=db).run("list_chats", "{}")
    assert json.loads(result) == [{"chat_id": 100, "name": "Alice"}]


# ==========================================================
#                    Chat approval gating
# ==========================================================


def make_approving_registry(tmp_path: Path, approved: int) -> ChatRegistry:
    """Build an enabled registry approving exactly one chat."""
    path = tmp_path / "chats.toml"
    path.write_text(f"{approved} = true\n", encoding="utf-8")
    return ChatRegistry(path, enabled=True)


async def test_unapproved_chat_is_refused_before_the_handler(
    tmp_path: Path,
) -> None:
    """No tool reaches Telegram or the DB for an unapproved chat."""
    bot = RecordingBot()
    db = FakeDB()
    toolbox = make_toolbox(
        db=db, bot=bot, registry=make_approving_registry(tmp_path, approved=100)
    )
    send = {
        "chat_id": 200,
        "text": "hi",
        "reply_to_message_id": None,
        "message_thread_id": None,
    }
    result = await toolbox.run("send_message", json.dumps(send))
    assert result == "error: chat 200 is not approved"
    assert bot.sent_messages == []
    history = {
        "chat_id": 200,
        "limit": None,
        "before_message_id": None,
        "message_thread_id": None,
    }
    result = await toolbox.run("get_recent_messages", json.dumps(history))
    assert result == "error: chat 200 is not approved"
    assert db.recent_calls == []
    assert toolbox.steps == 0


async def test_forward_message_gates_both_chats(tmp_path: Path) -> None:
    """A forward is refused when either end is unapproved."""
    bot = RecordingBot()
    toolbox = make_toolbox(
        bot=bot, registry=make_approving_registry(tmp_path, approved=100)
    )
    for args in (
        {
            "to_chat_id": 100,
            "from_chat_id": 200,
            "message_id": 1,
            "message_thread_id": None,
        },
        {
            "to_chat_id": 200,
            "from_chat_id": 100,
            "message_id": 1,
            "message_thread_id": None,
        },
    ):
        result = await toolbox.run("forward_message", json.dumps(args))
        assert result == "error: chat 200 is not approved"
    assert bot.forwards == []


async def test_approved_chat_passes_the_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gate lets approved chats through to the handler."""
    monkeypatch.setattr("libertati.tools.typing_delay", lambda text, cps: 0.0)
    bot = RecordingBot()
    toolbox = make_toolbox(
        bot=bot, registry=make_approving_registry(tmp_path, approved=100)
    )
    send = {
        "chat_id": 100,
        "text": "hi",
        "reply_to_message_id": None,
        "message_thread_id": None,
    }
    result = await toolbox.run("send_message", json.dumps(send))
    assert result.startswith("sent message 5 to chat 100")


async def test_list_chats_hides_unapproved_chats(tmp_path: Path) -> None:
    """Unapproved chats never show up in the chat list."""
    db = FakeDB()
    db.chats = [{"chat_id": 100, "name": "Alice"}, {"chat_id": 200, "name": "Eve"}]
    toolbox = make_toolbox(
        db=db, registry=make_approving_registry(tmp_path, approved=100)
    )
    result = await toolbox.run("list_chats", "{}")
    assert json.loads(result) == [{"chat_id": 100, "name": "Alice"}]


async def test_list_stickers_uses_only_approved_chat_ids(tmp_path: Path) -> None:
    """Sticker discovery cannot expose unapproved chat contents."""

    class RecordingStickerDB(FakeDB):
        def __init__(self) -> None:
            super().__init__()
            self.requested_chat_ids: list[int] | None = None

        async def known_stickers(
            self, limit: int, chat_ids: list[int] | None = None
        ) -> list[dict]:
            self.requested_chat_ids = chat_ids
            return []

    db = RecordingStickerDB()
    db.chats = [{"chat_id": 100}, {"chat_id": 200}]
    toolbox = make_toolbox(
        db=db, registry=make_approving_registry(tmp_path, approved=100)
    )
    await toolbox.run("list_stickers", "{}")
    assert db.requested_chat_ids == [100]


def test_every_chat_targeting_tool_is_gated() -> None:
    """Any schema parameter naming a chat id must be in the gate map.

    Guards the map against new tools that take a chat id but forget to
    register it — the gate is the only thing standing between the model
    and unapproved chats.
    """
    chat_keys = {"chat_id", "from_chat_id", "to_chat_id"}
    for tool in [*TOOLS, *SLEEP_TOOLS, *DREAM_API_TOOLS]:
        if tool["type"] != "function":
            continue
        schema = cast(dict[str, Any], tool)
        expected = tuple(
            key for key in schema["parameters"]["properties"] if key in chat_keys
        )
        gated = GATED_CHAT_ARGS.get(schema["name"], ())
        assert set(gated) == set(expected), schema["name"]


async def test_remember_appends_and_confirms(tmp_path: Path) -> None:
    """A remembered fact lands dated in INBOX.md and is confirmed."""
    mind = make_mind(tmp_path)
    result = await make_toolbox(mind=mind).run(
        "remember", json.dumps({"text": "  cat named Bober  "})
    )
    assert result == "remembered: cat named Bober"
    inbox = mind.inbox_path.read_text(encoding="utf-8")
    assert inbox.startswith("## [")
    assert inbox.endswith("]\ncat named Bober\n\n")


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
    mind.append_inbox("cat named Bober", "Sun 2026-08-02 12:00")
    client = FakeClient()
    toolbox = make_toolbox(client=client, mind=mind)
    result = await toolbox.run("recall", json.dumps({"query": "cat name?"}))
    assert result == "the cat is named Bober"
    (call,) = client.calls
    assert call["model"] == "recall-model"
    assert call["instructions"] == RECALL_PROMPT
    assert "# Inbox" in call["input"]
    assert "cat named Bober" in call["input"]
    assert "cat name?" in call["input"]
    assert call["store"] is False


async def test_recall_omits_reasoning_by_default(tmp_path: Path) -> None:
    """Without a configured effort the call sends no reasoning preference."""
    mind = make_mind(tmp_path)
    mind.append_inbox("cat named Bober", "Sun 2026-08-02 12:00")
    client = FakeClient()
    toolbox = make_toolbox(client=client, mind=mind)
    await toolbox.run("recall", json.dumps({"query": "cat name?"}))
    (call,) = client.calls
    assert isinstance(call["reasoning"], Omit)


async def test_recall_passes_configured_reasoning_effort(tmp_path: Path) -> None:
    """A configured effort rides along on both memory-reading calls."""
    mind = make_mind(tmp_path)
    mind.append_inbox("cat named Bober", "Sun 2026-08-02 12:00")
    client = FakeClient()
    toolbox = make_toolbox(client=client, mind=mind, recall_effort="none")
    await toolbox.run("recall", json.dumps({"query": "cat name?"}))
    await toolbox.run("summarize_memory", "{}")
    assert [call["reasoning"] for call in client.calls] == [
        {"effort": "none"},
        {"effort": "none"},
    ]


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
    mind.append_inbox("cat named Bober", "Sun 2026-08-02 12:00")
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


# ==========================================================
#                    Falling asleep
# ==========================================================


class FakeDreamGate:
    """Stands in for the real gate's budget check and request slot."""

    def __init__(self, left: int = 4, daily_budget: int = 4, cooldown: int = 0) -> None:
        """Start with a fixed budget and no request pending."""
        self.left = left
        self.daily_budget = daily_budget
        self.cooldown = cooldown
        self.note: str | None = None

    async def budget_left(self) -> int:
        """Return the canned remaining budget."""
        return self.left

    async def cooldown_left(self) -> int:
        """Return the canned remaining cooldown."""
        return self.cooldown

    def request(self, note: str) -> None:
        """Record the requested dream."""
        self.note = note


async def test_dream_tool_records_the_request() -> None:
    """A dream request is parked for the dream loop's next poll."""
    gate = FakeDreamGate()
    toolbox = make_toolbox(dream_gate=cast(Any, gate))

    result = await toolbox.run("dream", json.dumps({"note": "the trip"}))

    assert gate.note == "the trip"
    assert "falling asleep" in result
    assert "4 dreams left" in result


async def test_dream_tool_reports_a_spent_budget() -> None:
    """With no budget left the agent is told, not silently ignored."""
    gate = FakeDreamGate(left=0)
    toolbox = make_toolbox(dream_gate=cast(Any, gate))

    result = await toolbox.run("dream", json.dumps({"note": "the trip"}))

    assert result.startswith("error:")
    assert gate.note is None


async def test_dream_tool_reports_an_active_cooldown() -> None:
    """A request is rejected when sleep cannot start during cooldown."""
    gate = FakeDreamGate(cooldown=37)
    toolbox = make_toolbox(dream_gate=cast(Any, gate))

    result = await toolbox.run("dream", json.dumps({"note": "the trip"}))

    assert result == (
        "error: sleep is unavailable while dream cooldown is active (37 min left)"
    )
    assert gate.note is None


async def test_dream_tool_absent_without_a_gate() -> None:
    """Dreaming disabled means the tool exists but refuses."""
    result = await make_toolbox().run("dream", json.dumps({"note": "x"}))
    assert result == "error: dreaming is disabled"


def test_build_tools_dreaming_toggle() -> None:
    """The `dream` tool only appears when a dream budget exists."""
    assert build_tools(False) == TOOLS
    assert build_tools(False, dreaming=True) == [*TOOLS, *SLEEP_TOOLS]


# ==========================================================
#                        Dreaming
# ==========================================================


def make_dream_toolbox(mind: Mind, min_steps: int = 0) -> Toolbox:
    """Build the restricted toolbox the dreaming loop runs with."""
    return make_toolbox(
        mind=mind,
        allowed=DREAM_TOOL_NAMES,
        dream_min_steps=min_steps,
    )


def test_dream_tool_list_excludes_messaging() -> None:
    """A dream has no way to reach another person."""
    names = function_names(build_dream_tools(False))
    assert names.isdisjoint(function_names(MESSAGING_TOOLS))
    assert "wake_up" in names
    assert names == DREAM_TOOL_NAMES


async def test_dream_toolbox_refuses_messaging_by_name(tmp_path: Path) -> None:
    """A hallucinated send_message never reaches its handler."""
    toolbox = make_dream_toolbox(make_mind(tmp_path))

    result = await toolbox.run(
        "send_message",
        json.dumps({"chat_id": 1, "text": "hi", "reply_to_message_id": None}),
    )

    assert result == "error: unknown tool 'send_message'"


async def test_waking_toolbox_refuses_dream_writers(tmp_path: Path) -> None:
    """Undeclared dream-only calls cannot alter waking identity or memory."""
    mind = make_mind(tmp_path)
    toolbox = make_toolbox(mind=mind)

    result = await toolbox.run("write_soul", json.dumps({"text": "hijacked"}))

    assert result == "error: unknown tool 'write_soul'"
    assert mind.soul() == DEFAULT_SOUL.strip()


async def test_read_mind_returns_each_file(tmp_path: Path) -> None:
    """Every mind file is addressable by name; empty ones say so."""
    mind = make_mind(tmp_path)
    mind.memory_path.write_text("curated fact\n", encoding="utf-8")
    toolbox = make_dream_toolbox(mind)

    assert await toolbox.run("read_mind", json.dumps({"file": "memory"})) == (
        "curated fact"
    )
    assert await toolbox.run("read_mind", json.dumps({"file": "inbox"})) == (
        "inbox is empty"
    )


async def test_write_dream_appends_a_dated_entry(tmp_path: Path) -> None:
    """The journal entry lands in DREAMS.md under a timestamp."""
    mind = make_mind(tmp_path)
    toolbox = make_dream_toolbox(mind)

    await toolbox.run("write_dream", json.dumps({"text": "  thought about Alice  "}))

    journal = mind.dreams_path.read_text(encoding="utf-8")
    assert journal.startswith("## [")
    assert "thought about Alice" in journal


async def test_fold_inbox_consolidates(tmp_path: Path) -> None:
    """Memory is replaced and the inbox emptied by a single call."""
    mind = make_mind(tmp_path)
    mind.append_inbox("fresh fact", "Sun 2026-08-02 12:00")
    toolbox = make_dream_toolbox(mind)

    await toolbox.run(
        "fold_inbox", json.dumps({"memory": "# People\nAlice: likes tea"})
    )

    assert mind.read("memory") == "# People\nAlice: likes tea"
    assert mind.read("inbox") == ""


async def test_fold_inbox_over_cap_is_an_error_not_a_crash(tmp_path: Path) -> None:
    """An oversized memory comes back as a retryable error string."""
    mind = make_mind(tmp_path)
    mind.append_inbox("fresh fact", "Sun 2026-08-02 12:00")
    toolbox = make_dream_toolbox(mind)

    result = await toolbox.run(
        "fold_inbox", json.dumps({"memory": "x" * (MEMORY_MAX_CHARS + 1)})
    )

    assert result.startswith("error:")
    assert "fresh fact" in mind.read("inbox")


async def test_write_soul_snapshots_and_replaces(tmp_path: Path) -> None:
    """The soul is replaced only after the old one is filed away."""
    mind = make_mind(tmp_path)
    toolbox = make_dream_toolbox(mind)

    result = await toolbox.run("write_soul", json.dumps({"text": "a quieter person"}))

    assert mind.soul() == "a quieter person"
    assert "soul/" in result
    assert [path.name for path in mind.soul_dir.iterdir()]


async def test_write_soul_over_cap_is_an_error(tmp_path: Path) -> None:
    """A bloated soul is refused; the current one survives."""
    mind = make_mind(tmp_path)
    toolbox = make_dream_toolbox(mind)

    result = await toolbox.run(
        "write_soul", json.dumps({"text": "x" * (SOUL_MAX_CHARS + 1)})
    )

    assert result.startswith("error:")
    assert mind.soul() == DEFAULT_SOUL.strip()


async def test_wake_up_refused_before_the_minimum_steps(tmp_path: Path) -> None:
    """A dream that tidies up and leaves is sent back to wander."""
    toolbox = make_dream_toolbox(make_mind(tmp_path), min_steps=3)

    result = await toolbox.run("wake_up", json.dumps({"summary": "done"}))

    assert result.startswith("error:")
    assert toolbox.wake_summary is None


async def test_wake_up_accepted_once_the_dream_has_wandered(tmp_path: Path) -> None:
    """Past the minimum, wake_up ends the dream with its summary."""
    toolbox = make_dream_toolbox(make_mind(tmp_path), min_steps=3)
    await toolbox.run("read_mind", json.dumps({"file": "memory"}))
    await toolbox.run("read_mind", json.dumps({"file": "inbox"}))

    result = await toolbox.run("wake_up", json.dumps({"summary": "  say hi to Bob  "}))

    assert result == "waking up"
    assert toolbox.wake_summary == "say hi to Bob"
    assert toolbox.steps == 3


async def test_steps_count_only_dispatched_calls(tmp_path: Path) -> None:
    """Unknown tools and bad arguments are not progress."""
    toolbox = make_dream_toolbox(make_mind(tmp_path))

    await toolbox.run("send_message", "{}")
    await toolbox.run("read_mind", "not json")
    assert toolbox.steps == 0

    await toolbox.run("read_mind", json.dumps({"file": "soul"}))
    assert toolbox.steps == 1
