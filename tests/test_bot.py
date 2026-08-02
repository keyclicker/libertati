"""Tests for event formatting and routing of incoming Telegram messages."""

from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

from aiogram.types import Message, User

from libertati.bot import (
    EVENT_TEXT_LIMIT,
    deliver_wakeups,
    format_event,
    is_addressed,
    on_message,
)
from libertati.chats import ChatRegistry
from libertati.db import Database

UTC_TZ = ZoneInfo("UTC")

#: The bot's own identity as seen by handlers.
ME = User(id=999, is_bot=True, first_name="libertati", username="Libertati_bot")

#: Registry with approval mode off — allows everything, touches no file.
OPEN_REGISTRY = ChatRegistry(Path("unused-chats.toml"), enabled=False)

#: 2026-08-02 12:00:00 UTC as a Telegram unix timestamp.
STAMP = 1785672000


def make_message(**overrides: Any) -> Message:
    """Build a minimal private-chat text message, with field overrides."""
    data: dict[str, Any] = {
        "message_id": 42,
        "date": STAMP,
        "chat": {"id": 100, "type": "private", "first_name": "Alice"},
        "from": {"id": 7, "is_bot": False, "first_name": "Alice"},
        "text": "hello",
    }
    data.update(overrides)
    return Message.model_validate(data)


def test_format_event_private_text() -> None:
    """A plain private message renders time, chat, sender and body."""
    event = format_event(make_message(), UTC_TZ)
    assert event == "[Sun 2026-08-02 12:00] chat 100 (private) | Alice (msg 42): hello"


def test_format_event_group_title_and_username() -> None:
    """Group title and sender username are included when present."""
    message = make_message(
        chat={"id": -500, "type": "group", "title": "friends"},
        **{"from": {"id": 7, "is_bot": False, "first_name": "Bob", "username": "bob"}},
    )
    event = format_event(message, UTC_TZ)
    assert "chat -500 (group “friends”)" in event
    assert "Bob @bob" in event


def test_format_event_reply_reference() -> None:
    """Replies mention the id of the message being answered."""
    message = make_message(
        reply_to_message={
            "message_id": 41,
            "date": STAMP,
            "chat": {"id": 100, "type": "private", "first_name": "Alice"},
            "text": "earlier",
        }
    )
    assert "(msg 42, replying to msg 41)" in format_event(message, UTC_TZ)


def test_format_event_truncates_long_body() -> None:
    """Bodies beyond the limit are elided with a char count."""
    message = make_message(text="x" * (EVENT_TEXT_LIMIT + 250))
    event = format_event(message, UTC_TZ)
    assert "[…250 chars]" in event
    assert len(event) < EVENT_TEXT_LIMIT + 200


def test_format_event_caption_fallback() -> None:
    """Messages without text fall back to the caption."""
    message = make_message(text=None, caption="a photo caption")
    assert "a photo caption" in format_event(message, UTC_TZ)


class FakeAgent:
    """Records events pushed by the message handler."""

    def __init__(self) -> None:
        """Start with an empty event list."""
        self.events: list[str] = []

    async def push(self, event: str, *, activity: bool = True) -> None:
        """Store the event."""
        self.events.append(event)


def make_group_message(**overrides: Any) -> Message:
    """Build a minimal group-chat text message, with field overrides."""
    return make_message(
        chat={"id": -500, "type": "group", "title": "friends"}, **overrides
    )


def test_is_addressed_mention() -> None:
    """A case-insensitive @username mention addresses the bot."""
    assert is_addressed(make_group_message(text="hey @libertati_bot, hi"), ME)
    assert not is_addressed(make_group_message(text="hey @someone_else"), ME)


def test_is_addressed_mention_is_bounded() -> None:
    """Neither a longer username nor an email-like string is a mention."""
    assert not is_addressed(make_group_message(text="cc @libertati_bot_2"), ME)
    assert not is_addressed(make_group_message(text="mail me@libertati_bot"), ME)
    assert is_addressed(make_group_message(text="(@Libertati_Bot)"), ME)


def test_is_addressed_caption_mention() -> None:
    """Mentions in media captions count too."""
    message = make_group_message(text=None, caption="look @libertati_bot")
    assert is_addressed(message, ME)


def test_is_addressed_reply_to_bot() -> None:
    """Replying to one of the bot's messages addresses it."""
    message = make_group_message(
        reply_to_message={
            "message_id": 41,
            "date": STAMP,
            "chat": {"id": -500, "type": "group", "title": "friends"},
            "from": {"id": ME.id, "is_bot": True, "first_name": "libertati"},
            "text": "earlier",
        }
    )
    assert is_addressed(message, ME)


async def test_on_message_private_always_pushed() -> None:
    """Private messages always become events."""
    agent = FakeAgent()
    await on_message(make_message(), agent, UTC_TZ, ME, OPEN_REGISTRY)  # type: ignore[arg-type]
    assert len(agent.events) == 1


async def test_on_message_group_needs_address() -> None:
    """Group messages are dropped unless the bot is addressed."""
    agent = FakeAgent()
    await on_message(make_group_message(), agent, UTC_TZ, ME, OPEN_REGISTRY)  # type: ignore[arg-type]
    assert agent.events == []
    mention = make_group_message(text="ping @libertati_bot")
    await on_message(mention, agent, UTC_TZ, ME, OPEN_REGISTRY)  # type: ignore[arg-type]
    assert len(agent.events) == 1


async def test_deliver_wakeups_pushes_due_and_completes(db: Database) -> None:
    """A due wakeup becomes one event and leaves the pending set."""
    agent = FakeAgent()
    await db.add_wakeup("2000-01-01 00:00:00", "ping alice")
    await deliver_wakeups(cast(Any, agent), db, UTC_TZ)
    assert len(agent.events) == 1
    assert "ping alice" in agent.events[0]
    assert await db.pending_wakeups() == []


async def test_on_message_approval_gate(tmp_path: Path) -> None:
    """In approval mode a new chat is registered and its events dropped."""
    registry = ChatRegistry(tmp_path / "chats.toml", enabled=True)
    agent = FakeAgent()
    await on_message(make_message(), agent, UTC_TZ, ME, registry)  # type: ignore[arg-type]
    assert agent.events == []
    assert "100 = false  # Alice (private)" in registry.path.read_text(encoding="utf-8")
    text = registry.path.read_text(encoding="utf-8")
    registry.path.write_text(text.replace("false", "true"), encoding="utf-8")
    await on_message(make_message(), agent, UTC_TZ, ME, registry)  # type: ignore[arg-type]
    assert len(agent.events) == 1
