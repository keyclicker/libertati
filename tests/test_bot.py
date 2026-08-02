"""Tests for event formatting of incoming Telegram messages."""

from typing import Any
from zoneinfo import ZoneInfo

from aiogram.types import Message

from libertati.bot import EVENT_TEXT_LIMIT, format_event

UTC_TZ = ZoneInfo("UTC")

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
