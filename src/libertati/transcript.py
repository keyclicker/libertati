r"""How stored messages are rendered for the model.

Every tool that hands chat history back — and every event that carries
some along — goes through :func:`render_messages`, so the agent only ever
learns one transcript format. The format is deliberately line-oriented
rather than JSON: one message per line, no repeated keys, no columns that
are constant or empty for almost every row.

Message bodies are folded to a single line (newlines become a literal
``\\n``) so a message can never look like two, and a day header is
emitted whenever the date changes instead of stamping every line with it.
"""

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

#: Shown instead of a name when a message has no known sender.
UNKNOWN_SENDER = "unknown"

#: Shown as the sender of the agent's own messages.
SELF_SENDER = "you"


def _sender(row: dict[str, Any]) -> str:
    """Name the author of one stored message, as the model should see it."""
    if row.get("outgoing"):
        return SELF_SENDER
    name = row.get("first_name")
    username = row.get("username")
    if name and username:
        return f"{name} @{username}"
    if name:
        return name
    return f"@{username}" if username else UNKNOWN_SENDER


def media_body(content_type: str | None, note: str | None, body: str) -> str:
    """Prefix a body with what the message carries, described if known.

    ``<sticker>`` alone says a sticker happened; ``<sticker: a cat
    knocking a mug off a table>`` says what everyone else in the chat
    saw. The note comes from a model looking at the file (see
    ``media.py``) and is folded like any other untrusted text — it lives
    inside one transcript line and must not be able to become two.

    Text messages keep their body unadorned; the caller passes the body
    already folded and capped.
    """
    if not content_type or content_type == "text":
        return body
    label = content_type
    if note:
        label += ": " + " ".join(note.split())
    return f"<{label}> {body}".rstrip()


def _body(row: dict[str, Any], text_limit: int | None) -> str:
    """Render a message body, naming the media of non-text messages.

    Text and caption are folded onto one line the way events are, so a
    body with newlines in it cannot pass for several transcript lines.
    """
    text = row.get("text") or row.get("caption") or ""
    body = "\\n".join(text.splitlines())
    if text_limit is not None and len(body) > text_limit:
        body = body[:text_limit] + f" […{len(body) - text_limit} chars]"
    return media_body(row.get("content_type"), row.get("media_note"), body)


def render_message(
    row: dict[str, Any],
    tz: ZoneInfo,
    *,
    show_topic: bool = False,
    text_limit: int | None = None,
) -> str:
    """Render one stored message row as a single transcript line.

    The line is ``id HH:MM sender: body``, followed by ``↩id`` when the
    message replies to another one and by ``#id`` for its forum topic
    when ``show_topic`` says the surrounding query spanned several.
    """
    when = datetime.fromisoformat(row["date"]).astimezone(tz).strftime("%H:%M")
    line = f"{row['message_id']} {when} {_sender(row)}: {_body(row, text_limit)}"
    if row.get("reply_to_message_id"):
        line += f" ↩{row['reply_to_message_id']}"
    if show_topic and row.get("message_thread_id"):
        line += f" #{row['message_thread_id']}"
    return line


def render_messages(
    rows: list[dict[str, Any]],
    tz: ZoneInfo,
    *,
    show_topic: bool = False,
    text_limit: int | None = None,
) -> list[str]:
    """Render stored message rows as transcript lines, in the given order.

    A ``— Sat 2026-08-04 —`` header precedes the first message and every
    later one whose date differs from the line before it; the caller's
    ordering (oldest or newest first) is preserved either way.
    """
    lines: list[str] = []
    day = None
    for row in rows:
        moment = datetime.fromisoformat(row["date"]).astimezone(tz)
        if moment.date() != day:
            day = moment.date()
            lines.append(f"— {moment.strftime('%a %Y-%m-%d')} —")
        lines.append(
            render_message(row, tz, show_topic=show_topic, text_limit=text_limit)
        )
    return lines


def render_transcript(
    rows: list[dict[str, Any]],
    tz: ZoneInfo,
    *,
    show_topic: bool = False,
    text_limit: int | None = None,
) -> str:
    """Render message rows as one newline-joined transcript block."""
    return "\n".join(
        render_messages(rows, tz, show_topic=show_topic, text_limit=text_limit)
    )
