"""Per-kind rendering of context items for the spy TUI.

Every item type gets a layout of its own — chat events split into chat,
speaker and text, tool calls list their arguments field by field,
message-list results collapse to one row per message — and each layout
falls back to the generic JSON-ish body whenever an item does not have
the shape its renderer expects. The fallback is the format the viewer
used before there were layouts at all, so an unknown item still reads.

Nothing here touches the database or the terminal: a renderer takes one
decoded context item and returns a :class:`rich.text.Text`, which keeps
the whole module testable on plain dictionaries.
"""

import json
import re
from datetime import datetime
from typing import Any

from rich.style import Style
from rich.text import Text

#: Header style and label per item kind.
KINDS = {
    "event": ("cyan", "EVENT"),
    "internal_message": ("bright_black", "INTERNAL INPUT"),
    "reasoning": ("bright_black", "REASONING"),
    "message": ("yellow", "FINAL OUTPUT"),
    "function_call": ("magenta", "TOOL CALL"),
    "function_call_output": ("blue", "TOOL RESULT"),
    "web_search_call": ("green", "WEB SEARCH"),
    "other": ("white", "OTHER"),
}

#: Accents shared by every layout. Structure (who spoke, which field,
#: which id) is coloured; the text a human actually reads is not, so it
#: stays legible whatever the terminal palette is.
META = "bright_black"
SPEAKER = "bold"
HANDLE = "cyan"
FIELD = "bright_black"
NUMBER = "cyan"
QUERY = "green"
ERROR = "bold red"
INCOMING = "cyan"
OUTGOING = "green"

#: Every hit of the active search is reversed; the one the cursor sits
#: on is marked as well, so :mod:`libertati.spy` can find the line it
#: was rendered onto without redoing the layout by hand.
MATCH = Style(reverse=True)
CURRENT_MATCH = Style(
    reverse=True, bold=True, underline=True, meta={"spy_cursor": True}
)

#: Body truncation limit until ``f`` toggles full bodies.
TRUNCATE_AT = 600

#: Longest argument value still shown inline with the others.
INLINE_VALUE = 60

#: UTF-8 bytes per o200k token, by content shape. ASCII in the context
#: is mostly dense JSON framing; multi-byte text is mostly Cyrillic and
#: emoji; reasoning items are base64 blobs, which pack far more tokens
#: per byte than anything else here.
ASCII_BYTES_PER_TOKEN = 2.8
WIDE_BYTES_PER_TOKEN = 5.0
BASE64_BYTES_PER_TOKEN = 1.5

_NON_ASCII = re.compile(r"[^\x00-\x7f]")

#: Event grammar. Events are written by the bot, so their shape is
#: known — but they are also free text, and a renderer that guesses
#: wrong must degrade to showing the line as it was written.
_TAG = re.compile(r"^\[(?P<tag>[^\]]*)\]\s*(?P<rest>.*)$", re.DOTALL)
_STAMP_TAG = re.compile(r"^[A-Za-z]{3} \d{4}-\d{2}-\d{2} \d{2}:\d{2}$")
_CHAT_HEAD = re.compile(r"^chat (?P<chat_id>-?\d+) \(")
_SPEAKER_LINE = re.compile(
    r"^(?P<who>.*?)(?: \(msg (?P<msg>\d+)\))?: (?P<text>.*)$", re.DOTALL
)
_HANDLE = re.compile(r"@\w+")
_WAKEUP_TAG = re.compile(r"^wakeup #(?P<id>\d+) (?P<detail>.*)$", re.DOTALL)

Row = tuple[int, str, str]


def estimate_tokens(raw: str, kind: str) -> int:
    """Estimate what one raw context item costs as API input.

    Byte-length based rather than tokenizer based: loading a real BPE
    table costs seconds of startup for numbers that are approximate
    anyway (encrypted reasoning bills as its hidden original, not as the
    base64 that is actually sent).
    """
    size = len(raw.encode())
    if kind == "reasoning":
        return round(size / BASE64_BYTES_PER_TOKEN)
    if raw.isascii():
        return round(size / ASCII_BYTES_PER_TOKEN)
    narrow = len(_NON_ASCII.sub("", raw))
    return round(
        narrow / ASCII_BYTES_PER_TOKEN + (size - narrow) / WIDE_BYTES_PER_TOKEN
    )


def classify(item: dict[str, Any]) -> str:
    """Map a raw context item to one of the ``KINDS``."""
    if item.get("role") == "user" and "type" not in item:
        return "event"
    if item.get("role") == "user" and item.get("type") == "message":
        return "internal_message"
    kind = item.get("type", "other")
    return kind if kind in KINDS else "other"


def decode(raw: str) -> dict[str, Any]:
    """Decode a stored context item, keeping unparsable rows viewable."""
    try:
        item = json.loads(raw)
    except json.JSONDecodeError:
        return {"type": "other", "unparsed": raw}
    return item if isinstance(item, dict) else {"type": "other", "unparsed": raw}


# ===== Fallback layout =====


def body_text(kind: str, item: dict[str, Any]) -> str:
    """Extract the human-interesting body of an item, by kind.

    The plain-text layout every renderer falls back to: no structure is
    assumed beyond the envelope the API itself guarantees.
    """
    if kind == "event":
        return str(item.get("content", ""))
    if kind == "reasoning":
        parts = [s.get("text", "") for s in item.get("summary", [])]
        return "\n".join(p for p in parts if p) or "(hidden)"
    if kind in {"message", "internal_message"}:
        parts = item.get("content", [])
        if isinstance(parts, str):
            return parts
        return "\n".join(
            p.get("text", "") or p.get("refusal", "")
            for p in parts
            if isinstance(p, dict)
            and p.get("type") in {"input_text", "output_text", "refusal"}
        )
    if kind == "function_call":
        args = item.get("arguments") or "{}"
        try:
            args = json.dumps(json.loads(args), ensure_ascii=False)
        except json.JSONDecodeError:
            pass
        return f"{item.get('name', '?')} {args}"
    if kind == "function_call_output":
        return str(item.get("output", ""))
    if kind == "web_search_call":
        return json.dumps(item.get("action", {}), ensure_ascii=False)
    return json.dumps(item, ensure_ascii=False)


# ===== Shared pieces =====


def _scalar(value: Any) -> Text:
    """Render one JSON scalar with a style that matches its type."""
    if value is None:
        return Text("null", style=META)
    if isinstance(value, bool):
        return Text("true" if value else "false", style=NUMBER)
    if isinstance(value, int | float):
        return Text(str(value), style=NUMBER)
    if isinstance(value, str):
        return Text(value)
    return Text(json.dumps(value, ensure_ascii=False))


def _is_short(value: Any) -> bool:
    """Whether a value still reads well next to the other fields."""
    if isinstance(value, str):
        return len(value) <= INLINE_VALUE and "\n" not in value
    return not isinstance(value, dict | list)


def _fields(data: dict[Any, Any], separator: str = "  ") -> Text:
    """Render short fields on one line, long ones as their own block."""
    line = Text()
    for key, value in data.items():
        if not _is_short(value):
            continue
        if line:
            line.append(separator)
        line.append(f"{key}=", style=FIELD)
        line.append_text(_scalar(value))
    for key, value in data.items():
        if _is_short(value):
            continue
        if line:
            line.append("\n")
        line.append(f"{key}:\n", style=FIELD)
        if isinstance(value, str):
            line.append(value)
        else:
            line.append(json.dumps(value, ensure_ascii=False, indent=2))
    return line


def _handles(text: str, style: str = SPEAKER) -> Text:
    """Style a speaker line, picking its ``@handle`` out of the name."""
    rendered = Text(text, style=style)
    for match in _HANDLE.finditer(text):
        rendered.stylize(HANDLE, match.start(), match.end())
    return rendered


def _local_time(iso: str) -> str:
    """Render an ISO timestamp as local ``HH:MM``, or ``?`` if malformed."""
    try:
        return datetime.fromisoformat(iso).astimezone().strftime("%H:%M")
    except ValueError:
        return "?"


def _balanced(text: str, start: int) -> tuple[str, int] | None:
    """Return the parenthesised run at ``start`` and the index after it.

    Chat titles contain parentheses of their own, so the closing one
    cannot be found by searching for the first ``)``.
    """
    if start >= len(text) or text[start] != "(":
        return None
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                return text[start + 1 : index], index + 1
    return None


def _split_tag(content: str) -> tuple[str | None, str]:
    """Split a leading ``[tag]`` off an event, if it has one."""
    match = _TAG.match(content)
    if match is None:
        return None, content
    return match.group("tag"), match.group("rest")


# ===== Per-kind layouts =====


def _event_body(item: dict[str, Any]) -> Text | None:
    """Lay out an event: speaker and text, or the tag's own payload."""
    content = item.get("content")
    if not isinstance(content, str) or not content:
        return None
    tag, rest = _split_tag(content)
    body = Text()
    if tag is not None and (wakeup := _WAKEUP_TAG.match(tag)):
        body.append(wakeup.group("detail"), style=META)
        body.append("\n")
    elif tag is not None and not _STAMP_TAG.match(tag):
        # Timestamps are dropped: the header already shows the clock.
        body.append(tag, style=META)
        body.append("\n")
    chat = _CHAT_HEAD.match(rest)
    if chat is None:
        body.append(rest)
        return body
    title = _balanced(rest, chat.end() - 1)
    if title is None or not rest[title[1] :].startswith(" | "):
        body.append(rest)
        return body
    said = _SPEAKER_LINE.match(rest[title[1] + 3 :])
    if said is None:
        body.append(rest[title[1] + 3 :])
        return body
    body.append_text(_handles(said.group("who")))
    if said.group("msg"):
        body.append(f"  msg {said.group('msg')}", style=META)
    body.append("\n")
    body.append(said.group("text"))
    return body


def _internal_body(item: dict[str, Any]) -> Text | None:
    """Lay out an injected user message: its tag, then its text."""
    text = body_text("internal_message", item)
    if not text:
        return None
    tag, rest = _split_tag(text)
    body = Text()
    if tag is not None:
        body.append(f"{tag}\n", style=META)
    body.append(rest, style=META)
    return body


def _reasoning_body(item: dict[str, Any]) -> Text | None:
    """Lay out a reasoning summary, emphasising its own headings."""
    parts = [
        s.get("text", "")
        for s in item.get("summary", [])
        if isinstance(s, dict) and s.get("text")
    ]
    if not parts:
        return Text("(hidden)", style=f"italic {META}")
    body = Text("\n".join(parts), style=META)
    for match in re.finditer(r"\*\*(.+?)\*\*", body.plain):
        body.stylize("bold", match.start(), match.end())
    return body


def _message_body(item: dict[str, Any]) -> Text | None:
    """Lay out the model's own text, marking its emphasis and code."""
    text = body_text("message", item)
    if not text:
        return None
    body = Text(text)
    for match in re.finditer(r"\*\*(.+?)\*\*", text):
        body.stylize("bold", match.start(), match.end())
    for match in re.finditer(r"`[^`\n]+`", text):
        body.stylize(NUMBER, match.start(), match.end())
    return body


def _call_body(item: dict[str, Any]) -> Text | None:
    """Lay out a tool call's arguments; the name lives in the header."""
    try:
        args = json.loads(item.get("arguments") or "{}")
    except json.JSONDecodeError:
        return None
    if not isinstance(args, dict):
        return None
    if not args:
        return Text("(no arguments)", style=META)
    return _fields(args)


def _messages_body(rows: list[Any]) -> Text | None:
    """Lay out a message list as one row per message, newest last."""
    if not rows or not all(
        isinstance(row, dict) and "message_id" in row and "date" in row for row in rows
    ):
        return None
    body = Text()
    for index, row in enumerate(rows):
        if index:
            body.append("\n")
        outgoing = bool(row.get("outgoing"))
        body.append(f"{row['message_id']} ", style=META)
        body.append(f"{_local_time(str(row['date']))} ", style=META)
        body.append(
            "→ " if outgoing else "← ", style=OUTGOING if outgoing else INCOMING
        )
        name = row.get("first_name") or row.get("username") or "?"
        body.append(str(name), style=SPEAKER)
        if row.get("username"):
            body.append(f" @{row['username']}", style=HANDLE)
        body.append(": ")
        said = str(row.get("text") or row.get("caption") or "").strip()
        # Messages carry newlines of their own; indenting them keeps the
        # row boundaries visible in a long dump.
        body.append(said.replace("\n", "\n  "))
    return body


def _output_body(item: dict[str, Any]) -> Text | None:
    """Lay out a tool result: message rows, fields, or its plain text."""
    output = item.get("output")
    if not isinstance(output, str) or not output:
        return None
    if output.startswith("error:"):
        return Text(output, style=ERROR)
    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        return Text(output)
    if isinstance(data, list):
        if (rows := _messages_body(data)) is not None:
            return rows
        body = Text()
        for index, entry in enumerate(data):
            if index:
                body.append("\n")
            if isinstance(entry, dict):
                body.append_text(_fields(entry))
            else:
                body.append_text(_scalar(entry))
        return body
    if isinstance(data, dict):
        return _fields(data, separator="\n")
    return Text(output)


def _search_body(item: dict[str, Any]) -> Text | None:
    """Lay out a web search: one line per query it ran."""
    action = item.get("action")
    if not isinstance(action, dict):
        return None
    queries = action.get("queries") or (
        [action["query"]] if action.get("query") else []
    )
    if not queries:
        return None
    body = Text()
    for index, query in enumerate(queries):
        if index:
            body.append("\n")
        body.append("? ", style=META)
        body.append(str(query), style=QUERY)
    return body


#: Layout per kind. Each one may return ``None`` — for an item that
#: does not fit its shape, or has nothing worth laying out — and the
#: generic body is used instead.
_LAYOUTS = {
    "event": _event_body,
    "internal_message": _internal_body,
    "reasoning": _reasoning_body,
    "message": _message_body,
    "function_call": _call_body,
    "function_call_output": _output_body,
    "web_search_call": _search_body,
}


def render_body(kind: str, item: dict[str, Any]) -> Text:
    """Render an item's body with its own layout, or the generic one.

    A layout that raises is a bug in this module, not a reason to lose
    the row: the viewer keeps showing the plain body either way.
    """
    layout = _LAYOUTS.get(kind)
    if layout is not None:
        try:
            body = layout(item)
        except Exception:  # noqa: BLE001 - a layout must never blank a row
            body = None
        if body is not None:
            return body
    color, _ = KINDS[kind]
    return Text(body_text(kind, item), style="default" if kind == "event" else color)


# ===== Blocks =====


def subject(kind: str, item: dict[str, Any], name: str | None = None) -> str:
    """Summarise an item for its header: chat, tool name or tag."""
    if kind == "function_call":
        return str(item.get("name") or "?")
    if kind == "function_call_output":
        return name or ""
    if kind == "web_search_call":
        return "search"
    if kind != "event":
        return ""
    content = item.get("content")
    if not isinstance(content, str):
        return ""
    tag, rest = _split_tag(content)
    if tag is not None and (wakeup := _WAKEUP_TAG.match(tag)):
        return f"wakeup #{wakeup.group('id')}"
    chat = _CHAT_HEAD.match(rest)
    if chat is None:
        return "" if tag is None or _STAMP_TAG.match(tag) else tag
    title = _balanced(rest, chat.end() - 1)
    return title[0] if title else f"chat {chat.group('chat_id')}"


def call_name(item: dict[str, Any]) -> tuple[str, str] | None:
    """Return the ``(call_id, name)`` a tool call announces, if any."""
    if item.get("type") != "function_call":
        return None
    call_id, name = item.get("call_id"), item.get("name")
    if isinstance(call_id, str) and isinstance(name, str):
        return call_id, name
    return None


def searchable(raw: str) -> str:
    """Return everything of one stored row a search may match.

    The headline comes first, then the body — the order the highlights
    are numbered in, and the reason ``/send_message`` finds a call whose
    name the layout moved out of the body and into the header. Only the
    part of the headline the item itself carries counts: a tool result
    is labelled with the name of the call that opened it, which lives in
    another row and so cannot be indexed from this one.
    """
    item = decode(raw)
    kind = classify(item)
    return f"{subject(kind, item)}\n{render_body(kind, item).plain}"


def build_block(
    row: Row,
    full: bool = False,
    pattern: re.Pattern[str] | None = None,
    clock: str = "",
    name: str | None = None,
    cursor: int | None = None,
) -> Text | None:
    """Render one context row, or ``None`` for empty output envelopes.

    ``cursor`` picks one occurrence of ``pattern`` inside this row to
    mark as the current search hit, numbered exactly as
    :func:`searchable` counts them: headline first, then body.
    """
    row_id, _, raw = row
    item = decode(raw)
    kind = classify(item)
    color, label = KINDS[kind]
    own = subject(kind, item)
    headline = subject(kind, item, name)
    body = render_body(kind, item)
    if kind == "message" and not body.plain:
        return None
    if pattern is not None and pattern.search(body.plain):
        full = True  # a hit the search found must be a hit you can see
    if not full and len(body.plain) > TRUNCATE_AT:
        cut = len(body.plain) - TRUNCATE_AT
        body = body[:TRUNCATE_AT]
        body.append(f" […{cut} chars]", style=META)
    head_spans = list(pattern.finditer(own)) if pattern is not None and own else []
    if pattern is not None:
        for index, span in enumerate(pattern.finditer(body.plain)):
            style = CURRENT_MATCH if index + len(head_spans) == cursor else MATCH
            body.stylize(style, span.start(), span.end())
    block = Text()
    block.append(f"#{row_id} ", style=f"bold {META}")
    block.append(label, style=f"bold {color}")
    if headline:
        offset = len(block.plain) + 2
        block.append(f"  {headline}", style=color)
        # Only the item's own headline is indexed, so only it is marked.
        for index, span in enumerate(head_spans if headline == own else []):
            style = CURRENT_MATCH if index == cursor else MATCH
            block.stylize(style, offset + span.start(), offset + span.end())
    block.append(f"  {clock}" if clock else "", style=META)
    block.append(f"  ~{estimate_tokens(raw, kind)} tok\n", style=META)
    block.append_text(body)
    return block
