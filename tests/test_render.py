"""Tests for the per-kind context rendering."""

import json

from rich.style import Style

from libertati.render import (
    TRUNCATE_AT,
    build_block,
    call_name,
    classify,
    render_body,
    subject,
)

STAMP = "2026-08-02 10:00:00"


def row(item: dict, row_id: int = 1) -> tuple[int, str, str]:
    """Build the stored-row triple the renderer takes."""
    return (row_id, STAMP, json.dumps(item, ensure_ascii=False))


def event(content: str) -> dict:
    """Build an external event item."""
    return {"role": "user", "content": content}


def styles_at(text, needle: str) -> list[Style]:
    """Return the styles covering the first occurrence of ``needle``."""
    start = text.plain.index(needle)
    return [
        span.style
        for span in text.spans
        if span.start <= start < span.end  # type: ignore[operator]
    ]


# ===== Events =====


def test_chat_event_splits_chat_speaker_and_text() -> None:
    """A message event reads as who said what, with the chat in the header."""
    item = event(
        "[Tue 2026-08-04 14:44] chat -1001309183227 (supergroup “ЛПУ (підвал)”)"
        " | Nick @nick_keyclicker (msg 956): не пиши в інші вітки"
    )

    block = build_block(row(item))

    assert block is not None
    header, speaker, text = block.plain.splitlines()
    assert "supergroup “ЛПУ (підвал)”" in header  # nested parens survive
    assert speaker == "Nick @nick_keyclicker  msg 956"
    assert text == "не пиши в інші вітки"


def test_wakeup_event_names_its_alarm() -> None:
    """A wakeup carries its number in the header and its task in the body."""
    item = event(
        "[wakeup #33 at Tue 2026-08-04 14:52 — you scheduled it for"
        " Tue 2026-08-04 14:50] Check the flood topic."
    )

    block = build_block(row(item))

    assert block is not None
    assert "wakeup #33" in block.plain.splitlines()[0]
    assert block.plain.endswith("Check the flood topic.")


def test_free_text_event_keeps_its_line() -> None:
    """An event that fits no known shape is shown as it was written."""
    item = event("something nobody planned for")

    block = build_block(row(item))

    assert block is not None
    assert block.plain.endswith("something nobody planned for")


# ===== Tool calls and results =====


def test_tool_call_shows_short_arguments_inline_and_long_ones_apart() -> None:
    """Scalars share a line; a long text argument gets its own block."""
    item = {
        "type": "function_call",
        "name": "send_message",
        "call_id": "call_1",
        "arguments": json.dumps(
            {"chat_id": -100123, "reply_to": None, "text": "hi\nthere " + "x" * 80}
        ),
    }

    block = build_block(row(item))

    assert block is not None
    assert "send_message" in block.plain.splitlines()[0]
    assert "chat_id=-100123  reply_to=null" in block.plain
    assert "text:\nhi\nthere " in block.plain


def test_tool_call_with_unparsable_arguments_falls_back() -> None:
    """Arguments that are not JSON still show, through the generic body."""
    item = {
        "type": "function_call",
        "name": "react",
        "call_id": "call_2",
        "arguments": "{not json",
    }

    block = build_block(row(item))

    assert block is not None
    assert "react {not json" in block.plain


def test_message_list_result_becomes_one_row_per_message() -> None:
    """A `get_recent_messages` dump reads as a conversation, not as JSON."""
    item = {
        "type": "function_call_output",
        "call_id": "call_3",
        "output": json.dumps(
            [
                {
                    "message_id": 94905,
                    "date": "2026-08-04T11:41:21+00:00",
                    "outgoing": 1,
                    "username": "libertati_bot",
                    "first_name": "Ana Tati",
                    "text": "нічого нового",
                },
                {
                    "message_id": 94906,
                    "date": "2026-08-04T11:42:37+00:00",
                    "outgoing": 0,
                    "username": "Efosamark",
                    "first_name": "Efosamark",
                    "text": "ok",
                },
            ]
        ),
    }

    block = build_block(row(item), name="get_recent_messages")

    assert block is not None
    header, first, second = block.plain.splitlines()
    assert "get_recent_messages" in header  # named after the call it answers
    assert first.startswith("94905 ") and "→ Ana Tati @libertati_bot: нічого нового"
    assert "← Efosamark @Efosamark: ok" in second


def test_failed_tool_result_is_marked_as_an_error() -> None:
    """An `error: …` result is styled apart from a successful one."""
    item = {
        "type": "function_call_output",
        "call_id": "call_4",
        "output": "error: chat 5 is not approved",
    }

    body = render_body("function_call_output", json.loads(json.dumps(item)))

    assert body.style == "bold red"


def test_plain_tool_result_is_left_alone() -> None:
    """A result that is neither JSON nor an error keeps its own wording."""
    item = {
        "type": "function_call_output",
        "call_id": "call_5",
        "output": "sent message 814 to chat 319238363 at Tue 2026-08-04 07:44",
    }

    block = build_block(row(item))

    assert block is not None
    assert block.plain.endswith(
        "sent message 814 to chat 319238363 at Tue 2026-08-04 07:44"
    )


# ===== Reasoning, output, search =====


def test_hidden_reasoning_says_so() -> None:
    """An encrypted reasoning item with no summary is labelled, not blank."""
    item = {"type": "reasoning", "summary": [], "encrypted_content": "AAAA"}

    block = build_block(row(item))

    assert block is not None
    assert block.plain.endswith("(hidden)")


def test_reasoning_headings_are_emphasised() -> None:
    """The model's own `**heading**` markers become real emphasis."""
    item = {
        "type": "reasoning",
        "summary": [{"type": "summary_text", "text": "**Weighing it up**\nmaybe"}],
    }

    body = render_body("reasoning", item)

    assert any(style == "bold" for style in styles_at(body, "**Weighing"))


def test_web_search_lists_its_queries() -> None:
    """A web search shows what it asked, one query per line."""
    item = {
        "type": "web_search_call",
        "action": {"type": "search", "queries": ["libertarian кіт", "SLOP"]},
    }

    block = build_block(row(item))

    assert block is not None
    assert block.plain.splitlines()[1:] == ["? libertarian кіт", "? SLOP"]


# ===== Truncation =====


def test_long_bodies_truncate_until_asked_for_in_full() -> None:
    """A long body is cut with a count, and `full` brings all of it back."""
    item = event("y" * (TRUNCATE_AT + 40))

    cut = build_block(row(item))
    whole = build_block(row(item), full=True)

    assert cut is not None and whole is not None
    assert "[…40 chars]" in cut.plain
    assert "chars]" not in whole.plain


# ===== Plumbing =====


def test_call_names_come_from_the_call_row() -> None:
    """Tool calls announce the name their result will be labelled with."""
    call = {"type": "function_call", "name": "recall", "call_id": "call_8"}

    assert call_name(call) == ("call_8", "recall")
    assert call_name({"type": "function_call_output", "call_id": "call_8"}) is None


def test_unknown_items_stay_visible() -> None:
    """An item of an unknown type is classified and dumped, not dropped."""
    block = build_block(row({"type": "brand_new_thing", "payload": 1}))

    assert classify({"type": "brand_new_thing"}) == "other"
    assert block is not None
    assert "OTHER" in block.plain
    assert "brand_new_thing" in block.plain


def test_unparsable_rows_stay_visible() -> None:
    """A row that is not JSON at all still renders."""
    block = build_block((3, STAMP, "{ not json"))

    assert block is not None
    assert "not json" in block.plain


def test_subject_reports_the_chat_a_message_came_from() -> None:
    """The headline of a chat event names its chat, not its text."""
    item = event("[Tue 2026-08-04 14:44] chat 319 (private) | Nick: hi")

    assert subject("event", item) == "private"
