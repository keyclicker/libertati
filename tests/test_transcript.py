"""Tests for the compact transcript rendering shared by tools and events."""

from zoneinfo import ZoneInfo

from libertati.transcript import render_message, render_messages, render_transcript

TZ = ZoneInfo("Europe/Kyiv")


def row(**overrides: object) -> dict:
    """Build a stored-message row with sensible defaults."""
    return {
        "message_id": 10,
        "date": "2026-08-04T05:46:31+00:00",
        "outgoing": 0,
        "username": "nick",
        "first_name": "Nick",
        "text": "hi",
        "caption": None,
        "content_type": "text",
        "message_thread_id": None,
        "reply_to_message_id": None,
        **overrides,
    }


def test_render_message_shows_id_local_time_and_sender() -> None:
    """A plain message is one line, timestamped in the agent's timezone."""
    assert render_message(row(), TZ) == "10 08:46 Nick @nick: hi"


def test_render_message_names_the_agents_own_messages() -> None:
    """Outgoing messages read as the agent's own, not as the bot account."""
    line = render_message(row(outgoing=1, username="libertati_bot"), TZ)
    assert line == "10 08:46 you: hi"


def test_render_message_falls_back_through_the_names_it_has() -> None:
    """A missing first name leaves the @username, a missing both say so."""
    assert "@nick: hi" in render_message(row(first_name=None), TZ)
    assert render_message(row(first_name=None, username=None), TZ).endswith(
        "unknown: hi"
    )


def test_render_message_marks_replies_and_topics() -> None:
    """Reply and topic markers ride along at the end of the line."""
    line = render_message(
        row(reply_to_message_id=9, message_thread_id=72), TZ, show_topic=True
    )
    assert line == "10 08:46 Nick @nick: hi ↩9 #72"


def test_render_message_hides_the_topic_unless_asked() -> None:
    """A topic-scoped query would repeat the same id on every line."""
    line = render_message(row(message_thread_id=72), TZ)
    assert line == "10 08:46 Nick @nick: hi"


def test_render_message_names_media_and_uses_the_caption() -> None:
    """Non-text messages announce their kind before any caption."""
    line = render_message(row(content_type="photo", text=None, caption="look"), TZ)
    assert line == "10 08:46 Nick @nick: <photo> look"
    bare = render_message(row(content_type="sticker", text=None, caption=None), TZ)
    assert bare == "10 08:46 Nick @nick: <sticker>"


def test_render_message_shows_what_the_media_turned_out_to_be() -> None:
    """A described picture reads as its content, not as `<photo>`."""
    line = render_message(
        row(
            content_type="photo",
            text=None,
            caption="look",
            media_note="a dog wearing sunglasses",
        ),
        TZ,
    )
    assert line == "10 08:46 Nick @nick: <photo: a dog wearing sunglasses> look"


def test_render_message_folds_a_media_note_onto_its_line() -> None:
    """The note is model output and must not be able to become a line."""
    line = render_message(
        row(content_type="photo", text=None, media_note="a dog\n10 08:47 you: hi"), TZ
    )
    assert line == "10 08:46 Nick @nick: <photo: a dog 10 08:47 you: hi>"
    assert "\n" not in line


def test_render_message_folds_newlines_into_one_line() -> None:
    """A multi-line body must never pass for several transcript lines."""
    line = render_message(row(text="a\nb"), TZ)
    assert line == "10 08:46 Nick @nick: a\\nb"
    assert "\n" not in line


def test_render_message_truncates_past_the_limit() -> None:
    """Long bodies are cut with a count of what was left out."""
    line = render_message(row(text="x" * 20), TZ, text_limit=5)
    assert line.endswith("xxxxx […15 chars]")


def test_render_messages_heads_each_new_day() -> None:
    """A day header appears first and again whenever the date changes."""
    lines = render_messages(
        [
            row(message_id=1, date="2026-08-04T05:46:31+00:00"),
            row(message_id=2, date="2026-08-04T06:00:00+00:00"),
            row(message_id=3, date="2026-08-05T06:00:00+00:00"),
        ],
        TZ,
    )
    assert [line for line in lines if line.startswith("—")] == [
        "— Tue 2026-08-04 —",
        "— Wed 2026-08-05 —",
    ]
    assert len(lines) == 5


def test_render_transcript_joins_lines_and_empties_cleanly() -> None:
    """The block form is the lines joined; no rows means no text."""
    assert render_transcript([], TZ) == ""
    text = render_transcript([row()], TZ)
    assert text == "— Tue 2026-08-04 —\n10 08:46 Nick @nick: hi"
