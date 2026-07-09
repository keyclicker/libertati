from __future__ import annotations

from libertati.behavior import mentions_bot, typing_delay_seconds


def test_mentions_bot_by_username_handle_name():
    names = ["@libertati_bot", "@libertati", "Ana Tati"]
    assert mentions_bot("гей @libertati_bot шо там", names) is True
    assert mentions_bot("libertati, ти тут?", names) is True
    assert mentions_bot("що думає ana tati про це", names) is True
    assert mentions_bot("просто балачки в чаті", names) is False


def test_mentions_bot_ignores_empty_names():
    assert mentions_bot("hello", [None, "", "  "]) is False


def test_typing_delay_scales_and_caps():
    assert typing_delay_seconds("", 5.0) == 0.0
    assert typing_delay_seconds("x" * 25, 5.0, chars_per_second=25.0) == 1.0
    assert typing_delay_seconds("x" * 10_000, 5.0) == 5.0  # capped
    assert typing_delay_seconds("whatever", 0.0) == 0.0  # disabled via cap
