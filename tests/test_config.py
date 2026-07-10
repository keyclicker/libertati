from __future__ import annotations

from libertati.config import Settings


def _settings(**kw) -> Settings:
    return Settings(openai_api_key="k", bot_token="1:x", **kw)


def test_allowed_chats_parsing_from_string():
    s = _settings(allowed_chats="@mygroup, -1001234 , @Other")
    assert s.allowed_chats == ["@mygroup", "-1001234", "@Other"]


def test_is_chat_allowed_empty_allows_all():
    s = _settings()
    assert s.is_chat_allowed(123, "@anything") is True
    assert s.is_chat_allowed(123, None) is True


def test_is_chat_allowed_by_id():
    s = _settings(allowed_chats=["-1001234"])
    assert s.is_chat_allowed(-1001234, None) is True
    assert s.is_chat_allowed(999, None) is False


def test_is_chat_allowed_by_username_case_insensitive():
    s = _settings(allowed_chats=["@MyGroup"])
    assert s.is_chat_allowed(1, "@mygroup") is True
    assert s.is_chat_allowed(1, "mygroup") is True  # normalization strips @
    assert s.is_chat_allowed(1, "@nope") is False


def test_browse_channels_parsing():
    s = _settings(browse_channels="@a,@b")
    assert s.browse_channels == ["@a", "@b"]


def test_browse_defaults():
    s = _settings()
    assert s.browse_enabled is False
    assert s.browse_public_only is True
    assert s.browse_times_per_day == 3
