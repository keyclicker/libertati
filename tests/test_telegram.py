from __future__ import annotations

from datetime import UTC

from libertati.telegram.base import IncomingMessage, TelegramClient


def test_bot_api_message_mapping():
    from datetime import datetime
    from types import SimpleNamespace

    from libertati.telegram.bot_api import _to_incoming

    msg = SimpleNamespace(
        chat=SimpleNamespace(id=5, type="supergroup", title="G"),
        message_id=9,
        text="hi",
        caption=None,
        from_user=SimpleNamespace(id=42, username="alice", full_name="Alice A"),
        reply_to_message=SimpleNamespace(message_id=8),
        date=datetime(2020, 1, 1, tzinfo=UTC),
    )
    inc = _to_incoming(msg, self_id=1)
    assert isinstance(inc, IncomingMessage)
    assert inc.chat_id == 5
    assert inc.user_handle == "@alice"
    assert inc.reply_to_id == 8
    assert inc.is_group is True
    assert inc.from_self is False


def test_bot_api_detects_self():
    from types import SimpleNamespace

    from libertati.telegram.bot_api import _to_incoming

    msg = SimpleNamespace(
        chat=SimpleNamespace(id=5, type="private", title=None),
        message_id=9,
        text="hi",
        caption=None,
        from_user=SimpleNamespace(id=1, username="bot", full_name="Bot"),
        reply_to_message=None,
        date=None,
    )
    inc = _to_incoming(msg, self_id=1)
    assert inc.from_self is True
    assert inc.is_group is False


async def test_dispatch_calls_handler():
    class Dummy(TelegramClient):
        async def start(self):  # pragma: no cover
            ...

        async def send_message(self, chat_id, text, reply_to_id=None):  # pragma: no cover
            return None

        async def stop(self):  # pragma: no cover
            ...

        @property
        def self_username(self):
            return "@x"

    received = []
    d = Dummy()

    async def handler(m):
        received.append(m)

    d.on_message(handler)
    await d._dispatch(IncomingMessage(chat_id=1, message_id=1, text="yo"))
    assert received and received[0].text == "yo"


def test_bot_api_maps_chat_username():
    from types import SimpleNamespace

    from libertati.telegram.bot_api import _to_incoming

    msg = SimpleNamespace(
        chat=SimpleNamespace(id=5, type="supergroup", title="G", username="publicgroup"),
        message_id=9,
        text="hi",
        caption=None,
        from_user=SimpleNamespace(id=42, username="alice", full_name="Alice A"),
        reply_to_message=None,
        date=None,
    )
    inc = _to_incoming(msg, self_id=1)
    assert inc.chat_username == "@publicgroup"
