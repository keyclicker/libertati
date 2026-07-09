from __future__ import annotations

from conftest import FakeLLM, assistant, make_message
from libertati.config import Settings
from libertati.llm.agent import Agent
from libertati.llm.tools import ToolBox
from libertati.main import App
from libertati.news.reader import NewsReader
from libertati.storage.history import HistoryStore
from libertati.telegram.base import IncomingMessage, TelegramClient


class StubClient(TelegramClient):
    def __init__(self) -> None:
        super().__init__()
        self.sent: list[tuple] = []

    async def start(self) -> None: ...
    async def stop(self) -> None: ...

    @property
    def self_username(self):
        return "@libertati_bot"

    async def send_message(self, chat_id, text, reply_to_id=None):
        self.sent.append((chat_id, text))
        return IncomingMessage(
            chat_id=chat_id, message_id=9000 + len(self.sent), text=text, from_self=True
        )


async def _make_app(tmp_path, **kw) -> App:
    settings = Settings(
        openai_api_key="x", bot_token="1:x",
        db_path=tmp_path / "db.sqlite", memory_dir=tmp_path / "memory",
        event_log_enabled=False, typing_delay_enabled=False, **kw,
    )
    app = App(settings)
    app.client = StubClient()
    await app.db.connect()
    app.history = HistoryStore(app.db, settings.max_thread_chars)
    tools = ToolBox(app.history, app.memory, NewsReader(feeds=[]))
    app.agent = Agent(settings, FakeLLM([assistant("йо")]), tools, app.history, app.memory)
    return app


async def test_ambient_group_message_is_skipped(tmp_path):
    app = await _make_app(tmp_path, group_reply_chance=0.0)
    await app._on_message(make_message("просто балачки в чаті", message_id=1))
    assert app.client.sent == []  # not addressed + 0 chance => silent
    await app.db.close()


async def test_addressed_group_message_gets_reply(tmp_path):
    app = await _make_app(tmp_path, group_reply_chance=0.0)
    await app._on_message(make_message("@libertati_bot агов", message_id=1))
    assert app.client.sent, "a message that name-drops the bot must get a reply"
    await app.db.close()


async def test_private_message_always_gets_reply(tmp_path):
    app = await _make_app(tmp_path, group_reply_chance=0.0)
    await app._on_message(make_message("привіт", message_id=1, is_group=False))
    assert app.client.sent
    await app.db.close()


async def test_reply_to_bot_counts_as_addressed(tmp_path):
    app = await _make_app(tmp_path, group_reply_chance=0.0)
    bot_msg = IncomingMessage(
        chat_id=100, message_id=5, text="я тут", from_self=True, is_group=True
    )
    await app.history.add_message(bot_msg, role="assistant")
    await app._on_message(make_message("ага, зрозуміла", message_id=6, reply_to=5))
    assert app.client.sent, "replying to the bot's own message must get a reply"
    await app.db.close()


async def test_ambient_message_still_logged_even_when_silent(tmp_path):
    app = await _make_app(tmp_path, group_reply_chance=0.0)
    await app._on_message(make_message("щось у чат", message_id=7))
    assert app.client.sent == []
    # message is still persisted for memory/history even though we stayed quiet
    recent = await app.history.recent(100)
    assert any("щось у чат" in m.text for m in recent)
    await app.db.close()
