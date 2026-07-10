from __future__ import annotations

from conftest import FakeLLM, assistant, make_message
from libertati.config import Settings
from libertati.llm.agent import Agent
from libertati.llm.tools import ToolBox
from libertati.news.reader import NewsReader
from libertati.scheduler import jobs
from libertati.telegram.base import IncomingMessage, Source, TelegramClient


class FakeClient(TelegramClient):
    def __init__(self, *, supports_reading: bool, messages=None):
        super().__init__()
        self._supports = supports_reading
        self._messages = messages or []
        self.sent: list[tuple] = []

    async def start(self): ...
    async def stop(self): ...

    @property
    def self_username(self):
        return "@bot"

    @property
    def supports_reading(self):
        return self._supports

    async def list_readable_sources(self, limit: int = 100):
        return [Source(id=1, title="Ch", username="@ch", kind="channel", is_public=True)]

    async def read_source(self, source, limit: int = 20):
        return self._messages

    async def send_message(self, chat_id, text, reply_to_id=None):
        self.sent.append((chat_id, text))
        return IncomingMessage(chat_id=chat_id, message_id=999, text=text, from_self=True)


def _agent(settings, history, memory, script):
    tools = ToolBox(history, memory, NewsReader(feeds=[]))
    return Agent(settings, FakeLLM(script), tools, history, memory)


async def test_browse_job_noop_without_reading(settings, history, memory):
    client = FakeClient(supports_reading=False)
    agent = _agent(settings, history, memory, [assistant("should not be used")])
    await jobs.browse_job(agent, client, history, settings)
    assert client.sent == []


async def test_browse_job_reads_and_discusses(history, memory):
    settings = Settings(
        openai_api_key="k", bot_token="1:x",
        browse_channels=["@ch"], allowed_chats=["777"],
    )
    client = FakeClient(
        supports_reading=True,
        messages=[make_message("ринки падають", message_id=1, handle="@a")],
    )
    agent = _agent(settings, history, memory, [assistant("ну і шо, купуй на дні 😏")])
    await jobs.browse_job(agent, client, history, settings)
    assert client.sent, "expected a discuss message"
    chat_id, text = client.sent[0]
    assert chat_id == 777
    assert "купуй" in text


async def test_browse_job_pass_sends_nothing(history, memory):
    settings = Settings(
        openai_api_key="k", bot_token="1:x",
        browse_channels=["@ch"], allowed_chats=["777"],
    )
    client = FakeClient(
        supports_reading=True,
        messages=[make_message("нудьга", message_id=1)],
    )
    agent = _agent(settings, history, memory, [assistant("PASS")])
    await jobs.browse_job(agent, client, history, settings)
    assert client.sent == []
