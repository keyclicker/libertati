from __future__ import annotations

import json

import pytest

from conftest import make_message
from libertati.llm.tools import ToolBox
from libertati.news.reader import NewsItem, NewsReader


class StubNews(NewsReader):
    def __init__(self):
        super().__init__(feeds=[])

    async def fetch(self, limit_per_feed: int = 5, topic: str | None = None):
        return [NewsItem(title="Big news", summary="stuff happened", link="x", source="Src")]


@pytest.fixture
def toolbox(history, memory):
    return ToolBox(history, memory, StubNews())


async def test_search_history_tool(toolbox, history):
    await history.add_message(make_message("libertarian ideas", message_id=1))
    out = json.loads(await toolbox.dispatch("search_history", {"query": "libertarian"}))
    assert out[0]["text"] == "libertarian ideas"


async def test_read_update_memory_tool(toolbox, memory):
    await toolbox.dispatch(
        "update_memory", {"path": "self.md", "content": "hi", "mode": "overwrite"}
    )
    out = json.loads(await toolbox.dispatch("read_memory", {"path": "self.md"}))
    assert out["content"].strip() == "hi"


async def test_remember_user_tool(toolbox, memory):
    res = json.loads(await toolbox.dispatch("remember_user", {"handle": "@z", "note": "cool"}))
    assert res["ok"] is True
    assert "cool" in memory.read_user("@z")


async def test_read_news_tool(toolbox):
    out = json.loads(await toolbox.dispatch("read_news", {}))
    assert "Big news" in out["digest"]


async def test_unknown_tool_returns_error(toolbox):
    out = json.loads(await toolbox.dispatch("nope", {}))
    assert "error" in out


async def test_bad_path_returns_error(toolbox):
    out = json.loads(await toolbox.dispatch("read_memory", {"path": "../oops.md"}))
    assert "error" in out


class FakeReader:
    supports_reading = True

    async def read_source(self, source, limit=15):
        return [make_message("interesting channel post", message_id=1, handle="@author")]


async def test_read_telegram_unavailable_without_reader(toolbox):
    out = json.loads(await toolbox.dispatch("read_telegram", {"source": "@chan"}))
    assert "error" in out


async def test_read_telegram_public_only_refuses_numeric(history, memory):
    tb = ToolBox(history, memory, StubNews(), reader=FakeReader(), public_only=True)
    out = json.loads(await tb.dispatch("read_telegram", {"source": "-100123"}))
    assert "error" in out


async def test_read_telegram_reads_public_channel(history, memory):
    tb = ToolBox(history, memory, StubNews(), reader=FakeReader(), public_only=True)
    out = json.loads(await tb.dispatch("read_telegram", {"source": "@chan"}))
    assert "interesting channel post" in out["transcript"]


async def test_read_telegram_numeric_allowed_when_public_only_off(history, memory):
    tb = ToolBox(history, memory, StubNews(), reader=FakeReader(), public_only=False)
    out = json.loads(await tb.dispatch("read_telegram", {"source": "-100123"}))
    assert "interesting channel post" in out["transcript"]
