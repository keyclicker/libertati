from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from libertati.config import Settings
from libertati.storage.db import Database
from libertati.storage.history import HistoryStore
from libertati.storage.memory import MemoryStore
from libertati.telegram.base import IncomingMessage


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        openai_api_key="test",
        bot_token="test",
        db_path=tmp_path / "test.db",
        memory_dir=tmp_path / "memory",
    )


@pytest.fixture
async def db(tmp_path) -> Database:
    database = Database(tmp_path / "test.db")
    await database.connect()
    yield database
    await database.close()


@pytest.fixture
async def history(db) -> HistoryStore:
    return HistoryStore(db, max_thread_chars=5000)


@pytest.fixture
def memory(tmp_path) -> MemoryStore:
    return MemoryStore(tmp_path / "memory")


def make_message(
    text: str,
    message_id: int = 1,
    chat_id: int = 100,
    handle: str = "@alice",
    reply_to: int | None = None,
    is_group: bool = True,
    from_self: bool = False,
) -> IncomingMessage:
    return IncomingMessage(
        chat_id=chat_id,
        message_id=message_id,
        text=text,
        user_handle=handle,
        user_name="Alice",
        reply_to_id=reply_to,
        is_group=is_group,
        chat_title="Test Group",
        ts=1000.0 + message_id,
        from_self=from_self,
    )


# --- OpenAI mock helpers ----------------------------------------------------
def _tool_call(call_id: str, name: str, arguments: dict) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def assistant(content: str | None = None, tool_calls=None) -> SimpleNamespace:
    return SimpleNamespace(content=content, tool_calls=tool_calls)


class FakeLLM:
    """Stand-in for LLMClient.chat yielding a scripted sequence of messages."""

    def __init__(self, script: list[SimpleNamespace]) -> None:
        self._script = list(script)
        self.calls: list[list[dict]] = []

    async def chat(self, messages, tools=None, max_retries: int = 4):
        self.calls.append(messages)
        if tools is None:
            # Emulate OpenAI: with no tools available the model must answer in text.
            for item in list(self._script):
                if not item.tool_calls:
                    self._script.remove(item)
                    return item
            return assistant("(forced answer)")
        if not self._script:
            return assistant("(fallback)")
        return self._script.pop(0)


@pytest.fixture
def tool_call():
    return _tool_call
