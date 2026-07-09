"""Entry point: assemble everything and run the bot."""

from __future__ import annotations

import asyncio
import contextlib

from .config import Settings, load_settings
from .llm.agent import Agent
from .llm.client import LLMClient
from .llm.tools import ToolBox
from .logging import get_logger, setup_logging
from .news.reader import NewsReader
from .scheduler.runner import Scheduler
from .storage.db import Database
from .storage.history import HistoryStore
from .storage.memory import MemoryStore
from .telegram.base import IncomingMessage
from .telegram.factory import build_client

log = get_logger("main")


class App:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.db = Database(settings.db_path)
        self.memory = MemoryStore(settings.memory_dir)
        self.news = NewsReader(settings.news_feeds)
        self.client = build_client(settings)
        # history/agent are built after the DB connects
        self.history: HistoryStore | None = None
        self.agent: Agent | None = None
        self.scheduler: Scheduler | None = None

    async def _on_message(self, incoming: IncomingMessage) -> None:
        assert self.history is not None and self.agent is not None
        # Never act on our own outgoing messages, but do log them.
        role = "assistant" if incoming.from_self else "user"
        await self.history.add_message(incoming, role=role)
        if incoming.from_self or not incoming.text.strip():
            return
        if not self.settings.respond_to_all:
            return
        try:
            reply = await self.agent.respond(incoming)
        except Exception as exc:  # noqa: BLE001
            log.exception("agent failed to respond: %s", exc)
            return
        if not reply:
            return
        sent = await self.client.send_message(
            incoming.chat_id, reply, reply_to_id=incoming.message_id
        )
        if sent is not None:
            await self.history.add_message(sent, role="assistant")

    async def run(self) -> None:
        await self.db.connect()
        self.history = HistoryStore(self.db, self.settings.max_thread_chars)
        llm = LLMClient(self.settings)
        tools = ToolBox(self.history, self.memory, self.news)
        self.agent = Agent(self.settings, llm, tools, self.history, self.memory)
        self.scheduler = Scheduler(
            self.settings, self.agent, self.client, self.history, self.memory, self.news
        )

        self.client.on_message(self._on_message)
        self.scheduler.start()
        log.info("libertati is up (mode=%s)", self.settings.telegram_mode)
        try:
            await self.client.start()
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        if self.scheduler is not None:
            self.scheduler.shutdown()
        with contextlib.suppress(Exception):
            await self.client.stop()
        await self.db.close()
        log.info("libertati stopped")


async def main() -> None:
    settings = load_settings()
    setup_logging(settings.log_level)
    settings.validate_runtime()
    app = App(settings)
    await app.run()


def run() -> None:
    """Console-script entry point."""
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass


if __name__ == "__main__":
    run()
