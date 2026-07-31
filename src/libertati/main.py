"""Entry point: assemble everything and run the bot."""

from __future__ import annotations

import asyncio
import contextlib
import random

from .behavior import mentions_bot, typing_delay_seconds
from .config import Settings, load_settings
from .llm.agent import Agent
from .llm.client import LLMClient
from .llm.tools import ToolBox
from .logging import get_logger, setup_logging
from .news.reader import NewsReader
from .observability import EventLogger
from .scheduler.runner import Scheduler
from .storage.db import Database
from .storage.history import HistoryStore
from .storage.memory import MemoryStore
from .telegram.base import IncomingMessage
from .telegram.bot_api import BotApiClient
from .telegram.webpreview import WebPreviewReader

log = get_logger("main")


class App:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.events = EventLogger(
            path=settings.event_log_path,
            enabled=settings.event_log_enabled,
            log_content=settings.log_message_content,
            content_max=settings.log_content_max_chars,
        )
        self.db = Database(settings.db_path)
        self.memory = MemoryStore(
            settings.memory_dir,
            events=self.events,
            max_file_chars=settings.memory_max_file_chars,
        )
        self.news = NewsReader(settings.news_feeds)
        self.client = BotApiClient(token=settings.bot_token)
        self.reader = WebPreviewReader()
        # history/agent are built after the DB connects
        self.history: HistoryStore | None = None
        self.agent: Agent | None = None
        self.scheduler: Scheduler | None = None

    async def _on_message(self, incoming: IncomingMessage) -> None:
        assert self.history is not None and self.agent is not None
        # Never act on our own outgoing messages, but do log them.
        role = "assistant" if incoming.from_self else "user"
        await self.history.add_message(incoming, role=role)

        allowed = self.settings.is_chat_allowed(incoming.chat_id, incoming.chat_username)
        self.events.emit(
            "inbound",
            chat_id=incoming.chat_id,
            chat_title=incoming.chat_title,
            user=incoming.user_handle or incoming.user_name,
            is_group=incoming.is_group,
            allowed=allowed,
            from_self=incoming.from_self,
            text_len=len(incoming.text),
            preview=self.events.redact(incoming.text),
        )

        skip = None
        if incoming.from_self or not incoming.text.strip():
            skip = "self_or_empty"
        elif not self.settings.respond_to_all:
            skip = "respond_to_all_off"
        elif not allowed:
            skip = "not_allowed"
        elif (
            incoming.is_group
            and not await self._is_addressed(incoming)
            and random.random() >= self.settings.group_reply_chance
        ):
            # Like a person, stay out of most ambient group chatter unless spoken to.
            skip = "ambient_group"
        if skip is not None:
            self.events.emit("skip", reason=skip, chat_id=incoming.chat_id)
            return

        try:
            reply = await self.agent.respond(incoming)
        except Exception as exc:  # noqa: BLE001
            log.exception("agent failed to respond: %s", exc)
            self.events.emit("respond_error", chat_id=incoming.chat_id, error=str(exc)[:200])
            return
        if not reply:
            return
        if self.settings.typing_delay_enabled:
            await asyncio.sleep(
                typing_delay_seconds(reply, self.settings.typing_delay_max_seconds)
            )
        # Quote-reply only in groups (to thread the answer); in DMs quoting every
        # message reads like a bot.
        sent = await self.client.send_message(
            incoming.chat_id,
            reply,
            reply_to_id=incoming.message_id if incoming.is_group else None,
        )
        self.events.emit(
            "outbound", chat_id=incoming.chat_id, ok=sent is not None, text_len=len(reply)
        )
        if sent is not None:
            await self.history.add_message(sent, role="assistant")

    async def _is_addressed(self, incoming: IncomingMessage) -> bool:
        """Whether this group message is aimed at the bot (mention or reply-to-us)."""
        s = self.settings
        if mentions_bot(incoming.text, [s.bot_username, s.bot_handle, s.bot_full_name]):
            return True
        if incoming.reply_to_id is not None and self.history is not None:
            replied = await self.history.get_message(incoming.chat_id, incoming.reply_to_id)
            if replied is not None and replied.role == "assistant":
                return True
        return False

    def _build_runtime(self) -> None:
        """Assemble history/agent/scheduler (assumes the DB is already connected)."""
        self.history = HistoryStore(self.db, self.settings.max_thread_chars)
        llm = LLMClient(self.settings, events=self.events)
        tools = ToolBox(
            self.history,
            self.memory,
            self.news,
            reader=self.reader,
            events=self.events,
        )
        self.agent = Agent(
            self.settings, llm, tools, self.history, self.memory, events=self.events
        )
        self.scheduler = Scheduler(
            self.settings, self.agent, self.client, self.history, self.memory, self.news,
            reader=self.reader,
        )

    async def run(self) -> None:
        await self.db.connect()
        self._build_runtime()
        assert self.scheduler is not None
        self.client.on_message(self._on_message)
        self.scheduler.start()
        log.info("libertati is up")
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
        self.events.close()
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
