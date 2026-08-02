"""Telegram bot wiring: dispatcher, handlers, persistence and schedulers.

Handlers don't answer anything themselves — every incoming message is
persisted, formatted as an event and pushed to the single agent loop,
which replies (or not) through its ``send_message`` tool. Background
loops feed the same queue with due wakeups and heartbeat status events.
"""

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

from aiogram import BaseMiddleware, Bot, Dispatcher, Router
from aiogram.types import Message, TelegramObject

from libertati import clock
from libertati.agent import Agent
from libertati.config import Settings
from libertati.db import Database

log = logging.getLogger(__name__)

router = Router()

#: How often the wakeup scheduler checks for due alarms.
WAKEUP_POLL_SECONDS = 30


class PersistMiddleware(BaseMiddleware):
    """Store every incoming (or edited) message before it is handled.

    Registered as outer middleware so messages are saved even when no
    handler matches them.
    """

    def __init__(self, db: Database) -> None:
        """Keep the database handle used for saving."""
        self.db = db

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        """Save the message, then pass the event on to the handlers."""
        if isinstance(event, Message):
            await self.db.save_message(event)
        return await handler(event, data)


def describe(message: Message, tz: ZoneInfo) -> str:
    """Format an incoming message as a one-line event for the agent."""
    chat = message.chat
    where = f"chat {chat.id} ({chat.type}"
    if chat.title:
        where += f" “{chat.title}”"
    where += ")"
    sender = "unknown"
    if message.from_user is not None:
        sender = message.from_user.full_name
        if message.from_user.username:
            sender += f" @{message.from_user.username}"
    ref = f"msg {message.message_id}"
    if message.reply_to_message is not None:
        ref += f", replying to msg {message.reply_to_message.message_id}"
    body = message.text or message.caption or f"<{message.content_type}>"
    return f"[{clock.fmt(message.date, tz)}] {where} | {sender} ({ref}): {body}"


@router.message()
async def on_message(message: Message, agent: Agent, tz: ZoneInfo) -> None:
    """Push any incoming message to the agent loop as an event."""
    await agent.push(describe(message, tz))


async def wakeup_loop(agent: Agent, db: Database, tz: ZoneInfo) -> None:
    """Deliver due self-scheduled wakeups to the agent queue."""
    while True:
        now = clock.utc_stamp(datetime.now(UTC))
        for wakeup_id, due_at, note in await db.due_wakeups(now):
            await db.complete_wakeup(wakeup_id)
            due_local = clock.fmt(
                datetime.strptime(due_at, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC), tz
            )
            await agent.push(
                f"[wakeup #{wakeup_id} at {clock.now(tz)} — you scheduled it "
                f"for {due_local}] {note}"
            )
        await asyncio.sleep(WAKEUP_POLL_SECONDS)


async def heartbeat_digest(db: Database, tz: ZoneInfo) -> str:
    """Build the status text attached to a heartbeat event."""
    parts = []
    unanswered = await db.unanswered_chats()
    if unanswered:
        chats = []
        for row in unanswered:
            name = row["title"] or row["first_name"] or "?"
            when = clock.fmt(datetime.fromisoformat(row["date"]), tz)
            chats.append(
                f"“{name}” (chat {row['chat_id']}, {row['type']}, last {when})"
            )
        parts.append("chats with unanswered last message: " + "; ".join(chats))
    else:
        parts.append("no unanswered chats")
    pending = await db.pending_wakeups()
    if pending:
        alarms = ", ".join(
            f"#{wid} at {clock.fmt(datetime.strptime(due, '%Y-%m-%d %H:%M:%S').replace(tzinfo=UTC), tz)}: {note}"
            for wid, due, note in pending
        )
        parts.append(f"pending wakeups: {alarms}")
    return ". ".join(parts)


async def heartbeat_loop(
    agent: Agent, db: Database, tz: ZoneInfo, minutes: int
) -> None:
    """Push periodic (jittered) heartbeat status events; 0 disables."""
    if minutes <= 0:
        return
    while True:
        await asyncio.sleep(minutes * 60 * random.uniform(0.8, 1.2))
        digest = await heartbeat_digest(db, tz)
        await agent.push(f"[heartbeat {clock.now(tz)}] {digest}")


async def run() -> None:
    """Assemble the bot and run long polling until cancelled.

    Loads settings, connects the database, starts the agent worker plus
    the wakeup and heartbeat loops, and routes all incoming messages to
    the agent. Background tasks are cancelled and the database closed on
    the way out.
    """
    settings = Settings()
    logging.basicConfig(level=settings.log_level)

    db = Database(settings.db_path)
    await db.connect()

    bot = Bot(token=settings.bot_token)
    agent = Agent(settings, db, bot)
    await agent.load()
    tz = agent.tz
    tasks = [
        asyncio.create_task(agent.worker()),
        asyncio.create_task(wakeup_loop(agent, db, tz)),
        asyncio.create_task(heartbeat_loop(agent, db, tz, settings.heartbeat_minutes)),
    ]

    dispatcher = Dispatcher(agent=agent, tz=tz)
    persist = PersistMiddleware(db)
    dispatcher.message.outer_middleware(persist)
    dispatcher.edited_message.outer_middleware(persist)
    dispatcher.include_router(router)
    try:
        await dispatcher.start_polling(bot)
    finally:
        for task in tasks:
            task.cancel()
        await db.close()
