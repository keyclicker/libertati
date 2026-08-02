"""Telegram bot wiring: dispatcher, handlers, persistence and schedulers.

Handlers don't answer anything themselves — every incoming message is
persisted, formatted as an event and pushed to the single agent loop,
which replies (or not) through its ``send_message`` tool. Background
loops feed the same queue with due wakeups and heartbeat status events,
and hand the agent over to the dreaming loop when it has been idle long
enough.
"""

import asyncio
import logging
import random
import re
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

from aiogram import BaseMiddleware, Bot, Dispatcher, Router
from aiogram.types import Message, TelegramObject, User

from libertati import clock
from libertati.agent import Agent
from libertati.chats import ChatRegistry
from libertati.config import Settings
from libertati.db import Database
from libertati.dream import Dreamer, DreamGate

log = logging.getLogger(__name__)

router = Router()

#: How often the wakeup scheduler checks for due alarms.
WAKEUP_POLL_SECONDS = 30

#: How often the dream loop checks whether it should take over.
DREAM_POLL_SECONDS = 60

#: Max message body length quoted into an event (rest is elided).
EVENT_TEXT_LIMIT = 1000


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


def format_event(message: Message, tz: ZoneInfo) -> str:
    """Format an incoming message as a one-line event for the agent."""
    chat = message.chat
    title = f" “{chat.title}”" if chat.title else ""
    where = f"chat {chat.id} ({chat.type}{title})"
    user = message.from_user
    if user is None:
        sender = "unknown"
    else:
        sender = (
            f"{user.full_name} @{user.username}" if user.username else user.full_name
        )
    ref = f"msg {message.message_id}"
    if message.reply_to_message is not None:
        ref += f", replying to msg {message.reply_to_message.message_id}"
    body = message.text or message.caption or f"<{message.content_type}>"
    if len(body) > EVENT_TEXT_LIMIT:
        body = body[:EVENT_TEXT_LIMIT] + f" […{len(body) - EVENT_TEXT_LIMIT} chars]"
    return (
        f"[{clock.format_local(message.date, tz)}] {where} | {sender} ({ref}): {body}"
    )


def is_addressed(message: Message, me: User) -> bool:
    """True when a message replies to the bot or @-mentions its username."""
    reply = message.reply_to_message
    if (
        reply is not None
        and reply.from_user is not None
        and reply.from_user.id == me.id
    ):
        return True
    if me.username:
        body = message.text or message.caption or ""
        # Bounded on both sides so "@name" doesn't match inside a longer
        # "@namesake" mention or an email-like "user@name".
        pattern = rf"(?<![\w@])@{re.escape(me.username)}(?![A-Za-z0-9_])"
        return re.search(pattern, body, re.IGNORECASE) is not None
    return False


def chat_label(message: Message) -> str:
    """Human-readable chat name for the approval registry comment."""
    chat = message.chat
    name = chat.title or chat.full_name or chat.username or "?"
    return f"{name} ({chat.type})"


@router.message()
async def on_message(
    message: Message,
    agent: Agent,
    tz: ZoneInfo,
    me: User,
    registry: ChatRegistry,
) -> None:
    """Push an incoming message to the agent loop as an event.

    Messages from unapproved chats are persisted but never become
    events (in approval mode the chat lands in chats.toml for review).
    Group messages become events only when the bot is mentioned or
    replied to — the agent catches up on the rest via history tools on
    heartbeats.
    """
    if not registry.register(message.chat.id, chat_label(message)):
        return
    if message.chat.type != "private" and not is_addressed(message, me):
        return
    await agent.push(format_event(message, tz))


async def deliver_wakeups(agent: Agent, db: Database, tz: ZoneInfo) -> None:
    """Push every due wakeup to the agent queue, one delivery pass.

    A wakeup is marked done only after it was pushed, so a crash in
    between redelivers it (at-least-once) rather than dropping it.
    """
    now = clock.utc_stamp(datetime.now(UTC))
    for wakeup in await db.due_wakeups(now):
        due_local = clock.format_local(clock.parse_utc_stamp(wakeup["due_at"]), tz)
        await agent.push(
            f"[wakeup #{wakeup['id']} at {clock.format_now(tz)} — you "
            f"scheduled it for {due_local}] {wakeup['note']}"
        )
        await db.complete_wakeup(wakeup["id"])


async def wakeup_loop(agent: Agent, db: Database, tz: ZoneInfo) -> None:
    """Deliver due self-scheduled wakeups to the agent queue.

    A transient failure must not take the loop down with it — wakeups
    would then silently never fire again until a restart.
    """
    while True:
        try:
            await deliver_wakeups(agent, db, tz)
        except Exception:
            log.exception("wakeup delivery failed")
        await asyncio.sleep(WAKEUP_POLL_SECONDS)


async def heartbeat_digest(db: Database, tz: ZoneInfo, registry: ChatRegistry) -> str:
    """Build the status text attached to a heartbeat event.

    Unapproved chats are left out — the agent shouldn't be nudged
    towards chats it isn't allowed to see.
    """
    parts = []
    unanswered = [
        row for row in await db.unanswered_chats() if registry.check(row["chat_id"])
    ]
    if unanswered:
        chats = []
        for row in unanswered:
            name = row["title"] or row["first_name"] or "?"
            when = clock.format_local(datetime.fromisoformat(row["date"]), tz)
            chats.append(
                f"“{name}” (chat {row['chat_id']}, {row['type']}, last {when})"
            )
        parts.append("chats with unanswered last message: " + "; ".join(chats))
    else:
        parts.append("no unanswered chats")
    pending = await db.pending_wakeups()
    if pending:
        alarms = ", ".join(
            f"#{row['id']} at "
            f"{clock.format_local(clock.parse_utc_stamp(row['due_at']), tz)}: "
            f"{row['note']}"
            for row in pending
        )
        parts.append(f"pending wakeups: {alarms}")
    return ". ".join(parts)


async def heartbeat_loop(
    agent: Agent, db: Database, tz: ZoneInfo, minutes: int, registry: ChatRegistry
) -> None:
    """Push periodic (jittered) heartbeat status events; 0 disables.

    Heartbeats are pushed as non-activity events: a heartbeat turn where
    the agent only reads leaves the dream idle clock alone, so regular
    heartbeats don't make the idle dream trigger unreachable. A transient
    failure is logged and the loop carries on.
    """
    if minutes <= 0:
        return
    while True:
        await asyncio.sleep(minutes * 60 * random.uniform(0.8, 1.2))
        try:
            digest = await heartbeat_digest(db, tz, registry)
            await agent.push(
                f"[heartbeat {clock.format_now(tz)}] {digest}", activity=False
            )
        except Exception:
            log.exception("heartbeat failed")


async def dream_loop(dreamer: Dreamer) -> None:
    """Let the dreaming loop take over whenever its conditions hold.

    Only polls: the conditions (idleness, cooldown, budget, a pending
    request) all live in :meth:`Dreamer.maybe_dream`. A failing dream
    must not take the loop down with it — the agent would then never
    sleep again until a restart.
    """
    if dreamer.gate.daily_budget <= 0:
        return
    while True:
        await asyncio.sleep(DREAM_POLL_SECONDS)
        try:
            await dreamer.maybe_dream()
        except Exception:
            log.exception("dream loop failed")


async def run() -> None:
    """Assemble the bot and run long polling until cancelled.

    Loads settings, connects the database, starts the agent worker plus
    the wakeup, heartbeat and dream loops, and routes all incoming
    messages to the agent. Background tasks are cancelled and the
    database closed on the way out.
    """
    settings = Settings()
    logging.basicConfig(level=settings.log_level)

    db = Database(settings.db_path)
    await db.connect()

    bot = Bot(token=settings.bot_token)
    dream_gate = DreamGate(db, settings.dream_daily_budget)
    agent = Agent(settings, db, bot, dream_gate)
    await agent.load()
    tz = agent.tz
    dreamer = Dreamer(settings, db, bot, agent, dream_gate)
    registry = ChatRegistry(settings.chats_path, settings.chat_approval)
    tasks = [
        asyncio.create_task(agent.run_forever()),
        asyncio.create_task(wakeup_loop(agent, db, tz)),
        asyncio.create_task(
            heartbeat_loop(agent, db, tz, settings.heartbeat_minutes, registry)
        ),
        asyncio.create_task(dream_loop(dreamer)),
    ]

    dispatcher = Dispatcher(agent=agent, tz=tz, me=await bot.me(), registry=registry)
    persist_middleware = PersistMiddleware(db)
    dispatcher.message.outer_middleware(persist_middleware)
    # Edits are persisted (updating the stored row) but deliberately not
    # pushed as events — the agent only reacts to new messages.
    dispatcher.edited_message.outer_middleware(persist_middleware)
    dispatcher.include_router(router)
    try:
        await dispatcher.start_polling(bot)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await db.close()
