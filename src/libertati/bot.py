"""Telegram bot wiring: dispatcher, handlers, persistence and schedulers.

Handlers don't answer anything themselves — every incoming message is
persisted, formatted as an event and pushed to the single agent loop,
which replies (or not) through its ``send_message`` tool. Background
loops feed the same queue with due wakeups, heartbeat status events and
whatever the operator typed at the spy console, and hand the agent over
to the dreaming loop when it has been idle long enough.
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
from aiogram.types import (
    Message,
    MessageReactionUpdated,
    ReactionType,
    ReactionTypeCustomEmoji,
    ReactionTypeEmoji,
    TelegramObject,
    User,
)

from libertati import clock
from libertati.agent import Agent, one_line
from libertati.chats import ChatRegistry
from libertati.config import Settings
from libertati.db import Database, effective_reply_to
from libertati.dream import Dreamer, DreamGate
from libertati.transcript import render_message, render_messages

log = logging.getLogger(__name__)

router = Router()

#: How often the wakeup scheduler checks for due alarms.
WAKEUP_POLL_SECONDS = 30

#: How often the dream loop checks whether it should take over.
DREAM_POLL_SECONDS = 60

#: How often the console is checked for instructions waiting in the
#: database. Far tighter than the other loops: an operator has just
#: typed and is watching for the turn to start.
STEERING_POLL_SECONDS = 2

#: Max message body length quoted into an event (rest is elided).
EVENT_TEXT_LIMIT = 1000

#: Most messages an event carries along as the context the agent has not
#: been shown yet, and how much of each body. Bounded because every event
#: stays in the agent's context for good: enough to answer a group that
#: moved on without it, not a second copy of the chat history.
EVENT_CONTEXT_LIMIT = 8
EVENT_CONTEXT_TEXT_LIMIT = 300

#: Most chats/topics and pending wakeups one heartbeat spells out. Every
#: heartbeat is appended to the agent's context for good, so an unbounded
#: digest would grow the prompt with each one; the rest is counted.
HEARTBEAT_DIGEST_LIMIT = 20


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


def format_event(message: Message, tz: ZoneInfo, topic_name: str | None = None) -> str:
    """Format an incoming message as a one-line event for the agent.

    Strictly one line: every interpolated field is sender-controlled, and
    a body (or name) containing a newline could otherwise forge extra
    event lines — a fake wakeup, a fake message from another chat — and
    steer the agent. :meth:`Agent.push` collapses whatever slips through;
    the body is folded here instead so its newlines survive visibly, as a
    literal backslash-n, rather than reading as ordinary spaces.
    Forum topic messages name their topic after the chat; the caller
    resolves ``topic_name`` (this function stays sync and DB-free).
    """
    chat = message.chat
    title = f" “{one_line(chat.title)}”" if chat.title else ""
    if message.is_topic_message and message.message_thread_id:
        topic = f"topic {message.message_thread_id}"
        if topic_name:
            topic += f" “{one_line(topic_name)}”"
        where = f"chat {chat.id} ({chat.type}{title}, {topic})"
    else:
        where = f"chat {chat.id} ({chat.type}{title})"
    user = message.from_user
    if user is None:
        sender = "unknown"
    else:
        sender = (
            f"{user.full_name} @{user.username}" if user.username else user.full_name
        )
    ref = f"msg {message.message_id}"
    reply_to = effective_reply_to(message)
    if reply_to is not None:
        ref += f", replying to msg {reply_to}"
    if message.forum_topic_created is not None:
        body = f"<forum_topic_created “{one_line(message.forum_topic_created.name)}”>"
    else:
        body = message.text or message.caption or f"<{message.content_type}>"
    body = "\\n".join(body.splitlines())
    if len(body) > EVENT_TEXT_LIMIT:
        body = body[:EVENT_TEXT_LIMIT] + f" […{len(body) - EVENT_TEXT_LIMIT} chars]"
    return (
        f"[{clock.format_local(message.date, tz)}] {where} | {one_line(sender)}"
        f" ({ref}): {body}"
    )


def event_thread_id(message: Message) -> int | None:
    """Forum topic a message belongs to, or ``None`` outside topics."""
    if message.is_topic_message and message.message_thread_id:
        return message.message_thread_id
    return None


async def event_context(db: Database, message: Message, tz: ZoneInfo) -> list[str]:
    """Transcript lines for what the agent has not been shown yet.

    An incoming message rarely stands alone: in a group only the ones
    addressing the agent become events, so by the time it is pulled in
    the conversation has usually moved several messages on, and the
    message it answers may be one of them. Carrying that along means a
    turn starts knowing what happened instead of spending two lookups
    rediscovering it.

    Older messages beyond the cap are announced rather than quoted; the
    agent can page back to them with ``get_recent_messages``.
    """
    chat_id = message.chat.id
    thread_id = event_thread_id(message)
    rows = await db.messages_since_read(
        chat_id, EVENT_CONTEXT_LIMIT + 1, message.message_id, thread_id
    )
    dropped = max(0, len(rows) - EVENT_CONTEXT_LIMIT)
    rows = rows[dropped:]
    lines = []
    reply_to = effective_reply_to(message)
    if reply_to is not None and all(row["message_id"] != reply_to for row in rows):
        parent = await db.message_row(chat_id, reply_to)
        if parent is not None:
            quoted = render_message(parent, tz, text_limit=EVENT_CONTEXT_TEXT_LIMIT)
            lines.append(f"[replies to] {quoted}")
    if rows:
        lines.append("[earlier here, not shown to you yet]")
        if dropped:
            lines.append("[older ones skipped — get_recent_messages has them]")
        lines.extend(
            render_messages(
                rows,
                tz,
                show_topic=thread_id is None,
                text_limit=EVENT_CONTEXT_TEXT_LIMIT,
            )
        )
    return lines


def is_addressed(message: Message, me: User) -> bool:
    """True when a message replies to the bot or @-mentions its username."""
    reply = message.reply_to_message
    if (
        reply is not None
        and reply.from_user is not None
        and reply.from_user.id == me.id
        # A topic's creation service message is "from" whoever created the
        # topic; the pseudo-reply to it every topic message carries must
        # not make a bot-created topic address the bot wholesale.
        and reply.forum_topic_created is None
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


def format_reaction(reaction: ReactionType) -> str:
    """Format one Telegram reaction for an agent event."""
    if isinstance(reaction, ReactionTypeEmoji):
        return reaction.emoji
    if isinstance(reaction, ReactionTypeCustomEmoji):
        return f"custom emoji {reaction.custom_emoji_id}"
    return "paid reaction"


def format_reaction_event(update: MessageReactionUpdated, tz: ZoneInfo) -> str:
    """Format a reaction change as a one-line event for the agent."""
    chat = update.chat
    title = f" “{one_line(chat.title)}”" if chat.title else ""
    where = f"chat {chat.id} ({chat.type}{title})"
    if update.user is not None:
        actor = (
            f"{update.user.full_name} @{update.user.username}"
            if update.user.username
            else update.user.full_name
        )
    elif update.actor_chat is not None:
        actor = update.actor_chat.title or update.actor_chat.full_name or "unknown chat"
    else:
        actor = "unknown"
    old = {format_reaction(item) for item in update.old_reaction}
    new = {format_reaction(item) for item in update.new_reaction}
    changes = []
    if added := sorted(new - old):
        changes.append("added " + ", ".join(added))
    if removed := sorted(old - new):
        changes.append("removed " + ", ".join(removed))
    change = "; ".join(changes) or "reaction unchanged"
    return (
        f"[{clock.format_local(update.date, tz)}] {where} | "
        f"{one_line(actor)} reacted to your msg {update.message_id}: {change}"
    )


@router.message()
async def on_message(
    message: Message,
    agent: Agent,
    tz: ZoneInfo,
    me: User,
    registry: ChatRegistry,
    db: Database,
) -> None:
    """Push an incoming message to the agent loop as an event.

    Messages from unapproved chats are persisted but never become
    events (in approval mode the chat lands in chats.toml for review).
    Group messages become events only when the bot is mentioned or
    replied to — the rest lands in history and rides along with the next
    event as the context the agent has not seen. Forum topic messages
    resolve their topic name here (PersistMiddleware has already saved
    the message, so even the first message seen in a topic can name
    itself from its own payload).

    Everything the event carries counts as read, the skipped older
    messages included: they were deliberately left out, and leaving them
    unread would quote them into every later event instead. The cursor
    is handed to the agent rather than moved here, so it only advances
    once the event is persisted and the messages really have been shown.
    """
    if not registry.register(message.chat.id, chat_label(message)):
        return
    if message.chat.type != "private" and not is_addressed(message, me):
        return
    thread_id = event_thread_id(message)
    topic_name = await db.topic_name(message.chat.id, thread_id) if thread_id else None
    context = await event_context(db, message, tz)
    await agent.push(
        [format_event(message, tz, topic_name=topic_name), *context],
        read_mark=(message.chat.id, message.message_id, thread_id),
    )


@router.message_reaction()
async def on_message_reaction(
    event: MessageReactionUpdated,
    agent: Agent,
    tz: ZoneInfo,
    me: User,
    registry: ChatRegistry,
    db: Database,
) -> None:
    """Push human reactions to the bot's stored messages as events."""
    name = event.chat.title or event.chat.full_name or event.chat.username or "?"
    if not registry.register(event.chat.id, f"{name} ({event.chat.type})"):
        return
    if event.user is not None and event.user.id == me.id:
        return
    if not await db.message_is_outgoing(event.chat.id, event.message_id):
        return
    await agent.push(format_reaction_event(event, tz))


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


async def deliver_steering(agent: Agent, dreamer: Dreamer) -> None:
    """Queue the console's waiting instructions, one delivery pass.

    Urgent ones are left where they are while a waking turn holds the
    lock: that turn collects them itself between rounds, which is the
    whole point of marking one urgent. Should the lock be taken right
    after it was read as free, the instruction is queued instead and
    arrives one turn later — later than asked for, but never twice and
    never lost.

    A dream holds the same lock and has no round boundary to collect
    anything at, so during one there is nobody to leave them for: they
    are queued like the rest and land in the first batch after waking.
    """
    collected = agent.turn_lock.locked() and dreamer.dream_id is None
    await agent.deliver_steering(False if collected else None)


async def steering_loop(agent: Agent, dreamer: Dreamer) -> None:
    """Deliver instructions typed at the operator console.

    A transient failure must not take the loop down: the console would
    then go quiet until a restart, with nothing there to say so.
    """
    while True:
        try:
            await deliver_steering(agent, dreamer)
        except Exception:
            log.exception("steering delivery failed")
        await asyncio.sleep(STEERING_POLL_SECONDS)


def and_more(rendered: list[str], total: int, separator: str) -> str:
    """Join the entries that fit and count the ones left out."""
    hidden = total - len(rendered)
    text = separator.join(rendered)
    return f"{text} (+{hidden} more)" if hidden > 0 else text


async def heartbeat_digest(db: Database, tz: ZoneInfo, registry: ChatRegistry) -> str:
    """Build the status text attached to a heartbeat event.

    Unapproved chats are left out — the agent shouldn't be nudged
    towards chats it isn't allowed to see. Forum chats report per topic,
    so an answered topic can't hide an unanswered one.

    Both lists are capped at :data:`HEARTBEAT_DIGEST_LIMIT` entries. They
    arrive longest-waiting and soonest-due first, so the cap keeps what
    actually needs the agent — and the entries beyond it, whose topic
    names are never looked up, only cost a count.
    """
    parts = []
    rows = await db.unanswered_chats()
    allowed = registry.approved(row["chat_id"] for row in rows)
    unanswered = [row for row in rows if row["chat_id"] in allowed]
    if unanswered:
        chats = []
        for row in unanswered[:HEARTBEAT_DIGEST_LIMIT]:
            name = row["title"] or row["first_name"] or "?"
            when = clock.format_local(datetime.fromisoformat(row["date"]), tz)
            where = f"chat {row['chat_id']}, {row['type']}"
            thread_id = row["message_thread_id"]
            if thread_id:
                topic_name = await db.topic_name(row["chat_id"], thread_id)
                where += f", topic {thread_id}"
                if topic_name:
                    where += f" “{one_line(topic_name)}”"
            chats.append(f"“{name}” ({where}, last {when})")
        parts.append(
            "chats with unanswered last message: "
            + and_more(chats, len(unanswered), "; ")
        )
    else:
        parts.append("no unanswered chats")
    pending = await db.pending_wakeups()
    if pending:
        alarms = [
            f"#{row['id']} at "
            f"{clock.format_local(clock.parse_utc_stamp(row['due_at']), tz)}: "
            f"{row['note']}"
            for row in pending[:HEARTBEAT_DIGEST_LIMIT]
        ]
        parts.append(f"pending wakeups: {and_more(alarms, len(pending), ', ')}")
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


def polled_updates(dispatcher: Dispatcher) -> list[str]:
    """Update types to ask Telegram for, middleware-only ones included.

    Telegram sends nothing outside ``allowed_updates``, and aiogram's own
    resolution only counts observers carrying a *handler*. Edits are
    persisted by an outer middleware and have no handler, so without
    naming them here an edit would never reach the database at all.
    """
    return sorted({*dispatcher.resolve_used_update_types(), "edited_message"})


async def run() -> None:
    """Assemble the bot and run long polling until cancelled.

    Loads settings, connects the database, starts the agent worker plus
    the wakeup, heartbeat, dream and steering loops, and routes all incoming
    messages to the agent. Background tasks are cancelled and the
    database closed on the way out.
    """
    settings = Settings()
    logging.basicConfig(level=settings.log_level)

    db = Database(settings.db_path)
    await db.connect()

    bot = Bot(token=settings.bot_token)
    dream_gate = DreamGate(
        db, settings.dream_daily_budget, settings.dream_cooldown_minutes
    )
    registry = ChatRegistry(settings.chats_path, settings.chat_approval)
    agent = Agent(settings, db, bot, registry, dream_gate)
    await agent.load()
    tz = agent.tz
    dreamer = Dreamer(settings, db, bot, agent, dream_gate)
    tasks = [
        asyncio.create_task(agent.run_forever()),
        asyncio.create_task(wakeup_loop(agent, db, tz)),
        asyncio.create_task(
            heartbeat_loop(agent, db, tz, settings.heartbeat_minutes, registry)
        ),
        asyncio.create_task(dream_loop(dreamer)),
        asyncio.create_task(steering_loop(agent, dreamer)),
    ]

    dispatcher = Dispatcher(
        agent=agent, tz=tz, me=await bot.me(), registry=registry, db=db
    )
    persist_middleware = PersistMiddleware(db)
    dispatcher.message.outer_middleware(persist_middleware)
    # Edits are persisted (updating the stored row) but deliberately not
    # pushed as events — the agent only reacts to new messages.
    dispatcher.edited_message.outer_middleware(persist_middleware)
    dispatcher.include_router(router)
    try:
        await dispatcher.start_polling(bot, allowed_updates=polled_updates(dispatcher))
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await db.close()
