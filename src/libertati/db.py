"""SQLite persistence for messages, chats and users.

Every message flowing through the bot (incoming and its own replies) is
stored, along with the full raw Telegram payload for metadata not covered
by dedicated columns. Tables are declared in :mod:`libertati.schema`;
schema creation and upgrades are Alembic's job
(:func:`libertati.migrations.upgrade_to_head`), run before connecting.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aiogram.types import Chat, Message, User
from sqlalchemy import Select, delete, desc, event, func, or_, select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from .schema import (
    agent_turns,
    api_usage,
    chats,
    context,
    dream_context,
    dreams,
    media_notes,
    message_read_cursors,
    messages,
    steering,
    users,
    wakeups,
)

#: Forum housekeeping messages, which nobody is waiting on an answer to.
_FORUM_SERVICE_TYPES = (
    "forum_topic_created",
    "forum_topic_edited",
    "forum_topic_closed",
    "forum_topic_reopened",
)


def effective_reply_to(message: Message) -> int | None:
    """Return the real reply target of a message, or ``None``.

    Telegram makes every non-reply message in a forum topic "reply to"
    the topic-creation service message (backward compatibility); such
    pseudo-replies are not replies and are reported as ``None``.
    """
    reply = message.reply_to_message
    if reply is None or reply.forum_topic_created is not None:
        return None
    return reply.message_id


def _dump_context(item: dict) -> str:
    """Serialize context readably, escaping only invalid Unicode."""
    text = json.dumps(item, ensure_ascii=False)
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return json.dumps(item)
    return text


def _configure_connection(dbapi_connection: Any, _record: Any) -> None:
    """Set per-connection pragmas and register the casefold function.

    Runs for every connection the pool opens: pragmas and custom SQL
    functions are per-connection state, not per-engine.
    """
    # SQLite's own LIKE/NOCASE are case-insensitive for ASCII only;
    # message search needs real Unicode folding (Cyrillic etc.).
    dbapi_connection.create_function(
        "casefold",
        1,
        lambda text: text.casefold() if text else "",
        deterministic=True,
    )
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode = WAL")
    cursor.execute("PRAGMA foreign_keys = ON")
    # Pooled connections write concurrently, so a briefly locked
    # database is normal; wait it out instead of failing.
    cursor.execute("PRAGMA busy_timeout = 5000")
    cursor.close()


def _message_select() -> Select:
    """Message columns returned to the LLM as chat context.

    The media note rides along so a picture reads as what it depicts
    wherever a transcript is rendered, instead of as a bare ``<photo>``.
    """
    return select(
        messages.c.message_id,
        messages.c.date,
        messages.c.outgoing,
        users.c.username,
        users.c.first_name,
        messages.c.text,
        messages.c.caption,
        messages.c.content_type,
        messages.c.message_thread_id,
        messages.c.reply_to_message_id,
        media_notes.c.note.label("media_note"),
    ).select_from(
        messages.outerjoin(users, users.c.id == messages.c.from_user_id).outerjoin(
            media_notes, media_notes.c.file_unique_id == messages.c.media_uid
        )
    )


def _read_cursor(chat_id: int, thread_key: int):
    """Newest message of a chat/topic the agent has already been shown.

    A topic honours the whole-chat cursor too — a chat-wide history
    read exposed that topic's older messages just the same.
    """
    return (
        select(func.coalesce(func.max(message_read_cursors.c.message_id), 0))
        .where(
            message_read_cursors.c.chat_id == chat_id,
            message_read_cursors.c.message_thread_id.in_((0, thread_key)),
        )
        .scalar_subquery()
    )


class Database:
    """Async wrapper around the bot's SQLite database.

    Owns an async engine over a small connection pool; call
    :meth:`connect` before use and :meth:`close` on shutdown. Every
    method is one unit of work — a single transaction for writes, a
    single pooled connection for reads — and rows come back as plain
    dicts.
    """

    def __init__(self, path: Path) -> None:
        """Remember the database file location; no I/O happens here."""
        self.path = path
        self._engine: AsyncEngine | None = None

    @property
    def engine(self) -> AsyncEngine:
        """Return the live engine, or raise if not connected yet."""
        if self._engine is None:
            raise RuntimeError("database is not connected")
        return self._engine

    async def connect(self) -> None:
        """Open the engine and prove the database file is reachable."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._engine = create_async_engine(f"sqlite+aiosqlite:///{self.path}")
        event.listen(self._engine.sync_engine, "connect", _configure_connection)
        # Open one connection now so a bad path fails here, not on the
        # first query, and so the file exists before it is chmodded.
        async with self._engine.connect():
            pass
        # Full message payloads and model context are private even on a
        # multi-user host; do not leave the database world-readable.
        self.path.chmod(0o600)

    async def close(self) -> None:
        """Dispose the engine; safe to call when already closed."""
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None

    @staticmethod
    def _user_upsert(user: User):
        """Build the insert-or-refresh statement for one user row."""
        stmt = sqlite_insert(users).values(
            id=user.id,
            is_bot=int(user.is_bot),
            username=user.username,
            first_name=user.first_name,
            last_name=user.last_name,
            language_code=user.language_code,
            raw=user.model_dump_json(exclude_none=True),
        )
        return stmt.on_conflict_do_update(
            index_elements=[users.c.id],
            set_={
                "is_bot": stmt.excluded.is_bot,
                "username": stmt.excluded.username,
                "first_name": stmt.excluded.first_name,
                "last_name": stmt.excluded.last_name,
                "language_code": stmt.excluded.language_code,
                "raw": stmt.excluded.raw,
                "last_seen_at": func.datetime("now"),
            },
        )

    @staticmethod
    def _chat_upsert(chat: Chat):
        """Build the insert-or-refresh statement for one chat row."""
        stmt = sqlite_insert(chats).values(
            id=chat.id,
            type=chat.type,
            title=chat.title,
            username=chat.username,
            raw=chat.model_dump_json(exclude_none=True),
        )
        return stmt.on_conflict_do_update(
            index_elements=[chats.c.id],
            set_={
                "type": stmt.excluded.type,
                "title": stmt.excluded.title,
                "username": stmt.excluded.username,
                "raw": stmt.excluded.raw,
                "last_seen_at": func.datetime("now"),
            },
        )

    async def upsert_user(self, user: User) -> None:
        """Insert or refresh a user row, bumping ``last_seen_at``."""
        async with self.engine.begin() as conn:
            await conn.execute(self._user_upsert(user))

    async def upsert_chat(self, chat: Chat) -> None:
        """Insert or refresh a chat row, bumping ``last_seen_at``."""
        async with self.engine.begin() as conn:
            await conn.execute(self._chat_upsert(chat))

    async def append_context(self, item: dict) -> None:
        """Append one agent context item (as JSON) to the full history."""
        async with self.engine.begin() as conn:
            await conn.execute(context.insert().values(item=_dump_context(item)))

    async def latest_context_id(self) -> int:
        """Return newest persisted context id, or zero when empty."""
        stmt = select(func.coalesce(func.max(context.c.id), 0))
        async with self.engine.connect() as conn:
            return int((await conn.execute(stmt)).scalar_one())

    async def append_dream_context(self, dream_id: int, item: dict) -> None:
        """Append one dreaming context item (as JSON) to a dream's trace."""
        async with self.engine.begin() as conn:
            await conn.execute(
                dream_context.insert().values(
                    dream_id=dream_id, item=_dump_context(item)
                )
            )

    async def latest_dream_context_id(self) -> int:
        """Return newest persisted dreaming context id, or zero when empty.

        Deliberately not scoped to one dream: ids are monotonic, so an
        ``id > anchor`` comparison within a single dream holds either way.
        """
        stmt = select(func.coalesce(func.max(dream_context.c.id), 0))
        async with self.engine.connect() as conn:
            return int((await conn.execute(stmt)).scalar_one())

    async def start_agent_turn(self, start_context_id: int) -> int:
        """Open a turn and mark any crash-left turn interrupted."""
        async with self.engine.begin() as conn:
            await conn.execute(
                update(agent_turns)
                .where(agent_turns.c.status == "running")
                .values(status="interrupted", finished_at=func.datetime("now"))
            )
            result = await conn.execute(
                agent_turns.insert()
                .values(start_context_id=start_context_id)
                .returning(agent_turns.c.id)
            )
            return int(result.scalar_one())

    async def finish_agent_turn(
        self,
        turn_id: int,
        end_context_id: int,
        status: str,
    ) -> None:
        """Close a turn with its final context id and outcome."""
        async with self.engine.begin() as conn:
            await conn.execute(
                update(agent_turns)
                .where(agent_turns.c.id == turn_id)
                .values(
                    end_context_id=end_context_id,
                    status=status,
                    finished_at=func.datetime("now"),
                )
            )

    async def load_context(
        self,
        limit: int,
        exclude_types: tuple[str, ...] = (),
    ) -> list[dict]:
        """Return newest eligible context items, oldest first."""
        stmt = select(context.c.item)
        if exclude_types:
            item_type = func.coalesce(func.json_extract(context.c.item, "$.type"), "")
            stmt = stmt.where(item_type.not_in(exclude_types))
        stmt = stmt.order_by(context.c.id.desc()).limit(limit)
        async with self.engine.connect() as conn:
            rows = (await conn.execute(stmt)).scalars().all()
        return [json.loads(item) for item in reversed(rows)]

    async def append_api_usage(
        self,
        *,
        response_id: str | None,
        turn_id: int | None,
        input_context_id: int,
        model: str,
        input_tokens: int,
        cached_tokens: int,
        cache_write_tokens: int,
        output_tokens: int,
        reasoning_tokens: int,
        total_tokens: int,
        dream_id: int | None = None,
    ) -> None:
        """Persist exact token and prompt-cache usage for one API call.

        Exactly one of ``turn_id`` and ``dream_id`` is set: the row
        belongs either to a waking turn or to a dream.
        """
        async with self.engine.begin() as conn:
            await conn.execute(
                api_usage.insert().values(
                    response_id=response_id,
                    turn_id=turn_id,
                    dream_id=dream_id,
                    input_context_id=input_context_id,
                    model=model,
                    input_tokens=input_tokens,
                    cached_tokens=cached_tokens,
                    cache_write_tokens=cache_write_tokens,
                    output_tokens=output_tokens,
                    reasoning_tokens=reasoning_tokens,
                    total_tokens=total_tokens,
                )
            )

    async def start_dream(self, trigger: str) -> int:
        """Open a dream record and return its id.

        Unlike :meth:`start_agent_turn` this does not sweep older
        ``running`` rows: the waking and dreaming loops write here
        concurrently, and a stale row only ever costs one dream of
        budget.
        """
        async with self.engine.begin() as conn:
            result = await conn.execute(
                dreams.insert().values(trigger=trigger).returning(dreams.c.id)
            )
            return int(result.scalar_one())

    async def finish_dream(
        self,
        dream_id: int,
        status: str,
        steps: int,
        summary: str,
    ) -> None:
        """Close a dream with its outcome, tool-call count and summary."""
        async with self.engine.begin() as conn:
            await conn.execute(
                update(dreams)
                .where(dreams.c.id == dream_id)
                .values(
                    status=status,
                    steps=steps,
                    summary=summary,
                    finished_at=func.datetime("now"),
                )
            )

    async def dreams_since(self, since: str) -> int:
        """Count dreams started at or after a UTC stamp."""
        stmt = (
            select(func.count()).select_from(dreams).where(dreams.c.started_at >= since)
        )
        async with self.engine.connect() as conn:
            return int((await conn.execute(stmt)).scalar_one())

    async def last_dream_end(self) -> str | None:
        """Return the UTC stamp of the last finished dream, if any."""
        stmt = select(func.max(dreams.c.finished_at))
        async with self.engine.connect() as conn:
            return (await conn.execute(stmt)).scalar_one()

    async def add_wakeup(self, due_at: str, note: str) -> int:
        """Store a scheduled wakeup (``due_at`` as UTC stamp); return its id."""
        async with self.engine.begin() as conn:
            result = await conn.execute(
                wakeups.insert()
                .values(due_at=due_at, note=note)
                .returning(wakeups.c.id)
            )
            return int(result.scalar_one())

    async def due_wakeups(self, now: str) -> list[dict]:
        """Return id/due_at/note rows of undone wakeups due by ``now`` (UTC)."""
        stmt = (
            select(wakeups.c.id, wakeups.c.due_at, wakeups.c.note)
            .where(wakeups.c.done == 0, wakeups.c.due_at <= now)
            .order_by(wakeups.c.due_at)
        )
        async with self.engine.connect() as conn:
            return [dict(row) for row in (await conn.execute(stmt)).mappings()]

    async def pending_wakeups(self) -> list[dict]:
        """Return id/due_at/note rows of all undone wakeups, soonest first."""
        stmt = (
            select(wakeups.c.id, wakeups.c.due_at, wakeups.c.note)
            .where(wakeups.c.done == 0)
            .order_by(wakeups.c.due_at)
        )
        async with self.engine.connect() as conn:
            return [dict(row) for row in (await conn.execute(stmt)).mappings()]

    async def complete_wakeup(self, wakeup_id: int) -> None:
        """Mark a wakeup as done."""
        async with self.engine.begin() as conn:
            await conn.execute(
                update(wakeups).where(wakeups.c.id == wakeup_id).values(done=1)
            )

    async def cancel_wakeup(self, wakeup_id: int) -> bool:
        """Mark a pending wakeup as done; return whether one was cancelled."""
        async with self.engine.begin() as conn:
            result = await conn.execute(
                update(wakeups)
                .where(wakeups.c.id == wakeup_id, wakeups.c.done == 0)
                .values(done=1)
            )
            return result.rowcount > 0

    async def add_steering(self, text: str, urgent: bool = False) -> int:
        """Queue one operator instruction from the console; return its id."""
        async with self.engine.begin() as conn:
            result = await conn.execute(
                steering.insert()
                .values(text=text, urgent=int(urgent))
                .returning(steering.c.id)
            )
            return int(result.scalar_one())

    async def claim_steering(self, urgent: bool | None = None) -> list[dict]:
        """Take undelivered operator instructions, oldest first.

        ``urgent`` selects one class of them (``True`` only the urgent
        ones, ``False`` only the rest) or, left as ``None``, both.

        Claiming and returning are one step: two deliverers race for
        these rows — the background loop and the turn in flight — and a
        single ``UPDATE … RETURNING`` hands each row to exactly one of
        them (the loser's transaction sees ``done = 1`` and matches
        nothing), so an instruction cannot be delivered twice. The cost
        is the opposite guarantee wakeups have: a crash between the
        claim and the event being persisted drops it. Repeating an
        instruction is worse than losing one you can see is still
        unread.
        """
        stmt = (
            update(steering)
            .where(steering.c.done == 0)
            .values(done=1)
            .returning(steering.c.id, steering.c.text, steering.c.urgent)
        )
        if urgent is not None:
            stmt = stmt.where(steering.c.urgent == int(urgent))
        async with self.engine.begin() as conn:
            rows = (await conn.execute(stmt)).mappings().all()
        # RETURNING carries no ORDER BY; restore the oldest-first order.
        return sorted((dict(row) for row in rows), key=lambda row: row["id"])

    async def unanswered_chats(self) -> list[dict]:
        """Chats whose latest message is incoming (i.e. awaiting the agent).

        One row per chat — or per forum topic within a forum chat, so an
        answered topic cannot mask an unanswered one. Returns dicts with
        chat id/type/title, the topic id (``NULL`` outside topics), the
        sender's name and the date of that last message. Forum service
        messages never count as the awaiting message, so a freshly
        created topic doesn't nag forever.

        Ranking each chat/topic once beats asking "is anything newer?"
        per message: the correlated form re-scanned the chat for every
        row it considered, which is quadratic in history length.
        """
        latest = (
            select(
                messages.c.chat_id,
                messages.c.message_thread_id,
                messages.c.from_user_id,
                messages.c.date,
                messages.c.outgoing,
                func.row_number()
                .over(
                    partition_by=(
                        messages.c.chat_id,
                        func.coalesce(messages.c.message_thread_id, 0),
                    ),
                    order_by=(messages.c.date.desc(), messages.c.message_id.desc()),
                )
                .label("position"),
            )
            .where(messages.c.content_type.not_in(_FORUM_SERVICE_TYPES))
            .cte("latest")
        )
        stmt = (
            select(
                chats.c.id.label("chat_id"),
                chats.c.type,
                chats.c.title,
                latest.c.message_thread_id,
                users.c.first_name,
                users.c.username,
                latest.c.date,
            )
            .select_from(
                latest.join(chats, chats.c.id == latest.c.chat_id).outerjoin(
                    users, users.c.id == latest.c.from_user_id
                )
            )
            .where(latest.c.position == 1, latest.c.outgoing == 0)
            .order_by(latest.c.date)
        )
        async with self.engine.connect() as conn:
            return [dict(row) for row in (await conn.execute(stmt)).mappings()]

    async def recent_messages(
        self,
        chat_id: int,
        limit: int,
        before_message_id: int | None = None,
        message_thread_id: int | None = None,
    ) -> list[dict]:
        """Return the latest ``limit`` messages of a chat, oldest first.

        Each row is a small dict (id, date, sender, text/caption, content
        type, outgoing flag, forum topic id) suitable for feeding to the
        LLM as context. ``before_message_id`` pages into the past: only
        messages older than it are returned. ``message_thread_id``
        restricts the result to one forum topic; ``None`` means the whole
        chat.
        """
        stmt = _message_select().where(messages.c.chat_id == chat_id)
        if before_message_id is not None:
            stmt = stmt.where(messages.c.message_id < before_message_id)
        if message_thread_id is not None:
            stmt = stmt.where(messages.c.message_thread_id == message_thread_id)
        stmt = stmt.order_by(
            messages.c.date.desc(), messages.c.message_id.desc()
        ).limit(limit)
        async with self.engine.connect() as conn:
            rows = (await conn.execute(stmt)).mappings().all()
        return [dict(row) for row in reversed(rows)]

    async def unread_messages_count(
        self, chat_id: int, message_thread_id: int | None = None
    ) -> int:
        """Count messages past the read cursor of a chat/topic.

        Events advance that cursor too, so this counts what nobody has
        put in front of the agent — not what it neglected to fetch.

        Only incoming messages count: the agent's own replies are stored
        in the same table, and reporting them back as unread would make
        every answered chat look like it still needs reading.
        """
        stmt = (
            select(func.count())
            .select_from(messages)
            .where(
                messages.c.chat_id == chat_id,
                messages.c.outgoing == 0,
                messages.c.message_id > _read_cursor(chat_id, message_thread_id or 0),
            )
        )
        if message_thread_id is not None:
            stmt = stmt.where(messages.c.message_thread_id == message_thread_id)
        async with self.engine.connect() as conn:
            return int((await conn.execute(stmt)).scalar_one())

    async def messages_since_read(
        self,
        chat_id: int,
        limit: int,
        before_message_id: int | None = None,
        message_thread_id: int | None = None,
    ) -> list[dict]:
        """Return messages past the chat/topic read cursor, oldest first.

        The transcript an event carries, so the agent starts a turn
        already knowing what it missed. Unlike
        :meth:`unread_messages_count` this keeps the agent's own replies:
        the point is a readable conversation, and a group exchange with
        one side deleted reads as if nobody answered. Rows have the shape
        of :meth:`recent_messages`; when more than ``limit`` are pending,
        the newest are kept.
        """
        stmt = _message_select().where(
            messages.c.chat_id == chat_id,
            messages.c.message_id > _read_cursor(chat_id, message_thread_id or 0),
        )
        if before_message_id is not None:
            stmt = stmt.where(messages.c.message_id < before_message_id)
        if message_thread_id is not None:
            stmt = stmt.where(messages.c.message_thread_id == message_thread_id)
        stmt = stmt.order_by(
            messages.c.date.desc(), messages.c.message_id.desc()
        ).limit(limit)
        async with self.engine.connect() as conn:
            rows = (await conn.execute(stmt)).mappings().all()
        return [dict(row) for row in reversed(rows)]

    async def message_row(self, chat_id: int, message_id: int) -> dict | None:
        """Return one stored message in :meth:`recent_messages` shape."""
        stmt = _message_select().where(
            messages.c.chat_id == chat_id, messages.c.message_id == message_id
        )
        async with self.engine.connect() as conn:
            row = (await conn.execute(stmt)).mappings().first()
        return dict(row) if row else None

    async def mark_messages_read(
        self,
        chat_id: int,
        message_id: int,
        message_thread_id: int | None = None,
    ) -> None:
        """Advance a chat/topic history cursor through one exposed message."""
        stmt = sqlite_insert(message_read_cursors).values(
            chat_id=chat_id,
            message_thread_id=message_thread_id or 0,
            message_id=message_id,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[
                message_read_cursors.c.chat_id,
                message_read_cursors.c.message_thread_id,
            ],
            # The cursor only ever advances; two-argument MAX keeps the
            # newer of the stored and offered ids.
            set_={
                "message_id": func.max(
                    message_read_cursors.c.message_id, stmt.excluded.message_id
                )
            },
        )
        async with self.engine.begin() as conn:
            await conn.execute(stmt)

    async def search_messages(
        self,
        chat_id: int,
        needle: str,
        limit: int,
        message_thread_id: int | None = None,
    ) -> list[dict]:
        """Return a chat's messages containing ``needle``, newest first.

        Literal substring match over text and caption, case-insensitive
        via Unicode casefold (LIKE would only fold ASCII); rows have the
        same shape as :meth:`recent_messages`. ``message_thread_id``
        restricts the search to one forum topic; ``None`` means the whole
        chat.
        """
        stmt = _message_select().where(
            messages.c.chat_id == chat_id,
            or_(
                func.instr(func.casefold(messages.c.text), func.casefold(needle)) > 0,
                func.instr(func.casefold(messages.c.caption), func.casefold(needle))
                > 0,
            ),
        )
        if message_thread_id is not None:
            stmt = stmt.where(messages.c.message_thread_id == message_thread_id)
        stmt = stmt.order_by(
            messages.c.date.desc(), messages.c.message_id.desc()
        ).limit(limit)
        async with self.engine.connect() as conn:
            return [dict(row) for row in (await conn.execute(stmt)).mappings()]

    async def message_thread(
        self, chat_id: int, message_id: int, limit: int
    ) -> list[dict]:
        """Return the reply thread a message belongs to, oldest first.

        The thread is the connected component of reply links through the
        given message: what it replies to, replies to those, and every
        branch hanging off any of them. Rows have the shape of
        :meth:`recent_messages` plus ``reply_to_message_id`` so the
        reply structure is visible. When the thread exceeds ``limit``
        the newest messages are kept. Unknown messages yield no rows.
        """
        thread = (
            select(messages.c.message_id, messages.c.reply_to_message_id)
            .where(messages.c.chat_id == chat_id, messages.c.message_id == message_id)
            .cte("thread", recursive=True)
        )
        walker = messages.alias("m")
        # ``union`` (not ``union_all``): the walk goes both directions
        # through reply links, and deduplication is what terminates it.
        thread = thread.union(
            select(walker.c.message_id, walker.c.reply_to_message_id).select_from(
                walker.join(
                    thread,
                    (walker.c.chat_id == chat_id)
                    & (
                        (walker.c.message_id == thread.c.reply_to_message_id)
                        | (walker.c.reply_to_message_id == thread.c.message_id)
                    ),
                )
            )
        )
        stmt = (
            _message_select()
            .where(
                messages.c.chat_id == chat_id,
                messages.c.message_id.in_(select(thread.c.message_id)),
            )
            .order_by(messages.c.date.desc(), messages.c.message_id.desc())
            .limit(limit)
        )
        async with self.engine.connect() as conn:
            rows = (await conn.execute(stmt)).mappings().all()
        return [dict(row) for row in reversed(rows)]

    async def topic_name(self, chat_id: int, thread_id: int) -> str | None:
        """Return the latest known name of a forum topic, or ``None``.

        Three sources, all mined from stored raw payloads: the creation
        service message (if the bot saw it), rename service messages, and
        the pseudo-reply payload every non-reply topic message carries
        (covers topics created before the bot joined). Renames win over
        the creation-time name that pseudo-replies keep echoing forever;
        icon-only edits carry no name and are skipped.
        """
        named = (
            select(
                func.coalesce(
                    func.json_extract(messages.c.raw, "$.forum_topic_edited.name"),
                    func.json_extract(messages.c.raw, "$.forum_topic_created.name"),
                    func.json_extract(
                        messages.c.raw,
                        "$.reply_to_message.forum_topic_created.name",
                    ),
                ).label("name"),
                (messages.c.content_type == "forum_topic_edited").label("renamed"),
                messages.c.date,
                messages.c.message_id,
            )
            .where(
                messages.c.chat_id == chat_id,
                messages.c.message_thread_id == thread_id,
            )
            .subquery()
        )
        stmt = (
            select(named.c.name)
            .where(named.c.name.is_not(None))
            .order_by(
                named.c.renamed.desc(),
                named.c.date.desc(),
                named.c.message_id.desc(),
            )
            .limit(1)
        )
        async with self.engine.connect() as conn:
            return (await conn.execute(stmt)).scalar_one_or_none()

    async def topic_observed(self, chat_id: int, thread_id: int) -> bool:
        """True when any stored message of the chat belongs to the topic."""
        stmt = (
            select(1)
            .select_from(messages)
            .where(
                messages.c.chat_id == chat_id,
                messages.c.message_thread_id == thread_id,
            )
            .limit(1)
        )
        async with self.engine.connect() as conn:
            return (await conn.execute(stmt)).first() is not None

    async def list_topics(self, chat_id: int) -> list[dict]:
        """Return the forum topics seen in a chat, most recent first.

        Each row carries the topic id, its latest known name, message
        count, last activity date and a ``closed`` flag from the newest
        close/reopen service message. The General topic never appears:
        its messages carry no topic id.
        """
        events = messages.alias("e")
        last_gate = (
            select(events.c.content_type)
            .where(
                events.c.chat_id == messages.c.chat_id,
                events.c.message_thread_id == messages.c.message_thread_id,
                events.c.content_type.in_(
                    ("forum_topic_closed", "forum_topic_reopened")
                ),
            )
            .order_by(events.c.date.desc(), events.c.message_id.desc())
            .limit(1)
            .scalar_subquery()
        )
        stmt = (
            select(
                messages.c.message_thread_id.label("topic_id"),
                func.count().label("messages"),
                func.max(messages.c.date).label("last_date"),
                func.coalesce(last_gate == "forum_topic_closed", 0).label("closed"),
            )
            .where(
                messages.c.chat_id == chat_id,
                messages.c.message_thread_id.is_not(None),
            )
            .group_by(messages.c.message_thread_id)
            .order_by(desc("last_date"))
        )
        async with self.engine.connect() as conn:
            rows = [dict(row) for row in (await conn.execute(stmt)).mappings()]
        # Topic counts are tiny; a name lookup per row keeps the tricky
        # name-resolution logic in one place.
        for row in rows:
            row["name"] = await self.topic_name(chat_id, row["topic_id"])
            row["closed"] = bool(row["closed"])
        return rows

    async def list_chats(self) -> list[dict]:
        """Return every known chat with a display name and activity stats.

        For private chats (no title) the name falls back to the peer's
        first name or username — a private chat's id equals the user's id.
        """
        stmt = (
            select(
                chats.c.id.label("chat_id"),
                chats.c.type,
                func.coalesce(
                    chats.c.title,
                    users.c.first_name,
                    chats.c.username,
                    users.c.username,
                ).label("name"),
                func.count(messages.c.message_id).label("messages"),
                func.max(messages.c.date).label("last_date"),
            )
            .select_from(
                chats.outerjoin(users, users.c.id == chats.c.id).outerjoin(
                    messages, messages.c.chat_id == chats.c.id
                )
            )
            .group_by(chats.c.id)
            .order_by(desc("last_date"))
        )
        async with self.engine.connect() as conn:
            return [dict(row) for row in (await conn.execute(stmt)).mappings()]

    async def chat_members(self, chat_id: int) -> list[dict]:
        """Return users seen talking in a chat, most recently active first.

        Built from stored history — Telegram doesn't let bots fetch a
        group's full roster, so this is who has actually said something.
        """
        stmt = (
            select(
                users.c.id.label("user_id"),
                users.c.username,
                users.c.first_name,
                users.c.last_name,
                func.count().label("messages"),
                func.max(messages.c.date).label("last_date"),
            )
            .select_from(messages.join(users, users.c.id == messages.c.from_user_id))
            .where(messages.c.chat_id == chat_id)
            .group_by(users.c.id)
            .order_by(desc("last_date"))
        )
        async with self.engine.connect() as conn:
            return [dict(row) for row in (await conn.execute(stmt)).mappings()]

    async def known_stickers(
        self, limit: int, chat_ids: list[int] | None = None
    ) -> list[dict]:
        """Return distinct stickers from selected chats, newest first.

        Extracted from raw payloads; deduplicated by the sticker's stable
        ``file_unique_id``, returning a ``file_id`` the bot can resend.
        ``None`` selects every chat; an empty list selects none.
        """
        if chat_ids == []:
            return []
        stmt = (
            select(
                func.json_extract(messages.c.raw, "$.sticker.file_id").label("file_id"),
                func.json_extract(messages.c.raw, "$.sticker.emoji").label("emoji"),
                func.json_extract(messages.c.raw, "$.sticker.set_name").label(
                    "set_name"
                ),
                func.max(messages.c.date).label("last_date"),
            )
            .where(messages.c.content_type == "sticker")
            .group_by(func.json_extract(messages.c.raw, "$.sticker.file_unique_id"))
            .order_by(desc("last_date"))
            .limit(limit)
        )
        if chat_ids is not None:
            stmt = stmt.where(messages.c.chat_id.in_(chat_ids))
        async with self.engine.connect() as conn:
            return [dict(row) for row in (await conn.execute(stmt)).mappings()]

    async def sticker_is_known(self, file_id: str, chat_ids: list[int]) -> bool:
        """Whether a sticker file id was observed in selected chats."""
        if not chat_ids:
            return False
        stmt = (
            select(1)
            .select_from(messages)
            .where(
                messages.c.content_type == "sticker",
                func.json_extract(messages.c.raw, "$.sticker.file_id") == file_id,
                messages.c.chat_id.in_(chat_ids),
            )
            .limit(1)
        )
        async with self.engine.connect() as conn:
            return (await conn.execute(stmt)).first() is not None

    async def message_payload(self, chat_id: int, message_id: int) -> dict | None:
        """Return one stored message's raw Telegram payload, decoded.

        The dedicated columns cover what a transcript needs; everything
        else — a sticker's set, a video's duration, the file ids behind
        either — only exists here.
        """
        stmt = select(messages.c.raw).where(
            messages.c.chat_id == chat_id, messages.c.message_id == message_id
        )
        async with self.engine.connect() as conn:
            raw = (await conn.execute(stmt)).scalar_one_or_none()
        return json.loads(raw) if raw else None

    async def media_note(self, file_unique_id: str) -> str | None:
        """Return what a media file was described as, if anyone has."""
        stmt = select(media_notes.c.note).where(
            media_notes.c.file_unique_id == file_unique_id
        )
        async with self.engine.connect() as conn:
            return (await conn.execute(stmt)).scalar_one_or_none()

    async def save_media_note(
        self, file_unique_id: str, kind: str, note: str, model: str
    ) -> None:
        """Store a media file's description, replacing any older one.

        Replacing rather than ignoring the conflict: a re-description is
        only ever asked for deliberately (a better model, a bad note),
        and it should be the one that sticks.
        """
        stmt = sqlite_insert(media_notes).values(
            file_unique_id=file_unique_id, kind=kind, note=note, model=model
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[media_notes.c.file_unique_id],
            set_={
                "kind": stmt.excluded.kind,
                "note": stmt.excluded.note,
                "model": stmt.excluded.model,
                "created_at": func.datetime("now"),
            },
        )
        async with self.engine.begin() as conn:
            await conn.execute(stmt)

    async def set_message_media(
        self, chat_id: int, message_id: int, file_unique_id: str
    ) -> None:
        """Link a stored message to the media file it carries."""
        async with self.engine.begin() as conn:
            await conn.execute(
                update(messages)
                .where(
                    messages.c.chat_id == chat_id,
                    messages.c.message_id == message_id,
                )
                .values(media_uid=file_unique_id)
            )

    async def message_exists(self, chat_id: int, message_id: int) -> bool:
        """Whether a message was observed and stored in one chat."""
        stmt = (
            select(1)
            .select_from(messages)
            .where(messages.c.chat_id == chat_id, messages.c.message_id == message_id)
        )
        async with self.engine.connect() as conn:
            return (await conn.execute(stmt)).first() is not None

    async def message_is_outgoing(self, chat_id: int, message_id: int) -> bool:
        """Whether a stored message was sent by the bot itself.

        Unknown messages are not outgoing: every message the bot sends is
        persisted, so "not stored" means "not ours to touch".
        """
        stmt = select(messages.c.outgoing).where(
            messages.c.chat_id == chat_id, messages.c.message_id == message_id
        )
        async with self.engine.connect() as conn:
            return bool((await conn.execute(stmt)).scalar_one_or_none())

    async def delete_message(self, chat_id: int, message_id: int) -> None:
        """Remove a message row (mirrors a deletion done on Telegram)."""
        async with self.engine.begin() as conn:
            await conn.execute(
                delete(messages).where(
                    messages.c.chat_id == chat_id,
                    messages.c.message_id == message_id,
                )
            )

    async def save_message(self, message: Message, *, outgoing: bool = False) -> None:
        """Persist a message together with its chat and sender.

        Stores common fields in dedicated columns and the full serialized
        payload in ``raw``. Re-saving the same ``(chat_id, message_id)``
        (e.g. an edit) updates the mutable columns in place. Set
        ``outgoing=True`` for messages the bot itself sent. The chat and
        sender upserts share the message's transaction.
        """
        stmt = sqlite_insert(messages).values(
            chat_id=message.chat.id,
            message_id=message.message_id,
            from_user_id=message.from_user.id if message.from_user else None,
            date=message.date.isoformat(),
            # Telegram sends edit_date as a unix timestamp; normalize to
            # ISO so it compares with the `date` column.
            edit_date=datetime.fromtimestamp(message.edit_date, tz=UTC).isoformat()
            if message.edit_date
            else None,
            content_type=message.content_type,
            text=message.text,
            caption=message.caption,
            reply_to_message_id=effective_reply_to(message),
            # Bot API also sets message_thread_id on plain reply chains;
            # is_topic_message discriminates real forum topics.
            message_thread_id=message.message_thread_id
            if message.is_topic_message
            else None,
            media_group_id=message.media_group_id,
            outgoing=int(outgoing),
            raw=message.model_dump_json(exclude_none=True),
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[messages.c.chat_id, messages.c.message_id],
            set_={
                "edit_date": stmt.excluded.edit_date,
                "content_type": stmt.excluded.content_type,
                "text": stmt.excluded.text,
                "caption": stmt.excluded.caption,
                "raw": stmt.excluded.raw,
                "saved_at": func.datetime("now"),
            },
        )
        async with self.engine.begin() as conn:
            await conn.execute(self._chat_upsert(message.chat))
            if message.from_user is not None:
                await conn.execute(self._user_upsert(message.from_user))
            await conn.execute(stmt)
