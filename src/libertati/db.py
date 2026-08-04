"""SQLite persistence for messages, chats and users.

Every message flowing through the bot (incoming and its own replies) is
stored, along with the full raw Telegram payload for metadata not covered
by dedicated columns.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import aiosqlite
from aiogram.types import Chat, Message, User

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY,
    is_bot        INTEGER NOT NULL DEFAULT 0,
    username      TEXT,
    first_name    TEXT,
    last_name     TEXT,
    language_code TEXT,
    raw           TEXT NOT NULL,
    first_seen_at TEXT NOT NULL DEFAULT (datetime('now')),
    last_seen_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS chats (
    id            INTEGER PRIMARY KEY,
    type          TEXT NOT NULL,
    title         TEXT,
    username      TEXT,
    raw           TEXT NOT NULL,
    first_seen_at TEXT NOT NULL DEFAULT (datetime('now')),
    last_seen_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS messages (
    chat_id             INTEGER NOT NULL REFERENCES chats(id),
    message_id          INTEGER NOT NULL,
    from_user_id        INTEGER REFERENCES users(id),
    -- Full ISO datetime; named "date" for parity with the Telegram API field.
    date                TEXT NOT NULL,
    edit_date           TEXT,
    content_type        TEXT NOT NULL,
    text                TEXT,
    caption             TEXT,
    reply_to_message_id INTEGER,
    -- Forum topic id; NULL outside forum topics (incl. the General topic).
    message_thread_id   INTEGER,
    media_group_id      TEXT,
    outgoing            INTEGER NOT NULL DEFAULT 0,
    raw                 TEXT NOT NULL,
    saved_at            TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (chat_id, message_id)
);

CREATE INDEX IF NOT EXISTS idx_messages_chat_date
    ON messages (chat_id, date);
CREATE INDEX IF NOT EXISTS idx_messages_from_user
    ON messages (from_user_id);

CREATE TABLE IF NOT EXISTS context (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    item       TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS agent_turns (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    start_context_id INTEGER NOT NULL,
    end_context_id   INTEGER,
    status           TEXT NOT NULL DEFAULT 'running',
    started_at       TEXT NOT NULL DEFAULT (datetime('now')),
    finished_at      TEXT
);

CREATE TABLE IF NOT EXISTS api_usage (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    response_id        TEXT,
    turn_id            INTEGER,
    dream_id           INTEGER,
    input_context_id   INTEGER NOT NULL DEFAULT 0,
    model              TEXT NOT NULL,
    input_tokens       INTEGER NOT NULL,
    cached_tokens      INTEGER NOT NULL,
    cache_write_tokens INTEGER NOT NULL,
    output_tokens      INTEGER NOT NULL,
    reasoning_tokens   INTEGER NOT NULL,
    total_tokens       INTEGER NOT NULL,
    created_at         TEXT NOT NULL DEFAULT (datetime('now'))
);

-- One row per dream: its budget has to survive restarts, and a dream
-- that dies before writing its journal entry still has to count against
-- that budget.
CREATE TABLE IF NOT EXISTS dreams (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    trigger     TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'running',
    steps       INTEGER NOT NULL DEFAULT 0,
    summary     TEXT,
    started_at  TEXT NOT NULL DEFAULT (datetime('now')),
    finished_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_dreams_started ON dreams (started_at);

-- A dream's context, kept apart from the waking one: it is written for
-- inspection only and never read back, so nothing here can leak into
-- what the waking agent is sent.
CREATE TABLE IF NOT EXISTS dream_context (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    dream_id   INTEGER NOT NULL REFERENCES dreams(id),
    item       TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_dream_context_dream ON dream_context (dream_id, id);

CREATE TABLE IF NOT EXISTS wakeups (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    due_at     TEXT NOT NULL,
    note       TEXT NOT NULL,
    done       INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_wakeups_due ON wakeups (done, due_at);
"""


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


class Database:
    """Async wrapper around the bot's SQLite database.

    Owns a single :mod:`aiosqlite` connection; call :meth:`connect` before
    use and :meth:`close` on shutdown.
    """

    def __init__(self, path: Path) -> None:
        """Remember the database file location; no I/O happens here."""
        self.path = path
        self._conn: aiosqlite.Connection | None = None

    @property
    def conn(self) -> aiosqlite.Connection:
        """Return the live connection, or raise if not connected yet."""
        if self._conn is None:
            raise RuntimeError("database is not connected")
        return self._conn

    async def connect(self) -> None:
        """Open the database, enable WAL and foreign keys, create schema."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self.path)
        # Full message payloads and model context are private even on a
        # multi-user host; do not leave the database world-readable.
        self.path.chmod(0o600)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode = WAL")
        await self._conn.execute("PRAGMA foreign_keys = ON")
        # SQLite's own LIKE/NOCASE are case-insensitive for ASCII only;
        # message search needs real Unicode folding (Cyrillic etc.).
        await self._conn.create_function(
            "casefold",
            1,
            lambda text: text.casefold() if text else "",
            deterministic=True,
        )
        await self._conn.executescript(SCHEMA)
        # ``CREATE TABLE IF NOT EXISTS`` does not add columns to DBs
        # created before usage snapshots gained context linkage.
        await self._ensure_column("api_usage", "turn_id", "INTEGER")
        await self._ensure_column(
            "api_usage", "input_context_id", "INTEGER NOT NULL DEFAULT 0"
        )
        await self._ensure_column("api_usage", "dream_id", "INTEGER")
        await self._ensure_column("messages", "message_thread_id", "INTEGER")
        # Backfill topic ids for rows saved before the column existed, and
        # strip Telegram's forum pseudo-replies (every non-reply message in
        # a topic "replies to" the topic-creation service message, which
        # would pollute reply-chain threads). Both idempotent.
        await self._conn.execute(
            """
            UPDATE messages
            SET message_thread_id = json_extract(raw, '$.message_thread_id')
            WHERE message_thread_id IS NULL
              AND json_extract(raw, '$.is_topic_message') = 1
            """
        )
        await self._conn.execute(
            """
            UPDATE messages
            SET reply_to_message_id = NULL
            WHERE reply_to_message_id IS NOT NULL
              AND json_extract(raw, '$.reply_to_message.forum_topic_created')
                  IS NOT NULL
            """
        )
        await self._conn.commit()

    async def _ensure_column(
        self,
        table: str,
        column: str,
        definition: str,
    ) -> None:
        """Add one trusted schema column when an older DB lacks it."""
        async with self.conn.execute(f"PRAGMA table_info({table})") as cursor:
            columns = {row[1] for row in await cursor.fetchall()}
        if column not in columns:
            await self.conn.execute(
                f"ALTER TABLE {table} ADD COLUMN {column} {definition}"
            )

    async def close(self) -> None:
        """Close the connection; safe to call when already closed."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    async def upsert_user(self, user: User) -> None:
        """Insert or refresh a user row, bumping ``last_seen_at``."""
        await self.conn.execute(
            """
            INSERT INTO users (id, is_bot, username, first_name, last_name,
                               language_code, raw)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (id) DO UPDATE SET
                is_bot = excluded.is_bot,
                username = excluded.username,
                first_name = excluded.first_name,
                last_name = excluded.last_name,
                language_code = excluded.language_code,
                raw = excluded.raw,
                last_seen_at = datetime('now')
            """,
            (
                user.id,
                int(user.is_bot),
                user.username,
                user.first_name,
                user.last_name,
                user.language_code,
                user.model_dump_json(exclude_none=True),
            ),
        )

    async def upsert_chat(self, chat: Chat) -> None:
        """Insert or refresh a chat row, bumping ``last_seen_at``."""
        await self.conn.execute(
            """
            INSERT INTO chats (id, type, title, username, raw)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (id) DO UPDATE SET
                type = excluded.type,
                title = excluded.title,
                username = excluded.username,
                raw = excluded.raw,
                last_seen_at = datetime('now')
            """,
            (
                chat.id,
                chat.type,
                chat.title,
                chat.username,
                chat.model_dump_json(exclude_none=True),
            ),
        )

    async def append_context(self, item: dict) -> None:
        """Append one agent context item (as JSON) to the full history."""
        await self.conn.execute(
            "INSERT INTO context (item) VALUES (?)",
            (_dump_context(item),),
        )
        await self.conn.commit()

    async def latest_context_id(self) -> int:
        """Return newest persisted context id, or zero when empty."""
        async with self.conn.execute(
            "SELECT COALESCE(MAX(id), 0) FROM context"
        ) as cursor:
            row = await cursor.fetchone()
        return int(row[0]) if row else 0

    async def append_dream_context(self, dream_id: int, item: dict) -> None:
        """Append one dreaming context item (as JSON) to a dream's trace."""
        await self.conn.execute(
            "INSERT INTO dream_context (dream_id, item) VALUES (?, ?)",
            (dream_id, _dump_context(item)),
        )
        await self.conn.commit()

    async def latest_dream_context_id(self) -> int:
        """Return newest persisted dreaming context id, or zero when empty.

        Deliberately not scoped to one dream: ids are monotonic, so an
        ``id > anchor`` comparison within a single dream holds either way.
        """
        async with self.conn.execute(
            "SELECT COALESCE(MAX(id), 0) FROM dream_context"
        ) as cursor:
            row = await cursor.fetchone()
        return int(row[0]) if row else 0

    async def start_agent_turn(self, start_context_id: int) -> int:
        """Open a turn and mark any crash-left turn interrupted."""
        await self.conn.execute(
            """
            UPDATE agent_turns
            SET status = 'interrupted', finished_at = datetime('now')
            WHERE status = 'running'
            """
        )
        cursor = await self.conn.execute(
            "INSERT INTO agent_turns (start_context_id) VALUES (?)",
            (start_context_id,),
        )
        await self.conn.commit()
        return cursor.lastrowid or 0

    async def finish_agent_turn(
        self,
        turn_id: int,
        end_context_id: int,
        status: str,
    ) -> None:
        """Close a turn with its final context id and outcome."""
        await self.conn.execute(
            """
            UPDATE agent_turns
            SET end_context_id = ?, status = ?, finished_at = datetime('now')
            WHERE id = ?
            """,
            (end_context_id, status, turn_id),
        )
        await self.conn.commit()

    async def load_context(
        self,
        limit: int,
        exclude_types: tuple[str, ...] = (),
    ) -> list[dict]:
        """Return newest eligible context items, oldest first."""
        query = "SELECT item FROM context"
        params: list[object] = []
        if exclude_types:
            placeholders = ", ".join("?" for _ in exclude_types)
            query += (
                " WHERE COALESCE(json_extract(item, '$.type'), '')"
                f" NOT IN ({placeholders})"
            )
            params.extend(exclude_types)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        async with self.conn.execute(query, params) as cursor:
            rows = list(await cursor.fetchall())
        return [json.loads(row["item"]) for row in reversed(rows)]

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
        await self.conn.execute(
            """
            INSERT INTO api_usage (
                response_id, turn_id, dream_id, input_context_id, model,
                input_tokens, cached_tokens, cache_write_tokens,
                output_tokens, reasoning_tokens, total_tokens
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                response_id,
                turn_id,
                dream_id,
                input_context_id,
                model,
                input_tokens,
                cached_tokens,
                cache_write_tokens,
                output_tokens,
                reasoning_tokens,
                total_tokens,
            ),
        )
        await self.conn.commit()

    async def start_dream(self, trigger: str) -> int:
        """Open a dream record and return its id.

        Unlike :meth:`start_agent_turn` this does not sweep older
        ``running`` rows: the waking and dreaming loops write here
        concurrently, and a stale row only ever costs one dream of
        budget.
        """
        cursor = await self.conn.execute(
            "INSERT INTO dreams (trigger) VALUES (?)", (trigger,)
        )
        await self.conn.commit()
        return cursor.lastrowid or 0

    async def finish_dream(
        self,
        dream_id: int,
        status: str,
        steps: int,
        summary: str,
    ) -> None:
        """Close a dream with its outcome, tool-call count and summary."""
        await self.conn.execute(
            """
            UPDATE dreams
            SET status = ?, steps = ?, summary = ?, finished_at = datetime('now')
            WHERE id = ?
            """,
            (status, steps, summary, dream_id),
        )
        await self.conn.commit()

    async def dreams_since(self, since: str) -> int:
        """Count dreams started at or after a UTC stamp."""
        async with self.conn.execute(
            "SELECT COUNT(*) FROM dreams WHERE started_at >= ?", (since,)
        ) as cursor:
            row = await cursor.fetchone()
        return int(row[0]) if row else 0

    async def last_dream_end(self) -> str | None:
        """Return the UTC stamp of the last finished dream, if any."""
        async with self.conn.execute("SELECT MAX(finished_at) FROM dreams") as cursor:
            row = await cursor.fetchone()
        return row[0] if row else None

    async def add_wakeup(self, due_at: str, note: str) -> int:
        """Store a scheduled wakeup (``due_at`` as UTC stamp); return its id."""
        cursor = await self.conn.execute(
            "INSERT INTO wakeups (due_at, note) VALUES (?, ?)", (due_at, note)
        )
        await self.conn.commit()
        return cursor.lastrowid or 0

    async def due_wakeups(self, now: str) -> list[aiosqlite.Row]:
        """Return id/due_at/note rows of undone wakeups due by ``now`` (UTC)."""
        async with self.conn.execute(
            "SELECT id, due_at, note FROM wakeups"
            " WHERE done = 0 AND due_at <= ? ORDER BY due_at",
            (now,),
        ) as cursor:
            return list(await cursor.fetchall())

    async def pending_wakeups(self) -> list[aiosqlite.Row]:
        """Return id/due_at/note rows of all undone wakeups, soonest first."""
        async with self.conn.execute(
            "SELECT id, due_at, note FROM wakeups WHERE done = 0 ORDER BY due_at"
        ) as cursor:
            return list(await cursor.fetchall())

    async def complete_wakeup(self, wakeup_id: int) -> None:
        """Mark a wakeup as done."""
        await self.conn.execute(
            "UPDATE wakeups SET done = 1 WHERE id = ?", (wakeup_id,)
        )
        await self.conn.commit()

    async def cancel_wakeup(self, wakeup_id: int) -> bool:
        """Mark a pending wakeup as done; return whether one was cancelled."""
        cursor = await self.conn.execute(
            "UPDATE wakeups SET done = 1 WHERE id = ? AND done = 0", (wakeup_id,)
        )
        await self.conn.commit()
        return cursor.rowcount > 0

    async def unanswered_chats(self) -> list[dict]:
        """Chats whose latest message is incoming (i.e. awaiting the agent).

        One row per chat — or per forum topic within a forum chat, so an
        answered topic cannot mask an unanswered one. Returns dicts with
        chat id/type/title, the topic id (``NULL`` outside topics), the
        sender's name and the date of that last message. Forum service
        messages never count as the awaiting message, so a freshly
        created topic doesn't nag forever.
        """
        query = """
            SELECT c.id AS chat_id, c.type, c.title, m.message_thread_id,
                   u.first_name, u.username, m.date
            FROM messages m
            JOIN chats c ON c.id = m.chat_id
            LEFT JOIN users u ON u.id = m.from_user_id
            WHERE m.outgoing = 0
              AND m.content_type NOT IN ('forum_topic_created',
                                         'forum_topic_edited',
                                         'forum_topic_closed',
                                         'forum_topic_reopened')
              AND NOT EXISTS (
                  SELECT 1 FROM messages n
                  WHERE n.chat_id = m.chat_id
                    AND COALESCE(n.message_thread_id, 0)
                        = COALESCE(m.message_thread_id, 0)
                    AND (n.date, n.message_id) > (m.date, m.message_id)
              )
            ORDER BY m.date
        """
        async with self.conn.execute(query) as cursor:
            return [dict(row) for row in await cursor.fetchall()]

    #: Message columns returned to the LLM as chat context.
    _MESSAGE_ROW = """
        SELECT m.message_id, m.date, m.outgoing, u.username, u.first_name,
               m.text, m.caption, m.content_type, m.message_thread_id
        FROM messages m LEFT JOIN users u ON u.id = m.from_user_id
    """

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
        query = (
            self._MESSAGE_ROW
            + """
            WHERE m.chat_id = ? AND (? IS NULL OR m.message_id < ?)
              AND (? IS NULL OR m.message_thread_id = ?)
            ORDER BY m.date DESC, m.message_id DESC LIMIT ?
        """
        )
        params = (
            chat_id,
            before_message_id,
            before_message_id,
            message_thread_id,
            message_thread_id,
            limit,
        )
        async with self.conn.execute(query, params) as cursor:
            rows = list(await cursor.fetchall())
        return [dict(row) for row in reversed(rows)]

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
        query = (
            self._MESSAGE_ROW
            + """
            WHERE m.chat_id = ? AND (instr(casefold(m.text), casefold(?))
                                     OR instr(casefold(m.caption), casefold(?)))
              AND (? IS NULL OR m.message_thread_id = ?)
            ORDER BY m.date DESC, m.message_id DESC LIMIT ?
        """
        )
        params = (chat_id, needle, needle, message_thread_id, message_thread_id, limit)
        async with self.conn.execute(query, params) as cursor:
            return [dict(row) for row in await cursor.fetchall()]

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
        query = r"""
            WITH RECURSIVE thread (message_id, reply_to_message_id) AS (
                SELECT message_id, reply_to_message_id FROM messages
                WHERE chat_id = :chat_id AND message_id = :message_id
                UNION
                SELECT m.message_id, m.reply_to_message_id
                FROM messages m JOIN thread t
                ON m.chat_id = :chat_id
                   AND (m.message_id = t.reply_to_message_id
                        OR m.reply_to_message_id = t.message_id)
            )
            SELECT m.message_id, m.date, m.outgoing, u.username, u.first_name,
                   m.text, m.caption, m.content_type, m.message_thread_id,
                   m.reply_to_message_id
            FROM messages m LEFT JOIN users u ON u.id = m.from_user_id
            WHERE m.chat_id = :chat_id
              AND m.message_id IN (SELECT message_id FROM thread)
            ORDER BY m.date DESC, m.message_id DESC
            LIMIT :limit
        """
        params = {"chat_id": chat_id, "message_id": message_id, "limit": limit}
        async with self.conn.execute(query, params) as cursor:
            rows = list(await cursor.fetchall())
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
        query = """
            SELECT name FROM (
                SELECT COALESCE(
                           json_extract(raw, '$.forum_topic_edited.name'),
                           json_extract(raw, '$.forum_topic_created.name'),
                           json_extract(
                               raw, '$.reply_to_message.forum_topic_created.name'
                           )
                       ) AS name,
                       (content_type = 'forum_topic_edited') AS renamed,
                       date, message_id
                FROM messages
                WHERE chat_id = ? AND message_thread_id = ?
            )
            WHERE name IS NOT NULL
            ORDER BY renamed DESC, date DESC, message_id DESC
            LIMIT 1
        """
        async with self.conn.execute(query, (chat_id, thread_id)) as cursor:
            row = await cursor.fetchone()
        return row["name"] if row else None

    async def topic_observed(self, chat_id: int, thread_id: int) -> bool:
        """True when any stored message of the chat belongs to the topic."""
        query = """
            SELECT 1 FROM messages
            WHERE chat_id = ? AND message_thread_id = ? LIMIT 1
        """
        async with self.conn.execute(query, (chat_id, thread_id)) as cursor:
            return await cursor.fetchone() is not None

    async def list_topics(self, chat_id: int) -> list[dict]:
        """Return the forum topics seen in a chat, most recent first.

        Each row carries the topic id, its latest known name, message
        count, last activity date and a ``closed`` flag from the newest
        close/reopen service message. The General topic never appears:
        its messages carry no topic id.
        """
        query = """
            SELECT m.message_thread_id AS topic_id,
                   COUNT(*) AS messages,
                   MAX(m.date) AS last_date,
                   COALESCE((SELECT e.content_type FROM messages e
                             WHERE e.chat_id = m.chat_id
                               AND e.message_thread_id = m.message_thread_id
                               AND e.content_type IN ('forum_topic_closed',
                                                      'forum_topic_reopened')
                             ORDER BY e.date DESC, e.message_id DESC LIMIT 1
                            ) = 'forum_topic_closed', 0) AS closed
            FROM messages m
            WHERE m.chat_id = ? AND m.message_thread_id IS NOT NULL
            GROUP BY m.message_thread_id
            ORDER BY last_date DESC
        """
        async with self.conn.execute(query, (chat_id,)) as cursor:
            rows = [dict(row) for row in await cursor.fetchall()]
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
        query = """
            SELECT c.id AS chat_id, c.type,
                   COALESCE(c.title, u.first_name, c.username, u.username) AS name,
                   COUNT(m.message_id) AS messages,
                   MAX(m.date) AS last_date
            FROM chats c
            LEFT JOIN users u ON u.id = c.id
            LEFT JOIN messages m ON m.chat_id = c.id
            GROUP BY c.id
            ORDER BY last_date DESC
        """
        async with self.conn.execute(query) as cursor:
            return [dict(row) for row in await cursor.fetchall()]

    async def chat_members(self, chat_id: int) -> list[dict]:
        """Return users seen talking in a chat, most recently active first.

        Built from stored history — Telegram doesn't let bots fetch a
        group's full roster, so this is who has actually said something.
        """
        query = """
            SELECT u.id AS user_id, u.username, u.first_name, u.last_name,
                   COUNT(*) AS messages, MAX(m.date) AS last_date
            FROM messages m
            JOIN users u ON u.id = m.from_user_id
            WHERE m.chat_id = ?
            GROUP BY u.id
            ORDER BY last_date DESC
        """
        async with self.conn.execute(query, (chat_id,)) as cursor:
            return [dict(row) for row in await cursor.fetchall()]

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
        chat_filter = ""
        params: list[object] = []
        if chat_ids is not None:
            placeholders = ", ".join("?" for _ in chat_ids)
            chat_filter = f" AND chat_id IN ({placeholders})"
            params.extend(chat_ids)
        query = (
            """
            SELECT json_extract(raw, '$.sticker.file_id') AS file_id,
                   json_extract(raw, '$.sticker.emoji') AS emoji,
                   json_extract(raw, '$.sticker.set_name') AS set_name,
                   MAX(date) AS last_date
            FROM messages
            WHERE content_type = 'sticker'
            """
            + chat_filter
            + """
            GROUP BY json_extract(raw, '$.sticker.file_unique_id')
            ORDER BY last_date DESC
            LIMIT ?
        """
        )
        params.append(limit)
        async with self.conn.execute(query, params) as cursor:
            return [dict(row) for row in await cursor.fetchall()]

    async def sticker_is_known(self, file_id: str, chat_ids: list[int]) -> bool:
        """Whether a sticker file id was observed in selected chats."""
        if not chat_ids:
            return False
        placeholders = ", ".join("?" for _ in chat_ids)
        query = f"""
            SELECT 1 FROM messages
            WHERE content_type = 'sticker'
              AND json_extract(raw, '$.sticker.file_id') = ?
              AND chat_id IN ({placeholders})
            LIMIT 1
        """
        async with self.conn.execute(query, [file_id, *chat_ids]) as cursor:
            return await cursor.fetchone() is not None

    async def message_exists(self, chat_id: int, message_id: int) -> bool:
        """Whether a message was observed and stored in one chat."""
        async with self.conn.execute(
            "SELECT 1 FROM messages WHERE chat_id = ? AND message_id = ?",
            (chat_id, message_id),
        ) as cursor:
            return await cursor.fetchone() is not None

    async def message_is_outgoing(self, chat_id: int, message_id: int) -> bool:
        """Whether a stored message was sent by the bot itself.

        Unknown messages are not outgoing: every message the bot sends is
        persisted, so "not stored" means "not ours to touch".
        """
        async with self.conn.execute(
            "SELECT outgoing FROM messages WHERE chat_id = ? AND message_id = ?",
            (chat_id, message_id),
        ) as cursor:
            row = await cursor.fetchone()
        return bool(row and row["outgoing"])

    async def delete_message(self, chat_id: int, message_id: int) -> None:
        """Remove a message row (mirrors a deletion done on Telegram)."""
        await self.conn.execute(
            "DELETE FROM messages WHERE chat_id = ? AND message_id = ?",
            (chat_id, message_id),
        )
        await self.conn.commit()

    async def save_message(self, message: Message, *, outgoing: bool = False) -> None:
        """Persist a message together with its chat and sender.

        Stores common fields in dedicated columns and the full serialized
        payload in ``raw``. Re-saving the same ``(chat_id, message_id)``
        (e.g. an edit) updates the mutable columns in place. Set
        ``outgoing=True`` for messages the bot itself sent.
        """
        await self.upsert_chat(message.chat)
        if message.from_user is not None:
            await self.upsert_user(message.from_user)
        await self.conn.execute(
            """
            INSERT INTO messages (chat_id, message_id, from_user_id, date,
                                  edit_date, content_type, text, caption,
                                  reply_to_message_id, message_thread_id,
                                  media_group_id, outgoing, raw)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (chat_id, message_id) DO UPDATE SET
                edit_date = excluded.edit_date,
                content_type = excluded.content_type,
                text = excluded.text,
                caption = excluded.caption,
                raw = excluded.raw,
                saved_at = datetime('now')
            """,
            (
                message.chat.id,
                message.message_id,
                message.from_user.id if message.from_user else None,
                message.date.isoformat(),
                # Telegram sends edit_date as a unix timestamp; normalize to
                # ISO so it compares with the `date` column.
                datetime.fromtimestamp(message.edit_date, tz=UTC).isoformat()
                if message.edit_date
                else None,
                message.content_type,
                message.text,
                message.caption,
                effective_reply_to(message),
                # Bot API also sets message_thread_id on plain reply chains;
                # is_topic_message discriminates real forum topics.
                message.message_thread_id if message.is_topic_message else None,
                message.media_group_id,
                int(outgoing),
                message.model_dump_json(exclude_none=True),
            ),
        )
        await self.conn.commit()
