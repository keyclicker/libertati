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

-- Bookkeeping only: a dream's context is never persisted, but its budget
-- has to survive restarts, and a dream that dies before writing its
-- journal entry still has to count against that budget.
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

CREATE TABLE IF NOT EXISTS wakeups (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    due_at     TEXT NOT NULL,
    note       TEXT NOT NULL,
    done       INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_wakeups_due ON wakeups (done, due_at);
"""


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
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode = WAL")
        await self._conn.execute("PRAGMA foreign_keys = ON")
        await self._conn.executescript(SCHEMA)
        # ``CREATE TABLE IF NOT EXISTS`` does not add columns to DBs
        # created before usage snapshots gained context linkage.
        await self._ensure_column("api_usage", "turn_id", "INTEGER")
        await self._ensure_column(
            "api_usage", "input_context_id", "INTEGER NOT NULL DEFAULT 0"
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
            (json.dumps(item, ensure_ascii=False),),
        )
        await self.conn.commit()

    async def latest_context_id(self) -> int:
        """Return newest persisted context id, or zero when empty."""
        async with self.conn.execute(
            "SELECT COALESCE(MAX(id), 0) FROM context"
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
        turn_id: int,
        input_context_id: int,
        model: str,
        input_tokens: int,
        cached_tokens: int,
        cache_write_tokens: int,
        output_tokens: int,
        reasoning_tokens: int,
        total_tokens: int,
    ) -> None:
        """Persist exact token and prompt-cache usage for one API call."""
        await self.conn.execute(
            """
            INSERT INTO api_usage (
                response_id, turn_id, input_context_id, model,
                input_tokens, cached_tokens, cache_write_tokens,
                output_tokens, reasoning_tokens, total_tokens
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                response_id,
                turn_id,
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

        Returns dicts with chat id/type/title, the sender's name and the
        date of that last message.
        """
        query = """
            SELECT c.id AS chat_id, c.type, c.title,
                   u.first_name, u.username, m.date
            FROM chats c
            JOIN messages m ON m.chat_id = c.id AND m.date = (
                SELECT MAX(date) FROM messages WHERE chat_id = c.id
            )
            LEFT JOIN users u ON u.id = m.from_user_id
            WHERE m.outgoing = 0
            ORDER BY m.date
        """
        async with self.conn.execute(query) as cursor:
            return [dict(row) for row in await cursor.fetchall()]

    #: Message columns returned to the LLM as chat context.
    _MESSAGE_ROW = """
        SELECT m.message_id, m.date, m.outgoing, u.username, u.first_name,
               m.text, m.caption, m.content_type
        FROM messages m LEFT JOIN users u ON u.id = m.from_user_id
    """

    async def recent_messages(
        self, chat_id: int, limit: int, before_message_id: int | None = None
    ) -> list[dict]:
        """Return the latest ``limit`` messages of a chat, oldest first.

        Each row is a small dict (id, date, sender, text/caption, content
        type, outgoing flag) suitable for feeding to the LLM as context.
        ``before_message_id`` pages into the past: only messages older
        than it are returned.
        """
        query = (
            self._MESSAGE_ROW
            + """
            WHERE m.chat_id = ? AND (? IS NULL OR m.message_id < ?)
            ORDER BY m.date DESC LIMIT ?
        """
        )
        params = (chat_id, before_message_id, before_message_id, limit)
        async with self.conn.execute(query, params) as cursor:
            rows = list(await cursor.fetchall())
        return [dict(row) for row in reversed(rows)]

    async def search_messages(
        self, chat_id: int, needle: str, limit: int
    ) -> list[dict]:
        """Return a chat's messages containing ``needle``, newest first.

        Case-insensitive substring match over text and caption; rows have
        the same shape as :meth:`recent_messages`.
        """
        escaped = needle.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        query = (
            self._MESSAGE_ROW
            + r"""
            WHERE m.chat_id = ? AND (m.text LIKE ? ESCAPE '\'
                                     OR m.caption LIKE ? ESCAPE '\')
            ORDER BY m.date DESC LIMIT ?
        """
        )
        pattern = f"%{escaped}%"
        params = (chat_id, pattern, pattern, limit)
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
                   m.text, m.caption, m.content_type, m.reply_to_message_id
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

    async def known_stickers(self, limit: int) -> list[dict]:
        """Return distinct stickers seen anywhere, most recently seen first.

        Extracted from raw payloads; deduplicated by the sticker's stable
        ``file_unique_id``, returning a ``file_id`` the bot can resend.
        """
        query = """
            SELECT json_extract(raw, '$.sticker.file_id') AS file_id,
                   json_extract(raw, '$.sticker.emoji') AS emoji,
                   json_extract(raw, '$.sticker.set_name') AS set_name,
                   MAX(date) AS last_date
            FROM messages
            WHERE content_type = 'sticker'
            GROUP BY json_extract(raw, '$.sticker.file_unique_id')
            ORDER BY last_date DESC
            LIMIT ?
        """
        async with self.conn.execute(query, (limit,)) as cursor:
            return [dict(row) for row in await cursor.fetchall()]

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
                                  reply_to_message_id, media_group_id,
                                  outgoing, raw)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                message.reply_to_message.message_id
                if message.reply_to_message
                else None,
                message.media_group_id,
                int(outgoing),
                message.model_dump_json(exclude_none=True),
            ),
        )
        await self.conn.commit()
