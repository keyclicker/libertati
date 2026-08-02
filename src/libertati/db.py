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
        await self._conn.commit()

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

    async def load_context(self, limit: int) -> list[dict]:
        """Return the newest ``limit`` context items, oldest first."""
        async with self.conn.execute(
            "SELECT item FROM context ORDER BY id DESC LIMIT ?", (limit,)
        ) as cursor:
            rows = list(await cursor.fetchall())
        return [json.loads(row["item"]) for row in reversed(rows)]

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
