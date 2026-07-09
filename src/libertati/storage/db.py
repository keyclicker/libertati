"""SQLite database: schema, migrations, indexes and FTS5."""

from __future__ import annotations

from pathlib import Path

import aiosqlite

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chats (
    chat_id   INTEGER PRIMARY KEY,
    kind      TEXT,               -- 'private' | 'group' | 'channel'
    title     TEXT,
    updated_at REAL
);

CREATE TABLE IF NOT EXISTS users (
    handle     TEXT,
    name       TEXT,
    chat_id    INTEGER,
    last_seen  REAL,
    PRIMARY KEY (handle, chat_id)
);

CREATE TABLE IF NOT EXISTS messages (
    row_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id      INTEGER NOT NULL,
    message_id   INTEGER NOT NULL,
    user_handle  TEXT,
    user_name    TEXT,
    role         TEXT NOT NULL,   -- 'user' | 'assistant'
    text         TEXT,
    reply_to_id  INTEGER,
    ts           REAL NOT NULL,
    UNIQUE (chat_id, message_id)
);

CREATE INDEX IF NOT EXISTS idx_messages_chat_ts
    ON messages (chat_id, ts);
CREATE INDEX IF NOT EXISTS idx_messages_chat_reply
    ON messages (chat_id, reply_to_id);

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts
    USING fts5(text, content='messages', content_rowid='row_id');

CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, text) VALUES (new.row_id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, text)
        VALUES ('delete', old.row_id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, text)
        VALUES ('delete', old.row_id, old.text);
    INSERT INTO messages_fts(rowid, text) VALUES (new.row_id, new.text);
END;
"""


class Database:
    """Owns the aiosqlite connection and applies the schema."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._conn: aiosqlite.Connection | None = None

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Database not connected; call connect() first")
        return self._conn

    async def connect(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self.path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA foreign_keys=ON")
        await self._conn.executescript(_SCHEMA)
        await self._conn.commit()

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    async def __aenter__(self) -> Database:
        await self.connect()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()
