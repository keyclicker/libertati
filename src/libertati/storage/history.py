"""Message history store: persistence, thread reconstruction and FTS search."""

from __future__ import annotations

import time
from dataclasses import dataclass

from ..telegram.base import IncomingMessage
from .db import Database


@dataclass(slots=True)
class StoredMessage:
    chat_id: int
    message_id: int
    user_handle: str | None
    user_name: str | None
    role: str
    text: str
    reply_to_id: int | None
    ts: float


def _fts_query(raw: str) -> str:
    """Turn free text into a safe FTS5 MATCH query (quoted terms, OR-joined)."""
    terms = [t for t in "".join(c if c.isalnum() else " " for c in raw).split() if t]
    if not terms:
        return '""'
    return " OR ".join(f'"{t}"' for t in terms)


class HistoryStore:
    """Durable, indexed replacement for the old pickle message log."""

    def __init__(self, db: Database, max_thread_chars: int = 5000) -> None:
        self.db = db
        self.max_thread_chars = max_thread_chars

    async def add_message(self, msg: IncomingMessage, role: str = "user") -> None:
        ts = msg.ts or time.time()
        await self.db.conn.execute(
            """
            INSERT INTO messages
                (chat_id, message_id, user_handle, user_name, role, text, reply_to_id, ts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(chat_id, message_id) DO UPDATE SET
                text=excluded.text, role=excluded.role
            """,
            (
                msg.chat_id,
                msg.message_id,
                msg.user_handle,
                msg.user_name,
                role,
                msg.text,
                msg.reply_to_id,
                ts,
            ),
        )
        await self.db.conn.execute(
            """
            INSERT INTO chats (chat_id, kind, title, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(chat_id) DO UPDATE SET
                kind=excluded.kind, title=excluded.title, updated_at=excluded.updated_at
            """,
            (msg.chat_id, "group" if msg.is_group else "private", msg.chat_title, ts),
        )
        if msg.user_handle:
            await self.db.conn.execute(
                """
                INSERT INTO users (handle, name, chat_id, last_seen)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(handle, chat_id) DO UPDATE SET
                    name=excluded.name, last_seen=excluded.last_seen
                """,
                (msg.user_handle, msg.user_name, msg.chat_id, ts),
            )
        await self.db.conn.commit()

    async def _row_to_stored(self, row) -> StoredMessage:
        return StoredMessage(
            chat_id=row["chat_id"],
            message_id=row["message_id"],
            user_handle=row["user_handle"],
            user_name=row["user_name"],
            role=row["role"],
            text=row["text"] or "",
            reply_to_id=row["reply_to_id"],
            ts=row["ts"],
        )

    async def get_message(self, chat_id: int, message_id: int) -> StoredMessage | None:
        async with self.db.conn.execute(
            "SELECT * FROM messages WHERE chat_id=? AND message_id=?",
            (chat_id, message_id),
        ) as cur:
            row = await cur.fetchone()
        return await self._row_to_stored(row) if row else None

    async def get_thread(self, msg: IncomingMessage) -> list[StoredMessage]:
        """Walk the reply chain backwards up to max_thread_chars, oldest-first."""
        thread: list[StoredMessage] = []
        total = 0
        chat_id = msg.chat_id
        cur_id: int | None = msg.message_id
        seen: set[int] = set()
        while cur_id is not None and total < self.max_thread_chars and cur_id not in seen:
            seen.add(cur_id)
            stored = await self.get_message(chat_id, cur_id)
            if stored is None:
                break
            total += len(stored.text)
            thread.append(stored)
            cur_id = stored.reply_to_id
        thread.reverse()
        return thread

    async def recent(self, chat_id: int, limit: int = 25) -> list[StoredMessage]:
        async with self.db.conn.execute(
            "SELECT * FROM messages WHERE chat_id=? ORDER BY ts DESC LIMIT ?",
            (chat_id, limit),
        ) as cur:
            rows = await cur.fetchall()
        out = [await self._row_to_stored(r) for r in rows]
        out.reverse()
        return out

    async def search(self, query: str, limit: int = 20) -> list[StoredMessage]:
        async with self.db.conn.execute(
            """
            SELECT m.* FROM messages_fts f
            JOIN messages m ON m.row_id = f.rowid
            WHERE messages_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (_fts_query(query), limit),
        ) as cur:
            rows = await cur.fetchall()
        return [await self._row_to_stored(r) for r in rows]

    async def chat_id_for_handle(self, handle: str) -> int | None:
        """Most recent chat where a given user handle was seen."""
        async with self.db.conn.execute(
            "SELECT chat_id FROM users WHERE handle=? ORDER BY last_seen DESC LIMIT 1",
            (handle,),
        ) as cur:
            row = await cur.fetchone()
        return row["chat_id"] if row else None

    async def active_chats(self, limit: int = 50) -> list[dict]:
        async with self.db.conn.execute(
            "SELECT chat_id, kind, title FROM chats ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ) as cur:
            rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def _count(self, sql: str) -> int:
        async with self.db.conn.execute(sql) as cur:
            row = await cur.fetchone()
        return int(row["n"]) if row else 0

    async def stats(self) -> dict[str, int]:
        return {
            "messages": await self._count("SELECT COUNT(*) AS n FROM messages"),
            "chats": await self._count("SELECT COUNT(*) AS n FROM chats"),
            "users": await self._count("SELECT COUNT(DISTINCT handle) AS n FROM users"),
        }
