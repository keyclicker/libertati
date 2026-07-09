"""Telethon adapter — the MTProto (user account) backend.

Running as a real user account (rather than a bot) lets libertati read all
messages in its chats and proactively DM people during heartbeats.
"""

from __future__ import annotations

import asyncio

from telethon import TelegramClient as TelethonClient
from telethon import events

from ..logging import get_logger
from .base import IncomingMessage, Source, TelegramClient

log = get_logger("telegram.account")


def _sender_fields(sender: object) -> tuple[str | None, str | None]:
    """(handle, display name) from a Telethon sender-like object."""
    username = getattr(sender, "username", None)
    handle = f"@{username}" if username else None
    first = getattr(sender, "first_name", None)
    last = getattr(sender, "last_name", None)
    name = " ".join(p for p in (first, last) if p) or None
    return handle, name


def _message_to_incoming(
    msg: object, chat_id: int, title: str | None, username: str | None
) -> IncomingMessage:
    """Map a Telethon Message (from get_messages) to an IncomingMessage."""
    handle, name = _sender_fields(getattr(msg, "sender", None))
    date = getattr(msg, "date", None)
    return IncomingMessage(
        chat_id=chat_id,
        message_id=getattr(msg, "id", 0),
        text=getattr(msg, "message", "") or "",
        user_handle=handle,
        user_name=name,
        reply_to_id=getattr(msg, "reply_to_msg_id", None),
        is_group=True,
        chat_title=title,
        chat_username=username,
        ts=date.timestamp() if date else None,
        from_self=bool(getattr(msg, "out", False)),
    )


class AccountClient(TelegramClient):
    def __init__(self, api_id: int, api_hash: str, session: str) -> None:
        super().__init__()
        self._client = TelethonClient(session, api_id, api_hash)
        self._username: str | None = None
        self._stop = asyncio.Event()

    @property
    def self_username(self) -> str | None:
        return self._username

    async def _to_incoming(self, event: events.NewMessage.Event) -> IncomingMessage:
        msg = event.message
        sender = await event.get_sender()
        handle, name = _sender_fields(sender)
        chat = await event.get_chat()
        title = getattr(chat, "title", None) or name
        chat_username = f"@{chat.username}" if getattr(chat, "username", None) else None
        return IncomingMessage(
            chat_id=event.chat_id,
            message_id=msg.id,
            text=msg.message or "",
            user_handle=handle,
            user_name=name,
            reply_to_id=msg.reply_to_msg_id,
            is_group=bool(event.is_group),
            chat_title=title,
            chat_username=chat_username,
            ts=msg.date.timestamp() if msg.date else None,
            from_self=bool(event.out),
        )

    async def connect(self) -> None:
        await self._client.connect()
        if not await self._client.is_user_authorized():
            raise RuntimeError(
                "Telethon session is not authorized. Run an interactive login once to "
                "create the session file (see README)."
            )
        me = await self._client.get_me()
        self._username = f"@{me.username}" if me and me.username else None
        log.info("Account mode online as %s", self._username)

    async def start(self) -> None:
        await self.connect()

        @self._client.on(events.NewMessage)
        async def _handler(event: events.NewMessage.Event) -> None:  # pragma: no cover - I/O glue
            if not event.message.message:
                return
            await self._dispatch(await self._to_incoming(event))

        await self._stop.wait()

    async def send_message(
        self, chat_id: int, text: str, reply_to_id: int | None = None
    ) -> IncomingMessage | None:
        sent = await self._client.send_message(chat_id, text, reply_to=reply_to_id)
        return IncomingMessage(
            chat_id=chat_id,
            message_id=sent.id,
            text=text,
            user_handle=self._username,
            reply_to_id=reply_to_id,
            ts=sent.date.timestamp() if sent.date else None,
            from_self=True,
        )

    # -- reading -------------------------------------------------------------
    @property
    def supports_reading(self) -> bool:
        return True

    async def list_readable_sources(self, limit: int = 100) -> list[Source]:
        sources: list[Source] = []
        async for dialog in self._client.iter_dialogs(limit=limit):
            if not (dialog.is_group or dialog.is_channel):
                continue  # skip 1:1 user DMs
            entity = dialog.entity
            username = getattr(entity, "username", None)
            kind = "channel" if getattr(entity, "broadcast", False) else "group"
            sources.append(
                Source(
                    id=dialog.id,
                    title=dialog.name,
                    username=f"@{username}" if username else None,
                    kind=kind,
                    is_public=bool(username),
                )
            )
        return sources

    async def read_source(self, source: int | str, limit: int = 20) -> list[IncomingMessage]:
        entity = await self._client.get_entity(source)
        messages = await self._client.get_messages(entity, limit=limit)
        title = getattr(entity, "title", None)
        username = f"@{entity.username}" if getattr(entity, "username", None) else None
        chat_id = getattr(entity, "id", 0)
        out = [
            _message_to_incoming(msg, chat_id, title, username)
            for msg in reversed(list(messages))  # oldest first
            if getattr(msg, "message", None)
        ]
        return out

    async def stop(self) -> None:
        self._stop.set()
        await self._client.disconnect()
