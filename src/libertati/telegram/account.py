"""Telethon adapter — the MTProto (user account) backend.

Running as a real user account (rather than a bot) lets libertati read all
messages in its chats and proactively DM people during heartbeats.
"""

from __future__ import annotations

import asyncio

from telethon import TelegramClient as TelethonClient
from telethon import events

from ..logging import get_logger
from .base import IncomingMessage, TelegramClient

log = get_logger("telegram.account")


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
        handle = f"@{sender.username}" if getattr(sender, "username", None) else None
        first = getattr(sender, "first_name", None)
        last = getattr(sender, "last_name", None)
        name = " ".join(p for p in (first, last) if p) or None
        chat = await event.get_chat()
        title = getattr(chat, "title", None) or name
        return IncomingMessage(
            chat_id=event.chat_id,
            message_id=msg.id,
            text=msg.message or "",
            user_handle=handle,
            user_name=name,
            reply_to_id=msg.reply_to_msg_id,
            is_group=bool(event.is_group),
            chat_title=title,
            ts=msg.date.timestamp() if msg.date else None,
            from_self=bool(event.out),
        )

    async def start(self) -> None:
        await self._client.connect()
        if not await self._client.is_user_authorized():
            raise RuntimeError(
                "Telethon session is not authorized. Run an interactive login once to "
                "create the session file (see README)."
            )
        me = await self._client.get_me()
        self._username = f"@{me.username}" if me and me.username else None
        log.info("Account mode online as %s", self._username)

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

    async def stop(self) -> None:
        self._stop.set()
        await self._client.disconnect()
