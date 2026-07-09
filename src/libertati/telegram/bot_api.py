"""aiogram adapter — the HTTP Bot API backend."""

from __future__ import annotations

from aiogram import Bot, Dispatcher
from aiogram import types as agot

from ..logging import get_logger
from .base import IncomingMessage, TelegramClient

log = get_logger("telegram.bot_api")


def _to_incoming(msg: agot.Message, self_id: int | None) -> IncomingMessage:
    user = msg.from_user
    handle = f"@{user.username}" if user and user.username else None
    name = user.full_name if user else None
    reply_to = msg.reply_to_message.message_id if msg.reply_to_message else None
    is_group = msg.chat.type in ("group", "supergroup")
    return IncomingMessage(
        chat_id=msg.chat.id,
        message_id=msg.message_id,
        text=msg.text or msg.caption or "",
        user_handle=handle,
        user_name=name,
        reply_to_id=reply_to,
        is_group=is_group,
        chat_title=msg.chat.title or (name if not is_group else None),
        ts=msg.date.timestamp() if msg.date else None,
        from_self=bool(self_id and user and user.id == self_id),
    )


class BotApiClient(TelegramClient):
    def __init__(self, token: str) -> None:
        super().__init__()
        self._bot = Bot(token=token)
        self._dp = Dispatcher()
        self._self_id: int | None = None
        self._username: str | None = None

        @self._dp.message()
        async def _on_message(message: agot.Message) -> None:  # pragma: no cover - I/O glue
            if not (message.text or message.caption):
                return
            await self._dispatch(_to_incoming(message, self._self_id))

    @property
    def self_username(self) -> str | None:
        return self._username

    async def start(self) -> None:
        me = await self._bot.get_me()
        self._self_id = me.id
        self._username = f"@{me.username}" if me.username else None
        log.info("Bot API mode online as %s", self._username)
        await self._dp.start_polling(self._bot, handle_signals=False)

    async def send_message(
        self, chat_id: int, text: str, reply_to_id: int | None = None
    ) -> IncomingMessage | None:
        sent = await self._bot.send_message(
            chat_id=chat_id, text=text, reply_to_message_id=reply_to_id
        )
        return _to_incoming(sent, self._self_id)

    async def stop(self) -> None:
        await self._dp.stop_polling()
        await self._bot.session.close()
