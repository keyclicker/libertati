"""Backend-agnostic Telegram interface.

The rest of the application depends only on the types defined here, so it does
not care whether messages arrive over the HTTP Bot API (aiogram) or MTProto
(Telethon).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

MessageHandler = Callable[["IncomingMessage"], Awaitable[None]]


@dataclass(slots=True)
class IncomingMessage:
    chat_id: int
    message_id: int
    text: str
    user_handle: str | None = None      # e.g. "@alice"
    user_name: str | None = None        # display name
    reply_to_id: int | None = None
    is_group: bool = False
    chat_title: str | None = None
    ts: float | None = None
    from_self: bool = False             # message authored by the bot/account itself


@dataclass(slots=True)
class OutgoingMessage:
    chat_id: int
    text: str
    reply_to_id: int | None = None


class TelegramClient(ABC):
    """Common surface implemented by both backends."""

    def __init__(self) -> None:
        self._handler: MessageHandler | None = None

    def on_message(self, handler: MessageHandler) -> None:
        self._handler = handler

    async def _dispatch(self, message: IncomingMessage) -> None:
        if self._handler is not None:
            await self._handler(message)

    @abstractmethod
    async def start(self) -> None:
        """Connect and begin receiving updates (blocks until stopped)."""

    @abstractmethod
    async def send_message(
        self, chat_id: int, text: str, reply_to_id: int | None = None
    ) -> IncomingMessage | None:
        """Send a message; return it as an IncomingMessage (from_self=True)."""

    @abstractmethod
    async def stop(self) -> None:
        """Disconnect cleanly."""

    @property
    @abstractmethod
    def self_username(self) -> str | None:
        """The bot/account username, once connected."""
