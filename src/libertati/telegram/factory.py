"""Pick a Telegram backend from settings."""

from __future__ import annotations

from ..config import Settings
from .base import TelegramClient


def build_client(settings: Settings) -> TelegramClient:
    if settings.telegram_mode == "bot":
        from .bot_api import BotApiClient

        return BotApiClient(token=settings.bot_token)
    if settings.telegram_mode == "account":
        from .account import AccountClient

        return AccountClient(
            api_id=settings.tg_api_id,
            api_hash=settings.tg_api_hash,
            session=settings.tg_session,
        )
    raise ValueError(f"unknown telegram_mode: {settings.telegram_mode!r}")
