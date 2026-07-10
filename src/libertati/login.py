"""Interactive one-time login for account (MTProto) mode.

Run once to create the Telethon session file::

    uv run python -m libertati.login

It prompts for your phone number and the login code (and 2FA password if set),
then writes ``<LIBERTATI_TG_SESSION>.session`` which the bot reuses on startup.
"""

from __future__ import annotations

import asyncio

from telethon import TelegramClient

from .config import load_settings
from .logging import setup_logging


async def _login() -> None:
    settings = load_settings()
    setup_logging(settings.log_level)
    if not (settings.tg_api_id and settings.tg_api_hash):
        raise SystemExit("Set LIBERTATI_TG_API_ID and LIBERTATI_TG_API_HASH first.")
    client = TelegramClient(settings.tg_session, settings.tg_api_id, settings.tg_api_hash)
    await client.start()  # prompts interactively as needed
    me = await client.get_me()
    handle = f"@{me.username}" if me and me.username else me.id if me else "?"
    print(f"Logged in as {handle}. Session saved to {settings.tg_session}.session")
    await client.disconnect()


def main() -> None:
    asyncio.run(_login())


if __name__ == "__main__":
    main()
