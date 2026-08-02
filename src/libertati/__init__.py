"""Libertati — an agentic Telegram bot.

Package entry point exposing :func:`main`, which is wired to the
``libertati`` console script.
"""

import asyncio

from libertati.bot import run


def main() -> None:
    """Run the bot until interrupted (sync wrapper around :func:`run`)."""
    asyncio.run(run())
