"""Libertati — an agentic Telegram bot.

Package entry point exposing :func:`main`, which is wired to the
``libertati`` console script.
"""

import asyncio


def main() -> None:
    """Run the bot until interrupted (sync wrapper around :func:`run`)."""
    # Imported here, not at module level: importing the package pulls in
    # aiogram and openai otherwise, which costs seconds of startup for
    # side tools like `libertati-spy` that need none of it.
    from libertati.bot import run

    asyncio.run(run())
