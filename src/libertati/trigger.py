"""Manually run a background routine once, then exit.

Usage::

    uv run python -m libertati.trigger heartbeat
    uv run python -m libertati.trigger dream
    uv run python -m libertati.trigger browse
    uv run python -m libertati.trigger news

Useful for testing without waiting for the scheduler. Uses the same config/env
as the bot. ``heartbeat`` and ``browse`` need to send messages, so they connect
the Telegram client; ``dream`` and ``news`` only touch memory/DB.
"""

from __future__ import annotations

import asyncio
import sys

from .config import load_settings
from .logging import get_logger, setup_logging
from .main import App

log = get_logger("trigger")

ROUTINES = ("heartbeat", "dream", "browse", "news")


async def _run(routine: str) -> None:
    settings = load_settings()
    setup_logging(settings.log_level)
    settings.validate_runtime()

    app = App(settings)
    await app.db.connect()
    app._build_runtime()  # history, agent, tools, scheduler wiring (no polling)

    from .scheduler import jobs

    needs_client = routine in ("heartbeat", "browse")
    if needs_client:
        await app.client.connect()

    try:
        assert app.agent is not None and app.history is not None
        if routine == "heartbeat":
            await jobs.heartbeat_job(app.agent, app.client, app.history)
        elif routine == "dream":
            await jobs.dream_job(app.agent, app.memory)
        elif routine == "browse":
            await jobs.browse_job(app.agent, app.client, app.history, settings)
        elif routine == "news":
            await jobs.news_refresh_job(app.news, app.memory)
    finally:
        if needs_client:
            await app.client.stop()
        await app.db.close()
        app.events.close()
    log.info("routine %s complete", routine)


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in ROUTINES:
        print(f"usage: python -m libertati.trigger {{{'|'.join(ROUTINES)}}}")
        raise SystemExit(2)
    asyncio.run(_run(sys.argv[1]))


if __name__ == "__main__":
    main()
