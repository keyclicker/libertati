"""Scheduled background routines: heartbeat, dreaming, news refresh."""

from __future__ import annotations

import time

from ..llm.agent import Agent
from ..logging import get_logger
from ..news.reader import NewsReader, format_digest
from ..storage.history import HistoryStore
from ..storage.memory import MemoryStore
from ..telegram.base import TelegramClient

log = get_logger("scheduler.jobs")


async def _resolve_target(target: str, history: HistoryStore) -> int | str | None:
    target = target.strip()
    if target.lstrip("-").isdigit():
        return int(target)
    chat_id = await history.chat_id_for_handle(target)
    return chat_id if chat_id is not None else target  # Telethon accepts @username


async def heartbeat_job(agent: Agent, client: TelegramClient, history: HistoryStore) -> None:
    log.info("heartbeat: thinking...")
    action = await agent.heartbeat()
    if action is None:
        log.info("heartbeat: nothing to say")
        return
    target = await _resolve_target(action.target, history)
    if target is None:
        log.info("heartbeat: could not resolve target %s", action.target)
        return
    try:
        sent = await client.send_message(target, action.text)  # type: ignore[arg-type]
    except Exception as exc:  # noqa: BLE001
        log.warning("heartbeat send failed: %s", exc)
        return
    if sent is not None:
        await history.add_message(sent, role="assistant")
    log.info("heartbeat: messaged %s", action.target)


async def dream_job(agent: Agent, memory: MemoryStore) -> None:
    log.info("dreaming...")
    reflection = await agent.dream()
    log.info("dream complete (%d chars)", len(reflection))


async def news_refresh_job(news: NewsReader, memory: MemoryStore) -> None:
    log.info("refreshing news...")
    items = await news.fetch()
    if not items:
        return
    stamp = time.strftime("%Y-%m-%d %H:%M")
    memory.append("world.md", f"\n## Новини ({stamp})\n{format_digest(items)}")
    log.info("news: appended %d items to world.md", len(items))
