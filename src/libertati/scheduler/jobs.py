"""Scheduled background routines: heartbeat, dreaming, news refresh."""

from __future__ import annotations

import random
import time

from ..config import Settings
from ..llm.agent import Agent
from ..llm.tools import format_transcript
from ..logging import get_logger
from ..news.reader import NewsReader, format_digest
from ..storage.history import HistoryStore
from ..storage.memory import MemoryStore
from ..telegram.base import TelegramClient
from ..telegram.webpreview import WebPreviewReader

log = get_logger("scheduler.jobs")


async def _resolve_target(target: str, history: HistoryStore) -> int | str | None:
    target = target.strip()
    if target.lstrip("-").isdigit():
        return int(target)
    chat_id = await history.chat_id_for_handle(target)
    return chat_id if chat_id is not None else target  # Bot API accepts @channelusername


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
        sent = await client.send_message(target, action.text)
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


def _pick_browse_source(settings: Settings) -> tuple[str, str] | None:
    """Choose a public channel to read from the configured list."""
    if not settings.browse_channels:
        return None
    channel = random.choice(settings.browse_channels)
    return channel, f"channel {channel}"


async def _pick_discuss_target(history: HistoryStore, settings: Settings) -> int | str | None:
    if settings.allowed_chats:
        return await _resolve_target(random.choice(settings.allowed_chats), history)
    chats = await history.active_chats(limit=20)
    groups = [c for c in chats if c.get("kind") == "group"] or chats
    return random.choice(groups)["chat_id"] if groups else None


async def browse_job(
    agent: Agent,
    client: TelegramClient,
    reader: WebPreviewReader,
    history: HistoryStore,
    settings: Settings,
) -> None:
    picked = _pick_browse_source(settings)
    if picked is None:
        log.info("browse: no channels configured (LIBERTATI_BROWSE_CHANNELS)")
        return
    ref, desc = picked
    messages = await reader.read_source(ref, limit=settings.browse_read_limit)
    if not messages:
        log.info("browse: %s had nothing to read", desc)
        return
    log.info("browse: read %d messages from %s", len(messages), desc)
    remark = await agent.browse(desc, format_transcript(messages))
    if not remark or remark.strip().upper() == "PASS":
        log.info("browse: nothing worth discussing")
        return
    target = await _pick_discuss_target(history, settings)
    if target is None:
        log.info("browse: no chat to discuss in")
        return
    try:
        sent = await client.send_message(target, remark)
    except Exception as exc:  # noqa: BLE001
        log.warning("browse send failed: %s", exc)
        return
    if sent is not None:
        await history.add_message(sent, role="assistant")
    log.info("browse: discussed %s in %s", desc, target)
