"""APScheduler wiring.

- Heartbeat: ``heartbeat_times_per_day`` randomised times (re-rolled daily).
- Dreaming: ``dream_times_per_day`` randomised early-morning times.
- News refresh: every ``news_refresh_hours`` hours.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import IntervalTrigger

from ..config import Settings
from ..llm.agent import Agent
from ..logging import get_logger
from ..news.reader import NewsReader
from ..storage.history import HistoryStore
from ..storage.memory import MemoryStore
from ..telegram.base import TelegramClient
from ..telegram.webpreview import WebPreviewReader
from . import jobs

log = get_logger("scheduler.runner")


class Scheduler:
    def __init__(
        self,
        settings: Settings,
        agent: Agent,
        client: TelegramClient,
        history: HistoryStore,
        memory: MemoryStore,
        news: NewsReader,
        reader: WebPreviewReader | None = None,
    ) -> None:
        self.settings = settings
        self.agent = agent
        self.client = client
        self.history = history
        self.memory = memory
        self.news = news
        self.reader = reader
        self._scheduler = AsyncIOScheduler()

    def start(self) -> None:
        s = self.settings
        if s.heartbeat_enabled and s.heartbeat_times_per_day > 0:
            # Re-plan the random heartbeat times every day just after midnight,
            # and plan today's remaining ones immediately.
            self._scheduler.add_job(
                self._plan_heartbeats,
                CronTrigger(hour=0, minute=1),
                id="plan_heartbeats",
                replace_existing=True,
            )
            self._plan_heartbeats()
        if s.dream_enabled and s.dream_times_per_day > 0:
            self._scheduler.add_job(
                self._plan_dreams,
                CronTrigger(hour=0, minute=1),
                id="plan_dreams",
                replace_existing=True,
            )
            self._plan_dreams()
        if s.news_feeds:
            self._scheduler.add_job(
                jobs.news_refresh_job,
                IntervalTrigger(hours=max(1, s.news_refresh_hours)),
                args=[self.news, self.memory],
                id="news",
                replace_existing=True,
                next_run_time=datetime.now() + timedelta(seconds=30),
            )
        if s.browse_enabled:
            if self.reader is not None and s.browse_channels:
                self._scheduler.add_job(
                    self._plan_browses,
                    CronTrigger(hour=0, minute=2),
                    id="plan_browses",
                    replace_existing=True,
                )
                self._plan_browses()
            else:
                log.warning(
                    "browse_enabled is set but no reader/browse_channels configured; skipping"
                )
        self._scheduler.start()
        log.info("scheduler started")

    def _plan_heartbeats(self) -> None:
        now = datetime.now()
        total = self.settings.heartbeat_times_per_day
        planned = 0
        for slot in range(total):
            when = _slot_time(now, slot, total, start_hour=9, end_hour=22)
            if when <= now:
                continue
            self._scheduler.add_job(
                jobs.heartbeat_job,
                DateTrigger(run_date=when),
                args=[self.agent, self.client, self.history],
                id=f"heartbeat_{slot}",
                replace_existing=True,
            )
            planned += 1
            log.info("heartbeat scheduled at %s", when.strftime("%H:%M"))
        if planned == 0:
            log.info("no heartbeat slots left today")

    def _plan_dreams(self) -> None:
        now = datetime.now()
        total = self.settings.dream_times_per_day
        planned = 0
        for slot in range(total):
            when = _slot_time(now, slot, total, start_hour=3, end_hour=7)
            if when <= now:
                continue
            self._scheduler.add_job(
                jobs.dream_job,
                DateTrigger(run_date=when),
                args=[self.agent, self.memory],
                id=f"dream_{slot}",
                replace_existing=True,
            )
            planned += 1
            log.info("dreaming scheduled at %s", when.strftime("%H:%M"))
        if planned == 0:
            log.info("no dream slots left today")

    def _plan_browses(self) -> None:
        now = datetime.now()
        planned = 0
        for i in range(max(1, self.settings.browse_times_per_day)):
            when = _random_time_in_day(now)
            if when <= now:
                continue
            self._scheduler.add_job(
                jobs.browse_job,
                DateTrigger(run_date=when),
                args=[self.agent, self.client, self.reader, self.history, self.settings],
                id=f"browse_{i}",
                replace_existing=True,
            )
            planned += 1
            log.info("browse scheduled at %s", when.strftime("%H:%M"))
        if planned == 0:
            log.info("no browse slots left today")

    def shutdown(self) -> None:
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)


def _slot_time(
    now: datetime, slot: int, total: int, start_hour: int, end_hour: int
) -> datetime:
    """Random time in the ``slot``-th of ``total`` equal windows of [start, end] today.

    Splitting the window keeps the times spread over the day instead of clustering.
    """
    span = (end_hour - start_hour) * 60
    lo = span * slot // total
    hi = span * (slot + 1) // total
    minutes = random.randint(lo, max(lo, hi - 1))
    base = now.replace(hour=start_hour, minute=0, second=0, microsecond=0)
    return base + timedelta(minutes=minutes)


def _random_time_in_day(now: datetime, start_hour: int = 8, end_hour: int = 23) -> datetime:
    """A random time within the [start_hour, end_hour] window today."""
    minutes = random.randint(0, (end_hour - start_hour) * 60)
    base = now.replace(hour=start_hour, minute=0, second=0, microsecond=0)
    return base + timedelta(minutes=minutes)
