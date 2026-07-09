"""APScheduler wiring.

- Heartbeat: twice a day at randomised times (re-rolled daily).
- Dreaming: once a day at a randomised early-morning time.
- News refresh: every ``news_refresh_hours`` hours.
"""

from __future__ import annotations

import random
from datetime import datetime, time, timedelta

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
    ) -> None:
        self.settings = settings
        self.agent = agent
        self.client = client
        self.history = history
        self.memory = memory
        self.news = news
        self._scheduler = AsyncIOScheduler()

    def start(self) -> None:
        s = self.settings
        if s.heartbeat_enabled:
            # Re-plan the two random heartbeat times every day just after midnight,
            # and plan today's remaining ones immediately.
            self._scheduler.add_job(
                self._plan_heartbeats,
                CronTrigger(hour=0, minute=1),
                id="plan_heartbeats",
                replace_existing=True,
            )
            self._plan_heartbeats()
        if s.dream_enabled:
            hour = random.randint(3, 6)
            minute = random.randint(0, 59)
            self._scheduler.add_job(
                jobs.dream_job,
                CronTrigger(hour=hour, minute=minute),
                args=[self.agent, self.memory],
                id="dream",
                replace_existing=True,
            )
            log.info("dreaming scheduled daily at %02d:%02d", hour, minute)
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
            if self.client.supports_reading:
                self._scheduler.add_job(
                    self._plan_browses,
                    CronTrigger(hour=0, minute=2),
                    id="plan_browses",
                    replace_existing=True,
                )
                self._plan_browses()
            else:
                log.warning(
                    "browse_enabled is set but reading is unsupported in %s mode; skipping",
                    s.telegram_mode,
                )
        self._scheduler.start()
        log.info("scheduler started")

    def _plan_heartbeats(self) -> None:
        now = datetime.now()
        planned = 0
        for slot in range(2):
            when = _random_time_today(now, slot)
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
                args=[self.agent, self.client, self.history, self.settings],
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


def _random_time_today(now: datetime, slot: int) -> datetime:
    """Random time in the morning (slot 0) or evening (slot 1) window."""
    if slot == 0:
        start, end = time(9, 0), time(13, 0)
    else:
        start, end = time(17, 0), time(22, 0)
    minutes = random.randint(0, (end.hour - start.hour) * 60 + (end.minute - start.minute))
    base = now.replace(hour=start.hour, minute=start.minute, second=0, microsecond=0)
    return base + timedelta(minutes=minutes)


def _random_time_in_day(now: datetime, start_hour: int = 8, end_hour: int = 23) -> datetime:
    """A random time within the [start_hour, end_hour] window today."""
    minutes = random.randint(0, (end_hour - start_hour) * 60)
    base = now.replace(hour=start_hour, minute=0, second=0, microsecond=0)
    return base + timedelta(minutes=minutes)
