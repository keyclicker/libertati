"""The dreaming loop: the agent's offline pass over its own mind.

When nothing has needed the agent for a while (or it asked to sleep), the
waking loop is paused and this one takes over for a single long session.
It is handed its mind files, wanders wherever its curiosity and the
read-only tools take it, then before waking writes a reflection into
DREAMS.md, folds INBOX.md into a rewritten MEMORY.md and may revise
SOUL.md.

A dream's context is written to ``dream_context`` for inspection but
never read back: the live window is dropped on waking and nothing from it
rejoins the waking agent's. What actually carries over is what the dream
chose to write into the mind files, plus a ledger row for budget
accounting and the ``[dream ended …]`` event handed back to the agent.
"""

import logging
import math
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from aiogram import Bot

from libertati import clock
from libertati.config import Settings
from libertati.db import Database
from libertati.loop import ModelLoop
from libertati.tools import DREAM_TOOL_NAMES, Toolbox, build_dream_tools

if TYPE_CHECKING:  # pragma: no cover - import cycle broken for runtime
    from libertati.agent import Agent

log = logging.getLogger(__name__)

#: How much of the dream journal is handed to a new dream, in characters.
DREAMS_TAIL_CHARS = 8000

#: Placeholder for a mind file that has nothing in it yet.
EMPTY = "(empty)"


class DreamGate:
    """Dream budget accounting and the awake agent's request slot.

    Owned jointly by the ``dream`` tool (which requests a dream and needs
    to know whether there is budget for one) and by :class:`Dreamer`
    (which honours the request). Keeping it separate from both is what
    lets the toolbox ask about dreaming without knowing the loop.
    """

    def __init__(
        self, db: Database, daily_budget: int, cooldown_minutes: int = 0
    ) -> None:
        """Keep the ledger handle and the rolling 24h budget."""
        self.db = db
        self.daily_budget = daily_budget
        self.cooldown_minutes = cooldown_minutes
        self.note: str | None = None

    @property
    def requested(self) -> bool:
        """Whether a dream was requested and not yet consumed."""
        return self.note is not None

    def request(self, note: str) -> None:
        """Record what the awake agent wants to sleep on."""
        self.note = note

    def take(self) -> str | None:
        """Consume the pending request, if there is one."""
        note, self.note = self.note, None
        return note

    async def budget_left(self) -> int:
        """Return how many dreams are still allowed in the next 24h."""
        since = clock.utc_stamp(datetime.now(UTC) - timedelta(hours=24))
        return max(0, self.daily_budget - await self.db.dreams_since(since))

    async def cooldown_left(self) -> int:
        """Return whole minutes until another dream may start."""
        last = await self.db.last_dream_end()
        if last is None:
            return 0
        elapsed = datetime.now(UTC) - clock.parse_utc_stamp(last)
        remaining = timedelta(minutes=self.cooldown_minutes) - elapsed
        return max(0, math.ceil(remaining.total_seconds() / 60))


class Dreamer(ModelLoop):
    """Runs one dream at a time, with the waking agent held still.

    Shares the round engine and the client with :class:`~libertati.agent.Agent`
    but nothing else: its own prompt, its own restricted toolbox, its own
    throwaway context.
    """

    def __init__(
        self,
        settings: Settings,
        db: Database,
        bot: Bot,
        agent: "Agent",
        gate: DreamGate,
    ) -> None:
        """Build the restricted toolbox and take the dream settings."""
        super().__init__(
            client=agent.client,
            model=settings.dream_model or settings.model,
            db=db,
            tools=Toolbox(
                db,
                bot,
                agent.tz,
                agent.client,
                settings.recall_model or settings.model,
                agent.mind,
                settings.typing_chars_per_second,
                agent.prompts.recall,
                agent.prompts.summary,
                registry=agent.registry,
                recall_effort=settings.recall_reasoning_effort,
                allowed=DREAM_TOOL_NAMES,
                dream_min_steps=settings.dream_min_steps,
                track_reads=False,
            ),
            api_tools=build_dream_tools(settings.web_search),
            reasoning=agent.reasoning,
        )
        self.agent = agent
        self.gate = gate
        self.mind = agent.mind
        self.tz = agent.tz
        self.prompt = agent.prompts.dream
        self.nudge = agent.prompts.dream_nudge
        self.idle_minutes = settings.dream_idle_minutes
        self.max_rounds = settings.dream_max_rounds

    async def _remember(self, item: dict[str, Any]) -> None:
        """Append an item to the window and to this dream's trace.

        The trace is write-only — a dream never loads it back — so a call
        outside a session has nothing to attach to and is simply not
        recorded.
        """
        await super()._remember(item)
        if self.dream_id is not None:
            await self.db.append_dream_context(self.dream_id, item)

    async def _anchor_id(self) -> int:
        """Newest persisted dreaming context id."""
        return await self.db.latest_dream_context_id()

    async def maybe_dream(self) -> None:
        """Run a dream when idleness or a request and the budget allow it."""
        if self.gate.daily_budget <= 0:
            return
        if self.agent.turn_lock.locked():
            # Mid-turn the agent is busy by definition, and last_active
            # is stale (it only moves once a turn settles). Waiting on
            # the lock here would spend budget on a dream that starts the
            # moment someone is still talking. Try again next poll.
            return
        requested = self.gate.requested
        if not requested and self._idle_minutes() < self.idle_minutes:
            return
        if await self.gate.cooldown_left() > 0:
            return
        if await self.gate.budget_left() <= 0:
            if requested:
                self.gate.take()
                log.info("dream requested but the daily budget is spent")
            return
        note = self.gate.take()
        await self.dream("requested" if note is not None else "idle", note)

    async def dream(self, trigger: str, note: str | None) -> None:
        """Run one full dream and hand a wake-up event to the agent.

        The waking loop's turn lock is held throughout: an in-flight turn
        finishes first, then no new one starts until this returns. A dream
        that fails still closes its ledger row and still wakes the agent —
        silently losing the loop would be worse than a confusing event.
        """
        started = datetime.now(UTC)
        dream_id = await self.db.start_dream(trigger)
        self.dream_id = dream_id
        status = "failed"
        try:
            async with self.agent.turn_lock:
                status = await self._session(dream_id, note)
        except Exception:
            log.exception("dream #%d failed", dream_id)
        finally:
            summary = self.tools.wake_summary or "(the dream ran out before waking)"
            steps = self.tools.steps
            minutes = round((datetime.now(UTC) - started).total_seconds() / 60)
            self._context = []
            await self.db.finish_dream(dream_id, status, steps, summary)
            self.dream_id = None
            log.info("dream #%d %s after %d steps", dream_id, status, steps)
            await self.agent.push(
                f"[dream #{dream_id} ended {clock.format_now(self.tz)} — you slept"
                f" {minutes} min, {steps} steps] {summary}"
            )

    async def _session(self, dream_id: int, note: str | None) -> str:
        """Drive the rounds of one dream; return its ledger status.

        ``prune_completed_reasoning`` deliberately does not apply here:
        with ``store=False`` the encrypted reasoning has to ride the whole
        session, which is one unbroken chain of tool rounds.
        """
        self.tools.steps = 0
        self.tools.wake_summary = None
        self.server_tools_failed = False
        self._context = []
        await self._remember({"role": "user", "content": self._opening(dream_id, note)})
        instructions = f"{self.prompt}\n\n## Soul\n{self.mind.soul()}"
        for _ in range(self.max_rounds):
            acted = await self._round(instructions, turn_id=None)
            if self.tools.wake_summary is not None:
                return "woke"
            if not acted:
                # Settling with no tool call would end the dream after one
                # round; a dream is supposed to wander, so nudge it on.
                await self._remember({"role": "user", "content": self.nudge})
        log.warning("dream #%d hit max_rounds without waking", dream_id)
        return "max_rounds"

    def _opening(self, dream_id: int, note: str | None) -> str:
        """Build the event that opens a dream, mind files included.

        Handing over all four files up front costs one prompt instead of
        four rounds spent reading them back.
        """
        idle = round(self._idle_minutes())
        opening = (
            f"[dream #{dream_id} at {clock.format_now(self.tz)} — you fall"
            f" asleep after {idle} min awake with nothing to do]"
        )
        parts = [opening]
        if note:
            parts.append(f"You went to sleep on this: {note}")
        parts.append(f"# Dreams\n{self.mind.dreams_tail(DREAMS_TAIL_CHARS) or EMPTY}")
        parts.append(f"# Memory\n{self.mind.read('memory') or EMPTY}")
        parts.append(f"# Inbox\n{self.mind.read('inbox') or EMPTY}")
        return "\n\n".join(parts)

    def _idle_minutes(self) -> float:
        """Minutes since the waking agent last finished a turn."""
        return (datetime.now(UTC) - self.agent.last_active).total_seconds() / 60
