"""Single-brain agent loop shared by all chats and triggers.

The agent emulates one person texting from one account: every external
event (an incoming message from any chat, later timers/news/…) is appended
to a single running context, and the model decides — deliberately, via the
``send_message`` tool — whether and where to reply. Plain text output is
treated as private thinking and sends nothing.

Full context history (events, reasoning, tool calls/results, final output)
is persisted append-only in the database. Only a capped, optionally
pruned tail window is kept in memory and sent to the API.
"""

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any, cast
from zoneinfo import ZoneInfo

from aiogram import Bot
from openai import AsyncOpenAI, omit
from openai.types.shared_params import Reasoning

from libertati.chats import ChatRegistry
from libertati.config import Settings
from libertati.db import Database
from libertati.dream import DreamGate
from libertati.loop import ModelLoop
from libertati.memory import Mind
from libertati.prompts import load_prompts
from libertati.tools import Toolbox, build_tools

log = logging.getLogger(__name__)


class Agent(ModelLoop):
    """One persistent agentic loop consuming events from all sources.

    Events are pushed onto an internal queue with :meth:`push` and
    processed strictly one batch at a time by :meth:`run_forever`, so the
    agent never talks over itself. Every context item is written through
    to the database (full history); the in-memory window is what the model
    sees. Call :meth:`load` on startup to restore the window after a
    restart.
    """

    def __init__(
        self,
        settings: Settings,
        db: Database,
        bot: Bot,
        registry: ChatRegistry,
        dream_gate: DreamGate | None = None,
    ) -> None:
        """Create the API client, toolbox and the (empty) context window."""
        client = AsyncOpenAI(api_key=settings.api_key, base_url=settings.base_url)
        self.registry = registry
        self.prompts = load_prompts(settings.prompts_path)
        self.base_prompt = self.prompts.system
        if settings.roleplay:
            self.base_prompt += "\n" + self.prompts.roleplay
        if settings.web_search:
            self.base_prompt += "\n" + self.prompts.web_search
        self.tz = ZoneInfo(settings.timezone)
        self.mind = Mind(settings.memory_dir)
        self.mind.ensure()
        dreaming = dream_gate is not None and settings.dream_daily_budget > 0
        if dreaming:
            self.base_prompt += "\n" + self.prompts.dream_tool
        reasoning: dict[str, Any] = {}
        if settings.reasoning_effort is not None:
            reasoning["effort"] = settings.reasoning_effort
        if settings.reasoning_context != "omit":
            reasoning["context"] = settings.reasoning_context
        super().__init__(
            client=client,
            model=settings.model,
            db=db,
            tools=Toolbox(
                db,
                bot,
                self.tz,
                client,
                settings.recall_model or settings.model,
                self.mind,
                settings.typing_chars_per_second,
                self.prompts.recall,
                self.prompts.summary,
                registry=registry,
                recall_effort=settings.recall_reasoning_effort,
                dream_gate=dream_gate if dreaming else None,
            ),
            api_tools=build_tools(settings.web_search, dreaming=dreaming),
            reasoning=cast(Reasoning, reasoning) if reasoning else omit,
        )
        # Max model/tool rounds per turn (one turn per batch of events).
        self.max_rounds = settings.max_rounds
        # Overflow threshold and post-trim size of the context window.
        # Trimming in chunks (not one-by-one) keeps the context prefix
        # byte-stable between trims instead of rewriting it on every
        # append. Cache keys/breakpoints still determine actual hits.
        self.max_context_items = settings.context_max_items
        self.trim_context_items = settings.context_trim_items
        self._queue: asyncio.Queue[tuple[str, bool]] = asyncio.Queue()
        # Held for the whole of a turn. The dreaming loop takes the same
        # lock, which is how "the agent sleeps while it dreams" works:
        # a dream waits for the turn in flight and blocks the next one.
        self.turn_lock = asyncio.Lock()
        # When the agent last did something real — the dream idle
        # trigger. Heartbeat-only turns with no outward action leave it
        # alone, or regular heartbeats would keep idleness at zero.
        self.last_active = datetime.now(UTC)
        self.prune_completed_reasoning = settings.prune_completed_reasoning

    async def load(self) -> None:
        """Restore the context window from the persisted history tail.

        Besides trimming to a window boundary, drops any trailing items
        left dangling by a crash (a function call without its output, a
        reasoning item without its follow-up) — the API rejects such
        context outright, which would wedge the agent permanently.
        Everything up to the last legacy reasoning item (from before
        ``store=False``) is dropped for the same reason: the item itself
        has no encrypted content to send, and a function call whose
        paired reasoning is missing is rejected just the same.
        """
        excluded_types = (
            ("reasoning", "message") if self.prune_completed_reasoning else ()
        )
        items = await self.db.load_context(
            self.trim_context_items,
            exclude_types=excluded_types,
        )
        items = self._drop_legacy_reasoning(items)
        self._context = self._trim_dangling(self._trim_to_boundary(items))
        log.info("restored %d context items", len(self._context))

    async def push(self, event: str, *, activity: bool = True) -> None:
        """Queue an external event (formatted as text) for the agent.

        ``activity=False`` marks events (heartbeats) that should not by
        themselves reset the dream idle clock; the clock still moves
        when the turn they trigger reaches out to anyone.
        """
        await self._queue.put((event, activity))

    async def run_forever(self) -> None:
        """Consume events forever; cancel the task to stop.

        All events queued by the time one is picked up are appended as a
        single batch, then the agent takes one thinking/acting turn.
        While the dreaming loop holds the turn lock, events simply pile
        up in the queue and land as one batch on waking.
        """
        while True:
            batch = [await self._queue.get()]
            while not self._queue.empty():
                batch.append(self._queue.get_nowait())
            await self._process(batch)

    async def _process(self, batch: list[tuple[str, bool]]) -> None:
        """Run one turn over a batch of events and update the idle clock.

        Failures are logged, dangling context is repaired and the caller
        moves on. ``last_active`` moves only when the batch held real
        activity or the turn acted outward — a heartbeat turn spent just
        reading leaves it alone, so idleness can actually accumulate.
        """
        outward_before = self.tools.outward_calls
        async with self.turn_lock:
            for event, _ in batch:
                await self._remember({"role": "user", "content": event})
            try:
                await self._turn()
            except Exception:
                log.exception("agent turn failed")
                self._context = self._trim_dangling(self._context)
        acted = self.tools.outward_calls > outward_before
        if acted or any(activity for _, activity in batch):
            self.last_active = datetime.now(UTC)

    async def _remember(self, item: dict[str, Any]) -> None:
        """Append an item to the window and the persistent history."""
        await super()._remember(item)
        await self.db.append_context(item)
        if len(self._context) > self.max_context_items:
            self._context = self._trim_to_boundary(
                self._context[-self.trim_context_items :]
            )

    @staticmethod
    def _trim_to_boundary(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Drop leading items up to the first external event.

        Windows must start at a ``role: user`` event so no function call is
        separated from its output and no reply is left half-orphaned.
        """
        start = next(
            (
                i
                for i, item in enumerate(items)
                if item.get("role") == "user" and "type" not in item
            ),
            None,
        )
        if start is None:
            if items:
                log.warning("no event boundary in %d items; window emptied", len(items))
            return []
        return items[start:]

    @staticmethod
    def _drop_legacy_reasoning(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Drop everything up to the last reasoning item lacking content.

        Reasoning items persisted before ``store=False`` carry no
        encrypted content, and the API rejects both such an item and any
        function call whose paired reasoning item is missing — so the
        window is cut just after the last one instead of filtering it
        out in place.
        """
        last = next(
            (
                i
                for i in range(len(items) - 1, -1, -1)
                if items[i].get("type") == "reasoning"
                and not items[i].get("encrypted_content")
            ),
            None,
        )
        if last is None:
            return items
        log.warning("dropping %d items up to a legacy reasoning item", last + 1)
        return items[last + 1 :]

    @staticmethod
    def _trim_dangling(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Drop trailing items the API would reject as incomplete.

        A well-formed tail ends with an event, a message or a function
        call output — never with an unanswered function call or a bare
        reasoning item (both happen when a turn dies halfway through).
        """
        items = list(items)
        answered = {
            item["call_id"]
            for item in items
            if item.get("type") == "function_call_output"
        }
        while items and (
            items[-1].get("type") == "reasoning"
            or (
                items[-1].get("type") == "function_call"
                and items[-1].get("call_id") not in answered
            )
        ):
            dropped = items.pop()
            log.warning("dropping dangling context item: %s", dropped.get("type"))
        return items

    async def _turn(self) -> None:
        """Run one agentic turn: call the model, execute tools, repeat.

        The turn ends when the model produces no tool calls (its text, if
        any, is logged as private final output) or ``max_rounds`` is
        reached. Everything the model produces is remembered. SOUL.md is
        re-read every turn so personality edits apply live.
        """
        instructions = f"{self.base_prompt}\n\n## Soul\n{self.mind.soul()}"
        turn_start = len(self._context)
        turn_id = await self.db.start_agent_turn(await self.db.latest_context_id())
        turn_status = "failed"
        try:
            for _ in range(self.max_rounds):
                if not await self._round(instructions, turn_id):
                    turn_status = "completed"
                    return
            turn_status = "max_rounds"
            log.warning("agent hit max_rounds without settling")
        finally:
            try:
                await self.db.finish_agent_turn(
                    turn_id,
                    await self.db.latest_context_id(),
                    turn_status,
                )
            finally:
                self._finish_turn(turn_start)

    def _finish_turn(self, start: int) -> None:
        """Prune ephemeral outputs from one settled live-context turn."""
        if not self.prune_completed_reasoning:
            return
        kept = [
            item
            for item in self._context[start:]
            if item.get("type") not in {"reasoning", "message"}
        ]
        removed = len(self._context) - start - len(kept)
        self._context[start:] = kept
        if removed:
            log.debug("pruned %d completed-turn reasoning/output items", removed)
