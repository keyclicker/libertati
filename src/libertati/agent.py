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
from typing import Any, cast
from zoneinfo import ZoneInfo

from aiogram import Bot
from openai import AsyncOpenAI, Omit, omit
from openai.types.responses import ResponseInputParam
from openai.types.shared_params import Reasoning

from libertati.config import Settings
from libertati.db import Database
from libertati.memory import Mind
from libertati.prompts import load_prompts
from libertati.tools import Toolbox, build_tools

log = logging.getLogger(__name__)


class Agent:
    """One persistent agentic loop consuming events from all sources.

    Events are pushed onto an internal queue with :meth:`push` and
    processed strictly one batch at a time by :meth:`run_forever`, so the
    agent never talks over itself. Every context item is written through
    to the database (full history); the in-memory window is what the model
    sees. Call :meth:`load` on startup to restore the window after a
    restart.
    """

    def __init__(self, settings: Settings, db: Database, bot: Bot) -> None:
        """Create the API client, toolbox and the (empty) context window."""
        self.client = AsyncOpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
        )
        self.model = settings.model
        prompts = load_prompts(settings.prompts_path)
        self.base_prompt = prompts.system
        if settings.roleplay:
            self.base_prompt += "\n" + prompts.roleplay
        if settings.web_search:
            self.base_prompt += "\n" + prompts.web_search
        self.db = db
        self.tz = ZoneInfo(settings.timezone)
        self.mind = Mind(settings.memory_dir)
        self.mind.ensure()
        self.tools = Toolbox(
            db,
            bot,
            self.tz,
            self.client,
            settings.recall_model or settings.model,
            self.mind,
            settings.typing_chars_per_second,
            prompts.recall,
            prompts.summary,
        )
        # Max model/tool rounds per turn (one turn per batch of events).
        self.max_rounds = settings.max_rounds
        # Overflow threshold and post-trim size of the context window.
        # Trimming in chunks (not one-by-one) keeps the context prefix
        # byte-stable between trims instead of rewriting it on every
        # append. Cache keys/breakpoints still determine actual hits.
        self.max_context_items = settings.context_max_items
        self.trim_context_items = settings.context_trim_items
        self._context: list[dict[str, Any]] = []
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._api_tools = build_tools(settings.web_search)
        reasoning: dict[str, Any] = {}
        if settings.reasoning_effort is not None:
            reasoning["effort"] = settings.reasoning_effort
        if settings.reasoning_context != "omit":
            reasoning["context"] = settings.reasoning_context
        self._reasoning: Reasoning | Omit = (
            cast(Reasoning, reasoning) if reasoning else omit
        )
        self.prune_completed_reasoning = settings.prune_completed_reasoning

    async def load(self) -> None:
        """Restore the context window from the persisted history tail.

        Besides trimming to a window boundary, drops any trailing items
        left dangling by a crash (a function call without its output, a
        reasoning item without its follow-up) — the API rejects such
        context outright, which would wedge the agent permanently. Legacy
        reasoning items without encrypted content (from before
        ``store=False``) are dropped for the same reason.
        """
        excluded_types = (
            ("reasoning", "message") if self.prune_completed_reasoning else ()
        )
        items = await self.db.load_context(
            self.trim_context_items,
            exclude_types=excluded_types,
        )
        items = [
            item
            for item in items
            if not (
                item.get("type") == "reasoning" and not item.get("encrypted_content")
            )
        ]
        self._context = self._trim_dangling(self._trim_to_boundary(items))
        log.info("restored %d context items", len(self._context))

    async def push(self, event: str) -> None:
        """Queue an external event (formatted as text) for the agent."""
        await self._queue.put(event)

    async def run_forever(self) -> None:
        """Consume events forever; cancel the task to stop.

        All events queued by the time one is picked up are appended as a
        single batch, then the agent takes one thinking/acting turn.
        Failures are logged, dangling context is repaired and the loop
        moves on.
        """
        while True:
            events = [await self._queue.get()]
            while not self._queue.empty():
                events.append(self._queue.get_nowait())
            for event in events:
                await self._remember({"role": "user", "content": event})
            try:
                await self._turn()
            except Exception:
                log.exception("agent turn failed")
                self._context = self._trim_dangling(self._context)

    async def _remember(self, item: dict[str, Any]) -> None:
        """Append an item to the window and the persistent history."""
        self._context.append(item)
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
                input_context_id = await self.db.latest_context_id()
                response = await self.client.responses.create(
                    model=self.model,
                    instructions=instructions,
                    input=cast(ResponseInputParam, self._context),
                    tools=self._api_tools,
                    # Nothing is stored server-side; encrypted reasoning must
                    # ride along in the context for multi-round tool turns.
                    store=False,
                    include=["reasoning.encrypted_content"],
                    reasoning=self._reasoning,
                )
                await self._record_usage(response, turn_id, input_context_id)
                for item in response.output:
                    await self._remember(
                        item.model_dump(mode="json", exclude_none=True)
                    )
                calls = [
                    item for item in response.output if item.type == "function_call"
                ]
                if not calls:
                    if response.output_text:
                        log.info("agent final output: %s", response.output_text)
                    turn_status = "completed"
                    return
                for call in calls:
                    result = await self.tools.run(call.name, call.arguments)
                    await self._remember(
                        {
                            "type": "function_call_output",
                            "call_id": call.call_id,
                            "output": result,
                        }
                    )
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

    async def _record_usage(
        self,
        response: Any,
        turn_id: int,
        input_context_id: int,
    ) -> None:
        """Persist and log authoritative usage returned by OpenAI."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        input_details = getattr(usage, "input_tokens_details", None)
        output_details = getattr(usage, "output_tokens_details", None)
        cached_tokens = getattr(input_details, "cached_tokens", 0) or 0
        cache_write_tokens = getattr(input_details, "cache_write_tokens", 0) or 0
        reasoning_tokens = getattr(output_details, "reasoning_tokens", 0) or 0
        await self.db.append_api_usage(
            response_id=getattr(response, "id", None),
            turn_id=turn_id,
            input_context_id=input_context_id,
            model=getattr(response, "model", None) or self.model,
            input_tokens=usage.input_tokens,
            cached_tokens=cached_tokens,
            cache_write_tokens=cache_write_tokens,
            output_tokens=usage.output_tokens,
            reasoning_tokens=reasoning_tokens,
            total_tokens=usage.total_tokens,
        )
        log.info(
            "api usage: input=%d cached=%d cache_write=%d "
            "output=%d reasoning=%d total=%d",
            usage.input_tokens,
            cached_tokens,
            cache_write_tokens,
            usage.output_tokens,
            reasoning_tokens,
            usage.total_tokens,
        )
