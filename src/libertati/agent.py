"""Single-brain agent loop shared by all chats and triggers.

The agent emulates one person texting from one account: every external
event (an incoming message from any chat, later timers/news/…) is appended
to a single running context, and the model decides — deliberately, via the
``send_message`` tool — whether and where to reply. Plain text output is
treated as private thinking and sends nothing.

The full context history (events, reasoning, tool calls/results, replies)
is persisted append-only in the database; only a capped tail window is
kept in memory and sent to the API.
"""

import asyncio
import logging
from typing import Any
from zoneinfo import ZoneInfo

from aiogram import Bot
from openai import AsyncOpenAI

from libertati.config import Settings
from libertati.db import Database
from libertati.tools import TOOLS, Toolbox

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are libertati, a person texting from their own Telegram account. "
    "You receive a stream of events: incoming messages from different "
    "chats (with chat ids), and other triggers. You live in one continuous "
    "timeline across all conversations.\n"
    "To say something, call send_message with the target chat_id — that is "
    "the only way to reply. Your plain text output is your private "
    "thinking and is never shown to anyone.\n"
    "You don't have to react to every event: ignore group chatter that "
    "isn't addressed to you; always answer direct/private messages. Match "
    "the sender's language, text like a human — short, casual, no "
    "assistant-speak. Use get_recent_messages when you need older context.\n"
    "Be proactive deliberately: when you intend to do something later, call "
    "schedule_wakeup — your future self receives the note as a wakeup "
    "event. Heartbeat events are routine status checks: usually no action; "
    "start conversations rarely and only with a real reason."
)

#: Max model/tool rounds per processed batch of events.
MAX_STEPS = 8

#: Cap on context items sent to the API (and kept in memory).
MAX_CONTEXT_ITEMS = 200


class Agent:
    """One persistent agentic loop consuming events from all sources.

    Events are pushed onto an internal queue with :meth:`push` and
    processed strictly one batch at a time by :meth:`worker`, so the agent
    never talks over itself. Every context item is written through to the
    database (full history); the in-memory window is what the model sees.
    Call :meth:`load` on startup to restore the window after a restart.
    """

    def __init__(self, settings: Settings, db: Database, bot: Bot) -> None:
        """Create the API client, toolbox and the (empty) context window."""
        self.client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.base_url,
        )
        self.model = settings.model
        self.db = db
        self.tz = ZoneInfo(settings.timezone)
        self.tools = Toolbox(db, bot, self.tz)
        self._context: list[Any] = []
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._extra: dict[str, Any] = {}
        if settings.reasoning_effort is not None:
            self._extra["reasoning"] = {"effort": settings.reasoning_effort}

    async def load(self) -> None:
        """Restore the context window from the persisted history tail."""
        self._context = self._at_boundary(await self.db.load_context(MAX_CONTEXT_ITEMS))
        log.info("restored %d context items", len(self._context))

    async def push(self, event: str) -> None:
        """Queue an external event (formatted as text) for the agent."""
        await self._queue.put(event)

    async def worker(self) -> None:
        """Consume events forever; cancel the task to stop.

        All events queued by the time one is picked up are appended as a
        single batch, then the agent takes one thinking/acting turn.
        Failures are logged and the loop moves on.
        """
        while True:
            events = [await self._queue.get()]
            while not self._queue.empty():
                events.append(self._queue.get_nowait())
            for event in events:
                await self._remember({"role": "user", "content": event})
            try:
                await self._step()
            except Exception:
                log.exception("agent step failed")

    async def _remember(self, item: dict[str, Any]) -> None:
        """Append an item to the window and the persistent history."""
        self._context.append(item)
        await self.db.append_context(item)
        if len(self._context) > MAX_CONTEXT_ITEMS:
            self._context = self._at_boundary(self._context[-MAX_CONTEXT_ITEMS:])

    @staticmethod
    def _at_boundary(items: list[Any]) -> list[Any]:
        """Drop leading items up to the first external event.

        Windows must start at a ``role: user`` event so no function call is
        separated from its output and no reply is left half-orphaned.
        """
        start = 0
        for i, item in enumerate(items):
            if item.get("role") == "user" and "type" not in item:
                start = i
                break
        else:
            return []
        return items[start:]

    async def _step(self) -> None:
        """Run one agentic turn: call the model, execute tools, repeat.

        The turn ends when the model produces no tool calls (its text, if
        any, is logged as internal monologue) or ``MAX_STEPS`` is reached.
        Everything the model produces is remembered.
        """
        for _ in range(MAX_STEPS):
            response = await self.client.responses.create(
                model=self.model,
                instructions=SYSTEM_PROMPT,
                input=self._context,
                tools=TOOLS,
                **self._extra,
            )
            for item in response.output:
                await self._remember(item.model_dump(mode="json", exclude_none=True))
            calls = [item for item in response.output if item.type == "function_call"]
            if not calls:
                if response.output_text:
                    log.info("agent monologue: %s", response.output_text)
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
        log.warning("agent hit MAX_STEPS without settling")
