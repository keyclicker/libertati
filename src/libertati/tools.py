"""Agent tool definitions (schemas) and their execution.

Adding a tool: write its JSON schema into ``TOOLS``, implement an async
handler on :class:`Toolbox`, and register it in ``self._handlers``.

Schemas use strict mode, so argument types are guaranteed by the API and
handlers don't need defensive casts; optional parameters are nullable.

Built-in tools (web search) execute server-side: they produce no
``function_call`` items, so the agent loop needs no handler for them —
they just have to be listed, which :func:`build_tools` does.
"""

import json
import logging
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

from aiogram import Bot
from aiogram.types import ReplyParameters
from openai import AsyncOpenAI
from openai.types.responses import ToolParam

from libertati import clock
from libertati.db import Database
from libertati.memory import Mind

log = logging.getLogger(__name__)

#: Instructions for the one-shot recall extraction call.
RECALL_PROMPT = (
    "You are the long-term memory of a person texting on Telegram. Below "
    "are their dated notes, then a query. Answer the query from the notes "
    "only: quote or paraphrase the relevant entries (with dates when they "
    "matter) and say plainly when the notes contain nothing relevant."
)

#: Instructions for the one-shot memory overview call.
SUMMARY_PROMPT = (
    "You are the long-term memory of a person texting on Telegram. Below "
    "are their dated notes. Give a short general overview of what is "
    "remembered: the people and key facts, recurring themes, open plans "
    "and promises, and the time span covered. A map, not the details — "
    "specifics can be fetched later with targeted recall."
)

TOOLS: list[ToolParam] = [
    {
        "type": "function",
        "name": "send_message",
        "description": (
            "Send a Telegram message to a chat. This is the ONLY way to "
            "actually say something; plain text output sends nothing."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "chat_id": {
                    "type": "integer",
                    "description": "Target chat id (shown in incoming events).",
                },
                "text": {
                    "type": "string",
                    "description": "Message text to send.",
                },
                "reply_to_message_id": {
                    "type": ["integer", "null"],
                    "description": (
                        "Message id to quote-reply to, or null. Quote only "
                        "when it clarifies what you're answering: the message "
                        "is buried under newer ones, or a group conversation "
                        "has several threads going. When you're answering the "
                        "latest message — always in private chats, usually in "
                        "groups with one active thread — pass null: quoting "
                        "every message reads robotic."
                    ),
                },
            },
            "required": ["chat_id", "text", "reply_to_message_id"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "schedule_wakeup",
        "description": (
            "Set an alarm for your future self: at the given time you will "
            "receive a wakeup event carrying your note. Use it whenever you "
            "intend to do something later — don't rely on remembering."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "when": {
                    "type": "string",
                    "description": (
                        "When to wake up, 'YYYY-MM-DD HH:MM' in your own timezone."
                    ),
                },
                "note": {
                    "type": "string",
                    "description": "Note to your future self: what and why.",
                },
            },
            "required": ["when", "note"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "get_recent_messages",
        "description": (
            "Fetch the most recent messages stored for a chat, newest "
            "last. Use to recall context beyond what you remember."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "chat_id": {
                    "type": "integer",
                    "description": "Chat id to read history from.",
                },
                "limit": {
                    "type": ["integer", "null"],
                    "description": "How many messages to fetch (max 50); null = 20.",
                },
            },
            "required": ["chat_id", "limit"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "remember",
        "description": (
            "Save one durable fact to your long-term memory (survives "
            "forever, unlike this context). Use for things worth keeping: "
            "people, preferences, promises, your own plans. One short "
            "fact per call."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The fact to remember, one short sentence.",
                },
            },
            "required": ["text"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "recall",
        "description": (
            "Ask your long-term memory a question. Use before answering "
            "anything that depends on the past beyond what you currently "
            "see: names, preferences, promises, earlier plans."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What you are trying to remember, as a question.",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "summarize_memory",
        "description": (
            "Get a general overview of everything in your long-term "
            "memory — what do you even remember? Use to orient yourself; "
            "follow up with recall for specifics."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


def build_tools(web_search: bool) -> list[ToolParam]:
    """Return the tool list for the API, optionally with built-in web search.

    Web search runs on OpenAI's side; it is opt-in because most
    OpenAI-compatible endpoints don't support it.
    """
    if web_search:
        return [*TOOLS, {"type": "web_search"}]
    return list(TOOLS)


class Toolbox:
    """Executes the agent's tool calls against application resources.

    One instance serves the single agent loop; tools that target a chat
    take an explicit ``chat_id`` argument from the model.
    """

    def __init__(
        self,
        db: Database,
        bot: Bot,
        tz: ZoneInfo,
        client: AsyncOpenAI,
        recall_model: str,
        mind: Mind,
    ) -> None:
        """Keep resource handles and build the name-to-handler dispatch."""
        self.db = db
        self.bot = bot
        self.tz = tz
        self.client = client
        self.recall_model = recall_model
        self.mind = mind
        self._handlers = {
            "send_message": self._send_message,
            "get_recent_messages": self._get_recent_messages,
            "schedule_wakeup": self._schedule_wakeup,
            "remember": self._remember,
            "recall": self._recall,
            "summarize_memory": self._summarize_memory,
        }

    async def run(self, name: str, arguments: str | None) -> str:
        """Execute one tool call and return its result as a string.

        Never raises: unknown tools, malformed arguments and handler
        failures all come back as ``error: …`` strings so the model can
        recover on the next loop step.
        """
        handler = self._handlers.get(name)
        if handler is None:
            return f"error: unknown tool {name!r}"
        try:
            args = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            return "error: invalid tool arguments"
        log.info("tool call: %s(%s)", name, args)
        try:
            return await handler(args)
        except Exception as exc:
            log.exception("tool %s failed", name)
            return f"error: {exc}"

    async def _send_message(self, args: dict[str, Any]) -> str:
        """Send a message (optionally as a reply) and persist it as outgoing."""
        reply_to = args.get("reply_to_message_id")
        sent = await self.bot.send_message(
            args["chat_id"],
            args["text"],
            reply_parameters=(
                ReplyParameters(message_id=reply_to) if reply_to is not None else None
            ),
        )
        await self.db.save_message(sent, outgoing=True)
        return (
            f"sent message {sent.message_id} to chat {sent.chat.id}"
            f" at {clock.format_now(self.tz)}"
        )

    async def _schedule_wakeup(self, args: dict[str, Any]) -> str:
        """Store a future wakeup for the agent itself."""
        try:
            due = clock.parse_local(args["when"], self.tz)
        except ValueError:
            return "error: 'when' must be 'YYYY-MM-DD HH:MM'"
        if due <= datetime.now(UTC):
            return (
                f"error: {args['when']} is in the past,"
                f" now is {clock.format_now(self.tz)}"
            )
        wakeup_id = await self.db.add_wakeup(clock.utc_stamp(due), args["note"])
        return f"wakeup #{wakeup_id} scheduled for {clock.format_local(due, self.tz)}"

    async def _get_recent_messages(self, args: dict[str, Any]) -> str:
        """Return recent messages of the given chat as JSON."""
        limit = max(1, min(args.get("limit") or 20, 50))
        rows = await self.db.recent_messages(args["chat_id"], limit)
        return json.dumps(rows, ensure_ascii=False)

    async def _remember(self, args: dict[str, Any]) -> str:
        """Append one stamped fact to the long-term memory file."""
        text = args["text"].strip()
        self.mind.append_memory(text, clock.format_now(self.tz))
        return f"remembered: {text}"

    async def _read_memory(self, instructions: str, input_text: str) -> str:
        """Run one no-loop extraction call over the memory notes."""
        response = await self.client.responses.create(
            model=self.recall_model,
            instructions=instructions,
            input=input_text,
            store=False,
        )
        return response.output_text or "recall came back empty"

    async def _recall(self, args: dict[str, Any]) -> str:
        """Answer a query from MEMORY.md via a one-shot extraction call."""
        notes = self.mind.memory()
        if not notes:
            return "memory is empty"
        return await self._read_memory(
            RECALL_PROMPT, f"Notes:\n{notes}\n\nQuery: {args['query']}"
        )

    async def _summarize_memory(self, args: dict[str, Any]) -> str:
        """Return a general overview of everything in MEMORY.md."""
        notes = self.mind.memory()
        if not notes:
            return "memory is empty"
        return await self._read_memory(SUMMARY_PROMPT, f"Notes:\n{notes}")
