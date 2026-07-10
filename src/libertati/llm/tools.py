"""Agent tools: schemas (OpenAI function-calling) and dispatch to real code."""

from __future__ import annotations

import json
import time
from typing import Any

from ..logging import get_logger
from ..news.reader import NewsReader, format_digest
from ..observability import NULL_EVENTS, EventLogger
from ..storage.history import HistoryStore
from ..storage.memory import MemoryStore
from ..telegram.base import IncomingMessage, TelegramClient

log = get_logger("llm.tools")

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_history",
            "description": "Full-text search the bot's own message history for past context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Keywords to search for."},
                    "limit": {"type": "integer", "default": 10},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_memory",
            "description": (
                "Read a markdown memory file. Paths like 'self.md', 'world.md', 'todo.md', "
                "'user/<handle>.md', 'group/<slug>.md'."
            ),
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_memory",
            "description": (
                "Write to a markdown memory file. mode 'append' adds a line, 'overwrite' "
                "replaces the whole file. Use for lasting facts about users, groups, the "
                "world or yourself."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "mode": {
                        "type": "string",
                        "enum": ["append", "overwrite"],
                        "default": "append",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember_user",
            "description": "Append a short fact about a user to user/<handle>.md.",
            "parameters": {
                "type": "object",
                "properties": {
                    "handle": {"type": "string"},
                    "note": {"type": "string"},
                },
                "required": ["handle", "note"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_news",
            "description": "Fetch recent news headlines, optionally filtered by a topic keyword.",
            "parameters": {
                "type": "object",
                "properties": {"topic": {"type": "string"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_telegram",
            "description": (
                "Read the most recent messages of a Telegram channel or chat (by @username or "
                "numeric id). Only works when running as a user account; may be limited to "
                "public @channels."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "@username or numeric chat id"},
                    "limit": {"type": "integer", "default": 15},
                },
                "required": ["source"],
            },
        },
    },
]


class ToolBox:
    """Holds tool dependencies and dispatches tool calls by name."""

    def __init__(
        self,
        history: HistoryStore,
        memory: MemoryStore,
        news: NewsReader,
        reader: TelegramClient | None = None,
        public_only: bool = True,
        events: EventLogger | None = None,
    ) -> None:
        self.history = history
        self.memory = memory
        self.news = news
        self.reader = reader
        self.public_only = public_only
        self.events = events or NULL_EVENTS

    @property
    def schemas(self) -> list[dict[str, Any]]:
        return TOOL_SCHEMAS

    async def dispatch(self, name: str, arguments: dict[str, Any]) -> str:
        started = time.perf_counter()
        error: str | None = None
        try:
            handler = getattr(self, f"_tool_{name}", None)
            if handler is None:
                error = f"unknown tool {name}"
                return json.dumps({"error": error})
            result = await handler(arguments)
            return result
        except Exception as exc:  # noqa: BLE001 - surface tool errors to the model
            log.warning("tool %s failed: %s", name, exc)
            error = str(exc)
            return json.dumps({"error": error})
        finally:
            acc = self.events.current_acc()
            if acc is not None:
                acc.tool_calls += 1
            self.events.emit(
                "tool_call",
                name=name,
                args=self.events.redact(json.dumps(arguments, ensure_ascii=False)),
                ok=error is None,
                error=error,
                latency_ms=round((time.perf_counter() - started) * 1000),
                result_len=len(result) if error is None else 0,
            )

    async def _tool_search_history(self, args: dict[str, Any]) -> str:
        results = await self.history.search(args["query"], limit=int(args.get("limit", 10)))
        return json.dumps(
            [
                {
                    "user": m.user_handle or m.user_name,
                    "role": m.role,
                    "text": m.text,
                }
                for m in results
            ],
            ensure_ascii=False,
        )

    async def _tool_read_memory(self, args: dict[str, Any]) -> str:
        content = self.memory.read(args["path"])
        return json.dumps({"path": args["path"], "content": content}, ensure_ascii=False)

    async def _tool_update_memory(self, args: dict[str, Any]) -> str:
        path = args["path"]
        content = args["content"]
        mode = args.get("mode", "append")
        if mode == "overwrite":
            self.memory.overwrite(path, content)
        else:
            self.memory.append(path, content)
        return json.dumps({"ok": True, "path": path, "mode": mode})

    async def _tool_remember_user(self, args: dict[str, Any]) -> str:
        path = self.memory.user_path(args["handle"])
        self.memory.append(path, f"- {args['note']}")
        return json.dumps({"ok": True, "path": path})

    async def _tool_read_news(self, args: dict[str, Any]) -> str:
        items = await self.news.fetch(topic=args.get("topic"))
        return json.dumps({"digest": format_digest(items)}, ensure_ascii=False)

    async def _tool_read_telegram(self, args: dict[str, Any]) -> str:
        if self.reader is None or not self.reader.supports_reading:
            return json.dumps(
                {"error": "reading Telegram is only available in account mode"}
            )
        source = str(args["source"]).strip()
        if self.public_only and not source.startswith("@"):
            return json.dumps(
                {"error": "only public @channels may be read (browse_public_only is on)"}
            )
        messages = await self.reader.read_source(source, limit=int(args.get("limit", 15)))
        return json.dumps(
            {"source": source, "transcript": format_transcript(messages)},
            ensure_ascii=False,
        )


def format_transcript(messages: list[IncomingMessage], limit: int = 40) -> str:
    if not messages:
        return "(no messages)"
    lines = []
    for m in messages[-limit:]:
        who = m.user_name or m.user_handle or "?"
        text = " ".join((m.text or "").split())
        if text:
            lines.append(f"{who}: {text}")
    return "\n".join(lines) if lines else "(no text messages)"
