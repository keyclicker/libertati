"""The agent: assembles context, runs the tool-call loop, produces output."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from typing import Any

from ..config import Settings
from ..logging import get_logger
from ..storage.history import HistoryStore, StoredMessage
from ..storage.memory import MemoryStore
from ..telegram.base import IncomingMessage
from . import prompts
from .client import LLMClient
from .tools import ToolBox

log = get_logger("llm.agent")


@dataclass(slots=True)
class HeartbeatAction:
    target: str
    text: str


def _assistant_dict(message: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in tool_calls
        ]
    return out


class Agent:
    def __init__(
        self,
        settings: Settings,
        llm: LLMClient,
        tools: ToolBox,
        history: HistoryStore,
        memory: MemoryStore,
    ) -> None:
        self.settings = settings
        self.llm = llm
        self.tools = tools
        self.history = history
        self.memory = memory

    # -- tool loop -----------------------------------------------------------
    async def _run_loop(self, messages: list[dict[str, Any]]) -> str:
        for _ in range(self.settings.openai_max_tool_iterations):
            message = await self.llm.chat(messages, tools=self.tools.schemas)
            messages.append(_assistant_dict(message))
            tool_calls = getattr(message, "tool_calls", None)
            if not tool_calls:
                return (message.content or "").strip()
            for tc in tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = await self.tools.dispatch(tc.function.name, args)
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": result}
                )
        # Exhausted iterations: ask for a final plain answer.
        messages.append(
            {"role": "user", "content": "Заверши коротко, без інструментів."}
        )
        message = await self.llm.chat(messages, tools=None)
        return (message.content or "").strip()

    # -- context helpers -----------------------------------------------------
    def _memory_context(self, incoming: IncomingMessage) -> str:
        parts = ["## Памʼять (нотатки)"]
        snap = self.memory.snapshot()
        for name, content in snap.items():
            if content.strip():
                parts.append(f"### {name}\n{content.strip()}")
        if incoming.user_handle:
            user_mem = self.memory.read_user(incoming.user_handle)
            if user_mem.strip():
                parts.append(f"### user/{incoming.user_handle}\n{user_mem.strip()}")
        if incoming.is_group and incoming.chat_title:
            group_mem = self.memory.read_group(incoming.chat_title)
            if group_mem.strip():
                parts.append(f"### group/{incoming.chat_title}\n{group_mem.strip()}")
        return "\n\n".join(parts)

    def _thread_to_messages(self, thread: list[StoredMessage]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in thread:
            if m.role == "assistant":
                out.append({"role": "assistant", "content": m.text})
            else:
                who = m.user_name or m.user_handle or "user"
                out.append({"role": "user", "content": f"{who}: {m.text}"})
        return out

    # -- public entry points -------------------------------------------------
    async def respond(self, incoming: IncomingMessage) -> str:
        thread = await self.history.get_thread(incoming)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": prompts.system_prompt(self.settings)},
            {"role": "system", "content": self._memory_context(incoming)},
            *self._thread_to_messages(thread),
            {"role": "system", "content": prompts.response_instruction(self.settings)},
        ]
        reply = await self._run_loop(messages)
        return reply

    async def heartbeat(self) -> HeartbeatAction | None:
        chats = await self.history.active_chats(limit=10)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": prompts.system_prompt(self.settings)},
            {"role": "system", "content": self._memory_context_general()},
            {
                "role": "system",
                "content": "Активні чати: " + json.dumps(chats, ensure_ascii=False),
            },
            {"role": "user", "content": prompts.heartbeat_instruction(self.settings)},
        ]
        result = await self._run_loop(messages)
        if not result or result.strip().upper() == "PASS":
            return None
        try:
            data = json.loads(result)
            return HeartbeatAction(target=str(data["target"]), text=str(data["text"]))
        except (json.JSONDecodeError, KeyError, TypeError):
            log.info("heartbeat produced non-actionable output: %s", result[:120])
            return None

    async def dream(self) -> str:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": prompts.system_prompt(self.settings)},
            {"role": "system", "content": self._memory_context_general()},
            {"role": "user", "content": prompts.dream_instruction(self.settings)},
        ]
        reflection = await self._run_loop(messages)
        if reflection:
            entry = f"# {date.today().isoformat()}\n\n{reflection}"
            self.memory.append(self.memory.diary_path(), entry)
        return reflection

    def _memory_context_general(self) -> str:
        parts = ["## Памʼять (нотатки)"]
        for name, content in self.memory.snapshot().items():
            if content.strip():
                parts.append(f"### {name}\n{content.strip()}")
        return "\n\n".join(parts)
