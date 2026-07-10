"""The agent: assembles context, runs the tool-call loop, produces output."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from typing import Any

from ..config import Settings
from ..logging import get_logger
from ..observability import NULL_EVENTS, EventLogger
from ..storage.history import HistoryStore, StoredMessage
from ..storage.memory import MemoryStore
from ..telegram.base import IncomingMessage
from . import prompts
from .client import LLMClient
from .tools import ToolBox

log = get_logger("llm.agent")


def _clip(text: str, limit: int) -> str:
    """Keep a note's most recent ``limit`` chars, marking a head trim with an ellipsis."""
    text = text.strip()
    return text if len(text) <= limit else "…" + text[-limit:]


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
        events: EventLogger | None = None,
    ) -> None:
        self.settings = settings
        self.llm = llm
        self.tools = tools
        self.history = history
        self.memory = memory
        self.events = events or NULL_EVENTS

    # -- tool loop -----------------------------------------------------------
    async def _run_loop(self, messages: list[dict[str, Any]]) -> str:
        acc = self.events.current_acc()
        for i in range(self.settings.openai_max_tool_iterations):
            message = await self.llm.chat(messages, tools=self.tools.schemas)
            messages.append(_assistant_dict(message))
            tool_calls = getattr(message, "tool_calls", None)
            if not tool_calls:
                if acc is not None:
                    acc.context["iterations"] = i + 1
                return (message.content or "").strip()
            log.debug(
                "iteration %d: model requested tools %s",
                i + 1,
                [tc.function.name for tc in tool_calls],
            )
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
        if acc is not None:
            acc.context["hit_cap"] = True
            acc.context["iterations"] = self.settings.openai_max_tool_iterations
        self.events.emit("tool_loop_cap", limit=self.settings.openai_max_tool_iterations)
        messages.append(
            {"role": "user", "content": "Заверши коротко, без інструментів."}
        )
        message = await self.llm.chat(messages, tools=None)
        return (message.content or "").strip()

    # -- context helpers -----------------------------------------------------
    def _memory_context(self, incoming: IncomingMessage | None = None) -> str:
        """Build the memory snapshot for the prompt.

        General files plus (when replying) the relevant user/group note. Each file is
        clipped to a per-file budget so unbounded logs never blow up the context.
        """
        cap = self.settings.memory_context_file_chars
        files: dict[str, str] = dict(self.memory.snapshot())
        if incoming is not None:
            if incoming.user_handle:
                files[f"user/{incoming.user_handle}"] = self.memory.read_user(
                    incoming.user_handle
                )
            if incoming.is_group and incoming.chat_title:
                files[f"group/{incoming.chat_title}"] = self.memory.read_group(
                    incoming.chat_title
                )
        parts = ["## Памʼять (нотатки)"]
        for name, content in files.items():
            if content.strip():
                parts.append(f"### {name}\n{_clip(content, cap)}")
        return "\n\n".join(parts)

    def _stored_to_messages(self, stored: list[StoredMessage]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in stored:
            if m.role == "assistant":
                out.append({"role": "assistant", "content": m.text})
            else:
                who = m.user_name or m.user_handle or "user"
                out.append({"role": "user", "content": f"{who}: {m.text}"})
        return out

    async def _conversation(self, incoming: IncomingMessage) -> list[StoredMessage]:
        """The reply thread plus a bounded window of recent chat messages.

        The thread is the focused reply chain; on top of it we add the newest messages in
        the chat (up to ``recent_context_chars``) that aren't already in the thread, so the
        bot sees current activity even when replying to something old. The union is returned
        in chronological order.
        """
        thread = await self.history.get_thread(incoming)
        keys = {(m.chat_id, m.message_id) for m in thread}
        recent = await self.history.recent(
            incoming.chat_id, limit=self.settings.recent_context_messages
        )
        budget = self.settings.recent_context_chars
        for m in reversed(recent):  # newest first, fill the budget
            if (m.chat_id, m.message_id) in keys:
                continue
            if len(m.text) > budget:
                break
            budget -= len(m.text)
            keys.add((m.chat_id, m.message_id))
            thread.append(m)
        thread.sort(key=lambda m: (m.ts or 0.0, m.message_id))
        return thread

    # -- public entry points -------------------------------------------------
    async def respond(self, incoming: IncomingMessage) -> str:
        convo = await self._conversation(incoming)
        memory_ctx = self._memory_context(incoming)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": prompts.system_prompt(self.settings)},
            {"role": "system", "content": memory_ctx},
            *self._stored_to_messages(convo),
            {"role": "system", "content": prompts.response_instruction(self.settings)},
        ]
        with self.events.turn(
            "respond",
            chat_id=incoming.chat_id,
            user=incoming.user_handle or incoming.user_name,
            thread_len=len(convo),
            memory_bytes=len(memory_ctx),
        ):
            reply = await self._run_loop(messages)
            self.events.emit("reply", preview=self.events.redact(reply), reply_len=len(reply))
            return reply

    async def heartbeat(self) -> HeartbeatAction | None:
        chats = await self.history.active_chats(limit=10)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": prompts.system_prompt(self.settings)},
            {"role": "system", "content": self._memory_context()},
            {
                "role": "system",
                "content": "Активні чати: " + json.dumps(chats, ensure_ascii=False),
            },
            {"role": "user", "content": prompts.heartbeat_instruction(self.settings)},
        ]
        with self.events.turn("heartbeat", active_chats=len(chats)):
            result = await self._run_loop(messages)
            if not result or result.strip().upper() == "PASS":
                self.events.emit("heartbeat_result", action=False)
                return None
            try:
                data = json.loads(result)
                action = HeartbeatAction(target=str(data["target"]), text=str(data["text"]))
                self.events.emit(
                    "heartbeat_result", action=True, target=action.target,
                    preview=self.events.redact(action.text),
                )
                return action
            except (json.JSONDecodeError, KeyError, TypeError):
                log.info("heartbeat produced non-actionable output: %s", result[:120])
                self.events.emit("heartbeat_result", action=False, malformed=True)
                return None

    async def dream(self) -> str:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": prompts.system_prompt(self.settings)},
            {"role": "system", "content": self._memory_context()},
            {"role": "user", "content": prompts.dream_instruction(self.settings)},
        ]
        with self.events.turn("dream"):
            reflection = await self._run_loop(messages)
            if reflection:
                entry = f"# {date.today().isoformat()}\n\n{reflection}"
                self.memory.append(self.memory.diary_path(), entry)
            self.events.emit("dream_result", reflection_len=len(reflection))
            return reflection

    async def browse(self, source_desc: str, content: str) -> str:
        """React to messages read from another chat/channel.

        The model may save something to reading.md, then returns a chat remark or 'PASS'.
        """
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": prompts.system_prompt(self.settings)},
            {"role": "system", "content": self._memory_context()},
            {
                "role": "user",
                "content": prompts.browse_instruction(self.settings, source_desc, content),
            },
        ]
        with self.events.turn("browse", source=source_desc, content_bytes=len(content)):
            remark = await self._run_loop(messages)
            self.events.emit(
                "browse_result",
                passed=(not remark or remark.strip().upper() == "PASS"),
                preview=self.events.redact(remark),
            )
            return remark
