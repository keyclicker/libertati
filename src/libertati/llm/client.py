"""Thin async OpenAI wrapper with retry/backoff."""

from __future__ import annotations

import asyncio
import time
from typing import Any, cast

from openai import APIError, AsyncOpenAI, RateLimitError

from ..config import Settings
from ..logging import get_logger
from ..observability import NULL_EVENTS, EventLogger

log = get_logger("llm.client")


class LLMClient:
    def __init__(self, settings: Settings, events: EventLogger | None = None) -> None:
        self._settings = settings
        self._events = events or NULL_EVENTS
        self._client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_retries: int = 4,
    ) -> Any:
        """Return the assistant message object from a chat completion."""
        delay = 2.0
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                kwargs: dict[str, Any] = {
                    "model": self._settings.openai_model,
                    "temperature": self._settings.openai_temperature,
                    "messages": cast(Any, messages),
                }
                if tools:
                    kwargs["tools"] = cast(Any, tools)
                started = time.perf_counter()
                resp = await self._client.chat.completions.create(**kwargs)
                latency_ms = round((time.perf_counter() - started) * 1000)
                message = resp.choices[0].message
                self._record(resp, message, len(messages), bool(tools), latency_ms)
                return message
            except (RateLimitError, APIError) as exc:  # transient
                last_exc = exc
                log.warning("OpenAI call failed (attempt %d): %s", attempt + 1, exc)
                self._events.emit(
                    "llm_error", attempt=attempt + 1, error=str(exc)[:200]
                )
                await asyncio.sleep(delay)
                delay *= 2
        assert last_exc is not None
        raise last_exc

    def _record(
        self, resp: Any, message: Any, n_messages: int, had_tools: bool, latency_ms: int
    ) -> None:
        usage = getattr(resp, "usage", None)
        total = int(getattr(usage, "total_tokens", 0) or 0)
        n_tool_calls = len(getattr(message, "tool_calls", None) or [])
        acc = self._events.current_acc()
        if acc is not None:
            acc.llm_calls += 1
            acc.total_tokens += total
        self._events.emit(
            "llm_call",
            model=self._settings.openai_model,
            n_messages=n_messages,
            had_tools=had_tools,
            n_tool_calls=n_tool_calls,
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            total_tokens=total,
            latency_ms=latency_ms,
        )
