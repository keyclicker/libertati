"""Thin async OpenAI wrapper with retry/backoff."""

from __future__ import annotations

import asyncio
from typing import Any, cast

from openai import APIError, AsyncOpenAI, RateLimitError

from ..config import Settings
from ..logging import get_logger

log = get_logger("llm.client")


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
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
                resp = await self._client.chat.completions.create(**kwargs)
                return resp.choices[0].message
            except (RateLimitError, APIError) as exc:  # transient
                last_exc = exc
                log.warning("OpenAI call failed (attempt %d): %s", attempt + 1, exc)
                await asyncio.sleep(delay)
                delay *= 2
        assert last_exc is not None
        raise last_exc
