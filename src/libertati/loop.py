"""The model/tool round engine shared by the waking and dreaming loops.

At the lowest level both loops do the same thing: send the context to the
Responses API with a set of tools, remember everything that comes back,
execute any tool calls and remember their output too. They differ only in
what sits above that — the waking agent runs one turn per batch of
external events, the dreaming loop runs one long offline session — and in
which table each of them records what it said, so only the round itself
lives here.
"""

import asyncio
import logging
import random
from typing import TYPE_CHECKING, Any, cast

from openai import (
    APIConnectionError,
    AsyncOpenAI,
    BadRequestError,
    InternalServerError,
    Omit,
    RateLimitError,
)
from openai.types.responses import ResponseInputParam, ToolParam
from openai.types.shared_params import Reasoning

from libertati.db import Database

if TYPE_CHECKING:  # pragma: no cover - import cycle broken for runtime
    from libertati.tools import Toolbox

log = logging.getLogger(__name__)

#: Provider complaints about context this loop can still salvage: both
#: mean the window carries envelopes the endpoint won't take back (short
#: tool-call ids reused across turns, reasoning encrypted for another
#: provider), and both are answered by shedding history.
STALE_CONTEXT_ERRORS = (
    "Duplicate tool call id in assistant message",
    "Could not decrypt the provided encrypted_content",
)

#: Provider complaint about a built-in (server-side) tool, answered by
#: retrying with the locally executed function tools alone.
SERVER_TOOL_ERROR = "Server tool request failed"

#: Failures that say nothing about the request itself — the endpoint was
#: unreachable, overloaded or rate limiting. Waiting is the whole fix; a
#: turn that gives up on one drops the event that triggered it, since
#: nothing re-queues an event whose turn already ran.
TRANSIENT_ERRORS = (APIConnectionError, RateLimitError, InternalServerError)

#: Delay before the first retry of a transient failure; each further one
#: doubles it, with jitter so a burst of turns does not resynchronize on
#: the provider.
RETRY_BACKOFF_SECONDS = 4.0


class ModelLoop:
    """One model call plus its tool calls, and the usage bookkeeping.

    Subclasses drive :meth:`_round` in whatever shape they need and
    override :meth:`_remember` to decide what happens to context items
    beyond the in-memory window.
    """

    #: Set by the dreaming loop for the length of one dream, so usage
    #: rows can say which dream they belong to.
    dream_id: int | None = None

    def __init__(
        self,
        *,
        client: AsyncOpenAI,
        model: str,
        db: Database,
        tools: "Toolbox",
        api_tools: list[ToolParam],
        reasoning: Reasoning | Omit,
        api_retries: int = 0,
    ) -> None:
        """Keep the API handles and start with an empty context window."""
        self.client = client
        self.model = model
        self.db = db
        self.tools = tools
        self.reasoning = reasoning
        self.api_retries = api_retries
        self._api_tools = api_tools
        self._context: list[dict[str, Any]] = []
        self._last_output_text = ""

    async def _remember(self, item: dict[str, Any]) -> None:
        """Append one item to the in-memory context window."""
        self._context.append(item)

    async def _anchor_id(self) -> int:
        """Newest persisted context id the next API call builds on.

        Subclasses that persist somewhere other than the waking history
        point this at their own table.
        """
        return await self.db.latest_context_id()

    def _provider_fallback_context(self) -> list[dict[str, Any]]:
        """Return a provider-neutral view after historical context fails.

        Persistent waking agents override this. Dream context starts empty and
        belongs to one provider session, so there is no safe history to shed.
        """
        return self._context

    async def _round(self, instructions: str, turn_id: int | None) -> bool:
        """Call the model once and run whatever tools it asked for.

        Returns whether any tool ran — i.e. whether the loop has a reason
        to go around again. ``turn_id`` links persisted usage to a waking
        turn; a dreaming round passes ``None`` and is linked by
        :attr:`dream_id` instead.
        """
        input_context_id = await self._anchor_id()
        request_context = self._context

        async def create(tools: list[ToolParam], context: list[dict[str, Any]]) -> Any:
            attempt = 0
            while True:
                try:
                    return await self.client.responses.create(
                        model=self.model,
                        instructions=instructions,
                        input=cast(ResponseInputParam, context),
                        tools=tools,
                        # Nothing is stored server-side; encrypted reasoning
                        # must ride along for multi-round tool turns.
                        store=False,
                        include=["reasoning.encrypted_content"],
                        reasoning=self.reasoning,
                    )
                except TRANSIENT_ERRORS as exc:
                    if attempt >= self.api_retries:
                        raise
                    delay = (
                        RETRY_BACKOFF_SECONDS * 2**attempt * random.uniform(0.8, 1.2)
                    )
                    log.warning(
                        "transient API failure (%s); retrying in %.1fs",
                        type(exc).__name__,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    attempt += 1

        request_tools = self._api_tools
        # Each fallback is worth exactly one attempt: a second rejection
        # of the same kind means the shed did not help, and retrying it
        # would only spend another request to reach the same error.
        compacted = False
        dropped_server_tools = False
        while True:
            try:
                response = await create(request_tools, request_context)
                break
            except BadRequestError as exc:
                error = str(exc)
                if not compacted and any(m in error for m in STALE_CONTEXT_ERRORS):
                    compacted = True
                    fallback_context = self._provider_fallback_context()
                    if fallback_context == request_context:
                        raise
                    log.warning(
                        "provider rejected historical context; "
                        "compacting window and retrying"
                    )
                    # Keep the compatible view for later model/tool rounds.
                    # Full append-only history remains untouched in SQLite.
                    self._context = fallback_context
                    request_context = fallback_context
                    continue
                if not dropped_server_tools and SERVER_TOOL_ERROR in error:
                    dropped_server_tools = True
                    local_tools = [
                        tool for tool in request_tools if tool["type"] == "function"
                    ]
                    if local_tools == request_tools:
                        raise
                    log.warning(
                        "server tool failed; disabling built-in tools and retrying"
                    )
                    # Only for this round: a server tool usually fails
                    # because its backend hiccuped, not because the
                    # endpoint lacks it, and self._api_tools is what the
                    # next round offers the model again.
                    request_tools = local_tools
                    continue
                raise
        self._last_output_text = response.output_text or ""
        await self._record_usage(response, turn_id, input_context_id)
        for item in response.output:
            await self._remember(item.model_dump(mode="json", exclude_none=True))
        calls = [item for item in response.output if item.type == "function_call"]
        if not calls:
            if response.output_text:
                log.info(
                    "model produced private final output (%d chars)",
                    len(response.output_text),
                )
            return False
        for call in calls:
            result = await self.tools.run(call.name, call.arguments)
            await self._remember(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": result,
                }
            )
        return True

    async def _record_usage(
        self,
        response: Any,
        turn_id: int | None,
        input_context_id: int,
    ) -> None:
        """Log authoritative usage; persist it against its turn or dream."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        input_details = getattr(usage, "input_tokens_details", None)
        output_details = getattr(usage, "output_tokens_details", None)
        cached_tokens = getattr(input_details, "cached_tokens", 0) or 0
        cache_write_tokens = getattr(input_details, "cache_write_tokens", 0) or 0
        reasoning_tokens = getattr(output_details, "reasoning_tokens", 0) or 0
        if turn_id is not None or self.dream_id is not None:
            await self.db.append_api_usage(
                response_id=getattr(response, "id", None),
                turn_id=turn_id,
                dream_id=self.dream_id,
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
