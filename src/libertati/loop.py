"""The model/tool round engine shared by the waking and dreaming loops.

At the lowest level both loops do the same thing: send the context to the
Responses API with a set of tools, remember everything that comes back,
execute any tool calls and remember their output too. They differ only in
what sits above that — the waking agent runs one turn per batch of
external events and persists its whole context, the dreaming loop runs
one long offline session and persists nothing — so only the round itself
lives here.
"""

import logging
from typing import TYPE_CHECKING, Any, cast

from openai import AsyncOpenAI, Omit
from openai.types.responses import ResponseInputParam, ToolParam
from openai.types.shared_params import Reasoning

from libertati.db import Database

if TYPE_CHECKING:  # pragma: no cover - import cycle broken for runtime
    from libertati.tools import Toolbox

log = logging.getLogger(__name__)


class ModelLoop:
    """One model call plus its tool calls, and the usage bookkeeping.

    Subclasses drive :meth:`_round` in whatever shape they need and
    override :meth:`_remember` to decide what happens to context items
    beyond the in-memory window.
    """

    def __init__(
        self,
        *,
        client: AsyncOpenAI,
        model: str,
        db: Database,
        tools: "Toolbox",
        api_tools: list[ToolParam],
        reasoning: Reasoning | Omit,
    ) -> None:
        """Keep the API handles and start with an empty context window."""
        self.client = client
        self.model = model
        self.db = db
        self.tools = tools
        self.reasoning = reasoning
        self._api_tools = api_tools
        self._context: list[dict[str, Any]] = []

    async def _remember(self, item: dict[str, Any]) -> None:
        """Append one item to the in-memory context window."""
        self._context.append(item)

    async def _round(self, instructions: str, turn_id: int | None) -> bool:
        """Call the model once and run whatever tools it asked for.

        Returns whether any tool ran — i.e. whether the loop has a reason
        to go around again. ``turn_id`` links persisted usage to a turn;
        ``None`` means this round belongs to a loop that persists nothing
        (usage is still logged).
        """
        input_context_id = 0
        if turn_id is not None:
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
            reasoning=self.reasoning,
        )
        await self._record_usage(response, turn_id, input_context_id)
        for item in response.output:
            await self._remember(item.model_dump(mode="json", exclude_none=True))
        calls = [item for item in response.output if item.type == "function_call"]
        if not calls:
            if response.output_text:
                log.info("model final output: %s", response.output_text)
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
        """Log authoritative usage; persist it for persisted turns only."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        input_details = getattr(usage, "input_tokens_details", None)
        output_details = getattr(usage, "output_tokens_details", None)
        cached_tokens = getattr(input_details, "cached_tokens", 0) or 0
        cache_write_tokens = getattr(input_details, "cache_write_tokens", 0) or 0
        reasoning_tokens = getattr(output_details, "reasoning_tokens", 0) or 0
        if turn_id is not None:
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
