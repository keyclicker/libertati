"""The model/tool round engine shared by the waking and dreaming loops.

At the lowest level both loops do the same thing: send the context to the
Responses API with a set of tools, remember everything that comes back,
execute any tool calls and remember their output too. They differ only in
what sits above that — the waking agent runs one turn per batch of
external events, the dreaming loop runs one long offline session — and in
which table each of them records what it said, so only the round itself
lives here.
"""

import logging
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Any, cast

from openai import AsyncOpenAI, BadRequestError, Omit
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
    ) -> None:
        """Keep the API handles and start with an empty context window."""
        self.client = client
        self.model = model
        self.db = db
        self.tools = tools
        self.reasoning = reasoning
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

    @staticmethod
    def _unique_call_ids(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Rename repeated tool-call ids while preserving call/output pairs.

        Some compatible providers reuse short ids across turns while others
        require every id in the full input to be unique. Persistence keeps the
        provider's original values; only the request view is normalized.
        """
        counts: dict[str, int] = defaultdict(int)
        pending: dict[str, deque[str]] = defaultdict(deque)
        used: set[str] = set()
        normalized: list[dict[str, Any]] = []
        for item in items:
            kind = item.get("type")
            call_id = item.get("call_id")
            replacement = call_id
            if kind == "function_call" and isinstance(call_id, str):
                counts[call_id] += 1
                replacement = call_id
                suffix = counts[call_id]
                while replacement in used:
                    replacement = f"{call_id}__libertati_{suffix}"
                    suffix += 1
                used.add(replacement)
                pending[call_id].append(replacement)
            elif (
                kind == "function_call_output"
                and isinstance(call_id, str)
                and pending[call_id]
            ):
                replacement = pending[call_id].popleft()
            if replacement != call_id:
                item = {**item, "call_id": replacement}
            normalized.append(item)
        return normalized

    async def _round(self, instructions: str, turn_id: int | None) -> bool:
        """Call the model once and run whatever tools it asked for.

        Returns whether any tool ran — i.e. whether the loop has a reason
        to go around again. ``turn_id`` links persisted usage to a waking
        turn; a dreaming round passes ``None`` and is linked by
        :attr:`dream_id` instead.
        """
        input_context_id = await self._anchor_id()
        request_context = self._unique_call_ids(self._context)

        async def create(tools: list[ToolParam]) -> Any:
            return await self.client.responses.create(
                model=self.model,
                instructions=instructions,
                input=cast(ResponseInputParam, request_context),
                tools=tools,
                # Nothing is stored server-side; encrypted reasoning must
                # ride along in the context for multi-round tool turns.
                store=False,
                include=["reasoning.encrypted_content"],
                reasoning=self.reasoning,
            )

        try:
            response = await create(self._api_tools)
        except BadRequestError as exc:
            local_tools = [
                tool for tool in self._api_tools if tool["type"] == "function"
            ]
            if "Server tool request failed" not in str(exc) or len(local_tools) == len(
                self._api_tools
            ):
                raise
            log.warning("server tool failed; disabling built-in tools and retrying")
            self._api_tools = local_tools
            response = await create(local_tools)
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
