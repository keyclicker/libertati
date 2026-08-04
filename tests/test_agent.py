"""Tests for the agent's context-window trimming logic."""

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast

import httpx
from openai import BadRequestError

from libertati.agent import Agent
from libertati.db import Database

#: Window sizes used by the trim test (mirrors the settings defaults).
MAX_CONTEXT_ITEMS = 300
TRIM_CONTEXT_ITEMS = 200

EVENT: dict[str, Any] = {"role": "user", "content": "[event] hi"}
MESSAGE: dict[str, Any] = {
    "type": "message",
    "role": "assistant",
    "content": [{"type": "output_text", "text": "ok"}],
}
REASONING: dict[str, Any] = {"type": "reasoning", "summary": []}
CALL: dict[str, Any] = {
    "type": "function_call",
    "call_id": "call_1",
    "name": "send_message",
    "arguments": "{}",
}
CALL_OUTPUT: dict[str, Any] = {
    "type": "function_call_output",
    "call_id": "call_1",
    "output": "sent",
}


def test_trim_to_boundary_keeps_from_first_event() -> None:
    """Leading non-event items are dropped up to the first event."""
    items = [CALL_OUTPUT, MESSAGE, EVENT, MESSAGE]
    assert Agent._trim_to_boundary(items) == [EVENT, MESSAGE]


def test_trim_to_boundary_ignores_assistant_messages() -> None:
    """Assistant messages have a role too but are not event boundaries."""
    items = [MESSAGE, EVENT]
    assert Agent._trim_to_boundary(items) == [EVENT]


def test_trim_to_boundary_empty_when_no_event() -> None:
    """Without any event the window is emptied entirely."""
    assert Agent._trim_to_boundary([MESSAGE, CALL, CALL_OUTPUT]) == []


def test_trim_to_boundary_empty_input() -> None:
    """An empty window stays empty."""
    assert Agent._trim_to_boundary([]) == []


def test_trim_dangling_drops_unanswered_call() -> None:
    """A trailing function call without its output is removed."""
    items = [EVENT, REASONING, CALL]
    assert Agent._trim_dangling(items) == [EVENT]


def test_trim_dangling_keeps_answered_call() -> None:
    """A call followed by its output is complete and stays."""
    items = [EVENT, CALL, CALL_OUTPUT]
    assert Agent._trim_dangling(items) == items


def test_trim_dangling_drops_trailing_reasoning() -> None:
    """A bare trailing reasoning item is removed."""
    items = [EVENT, REASONING]
    assert Agent._trim_dangling(items) == [EVENT]


def test_trim_dangling_keeps_completed_turn() -> None:
    """Reasoning followed by a message is a complete turn."""
    items = [EVENT, REASONING, MESSAGE]
    assert Agent._trim_dangling(items) == items


def test_trim_dangling_does_not_mutate_input() -> None:
    """The original list is left untouched."""
    items = [EVENT, CALL]
    Agent._trim_dangling(items)
    assert items == [EVENT, CALL]


def test_drop_legacy_reasoning_cuts_paired_calls_too() -> None:
    """The window is cut after the last legacy item, not filtered in place.

    A function call whose paired reasoning item is missing is rejected by
    the API just like the content-less reasoning item itself.
    """
    items = [EVENT, REASONING, CALL, CALL_OUTPUT, EVENT, MESSAGE]
    assert Agent._drop_legacy_reasoning(items) == [CALL, CALL_OUTPUT, EVENT, MESSAGE]


def test_drop_legacy_reasoning_keeps_encrypted_items() -> None:
    """Reasoning items with encrypted content are valid input and stay."""
    encrypted = {"type": "reasoning", "summary": [], "encrypted_content": "x"}
    items = [EVENT, encrypted, MESSAGE]
    assert Agent._drop_legacy_reasoning(items) == items


def test_legacy_delivery_nudge_is_not_an_external_event() -> None:
    """Old untyped correction messages never become event boundaries."""
    nudge = {"role": "user", "content": "[delivery correction]\nretry"}
    assert not Agent._is_external_event(nudge)
    assert Agent._normalize_internal_nudge(nudge) == {**nudge, "type": "message"}


def test_provider_fallback_keeps_active_turn_suffix() -> None:
    """Provider-neutral history retains fresh tool calls and results."""
    agent = Agent.__new__(Agent)
    old_call = {**CALL, "call_id": "old"}
    old_output = {**CALL_OUTPUT, "call_id": "old"}
    current_call = {**CALL, "call_id": "current"}
    current_output = {**CALL_OUTPUT, "call_id": "current"}
    agent._context = [EVENT, old_call, old_output, EVENT, current_call, current_output]
    agent._active_turn_start = 3

    assert agent._provider_fallback_context() == [
        EVENT,
        EVENT,
        current_call,
        current_output,
    ]


def test_finish_turn_prunes_only_new_ephemeral_outputs() -> None:
    """Settling removes new reasoning/messages but keeps durable items."""
    old_message = dict(MESSAGE)
    agent = Agent.__new__(Agent)
    agent.prune_completed_reasoning = True
    agent._context = [EVENT, old_message, REASONING, CALL, CALL_OUTPUT, MESSAGE]

    agent._finish_turn(2)

    assert agent._context == [EVENT, old_message, CALL, CALL_OUTPUT]


def test_finish_turn_can_retain_complete_outputs() -> None:
    """Disabling pruning preserves reasoning and assistant output."""
    agent = Agent.__new__(Agent)
    agent.prune_completed_reasoning = False
    agent._context = [EVENT, REASONING, MESSAGE]

    agent._finish_turn(1)

    assert agent._context == [EVENT, REASONING, MESSAGE]


class FakeContextDB:
    """Persists nothing; satisfies _remember's write-through call."""

    def __init__(self) -> None:
        """Collect context and usage writes."""
        self.items: list[dict[str, Any]] = []
        self.usage: list[dict[str, Any]] = []
        self.turns: list[dict[str, Any]] = []

    async def append_context(self, item: dict[str, Any]) -> None:
        """Collect one full-history item."""
        self.items.append(item)

    async def append_api_usage(self, **usage: Any) -> None:
        """Collect one usage record."""
        self.usage.append(usage)

    async def latest_context_id(self) -> int:
        """Use collected history length as a stable fake id."""
        return len(self.items)

    async def start_agent_turn(self, start_context_id: int) -> int:
        """Collect a running turn and return its fake id."""
        self.turns.append({"start_context_id": start_context_id, "status": "running"})
        return len(self.turns)

    async def finish_agent_turn(
        self,
        turn_id: int,
        end_context_id: int,
        status: str,
    ) -> None:
        """Close one collected fake turn."""
        self.turns[turn_id - 1].update(
            end_context_id=end_context_id,
            status=status,
        )


async def test_remember_trims_in_chunks() -> None:
    """Overflow cuts the window back to TRIM_CONTEXT_ITEMS in one go.

    Chunked trimming keeps the context prefix stable between trims;
    one-by-one trimming would shift the prefix on every append.
    """
    agent = Agent.__new__(Agent)
    agent.db = cast(Database, FakeContextDB())
    agent.max_context_items = MAX_CONTEXT_ITEMS
    agent.trim_context_items = TRIM_CONTEXT_ITEMS
    agent._context = [EVENT] * MAX_CONTEXT_ITEMS
    await agent._remember(dict(EVENT))
    assert len(agent._context) == TRIM_CONTEXT_ITEMS
    head = agent._context[0]
    await agent._remember(dict(EVENT))
    assert len(agent._context) == TRIM_CONTEXT_ITEMS + 1
    assert agent._context[0] is head


async def test_remember_updates_active_turn_boundary_after_trim() -> None:
    """Chunk trimming keeps active-turn boundary attached to its first event."""
    agent = Agent.__new__(Agent)
    agent.db = cast(Database, FakeContextDB())
    agent.max_context_items = MAX_CONTEXT_ITEMS
    agent.trim_context_items = TRIM_CONTEXT_ITEMS
    agent._context = [dict(EVENT) for _ in range(MAX_CONTEXT_ITEMS)]
    agent._active_turn_start = len(agent._context)
    active_event = {"role": "user", "content": "[event] current"}

    await agent._remember(active_event)

    assert agent._context[-1] is active_event
    assert agent._active_turn_start == len(agent._context) - 1


STALE = datetime(2020, 1, 1, tzinfo=UTC)


def make_processing_agent(outward_calls_per_turn: int = 0) -> Agent:
    """Build a bare agent whose turn only makes fake outward tool calls."""
    agent = Agent.__new__(Agent)
    agent.db = cast(Database, FakeContextDB())
    agent.max_context_items = MAX_CONTEXT_ITEMS
    agent.trim_context_items = TRIM_CONTEXT_ITEMS
    agent._context = []
    agent.turn_lock = asyncio.Lock()
    agent.tools = cast(Any, SimpleNamespace(outward_calls=0))
    agent.last_active = STALE

    async def turn() -> None:
        agent.tools.outward_calls += outward_calls_per_turn

    cast(Any, agent)._turn = turn
    return agent


async def test_process_heartbeat_only_leaves_idle_clock() -> None:
    """A quiet heartbeat turn does not reset last_active.

    Otherwise a heartbeat interval below dream_idle_minutes would make
    the idle dream trigger unreachable.
    """
    agent = make_processing_agent()
    await agent._process([("[heartbeat] all quiet", False)])
    assert agent.last_active is STALE


async def test_process_activity_event_resets_idle_clock() -> None:
    """A batch with a real event moves last_active."""
    agent = make_processing_agent()
    await agent._process([("[heartbeat] quiet", False), ("[event] hi", True)])
    assert agent.last_active is not STALE


async def test_process_outward_action_resets_idle_clock() -> None:
    """A heartbeat turn that reached out to someone counts as activity."""
    agent = make_processing_agent(outward_calls_per_turn=1)
    await agent._process([("[heartbeat] quiet", False)])
    assert agent.last_active is not STALE


class FakeOutputItem:
    """Minimal Responses API output item used by the turn test."""

    def __init__(self, item: dict[str, Any]) -> None:
        """Expose its type and serialized payload."""
        self.item = item
        for key, value in item.items():
            setattr(self, key, value)

    def model_dump(self, **_: Any) -> dict[str, Any]:
        """Return the canned API payload."""
        return dict(self.item)


async def test_turn_persists_then_prunes_ephemeral_outputs() -> None:
    """Completed output remains in SQLite but leaves the live window."""
    response = SimpleNamespace(
        id="resp_1",
        model="gpt-test",
        output=[FakeOutputItem(REASONING), FakeOutputItem(MESSAGE)],
        output_text="",
        usage=None,
    )
    calls: list[dict[str, Any]] = []

    async def create(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return response

    db = FakeContextDB()
    agent = Agent.__new__(Agent)
    agent.client = cast(Any, SimpleNamespace(responses=SimpleNamespace(create=create)))
    agent.model = "gpt-test"
    agent.mind = cast(Any, SimpleNamespace(soul=lambda: "soul"))
    agent.base_prompt = "base"
    agent.max_rounds = 1
    agent.max_context_items = MAX_CONTEXT_ITEMS
    agent.trim_context_items = TRIM_CONTEXT_ITEMS
    agent._context = [EVENT]
    agent._api_tools = []
    agent.reasoning = {"effort": "low", "context": "current_turn"}
    agent.prune_completed_reasoning = True
    agent.db = cast(Database, db)

    await agent._turn()

    assert calls[0]["reasoning"]["context"] == "current_turn"
    assert db.items == [REASONING, MESSAGE]
    assert agent._context == [EVENT]
    assert db.turns == [
        {"start_context_id": 0, "end_context_id": 2, "status": "completed"}
    ]


async def test_turn_retries_private_final_output_once() -> None:
    """A provider mistaking final output for a reply gets one correction."""
    responses = iter(
        [
            SimpleNamespace(
                id="resp_1",
                model="gpt-test",
                output=[FakeOutputItem(MESSAGE)],
                output_text="This should have been sent",
                usage=None,
            ),
            SimpleNamespace(
                id="resp_2",
                model="gpt-test",
                output=[],
                output_text="",
                usage=None,
            ),
        ]
    )
    calls: list[dict[str, Any]] = []

    async def create(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return next(responses)

    db = FakeContextDB()
    agent = Agent.__new__(Agent)
    agent.client = cast(Any, SimpleNamespace(responses=SimpleNamespace(create=create)))
    agent.model = "gpt-test"
    agent.mind = cast(Any, SimpleNamespace(soul=lambda: "soul"))
    agent.base_prompt = "base"
    agent.max_rounds = 3
    agent.max_context_items = MAX_CONTEXT_ITEMS
    agent.trim_context_items = TRIM_CONTEXT_ITEMS
    agent._context = [EVENT]
    agent._api_tools = []
    agent.reasoning = {}
    agent.prune_completed_reasoning = False
    agent.db = cast(Database, db)

    await agent._turn()

    assert len(calls) == 2
    assert calls[1]["input"][-1]["role"] == "user"
    assert calls[1]["input"][-1]["type"] == "message"
    assert "call send_message now" in calls[1]["input"][-1]["content"]
    assert "This should have been sent" in calls[1]["input"][-1]["content"]
    assert db.turns == [
        {"start_context_id": 0, "end_context_id": 2, "status": "completed"}
    ]


async def test_turn_retries_without_failed_server_tool() -> None:
    """An unsupported built-in tool is removed while local tools remain."""
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/responses")
    failure = BadRequestError(
        "Server tool request failed",
        response=httpx.Response(400, request=request),
        body={"error": {"message": "Server tool request failed"}},
    )
    response = SimpleNamespace(
        id="resp_1", model="gpt-test", output=[], output_text="", usage=None
    )
    responses = iter([failure, response])
    calls: list[dict[str, Any]] = []

    async def create(**kwargs: Any) -> Any:
        calls.append(kwargs)
        result = next(responses)
        if isinstance(result, Exception):
            raise result
        return result

    db = FakeContextDB()
    agent = Agent.__new__(Agent)
    agent.client = cast(Any, SimpleNamespace(responses=SimpleNamespace(create=create)))
    agent.model = "gpt-test"
    agent.mind = cast(Any, SimpleNamespace(soul=lambda: "soul"))
    agent.base_prompt = "base"
    agent.max_rounds = 1
    agent.max_context_items = MAX_CONTEXT_ITEMS
    agent.trim_context_items = TRIM_CONTEXT_ITEMS
    agent._context = [EVENT]
    function_tool = cast(Any, {"type": "function", "name": "send_message"})
    agent._api_tools = [function_tool, cast(Any, {"type": "web_search"})]
    agent.reasoning = {}
    agent.prune_completed_reasoning = False
    agent.db = cast(Database, db)

    await agent._turn()

    assert len(calls) == 2
    assert calls[0]["tools"] == [function_tool, {"type": "web_search"}]
    assert calls[1]["tools"] == [function_tool]
    assert agent._api_tools == [function_tool, {"type": "web_search"}]


async def test_turn_retries_duplicate_tool_ids_with_events_only() -> None:
    """Mistral duplicate-id errors fall back to external event context."""
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/responses")
    failure = BadRequestError(
        "Duplicate tool call id in assistant message",
        response=httpx.Response(400, request=request),
        body={"error": {"message": "Duplicate tool call id in assistant message"}},
    )
    response = SimpleNamespace(
        id="resp_1", model="mistral-test", output=[], output_text="", usage=None
    )
    responses = iter([failure, response])
    calls: list[dict[str, Any]] = []

    async def create(**kwargs: Any) -> Any:
        calls.append(kwargs)
        result = next(responses)
        if isinstance(result, Exception):
            raise result
        return result

    db = FakeContextDB()
    agent = Agent.__new__(Agent)
    agent.client = cast(Any, SimpleNamespace(responses=SimpleNamespace(create=create)))
    agent.model = "mistral-test"
    agent.mind = cast(Any, SimpleNamespace(soul=lambda: "soul"))
    agent.base_prompt = "base"
    agent.max_rounds = 1
    agent.max_context_items = MAX_CONTEXT_ITEMS
    agent.trim_context_items = TRIM_CONTEXT_ITEMS
    agent._context = [EVENT, CALL, CALL_OUTPUT, MESSAGE, EVENT]
    agent._api_tools = []
    agent.reasoning = {}
    agent.prune_completed_reasoning = False
    agent.db = cast(Database, db)

    await agent._turn()

    assert len(calls) == 2
    assert calls[1]["input"] == [EVENT, EVENT]
    assert agent._context == [EVENT, EVENT]


async def test_turn_retries_encrypted_reasoning_with_provider_neutral_context() -> None:
    """Cross-provider encrypted reasoning errors compact historical context."""
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/responses")
    failure = BadRequestError(
        "Could not decrypt the provided encrypted_content",
        response=httpx.Response(400, request=request),
        body={"error": {"message": "Could not decrypt encrypted_content"}},
    )
    response = SimpleNamespace(
        id="resp_1", model="xai-test", output=[], output_text="", usage=None
    )
    responses = iter([failure, response])
    calls: list[dict[str, Any]] = []

    async def create(**kwargs: Any) -> Any:
        calls.append(kwargs)
        result = next(responses)
        if isinstance(result, Exception):
            raise result
        return result

    db = FakeContextDB()
    agent = Agent.__new__(Agent)
    agent.client = cast(Any, SimpleNamespace(responses=SimpleNamespace(create=create)))
    agent.model = "xai-test"
    agent.mind = cast(Any, SimpleNamespace(soul=lambda: "soul"))
    agent.base_prompt = "base"
    agent.max_rounds = 1
    agent.max_context_items = MAX_CONTEXT_ITEMS
    agent.trim_context_items = TRIM_CONTEXT_ITEMS
    encrypted = {"type": "reasoning", "encrypted_content": "opaque"}
    agent._context = [EVENT, encrypted, MESSAGE, EVENT]
    agent._api_tools = []
    agent.reasoning = {}
    agent.prune_completed_reasoning = False
    agent.db = cast(Database, db)

    await agent._turn()

    assert len(calls) == 2
    assert calls[1]["input"] == [EVENT, EVENT]
    assert agent._context == [EVENT, EVENT]


async def test_turn_chains_server_tool_and_duplicate_id_fallbacks() -> None:
    """Sequential compatibility failures both transform the next retry."""
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/responses")
    server_failure = BadRequestError(
        "Server tool request failed",
        response=httpx.Response(400, request=request),
        body={"error": {"message": "Server tool request failed"}},
    )
    duplicate_failure = BadRequestError(
        "Duplicate tool call id in assistant message",
        response=httpx.Response(400, request=request),
        body={"error": {"message": "Duplicate tool call id in assistant message"}},
    )
    response = SimpleNamespace(
        id="resp_1", model="mistral-test", output=[], output_text="", usage=None
    )
    responses = iter([server_failure, duplicate_failure, response])
    calls: list[dict[str, Any]] = []

    async def create(**kwargs: Any) -> Any:
        calls.append(kwargs)
        result = next(responses)
        if isinstance(result, Exception):
            raise result
        return result

    db = FakeContextDB()
    agent = Agent.__new__(Agent)
    agent.client = cast(Any, SimpleNamespace(responses=SimpleNamespace(create=create)))
    agent.model = "mistral-test"
    agent.mind = cast(Any, SimpleNamespace(soul=lambda: "soul"))
    agent.base_prompt = "base"
    agent.max_rounds = 1
    agent.max_context_items = MAX_CONTEXT_ITEMS
    agent.trim_context_items = TRIM_CONTEXT_ITEMS
    agent._context = [EVENT, CALL, CALL_OUTPUT, MESSAGE, EVENT]
    function_tool = cast(Any, {"type": "function", "name": "send_message"})
    agent._api_tools = [function_tool, cast(Any, {"type": "web_search"})]
    agent.reasoning = {}
    agent.prune_completed_reasoning = False
    agent.db = cast(Database, db)

    await agent._turn()

    assert len(calls) == 3
    assert calls[1]["tools"] == [function_tool]
    assert calls[1]["input"] != [EVENT, EVENT]
    assert calls[2]["tools"] == [function_tool]
    assert calls[2]["input"] == [EVENT, EVENT]


async def test_provider_fallback_preserves_tool_result_for_next_round() -> None:
    """Compacted context remains active through a multi-round tool turn."""
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/responses")
    failure = BadRequestError(
        "Duplicate tool call id in assistant message",
        response=httpx.Response(400, request=request),
        body={"error": {"message": "Duplicate tool call id in assistant message"}},
    )
    call_response = SimpleNamespace(
        id="resp_1",
        model="mistral-test",
        output=[FakeOutputItem(CALL)],
        output_text="",
        usage=None,
    )
    final_response = SimpleNamespace(
        id="resp_2", model="mistral-test", output=[], output_text="", usage=None
    )
    responses = iter([failure, call_response, final_response])
    calls: list[dict[str, Any]] = []

    async def create(**kwargs: Any) -> Any:
        calls.append(kwargs)
        result = next(responses)
        if isinstance(result, Exception):
            raise result
        return result

    class FakeTools:
        """Return one stable result for the model's function call."""

        async def run(self, name: str, arguments: str) -> str:
            """Record no side effects and return a visible tool result."""
            assert name == "send_message"
            assert arguments == "{}"
            return "sent"

    db = FakeContextDB()
    agent = Agent.__new__(Agent)
    agent.client = cast(Any, SimpleNamespace(responses=SimpleNamespace(create=create)))
    agent.model = "mistral-test"
    agent.mind = cast(Any, SimpleNamespace(soul=lambda: "soul"))
    agent.base_prompt = "base"
    agent.max_rounds = 2
    agent.max_context_items = MAX_CONTEXT_ITEMS
    agent.trim_context_items = TRIM_CONTEXT_ITEMS
    agent._context = [EVENT, {**CALL, "call_id": "old"}, EVENT]
    agent._api_tools = []
    agent.reasoning = {}
    agent.prune_completed_reasoning = True
    agent.db = cast(Database, db)
    agent.tools = cast(Any, FakeTools())

    await agent._turn()

    assert len(calls) == 3
    assert calls[2]["input"][-2:] == [CALL, CALL_OUTPUT]
    assert agent._context == [EVENT, EVENT, CALL, CALL_OUTPUT]


async def test_record_usage_maps_all_authoritative_counts() -> None:
    """Response usage fields map into persistent records unchanged."""
    db = FakeContextDB()
    agent = Agent.__new__(Agent)
    agent.db = cast(Database, db)
    agent.model = "fallback-model"
    response = SimpleNamespace(
        id="resp_1",
        model="actual-model",
        usage=SimpleNamespace(
            input_tokens=100,
            input_tokens_details=SimpleNamespace(
                cached_tokens=80,
                cache_write_tokens=20,
            ),
            output_tokens=30,
            output_tokens_details=SimpleNamespace(reasoning_tokens=25),
            total_tokens=130,
        ),
    )

    await agent._record_usage(response, turn_id=7, input_context_id=42)

    assert db.usage == [
        {
            "response_id": "resp_1",
            "turn_id": 7,
            "dream_id": None,
            "input_context_id": 42,
            "model": "actual-model",
            "input_tokens": 100,
            "cached_tokens": 80,
            "cache_write_tokens": 20,
            "output_tokens": 30,
            "reasoning_tokens": 25,
            "total_tokens": 130,
        }
    ]
