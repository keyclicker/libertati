"""Tests for the agent's context-window trimming logic."""

from typing import Any, cast

from libertati.agent import MAX_CONTEXT_ITEMS, TRIM_CONTEXT_ITEMS, Agent
from libertati.db import Database

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


class FakeContextDB:
    """Persists nothing; satisfies _remember's write-through call."""

    async def append_context(self, item: dict[str, Any]) -> None:
        """Discard the item."""


async def test_remember_trims_in_chunks() -> None:
    """Overflow cuts the window back to TRIM_CONTEXT_ITEMS in one go.

    Chunked trimming keeps the context prefix stable between trims so
    prompt caching stays effective; one-by-one trimming would shift the
    prefix on every append.
    """
    agent = Agent.__new__(Agent)
    agent.db = cast(Database, FakeContextDB())
    agent._context = [EVENT] * MAX_CONTEXT_ITEMS
    await agent._remember(dict(EVENT))
    assert len(agent._context) == TRIM_CONTEXT_ITEMS
    head = agent._context[0]
    await agent._remember(dict(EVENT))
    assert len(agent._context) == TRIM_CONTEXT_ITEMS + 1
    assert agent._context[0] is head
