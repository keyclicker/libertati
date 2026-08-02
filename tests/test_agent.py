"""Tests for the agent's context-window trimming logic."""

from typing import Any

from libertati.agent import Agent

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
