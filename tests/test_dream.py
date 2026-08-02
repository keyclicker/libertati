"""Tests for the dreaming loop: gating, session shape and waking up."""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from zoneinfo import ZoneInfo

import pytest

from libertati.agent import Agent
from libertati.db import Database
from libertati.dream import Dreamer, DreamGate
from libertati.memory import Mind
from libertati.tools import DREAM_TOOL_NAMES

UTC_TZ = ZoneInfo("UTC")

WAKE_CALL = {
    "type": "function_call",
    "call_id": "call_wake",
    "name": "wake_up",
    "arguments": json.dumps({"summary": "say hi to Bob"}),
}
READ_CALL = {
    "type": "function_call",
    "call_id": "call_read",
    "name": "read_mind",
    "arguments": json.dumps({"file": "memory"}),
}
MESSAGE = {
    "type": "message",
    "role": "assistant",
    "content": [{"type": "output_text", "text": "…"}],
}


class FakeOutputItem:
    """Minimal Responses API output item."""

    def __init__(self, item: dict[str, Any]) -> None:
        """Expose its type and serialized payload."""
        self.type = item["type"]
        self.item = item
        self.name = item.get("name")
        self.call_id = item.get("call_id")
        self.arguments = item.get("arguments")

    def model_dump(self, **_: Any) -> dict[str, Any]:
        """Return the canned API payload."""
        return dict(self.item)


class FakeClient:
    """Replays a canned sequence of model responses."""

    def __init__(self, rounds: list[list[dict[str, Any]]]) -> None:
        """Take one output-item list per round."""
        self.rounds = rounds
        self.calls: list[dict[str, Any]] = []
        self.responses = SimpleNamespace(create=self._create)

    async def _create(self, **kwargs: Any) -> Any:
        """Return the next canned round, repeating the last one."""
        self.calls.append(kwargs)
        index = min(len(self.calls) - 1, len(self.rounds) - 1)
        return SimpleNamespace(
            id=f"resp_{len(self.calls)}",
            model="dream-model",
            output=[FakeOutputItem(item) for item in self.rounds[index]],
            output_text="",
            usage=None,
        )


class FakeAgent:
    """The waking agent, reduced to what a Dreamer touches."""

    def __init__(self, mind: Mind, client: Any, idle_minutes: float = 90) -> None:
        """Expose the mind, lock, client and an idle 'last active'."""
        self.mind = mind
        self.client = client
        self.reasoning = {"effort": "low"}
        self.tz = UTC_TZ
        self.turn_lock = asyncio.Lock()
        self.last_active = datetime.now(UTC) - timedelta(minutes=idle_minutes)
        self.pushed: list[str] = []
        self.prompts = SimpleNamespace(
            recall="recall",
            summary="summary",
            dream="you are asleep",
            dream_nudge="[still asleep] keep going",
        )

    async def push(self, event: str) -> None:
        """Collect the wake-up event."""
        self.pushed.append(event)


def make_settings(**overrides: Any) -> Any:
    """Build the dream-relevant slice of Settings as a fake."""
    values: dict[str, Any] = {
        "model": "main-model",
        "dream_model": None,
        "recall_model": None,
        "recall_reasoning_effort": None,
        "web_search": False,
        "typing_chars_per_second": 15.0,
        "dream_idle_minutes": 45,
        "dream_cooldown_minutes": 120,
        "dream_min_steps": 0,
        "dream_max_rounds": 5,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def make_dreamer(
    db: Database,
    tmp_path: Path,
    rounds: list[list[dict[str, Any]]],
    **settings: Any,
) -> tuple[Dreamer, FakeAgent, FakeClient]:
    """Wire a Dreamer around fakes and a real Mind and Database."""
    mind = Mind(tmp_path / "mind")
    mind.ensure()
    client = FakeClient(rounds)
    agent = FakeAgent(mind, client, idle_minutes=settings.pop("idle_minutes", 90))
    gate = DreamGate(db, settings.pop("daily_budget", 4))
    dreamer = Dreamer(
        make_settings(**settings),
        db,
        cast(Any, SimpleNamespace(id=1)),
        cast(Any, agent),
        gate,
    )
    return dreamer, agent, client


async def test_dream_runs_a_session_and_wakes_the_agent(
    db: Database, tmp_path: Path
) -> None:
    """A dream that calls wake_up ends, is recorded and pushes an event."""
    dreamer, agent, client = make_dreamer(db, tmp_path, [[WAKE_CALL]])

    await dreamer.maybe_dream()

    assert len(client.calls) == 1
    assert client.calls[0]["instructions"].startswith("you are asleep")
    assert "## Soul" in client.calls[0]["instructions"]
    assert len(agent.pushed) == 1
    assert "dream #1 ended" in agent.pushed[0]
    assert "say hi to Bob" in agent.pushed[0]

    async with db.conn.execute("SELECT * FROM dreams") as cursor:
        rows = list(await cursor.fetchall())
    assert [row["status"] for row in rows] == ["woke"]
    assert rows[0]["trigger"] == "idle"
    assert rows[0]["steps"] == 1


async def test_dream_persists_no_context(db: Database, tmp_path: Path) -> None:
    """Nothing a dream thinks reaches the agent's history tables."""
    dreamer, _, _ = make_dreamer(db, tmp_path, [[WAKE_CALL]])

    await dreamer.maybe_dream()

    for table in ("context", "agent_turns", "api_usage"):
        async with db.conn.execute(f"SELECT COUNT(*) FROM {table}") as cursor:
            row = await cursor.fetchone()
        assert row is not None
        assert row[0] == 0, table
    assert dreamer._context == []


async def test_dream_opens_with_the_mind_files(db: Database, tmp_path: Path) -> None:
    """Falling asleep hands over all three files, no rounds wasted."""
    dreamer, agent, client = make_dreamer(db, tmp_path, [[WAKE_CALL]])
    agent.mind.memory_path.write_text("Alice likes tea\n", encoding="utf-8")
    agent.mind.append_inbox("Bob moved city", "Sun 2026-08-02 12:00")

    await dreamer.maybe_dream()

    opening = client.calls[0]["input"][0]["content"]
    assert "you fall asleep" in opening
    assert "Alice likes tea" in opening
    assert "Bob moved city" in opening
    assert "# Dreams\n(empty)" in opening


async def test_settling_early_nudges_instead_of_ending(
    db: Database, tmp_path: Path
) -> None:
    """A round with no tool call must not end the dream."""
    dreamer, _, client = make_dreamer(
        db, tmp_path, [[MESSAGE], [READ_CALL], [WAKE_CALL]]
    )

    await dreamer.maybe_dream()

    assert len(client.calls) == 3
    nudges = [
        item
        for item in client.calls[1]["input"]
        if item.get("role") == "user" and "still asleep" in item.get("content", "")
    ]
    assert len(nudges) == 1


async def test_max_rounds_ends_the_dream_without_waking(
    db: Database, tmp_path: Path
) -> None:
    """A dream that never calls wake_up still closes and wakes the agent."""
    dreamer, agent, client = make_dreamer(
        db, tmp_path, [[READ_CALL]], dream_max_rounds=3
    )

    await dreamer.maybe_dream()

    assert len(client.calls) == 3
    async with db.conn.execute("SELECT status FROM dreams") as cursor:
        row = await cursor.fetchone()
    assert row is not None
    assert row["status"] == "max_rounds"
    assert "ran out before waking" in agent.pushed[0]


async def test_min_steps_keeps_a_dream_wandering(db: Database, tmp_path: Path) -> None:
    """wake_up is refused until the dream has taken enough steps."""
    dreamer, agent, client = make_dreamer(
        db, tmp_path, [[WAKE_CALL]], dream_min_steps=3, dream_max_rounds=6
    )

    await dreamer.maybe_dream()

    # Rounds 1 and 2 are refused (1 and 2 steps), round 3 succeeds.
    assert len(client.calls) == 3
    assert "say hi to Bob" in agent.pushed[0]


async def test_idle_gate_keeps_a_busy_agent_awake(db: Database, tmp_path: Path) -> None:
    """No dream while someone was talking to the agent recently."""
    dreamer, agent, client = make_dreamer(db, tmp_path, [[WAKE_CALL]], idle_minutes=5)

    await dreamer.maybe_dream()

    assert client.calls == []
    assert agent.pushed == []


async def test_a_request_overrides_the_idle_gate(db: Database, tmp_path: Path) -> None:
    """The agent asking to sleep does not have to wait to be bored."""
    dreamer, _, client = make_dreamer(db, tmp_path, [[WAKE_CALL]], idle_minutes=5)
    dreamer.gate.request("the trip")

    await dreamer.maybe_dream()

    assert len(client.calls) == 1
    assert "the trip" in client.calls[0]["input"][0]["content"]
    assert dreamer.gate.requested is False
    async with db.conn.execute("SELECT trigger FROM dreams") as cursor:
        row = await cursor.fetchone()
    assert row is not None
    assert row["trigger"] == "requested"


async def test_budget_stops_the_fifth_dream(db: Database, tmp_path: Path) -> None:
    """A spent daily budget blocks both triggers, and drops the request."""
    dreamer, _, client = make_dreamer(
        db, tmp_path, [[WAKE_CALL]], daily_budget=2, dream_cooldown_minutes=0
    )

    await dreamer.maybe_dream()
    await dreamer.maybe_dream()
    assert await dreamer.gate.budget_left() == 0

    await dreamer.maybe_dream()
    dreamer.gate.request("please")
    await dreamer.maybe_dream()

    assert len(client.calls) == 2
    assert dreamer.gate.requested is False


async def test_a_zero_budget_disables_dreaming(db: Database, tmp_path: Path) -> None:
    """Budget 0 is the off switch, even for an explicit request."""
    dreamer, _, client = make_dreamer(db, tmp_path, [[WAKE_CALL]], daily_budget=0)
    dreamer.gate.request("please")

    await dreamer.maybe_dream()

    assert client.calls == []


async def test_cooldown_blocks_a_second_dream(db: Database, tmp_path: Path) -> None:
    """Back-to-back dreams are refused until the cooldown passes."""
    dreamer, _, client = make_dreamer(
        db, tmp_path, [[WAKE_CALL]], dream_cooldown_minutes=120
    )

    await dreamer.maybe_dream()
    await dreamer.maybe_dream()

    assert len(client.calls) == 1


async def test_no_dream_starts_mid_turn(db: Database, tmp_path: Path) -> None:
    """A busy agent is never idle, whatever its stale last_active says."""
    dreamer, agent, client = make_dreamer(db, tmp_path, [[WAKE_CALL]])
    await agent.turn_lock.acquire()

    await dreamer.maybe_dream()

    assert client.calls == []
    async with db.conn.execute("SELECT COUNT(*) FROM dreams") as cursor:
        row = await cursor.fetchone()
    assert row is not None
    assert row[0] == 0, "budget was spent on a dream that never started"

    agent.turn_lock.release()
    await dreamer.maybe_dream()
    assert len(client.calls) == 1


async def test_dream_holds_the_agents_turn_lock(db: Database, tmp_path: Path) -> None:
    """The waking loop cannot start a turn while the dream runs."""
    dreamer, agent, _ = make_dreamer(db, tmp_path, [[WAKE_CALL]])
    await agent.turn_lock.acquire()

    task = asyncio.create_task(dreamer.dream("idle", None))
    await asyncio.sleep(0)
    assert not task.done()

    agent.turn_lock.release()
    await task
    assert agent.pushed
    assert not agent.turn_lock.locked()


async def test_a_failing_dream_still_wakes_the_agent(
    db: Database, tmp_path: Path
) -> None:
    """Losing the dream loop silently would be worse than a odd event."""

    async def explode(**_: Any) -> Any:
        raise RuntimeError("boom")

    dreamer, agent, _ = make_dreamer(db, tmp_path, [[WAKE_CALL]])
    dreamer.client = cast(
        Any, SimpleNamespace(responses=SimpleNamespace(create=explode))
    )

    await dreamer.maybe_dream()

    async with db.conn.execute("SELECT status FROM dreams") as cursor:
        row = await cursor.fetchone()
    assert row is not None
    assert row["status"] == "failed"
    assert len(agent.pushed) == 1
    assert not agent.turn_lock.locked()


def test_dreamer_toolbox_is_the_restricted_one(db: Database, tmp_path: Path) -> None:
    """The dreaming loop can only execute its own tools."""
    dreamer, _, _ = make_dreamer(db, tmp_path, [[WAKE_CALL]])

    toolbox = dreamer.tools
    assert set(toolbox._handlers) == set(DREAM_TOOL_NAMES)
    assert "send_message" not in toolbox._handlers


async def test_budget_left_counts_the_rolling_window(db: Database) -> None:
    """Only dreams inside the last 24 hours spend budget."""
    gate = DreamGate(db, 4)
    assert await gate.budget_left() == 4

    await db.start_dream("idle")
    assert await gate.budget_left() == 3

    await db.conn.execute("UPDATE dreams SET started_at = '2020-01-01 00:00:00'")
    await db.conn.commit()
    assert await gate.budget_left() == 4


async def test_a_held_lock_stops_the_real_agent_loop(
    db: Database, tmp_path: Path
) -> None:
    """A dream in progress genuinely blocks the waking loop's next turn."""
    mind = Mind(tmp_path / "mind")
    mind.ensure()
    turns: list[Any] = []
    took_a_turn = asyncio.Event()

    async def create(**kwargs: Any) -> Any:
        turns.append(kwargs)
        took_a_turn.set()
        return SimpleNamespace(
            id="resp_1", model="m", output=[], output_text="", usage=None
        )

    agent = Agent.__new__(Agent)
    agent.client = cast(Any, SimpleNamespace(responses=SimpleNamespace(create=create)))
    agent.model = "m"
    agent.db = db
    agent.mind = mind
    agent.base_prompt = "base"
    agent.reasoning = {}
    agent.max_rounds = 1
    agent.max_context_items = 300
    agent.trim_context_items = 200
    agent._context = []
    agent._api_tools = []
    agent.prune_completed_reasoning = False
    agent._queue = asyncio.Queue()
    agent.tools = cast(Any, SimpleNamespace(outward_calls=0))
    agent.turn_lock = asyncio.Lock()
    agent.last_active = datetime.now(UTC) - timedelta(minutes=90)

    await agent.turn_lock.acquire()
    await agent.push("[event] hi")
    worker = asyncio.create_task(agent.run_forever())
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(took_a_turn.wait(), timeout=0.2)
    assert turns == [], "the agent took a turn while the lock was held"

    agent.turn_lock.release()
    await asyncio.wait_for(took_a_turn.wait(), timeout=2)
    worker.cancel()
    assert len(turns) == 1
