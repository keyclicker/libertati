from __future__ import annotations

import json

from libertati.observability import EventLogger


def _read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_disabled_logger_writes_nothing(tmp_path):
    ev = EventLogger(path=tmp_path / "e.jsonl", enabled=False)
    ev.emit("hello", x=1)
    assert not (tmp_path / "e.jsonl").exists()


def test_emit_writes_jsonl(tmp_path):
    path = tmp_path / "e.jsonl"
    ev = EventLogger(path=path, enabled=True)
    ev.emit("inbound", chat_id=5, text_len=10)
    ev.emit("outbound", chat_id=5)
    rows = _read(path)
    assert [r["type"] for r in rows] == ["inbound", "outbound"]
    assert rows[0]["chat_id"] == 5
    assert rows[0]["seq"] == 1 and rows[1]["seq"] == 2
    assert "ts" in rows[0]
    ev.close()


def test_redact_truncates_and_omits():
    ev = EventLogger(enabled=False, content_max=5)
    assert ev.redact("hello world") == "hello…"
    assert ev.redact(None) is None
    ev2 = EventLogger(enabled=False, log_content=False)
    assert ev2.redact("secret message") == "<14 chars>"


def test_turn_correlates_events_and_counts(tmp_path):
    path = tmp_path / "e.jsonl"
    ev = EventLogger(path=path, enabled=True)
    with ev.turn("respond", chat_id=1) as acc:
        assert ev.current_acc() is acc
        acc.llm_calls += 2
        acc.tool_calls += 1
        acc.total_tokens += 150
        ev.emit("tool_call", name="x")
    rows = _read(path)
    types = [r["type"] for r in rows]
    assert types == ["turn_start", "tool_call", "turn_end"]
    # all three share the same turn id
    turns = {r["turn"] for r in rows}
    assert turns == {"t1"}
    end = rows[-1]
    assert end["llm_calls"] == 2
    assert end["tool_calls"] == 1
    assert end["turn_tokens"] == 150
    assert end["session_tokens"] == 150
    ev.close()


def test_turn_resets_context_after_exit(tmp_path):
    ev = EventLogger(path=tmp_path / "e.jsonl", enabled=True)
    with ev.turn("dream"):
        pass
    assert ev.current_acc() is None
    # events outside a turn have turn=None
    ev.emit("standalone")
    rows = _read(tmp_path / "e.jsonl")
    assert rows[-1]["turn"] is None
    ev.close()
