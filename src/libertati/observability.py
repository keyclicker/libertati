"""Structured behavioral telemetry.

Writes one JSON object per line to an event log so the bot's real behavior can be
read back and analyzed — every LLM call (with token usage + latency), tool call,
memory write and message is correlated by a per-turn id, so a single grep
reconstructs a whole decision.

Everything is opt-in-safe: a disabled ``EventLogger`` is a cheap no-op, and every
component that takes one falls back to a disabled instance, so nothing breaks when
telemetry is off.
"""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .logging import get_logger

log = get_logger("events")

_current_turn: ContextVar[str | None] = ContextVar("libertati_turn", default=None)
_current_acc: ContextVar[TurnAccumulator | None] = ContextVar("libertati_acc", default=None)


@dataclass
class TurnAccumulator:
    turn_id: str
    kind: str
    started: float
    llm_calls: int = 0
    tool_calls: int = 0
    total_tokens: int = 0
    context: dict[str, Any] = field(default_factory=dict)


class EventLogger:
    def __init__(
        self,
        path: Path | str | None = None,
        enabled: bool = True,
        log_content: bool = True,
        content_max: int = 300,
    ) -> None:
        self.enabled = enabled and path is not None
        self.path = Path(path) if path is not None else None
        self.log_content = log_content
        self.content_max = content_max
        self._lock = threading.Lock()
        self._fh = None
        self._seq = 0
        self._turn_counter = 0
        self.session_tokens = 0
        if self.enabled and self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("a", encoding="utf-8")

    # -- redaction -----------------------------------------------------------
    def redact(self, text: str | None) -> str | None:
        if text is None:
            return None
        if not self.log_content:
            return f"<{len(text)} chars>"
        text = " ".join(text.split())
        if len(text) > self.content_max:
            return text[: self.content_max] + "…"
        return text

    # -- emit ----------------------------------------------------------------
    def emit(self, event_type: str, **fields: Any) -> None:
        if not self.enabled or self._fh is None:
            return
        with self._lock:
            self._seq += 1
            record = {
                "ts": round(time.time(), 3),
                "seq": self._seq,
                "turn": _current_turn.get(),
                "type": event_type,
                **fields,
            }
            self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._fh.flush()
        log.debug("%s %s", event_type, {k: v for k, v in fields.items() if k != "preview"})

    # -- turn correlation ----------------------------------------------------
    @contextmanager
    def turn(self, kind: str, **context: Any):
        self._turn_counter += 1
        turn_id = f"t{self._turn_counter}"
        acc = TurnAccumulator(
            turn_id=turn_id, kind=kind, started=time.perf_counter(), context=context
        )
        turn_token = _current_turn.set(turn_id)
        acc_token = _current_acc.set(acc)
        self.emit("turn_start", kind=kind, **context)
        try:
            yield acc
        finally:
            self.session_tokens += acc.total_tokens
            self.emit(
                "turn_end",
                kind=kind,
                latency_ms=round((time.perf_counter() - acc.started) * 1000),
                llm_calls=acc.llm_calls,
                tool_calls=acc.tool_calls,
                turn_tokens=acc.total_tokens,
                session_tokens=self.session_tokens,
            )
            _current_acc.reset(acc_token)
            _current_turn.reset(turn_token)

    @staticmethod
    def current_acc() -> TurnAccumulator | None:
        """The active turn accumulator, if a turn() is in progress on this task."""
        return _current_acc.get()

    def close(self) -> None:
        if self._fh is not None:
            with self._lock:
                self._fh.close()
                self._fh = None


# A shared disabled instance for default no-op wiring.
NULL_EVENTS = EventLogger(path=None, enabled=False)
