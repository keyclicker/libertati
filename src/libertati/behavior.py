"""Small, pure helpers that shape *when* and *how* the bot speaks — the human touch.

Kept dependency-free and side-effect-free so they are trivial to unit-test.
"""

from __future__ import annotations


def mentions_bot(text: str, names: list[str | None]) -> bool:
    """True if the text name-drops the bot (by @username, @handle or display name)."""
    low = (text or "").lower()
    for name in names:
        if not name:
            continue
        token = name.lstrip("@").lower().strip()
        if token and token in low:
            return True
    return False


def typing_delay_seconds(reply: str, max_seconds: float, chars_per_second: float = 25.0) -> float:
    """A human-like pause before sending, proportional to length and capped."""
    if not reply or max_seconds <= 0:
        return 0.0
    return min(max_seconds, len(reply) / chars_per_second)
