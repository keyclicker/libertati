"""Markdown memory store — the bot's self-maintained long-term memory.

Layout under ``memory_dir``::

    user/<handle>.md    per-user facts, tone, running summary
    group/<slug>.md     per-group dynamics, in-jokes, topics
    self.md             the bot's evolving self-concept / persona journal
    world.md            running digest of news & current events read
    social.md           relationship graph, who-knows-whom, open threads
    todo.md             follow-ups to raise with users during heartbeat
    reading.md          interesting things read while browsing other chats/channels
    diary/<date>.md     nightly dream reflections
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

from ..observability import NULL_EVENTS, EventLogger

_SLUG_RE = re.compile(r"[^a-z0-9_-]+")


def slugify(value: str) -> str:
    value = value.strip().lower().lstrip("@")
    value = value.replace(" ", "-")
    value = _SLUG_RE.sub("", value)
    return value or "unknown"


class MemoryStore:
    """Reads and writes markdown memory files, sandboxed to ``root``."""

    GENERAL_FILES = ("self.md", "world.md", "social.md", "todo.md", "reading.md")

    def __init__(
        self,
        root: Path | str,
        events: EventLogger | None = None,
        max_file_chars: int | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.events = events or NULL_EVENTS
        self.max_file_chars = max_file_chars

    # -- path safety ---------------------------------------------------------
    def _resolve(self, rel: str) -> Path:
        # Normalise and forbid escaping the memory root.
        candidate = (self.root / rel).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise ValueError(f"path escapes memory root: {rel!r}")
        if candidate.suffix != ".md":
            raise ValueError(f"memory files must be .md: {rel!r}")
        return candidate

    # -- generic markdown access --------------------------------------------
    def read(self, rel: str) -> str:
        path = self._resolve(rel)
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def _trim(self, text: str) -> str:
        """Keep a memory file under ``max_file_chars`` by dropping its oldest lines.

        A leading ``#`` header line (the file's title) is preserved; the rest is kept
        tail-first, since recent notes are the most relevant.
        """
        cap = self.max_file_chars
        if not cap or len(text) <= cap:
            return text
        lines = text.splitlines()
        header = [lines.pop(0)] if lines and lines[0].startswith("#") else []
        while lines and len("\n".join(header + lines)) + 1 > cap:
            lines.pop(0)
        return "\n".join(header + lines).rstrip() + "\n"

    def overwrite(self, rel: str, content: str, _mode: str = "overwrite") -> None:
        path = self._resolve(rel)
        old_len = len(self.read(rel))
        path.parent.mkdir(parents=True, exist_ok=True)
        new_text = self._trim(content.rstrip() + "\n")
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(new_text, encoding="utf-8")
        tmp.replace(path)
        self.events.emit(
            "memory_write",
            path=rel,
            mode=_mode,
            old_len=old_len,
            new_len=len(new_text),
            delta=len(new_text) - old_len,
        )

    def append(self, rel: str, content: str) -> None:
        existing = self.read(rel)
        joined = f"{existing.rstrip()}\n{content.strip()}" if existing else content.strip()
        self.overwrite(rel, joined, _mode="append")

    def exists(self, rel: str) -> bool:
        return self._resolve(rel).exists()

    # -- typed helpers -------------------------------------------------------
    def user_path(self, handle: str) -> str:
        return f"user/{slugify(handle)}.md"

    def group_path(self, name: str) -> str:
        return f"group/{slugify(name)}.md"

    def read_user(self, handle: str) -> str:
        return self.read(self.user_path(handle))

    def read_group(self, name: str) -> str:
        return self.read(self.group_path(name))

    def diary_path(self, day: date | None = None) -> str:
        day = day or date.today()
        return f"diary/{day.isoformat()}.md"

    def list_users(self) -> list[str]:
        user_dir = self.root / "user"
        if not user_dir.exists():
            return []
        return sorted(p.stem for p in user_dir.glob("*.md"))

    def snapshot(self) -> dict[str, str]:
        """All general memory files, for building agent context cheaply."""
        return {name: self.read(name) for name in self.GENERAL_FILES}
