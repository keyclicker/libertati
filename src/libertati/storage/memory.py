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

_SLUG_RE = re.compile(r"[^a-z0-9_-]+")


def slugify(value: str) -> str:
    value = value.strip().lower().lstrip("@")
    value = value.replace(" ", "-")
    value = _SLUG_RE.sub("", value)
    return value or "unknown"


class MemoryStore:
    """Reads and writes markdown memory files, sandboxed to ``root``."""

    GENERAL_FILES = ("self.md", "world.md", "social.md", "todo.md", "reading.md")

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

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

    def overwrite(self, rel: str, content: str) -> None:
        path = self._resolve(rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(content.rstrip() + "\n", encoding="utf-8")
        tmp.replace(path)

    def append(self, rel: str, content: str) -> None:
        existing = self.read(rel)
        joined = f"{existing.rstrip()}\n{content.strip()}" if existing else content.strip()
        self.overwrite(rel, joined)

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
