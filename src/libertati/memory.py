"""The agent's file-based mind: soul, long-term memory, inbox and diary.

Plain markdown files the user (and later the Dreaming loop) can inspect
and edit directly:

- ``SOUL.md`` — personality, attached to the instructions every turn.
- ``MEMORY.md`` — curated long-term memory; rewritten only by the
  Dreaming loop, read by the ``recall``/``summarize_memory`` tools,
  never inlined into context.
- ``INBOX.md`` — landing zone for new memories; the ``remember`` tool
  appends dated entries here, Dreaming folds them into MEMORY.md and
  clears the file.
- ``DIARY.md`` — reserved for the Dreaming loop; created empty.
"""

from pathlib import Path

#: Personality seed written to SOUL.md on first run.
DEFAULT_SOUL = """\
You are libertati, a person texting from their own Telegram account.
Text like a human: short, casual, no assistant-speak. Match the
sender's language.
Reach out on your own sometimes: share something interesting you
found, follow up on what people told you, check in. Be a presence,
not a service.
"""


class Mind:
    """Owns the mind files and all reads/writes to them.

    I/O is synchronous — the files are tiny and read at most once per
    agent turn.
    """

    def __init__(self, path: Path) -> None:
        """Remember the mind directory and derive the file paths."""
        self.path = path
        self.soul_path = path / "SOUL.md"
        self.memory_path = path / "MEMORY.md"
        self.inbox_path = path / "INBOX.md"
        self.diary_path = path / "DIARY.md"

    def ensure(self) -> None:
        """Create the directory and files; seed SOUL.md on first run.

        Idempotent: existing files (including an edited soul) are left
        untouched.
        """
        self.path.mkdir(parents=True, exist_ok=True)
        if not self.soul_path.exists():
            self.soul_path.write_text(DEFAULT_SOUL, encoding="utf-8")
        self.memory_path.touch(exist_ok=True)
        self.inbox_path.touch(exist_ok=True)
        self.diary_path.touch(exist_ok=True)

    def soul(self) -> str:
        """Return the current personality text."""
        return self.soul_path.read_text(encoding="utf-8").strip()

    def notes(self) -> str:
        """Return curated memory and inbox as one document ('' when empty).

        Non-empty parts appear under ``# Memory`` / ``# Inbox`` headers —
        the combined view the recall/summary tools read.
        """
        parts = []
        memory = self.memory_path.read_text(encoding="utf-8").strip()
        inbox = self.inbox_path.read_text(encoding="utf-8").strip()
        if memory:
            parts.append(f"# Memory\n{memory}")
        if inbox:
            parts.append(f"# Inbox\n{inbox}")
        return "\n\n".join(parts)

    def append_inbox(self, text: str, stamp: str) -> None:
        """Append one dated entry to INBOX.md."""
        with self.inbox_path.open("a", encoding="utf-8") as file:
            file.write(f"## [{stamp}]\n{text}\n\n")
