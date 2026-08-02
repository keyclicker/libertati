"""The agent's file-based mind: soul, long-term memory and diary.

Three plain markdown files the user (and later the Dreaming loop) can
inspect and edit directly:

- ``SOUL.md`` — personality, attached to the instructions every turn.
- ``MEMORY.md`` — append-only fact store; written by the ``remember``
  tool, read only by the ``recall`` tool, never inlined into context.
- ``DIARY.md`` — reserved for the Dreaming loop; created empty.
"""

from pathlib import Path

#: Personality seed written to SOUL.md on first run.
DEFAULT_SOUL = """\
You are libertati, a person texting from their own Telegram account.
Text like a human: short, casual, no assistant-speak. Match the
sender's language.
Be proactive deliberately: start conversations rarely and only with a
real reason.
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
        self.diary_path.touch(exist_ok=True)

    def soul(self) -> str:
        """Return the current personality text."""
        return self.soul_path.read_text(encoding="utf-8").strip()

    def memory(self) -> str:
        """Return the full long-term memory text ('' when empty)."""
        return self.memory_path.read_text(encoding="utf-8").strip()

    def append_memory(self, text: str, stamp: str) -> None:
        """Append one stamped fact to MEMORY.md."""
        with self.memory_path.open("a", encoding="utf-8") as file:
            file.write(f"- [{stamp}] {text}\n")
