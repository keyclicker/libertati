"""The agent's file-based mind: soul, long-term memory, inbox and dreams.

Plain markdown files the user (and the dreaming loop) can inspect and
edit directly:

- ``SOUL.md`` — personality, attached to the instructions every turn.
  Rewritten only while dreaming, and only after the previous text is
  snapshotted under ``soul/``.
- ``MEMORY.md`` — curated long-term memory; rewritten only while
  dreaming, read by the ``recall``/``summarize_memory`` tools, never
  inlined into context.
- ``INBOX.md`` — landing zone for new memories; the ``remember`` tool
  appends dated entries here, dreaming folds them into MEMORY.md and
  clears the file.
- ``DREAMS.md`` — the dream journal: one dated reflection per dream.
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

#: Largest SOUL.md a dream may write, in characters. The soul rides in
#: the instructions of every single turn, so it has to stay small.
SOUL_MAX_CHARS = 4000

#: Largest MEMORY.md a dream may write, in characters. Consolidation
#: that only ever grows the file is not consolidation.
MEMORY_MAX_CHARS = 20000

#: Mind files addressable by name from the dreaming loop.
MIND_FILES = ("soul", "memory", "inbox", "dreams")


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
        self.dreams_path = path / "DREAMS.md"
        self.soul_dir = path / "soul"

    def ensure(self) -> None:
        """Create the directory and files; seed SOUL.md on first run.

        Idempotent: existing files (including an edited soul) are left
        untouched. A ``DIARY.md`` left by an older version is renamed —
        it was reserved for exactly this journal.
        """
        self.path.mkdir(parents=True, exist_ok=True)
        legacy_diary = self.path / "DIARY.md"
        if legacy_diary.exists() and not self.dreams_path.exists():
            legacy_diary.rename(self.dreams_path)
        if not self.soul_path.exists():
            self.soul_path.write_text(DEFAULT_SOUL, encoding="utf-8")
        self.memory_path.touch(exist_ok=True)
        self.inbox_path.touch(exist_ok=True)
        self.dreams_path.touch(exist_ok=True)

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

    def _file(self, name: str) -> Path:
        """Resolve one of :data:`MIND_FILES` to its path."""
        paths = {
            "soul": self.soul_path,
            "memory": self.memory_path,
            "inbox": self.inbox_path,
            "dreams": self.dreams_path,
        }
        try:
            return paths[name]
        except KeyError:
            raise ValueError(
                f"unknown mind file {name!r}; expected one of {', '.join(MIND_FILES)}"
            ) from None

    def read(self, name: str) -> str:
        """Return one mind file's text ('' when empty)."""
        return self._file(name).read_text(encoding="utf-8").strip()

    def append_dreams(self, text: str, stamp: str) -> None:
        """Append one dated reflection to DREAMS.md."""
        with self.dreams_path.open("a", encoding="utf-8") as file:
            file.write(f"## [{stamp}]\n{text}\n\n")

    def dreams_tail(self, limit: int) -> str:
        """Return at most ``limit`` trailing characters of the journal.

        The journal grows forever; only the recent past is worth handing
        to a dream as context.
        """
        text = self.read("dreams")
        if len(text) <= limit:
            return text
        return "[…earlier entries elided…]\n" + text[-limit:]

    def fold_inbox(self, memory: str) -> None:
        """Replace MEMORY.md and clear INBOX.md in one step.

        Deliberately one operation: a dream that died between a separate
        write and clear would drop every unfolded memory on the floor.
        Raises when the new memory exceeds :data:`MEMORY_MAX_CHARS`, in
        which case neither file is touched.
        """
        if len(memory) > MEMORY_MAX_CHARS:
            raise ValueError(
                f"memory is {len(memory)} chars, over the "
                f"{MEMORY_MAX_CHARS} limit — consolidate harder"
            )
        self.memory_path.write_text(memory.strip() + "\n", encoding="utf-8")
        self.inbox_path.write_text("", encoding="utf-8")

    def write_soul(self, text: str, stamp: str) -> Path:
        """Snapshot the current soul, then replace it; return the snapshot.

        Every revision is kept under ``soul/`` so a dream that drifts the
        personality somewhere bad stays a file move away from rollback.
        Raises (touching nothing) on empty or oversized text.
        """
        if not text.strip():
            raise ValueError("the soul must not be empty")
        if len(text) > SOUL_MAX_CHARS:
            raise ValueError(
                f"soul is {len(text)} chars, over the {SOUL_MAX_CHARS} limit"
            )
        self.soul_dir.mkdir(parents=True, exist_ok=True)
        snapshot = self.soul_dir / f"{stamp}.md"
        snapshot.write_text(
            self.soul_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
        self.soul_path.write_text(text.strip() + "\n", encoding="utf-8")
        return snapshot
