"""The agent's file-based mind: soul, habits, long-term memory and more.

Plain markdown files the user (and the dreaming loop) can inspect and
edit directly:

- ``SOUL.md`` — personality, attached to the instructions every turn.
  Rewritten only while dreaming, and only after the previous text is
  snapshotted under ``soul/``.
- ``HABITS.md`` — behaviour learned from experience, attached to the
  instructions every turn next to the soul. Rewritten only while
  dreaming, and unlike the soul, most dreams touch it.
- ``MEMORY.md`` — curated long-term memory; rewritten only while
  dreaming, read by the ``recall``/``summarize_memory`` tools, never
  inlined into context.
- ``INBOX.md`` — landing zone for new memories; the ``remember`` tool
  appends dated entries here, dreaming folds them into MEMORY.md and
  clears the file.
- ``DREAMS.md`` — the dream journal: one dated reflection per dream.
"""

from pathlib import Path
from tempfile import NamedTemporaryFile

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

#: Largest HABITS.md a dream may write, in characters. It rides in
#: every turn's instructions too, and a habit list nobody prunes stops
#: being a list of habits.
HABITS_MAX_CHARS = 2000

#: Largest MEMORY.md a dream may write, in characters. Consolidation
#: that only ever grows the file is not consolidation.
MEMORY_MAX_CHARS = 20000

#: Most of DREAMS.md ``read_mind`` hands back, in characters. The
#: journal is the one mind file nothing caps on write — it only grows —
#: so the cap sits on the read instead, and a dream that asks for it
#: mid-session cannot bury the rest of that session in old entries.
DREAMS_READ_CHARS = 20000

#: Mind files addressable by name from the dreaming loop.
MIND_FILES = ("soul", "habits", "memory", "inbox", "dreams")


def _atomic_write(path: Path, text: str) -> None:
    """Replace one text file without exposing a truncated intermediate."""
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, delete=False
        ) as file:
            temporary = Path(file.name)
            file.write(text)
        temporary.replace(path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


class Mind:
    """Owns the mind files and all reads/writes to them.

    I/O is synchronous — the files are tiny and read at most once per
    agent turn.
    """

    def __init__(self, path: Path) -> None:
        """Remember the mind directory and derive the file paths."""
        self.path = path
        self.soul_path = path / "SOUL.md"
        self.habits_path = path / "HABITS.md"
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
        self.path.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path.chmod(0o700)
        legacy_diary = self.path / "DIARY.md"
        if legacy_diary.exists() and not self.dreams_path.exists():
            legacy_diary.rename(self.dreams_path)
        if not self.soul_path.exists():
            _atomic_write(self.soul_path, DEFAULT_SOUL)
        for file in (
            self.soul_path,
            self.habits_path,
            self.memory_path,
            self.inbox_path,
            self.dreams_path,
        ):
            file.touch(mode=0o600, exist_ok=True)
            file.chmod(0o600)

    def soul(self) -> str:
        """Return the current personality text."""
        return self.soul_path.read_text(encoding="utf-8").strip()

    def habits(self) -> str:
        """Return the learned-behaviour text ('' when nothing is learned)."""
        return self.habits_path.read_text(encoding="utf-8").strip()

    def resident(self) -> str:
        """Return the mind text that rides in every turn's instructions.

        Soul and habits under their own headers; an empty habits file is
        left out entirely rather than shown as a bare header, which reads
        to the model as a section it should fill.
        """
        parts = [f"## Soul\n{self.soul()}"]
        habits = self.habits()
        if habits:
            parts.append(f"## Habits\n{habits}")
        return "\n\n".join(parts)

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
            "habits": self.habits_path,
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
        """Replace MEMORY.md, then clear INBOX.md safely.

        Each replacement is atomic. A crash after MEMORY.md changes but
        before INBOX.md clears leaves duplicate pending facts for a later
        fold, never a truncated file. Oversized input touches neither file.
        """
        if len(memory) > MEMORY_MAX_CHARS:
            raise ValueError(
                f"memory is {len(memory)} chars, over the "
                f"{MEMORY_MAX_CHARS} limit — consolidate harder"
            )
        _atomic_write(self.memory_path, memory.strip() + "\n")
        _atomic_write(self.inbox_path, "")

    def write_habits(self, text: str) -> None:
        """Replace HABITS.md atomically; raise on oversized text.

        Emptying the file is allowed: a dream that decides nothing here
        was ever a habit should be able to say so. No snapshots, same as
        :meth:`fold_inbox` — the text a dream wrote is already in its
        ``dream_context`` trace, and unlike the soul this changes most
        nights, so a snapshot dir would be churn.
        """
        if len(text) > HABITS_MAX_CHARS:
            raise ValueError(
                f"habits are {len(text)} chars, over the "
                f"{HABITS_MAX_CHARS} limit — keep only what you act on"
            )
        stripped = text.strip()
        _atomic_write(self.habits_path, f"{stripped}\n" if stripped else "")

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
        self.soul_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.soul_dir.chmod(0o700)
        snapshot = self.soul_dir / f"{stamp}.md"
        suffix = 1
        while snapshot.exists():
            snapshot = self.soul_dir / f"{stamp}-{suffix}.md"
            suffix += 1
        _atomic_write(snapshot, self.soul_path.read_text(encoding="utf-8"))
        _atomic_write(self.soul_path, text.strip() + "\n")
        return snapshot
