"""Tests for the file-based mind (SOUL.md / MEMORY.md / INBOX.md / DIARY.md)."""

from pathlib import Path

from libertati.memory import DEFAULT_SOUL, Mind


def make_mind(tmp_path: Path) -> Mind:
    """Build an ensured Mind in a temporary directory."""
    mind = Mind(tmp_path / "mind")
    mind.ensure()
    return mind


def test_ensure_creates_and_seeds(tmp_path: Path) -> None:
    """First run creates the directory, seeds SOUL.md, touches the rest."""
    mind = make_mind(tmp_path)
    assert mind.soul_path.read_text(encoding="utf-8") == DEFAULT_SOUL
    assert mind.memory_path.read_text(encoding="utf-8") == ""
    assert mind.inbox_path.read_text(encoding="utf-8") == ""
    assert mind.diary_path.read_text(encoding="utf-8") == ""


def test_ensure_is_idempotent(tmp_path: Path) -> None:
    """Re-running ensure never overwrites edited files."""
    mind = make_mind(tmp_path)
    mind.soul_path.write_text("custom soul", encoding="utf-8")
    mind.append_inbox("a fact", "Sun 2026-08-02 12:00")
    mind.ensure()
    assert mind.soul() == "custom soul"
    assert "a fact" in mind.notes()


def test_append_inbox_format(tmp_path: Path) -> None:
    """Entries are appended as dated markdown sections, in order."""
    mind = make_mind(tmp_path)
    mind.append_inbox("first", "Sun 2026-08-02 12:00")
    mind.append_inbox("second", "Sun 2026-08-02 13:00")
    assert mind.inbox_path.read_text(encoding="utf-8") == (
        "## [Sun 2026-08-02 12:00]\nfirst\n\n## [Sun 2026-08-02 13:00]\nsecond\n\n"
    )


def test_notes_combines_memory_and_inbox(tmp_path: Path) -> None:
    """notes() concatenates the non-empty files under section headers."""
    mind = make_mind(tmp_path)
    assert mind.notes() == ""
    mind.append_inbox("fresh fact", "Sun 2026-08-02 12:00")
    assert mind.notes() == "# Inbox\n## [Sun 2026-08-02 12:00]\nfresh fact"
    mind.memory_path.write_text("curated fact\n", encoding="utf-8")
    assert mind.notes() == (
        "# Memory\ncurated fact\n\n# Inbox\n## [Sun 2026-08-02 12:00]\nfresh fact"
    )
