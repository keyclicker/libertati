"""Tests for the file-based mind (SOUL.md / MEMORY.md / INBOX.md / DREAMS.md)."""

from pathlib import Path

import pytest

from libertati.memory import (
    DEFAULT_SOUL,
    MEMORY_MAX_CHARS,
    SOUL_MAX_CHARS,
    Mind,
)


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
    assert mind.dreams_path.read_text(encoding="utf-8") == ""


def test_ensure_is_idempotent(tmp_path: Path) -> None:
    """Re-running ensure never overwrites edited files."""
    mind = make_mind(tmp_path)
    mind.soul_path.write_text("custom soul", encoding="utf-8")
    mind.append_inbox("a fact", "Sun 2026-08-02 12:00")
    mind.ensure()
    assert mind.soul() == "custom soul"
    assert "a fact" in mind.notes()


def test_ensure_migrates_legacy_diary(tmp_path: Path) -> None:
    """A DIARY.md from before the rename becomes the dream journal."""
    mind = Mind(tmp_path / "mind")
    mind.path.mkdir(parents=True)
    (mind.path / "DIARY.md").write_text("old entry\n", encoding="utf-8")

    mind.ensure()

    assert not (mind.path / "DIARY.md").exists()
    assert mind.read("dreams") == "old entry"


def test_ensure_keeps_existing_dreams_over_legacy_diary(tmp_path: Path) -> None:
    """A real journal is never clobbered by a stale DIARY.md."""
    mind = Mind(tmp_path / "mind")
    mind.path.mkdir(parents=True)
    (mind.path / "DIARY.md").write_text("old entry\n", encoding="utf-8")
    mind.dreams_path.write_text("real journal\n", encoding="utf-8")

    mind.ensure()

    assert mind.read("dreams") == "real journal"


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


def test_append_dreams_format(tmp_path: Path) -> None:
    """Journal entries mirror the inbox's dated-section shape."""
    mind = make_mind(tmp_path)
    mind.append_dreams("slept well", "Sun 2026-08-02 04:00")
    assert mind.dreams_path.read_text(encoding="utf-8") == (
        "## [Sun 2026-08-02 04:00]\nslept well\n\n"
    )


def test_read_rejects_unknown_file(tmp_path: Path) -> None:
    """An unknown mind file name names the valid ones."""
    mind = make_mind(tmp_path)
    with pytest.raises(ValueError, match="unknown mind file"):
        mind.read("secrets")


def test_dreams_tail_keeps_the_recent_end(tmp_path: Path) -> None:
    """Only the trailing slice of a long journal is handed to a dream."""
    mind = make_mind(tmp_path)
    mind.dreams_path.write_text("x" * 100 + "recent", encoding="utf-8")

    tail = mind.dreams_tail(20)

    assert tail.endswith("recent")
    assert tail.startswith("[…earlier entries elided…]")
    assert mind.dreams_tail(1000) == "x" * 100 + "recent"


def test_fold_inbox_rewrites_memory_and_clears_inbox(tmp_path: Path) -> None:
    """Consolidation replaces memory and empties the inbox in one step."""
    mind = make_mind(tmp_path)
    mind.memory_path.write_text("stale\n", encoding="utf-8")
    mind.append_inbox("fresh fact", "Sun 2026-08-02 12:00")

    mind.fold_inbox("consolidated")

    assert mind.read("memory") == "consolidated"
    assert mind.read("inbox") == ""


def test_fold_inbox_over_cap_touches_nothing(tmp_path: Path) -> None:
    """An oversized rewrite is refused before either file is written."""
    mind = make_mind(tmp_path)
    mind.memory_path.write_text("keep me\n", encoding="utf-8")
    mind.append_inbox("fresh fact", "Sun 2026-08-02 12:00")

    with pytest.raises(ValueError, match="over the"):
        mind.fold_inbox("x" * (MEMORY_MAX_CHARS + 1))

    assert mind.read("memory") == "keep me"
    assert "fresh fact" in mind.read("inbox")


def test_write_soul_snapshots_the_previous_version(tmp_path: Path) -> None:
    """The old soul survives as a dated file next to the new one."""
    mind = make_mind(tmp_path)

    snapshot = mind.write_soul("a new person", "20260802T040000Z")

    assert mind.soul() == "a new person"
    assert snapshot.read_text(encoding="utf-8") == DEFAULT_SOUL
    assert snapshot.parent == mind.soul_dir


def test_write_soul_rejects_oversized_and_empty(tmp_path: Path) -> None:
    """Neither an empty nor a bloated soul replaces the current one."""
    mind = make_mind(tmp_path)

    with pytest.raises(ValueError, match="over the"):
        mind.write_soul("x" * (SOUL_MAX_CHARS + 1), "20260802T040000Z")
    with pytest.raises(ValueError, match="must not be empty"):
        mind.write_soul("   ", "20260802T040000Z")

    assert mind.soul_path.read_text(encoding="utf-8") == DEFAULT_SOUL
    assert not mind.soul_dir.exists()
