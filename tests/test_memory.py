"""Tests for the file-based mind (SOUL.md / MEMORY.md / DIARY.md)."""

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
    assert mind.diary_path.read_text(encoding="utf-8") == ""


def test_ensure_is_idempotent(tmp_path: Path) -> None:
    """Re-running ensure never overwrites edited files."""
    mind = make_mind(tmp_path)
    mind.soul_path.write_text("custom soul", encoding="utf-8")
    mind.append_memory("a fact", "Sun 2026-08-02 12:00")
    mind.ensure()
    assert mind.soul() == "custom soul"
    assert "a fact" in mind.memory()


def test_append_memory_format(tmp_path: Path) -> None:
    """Facts are appended as stamped bullet lines, in order."""
    mind = make_mind(tmp_path)
    mind.append_memory("first", "Sun 2026-08-02 12:00")
    mind.append_memory("second", "Sun 2026-08-02 13:00")
    assert mind.memory_path.read_text(encoding="utf-8") == (
        "- [Sun 2026-08-02 12:00] first\n- [Sun 2026-08-02 13:00] second\n"
    )


def test_memory_empty_is_falsy(tmp_path: Path) -> None:
    """An untouched memory file reads as an empty string."""
    assert make_mind(tmp_path).memory() == ""
