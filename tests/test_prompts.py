"""Tests for user-editable prompt loading."""

from pathlib import Path

import pytest

from libertati.prompts import load_prompts


def test_repository_prompts_load() -> None:
    """Checked-in prompt file defines every runtime prompt."""
    prompts = load_prompts(Path("prompts.toml"))

    assert "schedule_wakeup" in prompts.system
    assert "Heartbeats" in prompts.system
    assert "Never break character" in prompts.roleplay
    assert "built-in web search" in prompts.web_search
    assert "long-term memory" in prompts.recall
    assert "general overview" in prompts.summary


def test_missing_prompt_is_rejected(tmp_path: Path) -> None:
    """Partial prompt files fail with a useful field name."""
    path = tmp_path / "prompts.toml"
    path.write_text('[agent]\nsystem = "hello"\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"\[agent\]\.roleplay"):
        load_prompts(path)
