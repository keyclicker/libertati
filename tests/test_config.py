"""Tests for settings validation."""

from pathlib import Path
from typing import Literal

import pytest
from pydantic import ValidationError

from libertati.config import Settings


def make_settings(
    max_items: int = 300,
    trim_items: int = 200,
    reasoning_context: Literal[
        "auto", "current_turn", "all_turns", "omit"
    ] = "current_turn",
    prune_completed_reasoning: bool = False,
) -> Settings:
    """Build settings with required secrets and given window sizes."""
    return Settings(
        bot_token="t",
        api_key="k",
        model="m",
        reasoning_context=reasoning_context,
        prune_completed_reasoning=prune_completed_reasoning,
        context_max_items=max_items,
        context_trim_items=trim_items,
    )


def test_context_window_defaults_valid() -> None:
    """The default window sizes pass validation."""
    settings = make_settings()
    assert settings.context_trim_items < settings.context_max_items
    assert settings.reasoning_context == "current_turn"
    assert settings.prune_completed_reasoning is False
    assert settings.prompts_path == Path("prompts.toml")
    assert settings.chat_approval is True


def test_context_trim_must_be_below_max() -> None:
    """A trim size at or above the max is rejected."""
    with pytest.raises(ValidationError):
        make_settings(max_items=200, trim_items=300)
    with pytest.raises(ValidationError):
        make_settings(max_items=200, trim_items=200)


def test_context_trim_must_be_positive() -> None:
    """A zero or negative trim size is rejected."""
    with pytest.raises(ValidationError):
        make_settings(trim_items=0)


def test_pruning_requires_current_turn_reasoning() -> None:
    """Cross-turn and provider-default reasoning cannot be pruned."""
    with pytest.raises(ValidationError):
        make_settings(
            reasoning_context="all_turns",
            prune_completed_reasoning=True,
        )
    with pytest.raises(ValidationError):
        make_settings(reasoning_context="auto", prune_completed_reasoning=True)
    with pytest.raises(ValidationError):
        make_settings(reasoning_context="omit", prune_completed_reasoning=True)


def test_media_models_are_named_and_share_the_one_endpoint() -> None:
    """The shipped file describes media, over base_url like everything."""
    settings = make_settings()
    assert settings.media_model
    assert settings.transcribe_model
    assert settings.media_dir == Path("data/media")
    # There is no second route or key to get them wrong: whatever is
    # named here has to be a model the configured provider serves.
    assert not hasattr(settings, "media_base_url")
    assert not hasattr(settings, "media_api_key")


def test_media_is_off_until_a_model_is_named() -> None:
    """Leaving the model unset is what turns the whole path off."""
    settings = Settings(bot_token="t", api_key="k", model="m", media_model=None)
    assert settings.media_model is None


def test_media_frames_and_note_length_must_be_positive() -> None:
    """Zero frames or a zero-length note would describe nothing."""
    with pytest.raises(ValidationError):
        Settings(bot_token="t", api_key="k", model="m", media_max_frames=0)
    with pytest.raises(ValidationError):
        Settings(bot_token="t", api_key="k", model="m", media_note_chars=0)
    with pytest.raises(ValidationError):
        Settings(bot_token="t", api_key="k", model="m", media_wait_seconds=-1)


def test_dream_defaults_valid() -> None:
    """The shipped dream knobs pass validation."""
    settings = make_settings()
    assert settings.dream_daily_budget > 0
    assert settings.dream_min_steps < settings.dream_max_rounds


def test_dreaming_can_be_disabled_with_a_zero_budget() -> None:
    """A zero budget is the off switch, not an invalid setting."""
    settings = Settings(bot_token="t", api_key="k", model="m", dream_daily_budget=0)
    assert settings.dream_daily_budget == 0


def test_negative_dream_budget_is_rejected() -> None:
    """Only zero disables dreaming; below that is a mistake."""
    with pytest.raises(ValidationError):
        Settings(bot_token="t", api_key="k", model="m", dream_daily_budget=-1)


def test_dream_min_steps_above_max_rounds_is_rejected() -> None:
    """A dream that could never satisfy wake_up would never wake up."""
    with pytest.raises(ValidationError):
        Settings(
            bot_token="t",
            api_key="k",
            model="m",
            dream_min_steps=30,
            dream_max_rounds=25,
        )


def test_dream_max_rounds_must_be_positive() -> None:
    """A dream with no rounds is not a dream."""
    with pytest.raises(ValidationError):
        Settings(
            bot_token="t",
            api_key="k",
            model="m",
            dream_min_steps=0,
            dream_max_rounds=0,
        )


def test_cross_turn_reasoning_allowed_when_pruning_disabled() -> None:
    """Operators can retain complete output history for all-turns use."""
    settings = make_settings(
        reasoning_context="all_turns",
    )
    assert settings.reasoning_context == "all_turns"
