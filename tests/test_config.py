"""Tests for settings validation."""

import pytest
from pydantic import ValidationError

from libertati.config import Settings


def make_settings(max_items: int = 300, trim_items: int = 200) -> Settings:
    """Build settings with required secrets and given window sizes."""
    return Settings(
        bot_token="t",
        api_key="k",
        model="m",
        context_max_items=max_items,
        context_trim_items=trim_items,
    )


def test_context_window_defaults_valid() -> None:
    """The default window sizes pass validation."""
    settings = make_settings()
    assert settings.context_trim_items < settings.context_max_items


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
