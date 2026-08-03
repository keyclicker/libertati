"""Application settings loaded from the environment and config files."""

from pathlib import Path
from typing import Literal

from pydantic import model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)


class Settings(BaseSettings):
    """Runtime configuration.

    Values are merged from several sources; the first one that provides a
    value wins: real environment variables (``LIBERTATI_*``), the ``.env``
    file (secrets), ``settings.toml`` (non-secret settings), then field
    defaults.

    ``settings.toml`` ships every non-secret setting explicitly and is the
    file to edit. The defaults below only keep the file optional (the
    secrets and ``model`` must then come from the environment); the two
    are grouped in the same order so they read side by side.
    """

    model_config = SettingsConfigDict(
        env_prefix="LIBERTATI_",
        env_file=".env",
        toml_file="settings.toml",
        extra="ignore",
    )

    # ==========================================================
    #                         Secrets
    # ==========================================================

    bot_token: str
    api_key: str

    # ==========================================================
    #                         Runtime
    # ==========================================================

    log_level: str = "INFO"
    # Timezone the agent lives in (event timestamps, wakeup scheduling).
    timezone: str = "UTC"

    # ==========================================================
    #                         Storage
    # ==========================================================

    db_path: Path = Path("data/libertati.db")
    # Directory holding SOUL.md / MEMORY.md / INBOX.md / DREAMS.md.
    memory_dir: Path = Path("data/memory")

    # ==========================================================
    #                     Model & reasoning
    # ==========================================================

    # Any OpenAI-compatible endpoint (OpenRouter, local, …); None = OpenAI.
    base_url: str | None = None
    model: str
    # For reasoning models: none/minimal/low/medium/high/xhigh/max
    # (model-dependent); None sends no preference.
    reasoning_effort: str | None = None
    # Which prior reasoning OpenAI may render into a new sample. "omit"
    # sends no context parameter for providers that do not support it.
    reasoning_context: Literal["auto", "current_turn", "all_turns", "omit"] = (
        "current_turn"
    )
    # Remove reasoning and assistant output envelopes from the live
    # window after each turn. Full history remains in SQLite; enabling
    # this can reduce prompt-cache reuse for the completed-turn suffix.
    prune_completed_reasoning: bool = False
    # Model for the one-shot recall extraction behind `recall` and
    # `summarize_memory`; None = main model.
    recall_model: str | None = None
    # Reasoning effort for that extraction call. It is a lookup, not a
    # problem to think about, so "none" is the sensible setting where the
    # model supports it; None sends no preference.
    recall_reasoning_effort: str | None = None
    # Model used while dreaming; None = main model.
    dream_model: str | None = None

    # ==========================================================
    #                      Context window
    # ==========================================================

    # Cut back to context_trim_items once the window grows past
    # context_max_items. Trimming in chunks keeps the prompt prefix
    # byte-stable between trims instead of rewriting it on every append.
    context_max_items: int = 300
    context_trim_items: int = 200
    # Max model/tool rounds per agent turn (one turn per event batch).
    max_rounds: int = 8

    # ==========================================================
    #                         Persona
    # ==========================================================

    # User-editable agent and memory prompts.
    prompts_path: Path = Path("prompts.toml")
    # Harden the persona: never break character, never admit to being a
    # bot/assistant, never discuss internals, no free assistant labor.
    roleplay: bool = False
    # Enable OpenAI's built-in web search tool (server-side; most
    # OpenAI-compatible endpoints don't support it).
    web_search: bool = False

    # ==========================================================
    #                    Chats & presence
    # ==========================================================

    # Approval mode: when on, only chats marked true in chats_path reach
    # the agent; new chats are appended there as false for review.
    chat_approval: bool = True
    # Chat approval registry (user-editable TOML, re-read live).
    chats_path: Path = Path("data/chats.toml")
    # Minutes between heartbeat status events (0 disables, ±20% jitter).
    heartbeat_minutes: int = 180
    # Simulated typing speed for outgoing messages, chars/second
    # (0 disables the typing emulation).
    typing_chars_per_second: float = 15.0

    # ==========================================================
    #                         Dreaming
    # ==========================================================

    # Dreams allowed in a rolling 24 hours (0 disables dreaming, which
    # also hides the `dream` tool from the waking agent).
    dream_daily_budget: int = 4
    # Quiet minutes before the agent falls asleep on its own. Heartbeat
    # turns count as quiet unless the agent reached out during one.
    dream_idle_minutes: int = 300
    # Minimum gap between the end of one dream and the start of the next.
    dream_cooldown_minutes: int = 120
    # Tool calls a dream must take (wake_up included) before wake_up is
    # accepted — a dream is meant to wander, not to tidy up and leave.
    dream_min_steps: int = 12
    # Hard cap on model/tool rounds in one dream.
    dream_max_rounds: int = 25

    @model_validator(mode="after")
    def _validate_context_window(self) -> "Settings":
        """Reject inconsistent context-management settings."""
        if not 0 < self.context_trim_items < self.context_max_items:
            raise ValueError(
                "context_trim_items must be positive and smaller than context_max_items"
            )
        if self.prune_completed_reasoning and self.reasoning_context != "current_turn":
            raise ValueError(
                "prune_completed_reasoning requires reasoning_context='current_turn'"
            )
        return self

    @model_validator(mode="after")
    def _validate_dreaming(self) -> "Settings":
        """Reject dream settings that could never produce a dream."""
        if self.dream_daily_budget < 0:
            raise ValueError("dream_daily_budget must not be negative")
        if self.dream_max_rounds < 1:
            raise ValueError("dream_max_rounds must be positive")
        if self.dream_min_steps > self.dream_max_rounds:
            raise ValueError(
                "dream_min_steps must not exceed dream_max_rounds,"
                " or no dream could ever wake up on its own"
            )
        return self

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Insert the TOML source below env/.env in the priority order."""
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            TomlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )
