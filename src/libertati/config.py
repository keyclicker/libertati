"""Application settings loaded from the environment and config files."""

from pathlib import Path

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
    """

    model_config = SettingsConfigDict(
        env_prefix="LIBERTATI_",
        env_file=".env",
        toml_file="settings.toml",
        extra="ignore",
    )

    bot_token: str
    api_key: str
    # Any OpenAI-compatible endpoint (OpenRouter, local, …); None = OpenAI.
    base_url: str | None = None
    model: str
    # For reasoning models: none/minimal/low/medium/high/xhigh/max
    # (model-dependent); None sends no preference.
    reasoning_effort: str | None = None
    # Persona/system prompt override; None uses the built-in default.
    system_prompt: str | None = None
    # Harden the persona: never break character, never admit to being a
    # bot/assistant, never discuss internals, no free assistant labor.
    roleplay: bool = False
    # Model for the one-shot recall extraction; None = main model.
    recall_model: str | None = None
    # Enable OpenAI's built-in web search tool (server-side; most
    # OpenAI-compatible endpoints don't support it).
    web_search: bool = False
    log_level: str = "INFO"
    db_path: Path = Path("data/libertati.db")
    # Directory holding SOUL.md / MEMORY.md / DIARY.md.
    memory_dir: Path = Path("data/memory")
    # Approval mode: when on, only chats marked true in chats_path reach
    # the agent; new chats are appended there as false for review.
    chat_approval: bool = False
    # Chat approval registry (user-editable TOML, re-read live).
    chats_path: Path = Path("data/chats.toml")
    # Max model/tool rounds per agent turn (one turn per event batch).
    max_rounds: int = 8
    # Context window: cut back to context_trim_items once it grows past
    # context_max_items. Trimming in chunks keeps the prompt prefix
    # byte-stable between trims, so OpenAI prompt caching keeps hitting.
    context_max_items: int = 300
    context_trim_items: int = 200
    # Timezone the agent lives in (event timestamps, wakeup scheduling).
    timezone: str = "UTC"
    # Minutes between heartbeat status events (0 disables, ±20% jitter).
    heartbeat_minutes: int = 180
    # Simulated typing speed for outgoing messages, chars/second
    # (0 disables the typing emulation).
    typing_chars_per_second: float = 15.0

    @model_validator(mode="after")
    def _validate_context_window(self) -> "Settings":
        """Reject window sizes where trimming could never fire."""
        if not 0 < self.context_trim_items < self.context_max_items:
            raise ValueError(
                "context_trim_items must be positive and smaller than context_max_items"
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
