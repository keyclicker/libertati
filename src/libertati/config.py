"""Application settings loaded from the environment and config files."""

from pathlib import Path

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
    log_level: str = "INFO"
    db_path: Path = Path("data/libertati.db")
    # Timezone the agent lives in (event timestamps, wakeup scheduling).
    timezone: str = "UTC"
    # Minutes between heartbeat status events (0 disables, ±20% jitter).
    heartbeat_minutes: int = 180

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
