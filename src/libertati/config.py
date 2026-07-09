"""Application configuration loaded from environment / .env."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

TelegramMode = Literal["bot", "account"]

DEFAULT_NEWS_FEEDS = [
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://www.pravda.com.ua/rss/",
    "https://hnrss.org/frontpage",
]


class Settings(BaseSettings):
    """Runtime settings. Validation errors fail fast at startup."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_prefix="LIBERTATI_",
    )

    # --- Telegram -----------------------------------------------------------
    telegram_mode: TelegramMode = "bot"
    bot_token: str = ""
    tg_api_id: int = 0
    tg_api_hash: str = ""
    tg_session: str = "libertati"  # session name (account mode)

    # --- OpenAI -------------------------------------------------------------
    openai_api_key: str = ""
    openai_base_url: str | None = None
    openai_model: str = "gpt-4o-mini"
    openai_temperature: float = 1.0
    openai_max_tool_iterations: int = 6

    # --- Storage ------------------------------------------------------------
    db_path: Path = Path("data/libertati.db")
    memory_dir: Path = Path("memory")
    max_thread_chars: int = 5000
    history_context_messages: int = 25

    # --- Behaviour ----------------------------------------------------------
    respond_to_all: bool = True
    heartbeat_enabled: bool = True
    dream_enabled: bool = True
    news_refresh_hours: int = 6
    news_feeds: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: list(DEFAULT_NEWS_FEEDS)
    )

    # --- Persona ------------------------------------------------------------
    bot_full_name: str = "Ana Tati"
    bot_handle: str = "@libertati"
    bot_username: str = "@libertati_bot"

    # --- Logging ------------------------------------------------------------
    log_level: str = "INFO"

    @field_validator("news_feeds", mode="before")
    @classmethod
    def _split_feeds(cls, v: object) -> object:
        # allow comma-separated env value
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

    def validate_runtime(self) -> None:
        """Assert the settings required for the chosen mode are present."""
        if not self.openai_api_key:
            raise ValueError("LIBERTATI_OPENAI_API_KEY is required")
        if self.telegram_mode == "bot":
            if not self.bot_token:
                raise ValueError("LIBERTATI_BOT_TOKEN is required in bot mode")
        elif self.telegram_mode == "account":
            if not (self.tg_api_id and self.tg_api_hash):
                raise ValueError(
                    "LIBERTATI_TG_API_ID and LIBERTATI_TG_API_HASH are required in account mode"
                )


def load_settings() -> Settings:
    return Settings()
