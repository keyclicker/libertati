"""Application configuration loaded from environment / .env."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

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
    bot_token: str = ""

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
    # On top of the reply thread, include a small window of recent chat messages so the
    # bot also sees what's happening *now*, not only the (possibly old) thread it replies to.
    recent_context_messages: int = 20
    recent_context_chars: int = 1200
    # Hard cap per memory file on disk; oldest lines are trimmed past this.
    memory_max_file_chars: int = 4000
    # Per-file budget when a memory file is injected into the prompt (keeps the tail).
    memory_context_file_chars: int = 700

    # --- Behaviour ----------------------------------------------------------
    respond_to_all: bool = True
    # In group chats, always reply when addressed (mentioned / replied-to); for other
    # ("ambient") group messages reply only with this small probability — it lurks like a
    # person and speaks up mostly when spoken to, not on every message.
    group_reply_chance: float = 0.05
    # Human-like pause before sending, scaled by reply length (seconds/char), capped.
    typing_delay_enabled: bool = True
    typing_delay_max_seconds: float = 5.0
    heartbeat_enabled: bool = True
    dream_enabled: bool = True
    news_refresh_hours: int = 6
    news_feeds: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: list(DEFAULT_NEWS_FEEDS)
    )

    # Chats the bot is allowed to talk in (ids and/or @usernames). Empty = everywhere.
    allowed_chats: Annotated[list[str], NoDecode] = Field(default_factory=list)

    # --- Browsing (reading public channels via the t.me/s/ web preview) -----
    browse_enabled: bool = False
    browse_channels: Annotated[list[str], NoDecode] = Field(default_factory=list)
    browse_times_per_day: int = 3
    browse_read_limit: int = 15

    # --- Persona ------------------------------------------------------------
    bot_full_name: str = "Ana Tati"
    bot_handle: str = "@libertati"
    bot_username: str = "@libertati_bot"

    # --- Logging / observability --------------------------------------------
    log_level: str = "INFO"
    event_log_enabled: bool = True
    event_log_path: Path = Path("data/events.jsonl")
    log_message_content: bool = True
    log_content_max_chars: int = 300

    @field_validator("news_feeds", "allowed_chats", "browse_channels", mode="before")
    @classmethod
    def _split_csv(cls, v: object) -> object:
        # allow comma-separated env value
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

    def is_chat_allowed(self, chat_id: int, username: str | None) -> bool:
        """Whether the bot may proactively talk in this chat."""
        if not self.allowed_chats:
            return True
        allowed = {entry.lstrip("@").lower() for entry in self.allowed_chats}
        if str(chat_id) in allowed:
            return True
        return bool(username and username.lstrip("@").lower() in allowed)

    def validate_runtime(self) -> None:
        """Assert the required settings are present."""
        if not self.openai_api_key:
            raise ValueError("LIBERTATI_OPENAI_API_KEY is required")
        if not self.bot_token:
            raise ValueError("LIBERTATI_BOT_TOKEN is required")


def load_settings() -> Settings:
    return Settings()
