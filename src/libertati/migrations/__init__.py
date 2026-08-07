"""Alembic wiring: programmatic upgrades to the newest schema revision.

Migration scripts ship as package data, so the bot can upgrade its
database on startup no matter where it is installed; the repo-root
``alembic.ini`` exists only for generating revisions during development.
"""

from pathlib import Path

from alembic import command
from alembic.config import Config


def build_config(db_path: Path) -> Config:
    """Build an Alembic config in code, no ini file required."""
    config = Config()
    config.set_main_option("script_location", "libertati:migrations")
    config.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    return config


def upgrade_to_head(db_path: Path) -> None:
    """Create or upgrade the SQLite database to the newest revision."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    command.upgrade(build_config(db_path), "head")
    # Full message payloads and model context are private even on a
    # multi-user host; do not leave the database world-readable.
    db_path.chmod(0o600)
