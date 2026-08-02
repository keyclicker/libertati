"""Time helpers: the agent perceives time in one configured timezone."""

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

#: The wire format shared with SQLite's ``datetime('now')``.
UTC_STAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


def format_local(dt: datetime, tz: ZoneInfo) -> str:
    """Format a datetime for the agent, e.g. ``Sun 2026-08-02 14:03``."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(tz).strftime("%a %Y-%m-%d %H:%M")


def format_now(tz: ZoneInfo) -> str:
    """Current time formatted for the agent (a display string)."""
    return format_local(datetime.now(UTC), tz)


def parse_local(text: str, tz: ZoneInfo) -> datetime:
    """Parse ``YYYY-MM-DD HH:MM`` given in the agent's timezone, to UTC."""
    return datetime.strptime(text, "%Y-%m-%d %H:%M").replace(tzinfo=tz).astimezone(UTC)


def utc_stamp(dt: datetime) -> str:
    """Format a UTC datetime like SQLite's ``datetime('now')`` does."""
    return dt.astimezone(UTC).strftime(UTC_STAMP_FORMAT)


def file_stamp(dt: datetime) -> str:
    """Format a UTC datetime for use inside a filename."""
    return dt.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")


def parse_utc_stamp(text: str) -> datetime:
    """Parse a :func:`utc_stamp` / SQLite ``datetime('now')`` string."""
    return datetime.strptime(text, UTC_STAMP_FORMAT).replace(tzinfo=UTC)
