"""Tests for the clock helpers (formatting, parsing, timezone math)."""

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pytest

from libertati import clock

KYIV = ZoneInfo("Europe/Kyiv")


def test_format_local_converts_timezone() -> None:
    """A UTC datetime is rendered in the target timezone."""
    dt = datetime(2026, 8, 2, 12, 0, tzinfo=UTC)
    assert clock.format_local(dt, KYIV) == "Sun 2026-08-02 15:00"


def test_format_local_treats_naive_as_utc() -> None:
    """Naive datetimes are assumed to be UTC."""
    naive = datetime(2026, 8, 2, 12, 0)  # noqa: DTZ001 — naiveté is the point
    aware = datetime(2026, 8, 2, 12, 0, tzinfo=UTC)
    assert clock.format_local(naive, KYIV) == clock.format_local(aware, KYIV)


def test_parse_local_summer_offset() -> None:
    """Kyiv summer time (UTC+3) parses back to the right UTC instant."""
    due = clock.parse_local("2026-08-02 15:00", KYIV)
    assert due == datetime(2026, 8, 2, 12, 0, tzinfo=UTC)


def test_parse_local_winter_offset() -> None:
    """Kyiv winter time (UTC+2) parses back to the right UTC instant."""
    due = clock.parse_local("2026-01-15 15:00", KYIV)
    assert due == datetime(2026, 1, 15, 13, 0, tzinfo=UTC)


def test_parse_local_rejects_garbage() -> None:
    """Malformed input raises ValueError (surfaced as a tool error)."""
    with pytest.raises(ValueError):
        clock.parse_local("tomorrow", KYIV)


def test_utc_stamp_roundtrip() -> None:
    """utc_stamp and parse_utc_stamp are inverses (to the second)."""
    dt = datetime(2026, 8, 2, 12, 34, 56, tzinfo=UTC)
    assert clock.parse_utc_stamp(clock.utc_stamp(dt)) == dt


def test_utc_stamp_matches_sqlite_format() -> None:
    """The stamp uses SQLite's datetime('now') layout."""
    dt = datetime(2026, 8, 2, 3, 4, 5, tzinfo=KYIV)
    assert clock.utc_stamp(dt) == "2026-08-02 00:04:05"
