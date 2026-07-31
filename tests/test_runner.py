from __future__ import annotations

from datetime import datetime

from libertati.scheduler.runner import _slot_time


def test_slot_times_stay_inside_window_and_ordered():
    now = datetime(2026, 7, 31, 0, 0)
    for total in (1, 2, 3, 5):
        times = [_slot_time(now, i, total, start_hour=9, end_hour=22) for i in range(total)]
        assert all(9 <= t.hour <= 22 for t in times)
        assert times == sorted(times)  # equal sub-windows keep slots in order


def test_slot_time_single_dream_window():
    now = datetime(2026, 7, 31, 0, 0)
    t = _slot_time(now, 0, 1, start_hour=3, end_hour=7)
    assert 3 <= t.hour < 7
