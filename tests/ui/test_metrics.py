from __future__ import annotations

from ui.components.metrics import datetime_text, local_date_text, time_text


def test_datetime_text_converts_utc_to_dhaka_time():
    assert datetime_text("2026-07-25T16:56:57+00:00") == "25 Jul 2026 · 10:56 PM BDT"
    assert local_date_text("2026-07-25T16:56:57+00:00") == "25 Jul 2026"
    assert time_text("2026-07-25T16:56:57+00:00") == "10:56 PM BDT"


def test_datetime_text_treats_naive_values_as_dhaka_time():
    assert datetime_text("2026-07-25 22:56") == "25 Jul 2026 · 10:56 PM BDT"


def test_datetime_text_handles_missing_values():
    assert datetime_text(None) == "—"
