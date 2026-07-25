from __future__ import annotations

from typing import Any

import pandas as pd

DISPLAY_TIMEZONE = "Asia/Dhaka"


def _localized_timestamp(value: Any, timezone: str) -> pd.Timestamp | None:
    if value is None:
        return None
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        return None
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(timezone)
    return timestamp.tz_convert(timezone)


def percent(value: Any, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value) * 100:.{digits}f}%"


def number(value: Any, digits: int = 0) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):,.{digits}f}"


def money(value: Any) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"৳{float(value):,.2f}"


def date_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return "—"
    timestamp = pd.Timestamp(value)
    return timestamp.strftime("%d %b %Y")


def datetime_text(value: Any, timezone: str = DISPLAY_TIMEZONE) -> str:
    if not value:
        return "—"
    try:
        timestamp = _localized_timestamp(value, timezone)
        if timestamp is None:
            return "—"
        return timestamp.strftime("%d %b %Y · %I:%M %p BDT")
    except (TypeError, ValueError, KeyError):
        return str(value)


def local_date_text(value: Any, timezone: str = DISPLAY_TIMEZONE) -> str:
    try:
        timestamp = _localized_timestamp(value, timezone)
        return timestamp.strftime("%d %b %Y") if timestamp is not None else "—"
    except (TypeError, ValueError, KeyError):
        return str(value)


def time_text(value: Any, timezone: str = DISPLAY_TIMEZONE) -> str:
    try:
        timestamp = _localized_timestamp(value, timezone)
        return timestamp.strftime("%I:%M %p BDT") if timestamp is not None else "—"
    except (TypeError, ValueError, KeyError):
        return str(value)
