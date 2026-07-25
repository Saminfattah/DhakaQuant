from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from dse_quant.processing.cleaning import CANONICAL_COLUMNS


@dataclass(frozen=True)
class ValidationReport:
    rows: int
    tickers: int
    first_date: str | None
    last_date: str | None
    duplicate_keys: int
    missing_required: int
    negative_values: int
    inconsistent_ohlc: int
    recent_thin_sessions: list[str]
    valid: bool

    def to_dict(self) -> dict:
        return asdict(self)


def validate_daily_prices(
    frame: pd.DataFrame, minimum_rows_per_recent_session: int = 50
) -> ValidationReport:
    missing_columns = [column for column in CANONICAL_COLUMNS if column not in frame]
    if missing_columns:
        raise ValueError(f"Canonical dataset is missing: {', '.join(missing_columns)}")

    duplicate_keys = int(frame.duplicated(["date", "ticker"]).sum())
    required = ["date", "ticker", "open", "high", "low", "close", "volume"]
    missing_required = int(frame[required].isna().any(axis=1).sum())
    price_columns = ["open", "high", "low", "close", "volume"]
    negative_values = int((frame[price_columns] < 0).any(axis=1).sum())
    inconsistent = (
        (frame["low"] > frame["open"])
        | (frame["low"] > frame["close"])
        | (frame["high"] < frame["open"])
        | (frame["high"] < frame["close"])
        | (frame["high"] < frame["low"])
    )
    recent_dates = sorted(pd.to_datetime(frame["date"]).dropna().unique())[-10:]
    counts = frame[frame["date"].isin(recent_dates)].groupby("date").size()
    thin = [str(pd.Timestamp(date).date()) for date, count in counts.items() if count < minimum_rows_per_recent_session]
    valid = duplicate_keys == 0 and missing_required == 0 and negative_values == 0 and not inconsistent.any()
    dates = pd.to_datetime(frame["date"], errors="coerce")
    return ValidationReport(
        rows=len(frame),
        tickers=int(frame["ticker"].nunique()),
        first_date=str(dates.min().date()) if dates.notna().any() else None,
        last_date=str(dates.max().date()) if dates.notna().any() else None,
        duplicate_keys=duplicate_keys,
        missing_required=missing_required,
        negative_values=negative_values,
        inconsistent_ohlc=int(np.asarray(inconsistent).sum()),
        recent_thin_sessions=thin,
        valid=bool(valid),
    )

