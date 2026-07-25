from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime

import numpy as np
import pandas as pd

CANONICAL_COLUMNS = [
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trade_count",
    "source",
    "ingested_at",
]

ALIASES = {
    "DATE": "date",
    "TRADING CODE": "ticker",
    "TRADING_CODE": "ticker",
    "SCRIP": "ticker",
    "OPENP*": "open",
    "OPEN": "open",
    "HIGH": "high",
    "LOW": "low",
    "CLOSEP*": "close",
    "CLOSE": "close",
    "VOLUME": "volume",
    "TRADE": "trade_count",
    "TRADES": "trade_count",
    "TRADE COUNT": "trade_count",
    "NO. OF TRADES": "trade_count",
}


def normalize_column_name(value: object) -> str:
    if isinstance(value, tuple):
        value = " ".join(str(part) for part in value if str(part) != "nan")
    text = " ".join(str(value).strip().upper().replace("_", " ").split())
    return ALIASES.get(text, text.lower().replace(" ", "_"))


def normalize_price_frame(
    frame: pd.DataFrame,
    *,
    source: str,
    reject_invalid: bool = True,
    ingested_at: datetime | None = None,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    result = frame.copy()
    result.columns = [normalize_column_name(column) for column in result.columns]
    required = {"date", "ticker", "open", "high", "low", "close", "volume"}
    missing = sorted(required - set(result.columns))
    if missing:
        raise ValueError(f"Price data is missing required columns: {', '.join(missing)}")

    selected = list(required)
    if "trade_count" in result:
        selected.append("trade_count")
    result = result[selected].copy()
    if "trade_count" not in result:
        result["trade_count"] = np.nan

    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    result["ticker"] = (
        result["ticker"].astype("string").str.strip().str.upper().replace({"": pd.NA, "NAN": pd.NA})
    )
    numeric = ["open", "high", "low", "close", "volume", "trade_count"]
    for column in numeric:
        result[column] = pd.to_numeric(
            result[column].astype("string").str.replace(",", "", regex=False).str.strip(),
            errors="coerce",
        )

    required_valid = result["date"].notna() & result["ticker"].notna()
    required_valid &= result[["open", "high", "low", "close", "volume"]].notna().all(axis=1)
    nonnegative = (result[["open", "high", "low", "close", "volume"]] >= 0).all(axis=1)
    ohlc_consistent = (
        (result["low"] <= result["open"])
        & (result["low"] <= result["close"])
        & (result["high"] >= result["open"])
        & (result["high"] >= result["close"])
        & (result["high"] >= result["low"])
    )
    valid = required_valid & nonnegative & ohlc_consistent
    if reject_invalid:
        result = result.loc[valid].copy()
    else:
        result["row_valid"] = valid

    timestamp = ingested_at or datetime.now(UTC)
    result["source"] = source
    result["ingested_at"] = pd.Timestamp(timestamp)
    result = result.sort_values(["date", "ticker", "ingested_at"])
    result = result.drop_duplicates(["date", "ticker"], keep="last")
    return result[CANONICAL_COLUMNS].reset_index(drop=True)


def merge_price_frames(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    usable = [frame for frame in frames if frame is not None and not frame.empty]
    if not usable:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    combined = pd.concat(usable, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce").dt.normalize()
    combined["ingested_at"] = pd.to_datetime(combined["ingested_at"], errors="coerce", utc=True)
    combined = combined.sort_values(["date", "ticker", "ingested_at"])
    combined = combined.drop_duplicates(["date", "ticker"], keep="last")
    return combined[CANONICAL_COLUMNS].sort_values(["ticker", "date"]).reset_index(drop=True)
