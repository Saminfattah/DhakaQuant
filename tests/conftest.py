from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def raw_prices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "DATE": ["2026-01-01", "2026-01-02", "bad-date", "2026-01-02"],
            "TRADING CODE": ["ACI", "ACI", "ACI", "ACI"],
            "OPENP*": ["100", "101", "5", "101"],
            "HIGH": ["102", "103", "4", "104"],
            "LOW": ["99", "100", "6", "100"],
            "CLOSEP*": ["101", "102", "5", "103"],
            "VOLUME": ["10,000", "12,000", "-1", "13,000"],
        }
    )


@pytest.fixture
def canonical_prices() -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-01", periods=80)
    rows = []
    for ticker, offset in (("ACI", 0), ("BATBC", 20)):
        for index, day in enumerate(dates):
            close = 100 + offset + index * 0.4
            rows.append(
                {
                    "date": day,
                    "ticker": ticker,
                    "open": close - 0.2,
                    "high": close + 1,
                    "low": close - 1,
                    "close": close,
                    "volume": 20_000 + index * 100,
                    "trade_count": 100,
                    "source": "test",
                    "ingested_at": pd.Timestamp("2026-01-01", tz="UTC"),
                }
            )
    return pd.DataFrame(rows)

