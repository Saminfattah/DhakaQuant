from __future__ import annotations

import pandas as pd

from dse_quant.processing.cleaning import merge_price_frames, normalize_price_frame
from dse_quant.processing.validation import validate_daily_prices


def test_normalization_rejects_bad_rows_and_deduplicates(raw_prices):
    result = normalize_price_frame(raw_prices, source="test")
    assert list(result["ticker"].unique()) == ["ACI"]
    assert len(result) == 2
    assert result.iloc[-1]["close"] == 103
    assert result.iloc[0]["volume"] == 10_000


def test_incremental_merge_retains_old_rows(canonical_prices):
    old = canonical_prices.iloc[:10].copy()
    new = canonical_prices.iloc[10:15].copy()
    merged = merge_price_frames([old, new])
    assert len(merged) == 15
    assert merged["date"].min() == old["date"].min()


def test_newer_duplicate_wins(canonical_prices):
    original = canonical_prices.iloc[[0]].copy()
    replacement = original.copy()
    replacement["close"] = 999
    replacement["high"] = 1000
    replacement["ingested_at"] = pd.Timestamp("2026-01-02", tz="UTC")
    merged = merge_price_frames([original, replacement])
    assert len(merged) == 1
    assert merged.iloc[0]["close"] == 999


def test_validation_accepts_canonical_data(canonical_prices):
    report = validate_daily_prices(canonical_prices, minimum_rows_per_recent_session=1)
    assert report.valid
    assert report.duplicate_keys == 0

