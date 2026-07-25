from __future__ import annotations

import os

import pandas as pd
import pytest

from ui.services.data_loader import (
    UIDataError,
    enforce_model_quality_gate,
    file_signature,
    filter_date_range,
    filter_signals,
    list_tickers,
    load_signals,
    preferred_artifact,
    require_columns,
)


def test_parquet_is_preferred(tmp_path):
    parquet = tmp_path / "artifact.parquet"
    csv = tmp_path / "artifact.csv"
    csv.write_text("ticker\nACI\n", encoding="utf-8")
    assert preferred_artifact(parquet, csv) == csv
    parquet.write_bytes(b"present")
    assert preferred_artifact(parquet, csv) == parquet


def test_required_column_validation():
    with pytest.raises(UIDataError, match="probability_up"):
        require_columns(pd.DataFrame({"ticker": ["ACI"]}), {"ticker", "probability_up"}, "Signals")


def test_file_signature_changes_with_modification_time(tmp_path):
    path = tmp_path / "sample.json"
    path.write_text("{}", encoding="utf-8")
    before = file_signature(path)
    os.utime(path, ns=(before.modified_ns + 1_000_000, before.modified_ns + 1_000_000))
    after = file_signature(path)
    assert before.as_tuple() != after.as_tuple()


def test_signal_filtering():
    frame = pd.DataFrame(
        {
            "ticker": ["ACI", "BATBC"],
            "signal": ["HOLD", "AVOID"],
            "probability_up": [0.7, 0.4],
            "probability_down": [0.3, 0.6],
            "liquidity_flag": [False, True],
            "stale_price_flag": [False, True],
            "floor_price_flag": [False, False],
            "model_version": ["v1", "v1"],
        }
    )
    result = filter_signals(
        frame,
        search="ACI",
        minimum_up=0.6,
        liquidity="Normal",
        stale="Normal",
    )
    assert result["ticker"].tolist() == ["ACI"]


def test_date_range_filtering():
    frame = pd.DataFrame(
        {"date": pd.date_range("2025-01-01", "2026-01-01", freq="D"), "close": 1}
    )
    result = filter_date_range(frame, "3M")
    assert result["date"].min() >= pd.Timestamp("2025-10-01")


def test_missing_signal_files_are_graceful(tmp_path):
    with pytest.raises(FileNotFoundError, match="No signal output"):
        load_signals(tmp_path)


def test_ticker_list_prefers_model_feature_universe(tmp_path):
    processed = tmp_path / "data/processed"
    features = tmp_path / "data/features"
    processed.mkdir(parents=True)
    features.mkdir(parents=True)
    pd.DataFrame({"ticker": ["ACI", "ABBLPBOND"]}).to_parquet(
        processed / "daily_prices.parquet"
    )
    pd.DataFrame({"ticker": ["ACI"]}).to_parquet(features / "daily_features.parquet")
    assert list_tickers(tmp_path) == ["ACI"]


def test_ui_suppresses_buy_watch_when_quality_gate_fails():
    frame = pd.DataFrame(
        {"signal": ["BUY WATCH"], "reason_codes": [""], "ticker": ["ACI"]}
    )
    result = enforce_model_quality_gate(
        frame,
        actual_precision=0.47,
        required_precision=0.65,
        require_quality=True,
    )
    assert result.iloc[0]["signal"] == "HOLD"
    assert "MODEL_BELOW_REQUIRED_PRECISION" in result.iloc[0]["reason_codes"]
