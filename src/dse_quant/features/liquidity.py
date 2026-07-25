from __future__ import annotations

import numpy as np
import pandas as pd


def _consecutive_true(values: pd.Series) -> pd.Series:
    clean = values.fillna(False).astype(bool)
    groups = (~clean).cumsum()
    return clean.astype("int64").groupby(groups).cumsum()


def add_liquidity_features(
    group: pd.DataFrame,
    *,
    lookback: int,
    stale_price_sessions: int,
    low_volume_threshold: float,
    low_turnover_threshold: float,
    floor_flat_sessions: int,
) -> pd.DataFrame:
    result = group.sort_values("date").copy()
    volume = result["volume"]
    close = result["close"]
    turnover = close * volume

    result["volume_change_1d"] = volume.pct_change(fill_method=None)
    result["volume_sma_5"] = volume.rolling(5, min_periods=5).mean()
    result["volume_sma_20"] = volume.rolling(20, min_periods=20).mean()
    result["relative_volume_20"] = volume / result["volume_sma_20"].replace(0, np.nan)
    result["price_volume_trend"] = (
        close.pct_change(fill_method=None).fillna(0) * volume.fillna(0)
    ).cumsum()
    result["turnover"] = turnover
    result["turnover_median_20"] = turnover.rolling(lookback, min_periods=lookback).median()
    result["amihud_illiquidity_20"] = (
        close.pct_change(fill_method=None).abs() / turnover.replace(0, np.nan)
    ).rolling(lookback, min_periods=lookback).mean()
    result["zero_volume_flag"] = volume.fillna(0).eq(0)
    result["low_volume_flag"] = (
        volume.rolling(lookback, min_periods=lookback).median() < low_volume_threshold
    )
    result["liquidity_flag"] = (
        result["low_volume_flag"]
        | (result["turnover_median_20"] < low_turnover_threshold)
        | result["zero_volume_flag"]
    )

    flat = close.diff().fillna(np.nan).eq(0)
    result["unchanged_price_streak"] = _consecutive_true(flat)
    result["stale_price_flag"] = result["unchanged_price_streak"] >= stale_price_sessions
    result["floor_price_flag"] = (
        (result["unchanged_price_streak"] >= floor_flat_sessions)
        & (volume > 0)
        & (result["intraday_return"].abs() < 1e-12)
    )
    return result
