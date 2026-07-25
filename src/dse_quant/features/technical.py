from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period, min_periods=period).mean()
    loss = -delta.clip(upper=0).rolling(period, min_periods=period).mean()
    relative_strength = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + relative_strength))


def add_technical_features(group: pd.DataFrame) -> pd.DataFrame:
    result = group.sort_values("date").copy()
    close = result["close"]
    high = result["high"]
    low = result["low"]
    open_price = result["open"]

    for period in (1, 3, 5, 10, 20):
        result[f"return_{period}d"] = close.pct_change(period, fill_method=None)
    for period in (5, 10, 20, 50, 200):
        average = close.rolling(period, min_periods=period).mean()
        result[f"sma_{period}"] = average
        result[f"distance_sma_{period}"] = close / average - 1

    result["ema_12"] = close.ewm(span=12, adjust=False, min_periods=12).mean()
    result["ema_26"] = close.ewm(span=26, adjust=False, min_periods=26).mean()
    result["macd"] = result["ema_12"] - result["ema_26"]
    result["macd_signal"] = result["macd"].ewm(span=9, adjust=False, min_periods=9).mean()
    result["macd_histogram"] = result["macd"] - result["macd_signal"]
    result["rsi_14"] = _rsi(close, 14)

    middle = close.rolling(20, min_periods=20).mean()
    standard_deviation = close.rolling(20, min_periods=20).std()
    upper = middle + 2 * standard_deviation
    lower = middle - 2 * standard_deviation
    result["bollinger_position"] = (close - lower) / (upper - lower).replace(0, np.nan)
    result["bollinger_width"] = (upper - lower) / middle.replace(0, np.nan)

    previous_close = close.shift(1)
    true_range = pd.concat(
        [(high - low), (high - previous_close).abs(), (low - previous_close).abs()], axis=1
    ).max(axis=1)
    result["atr_14"] = true_range.rolling(14, min_periods=14).mean()
    result["atr_14_pct"] = result["atr_14"] / close.replace(0, np.nan)

    result["volatility_20"] = result["return_1d"].rolling(20, min_periods=20).std()
    result["momentum_10"] = close - close.shift(10)
    result["roc_10"] = close.pct_change(10, fill_method=None)
    rolling_high = high.rolling(20, min_periods=20).max()
    rolling_low = low.rolling(20, min_periods=20).min()
    result["distance_rolling_high_20"] = close / rolling_high - 1
    result["distance_rolling_low_20"] = close / rolling_low - 1
    result["drawdown_252"] = close / close.rolling(252, min_periods=60).max() - 1
    result["overnight_gap"] = open_price / previous_close - 1
    result["intraday_return"] = close / open_price.replace(0, np.nan) - 1
    return result

