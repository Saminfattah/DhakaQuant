from __future__ import annotations

import pandas as pd

from dse_quant.features.liquidity import _consecutive_true, add_liquidity_features
from dse_quant.features.pipeline import _breadth
from dse_quant.features.technical import add_technical_features
from dse_quant.modeling.dataset import add_target


def test_target_uses_exact_future_ticker_session(canonical_prices):
    result = add_target(canonical_prices, horizon=3, minimum_return=0)
    aci = result[result["ticker"] == "ACI"].reset_index(drop=True)
    expected = aci.loc[3, "close"] / aci.loc[0, "close"] - 1
    assert aci.loc[0, "future_return"] == expected
    assert pd.isna(aci.iloc[-1]["target"])


def test_target_handles_nullable_price_dtype(canonical_prices):
    frame = canonical_prices.copy()
    frame["close"] = frame["close"].astype("Float64")
    result = add_target(frame, horizon=3, minimum_return=0)
    assert str(result["target"].dtype) == "Int8"
    assert result.groupby("ticker")["target"].tail(3).isna().all()


def test_future_price_change_does_not_change_past_features(canonical_prices):
    group = canonical_prices[canonical_prices["ticker"] == "ACI"].copy()
    before = add_technical_features(group)
    changed = group.copy()
    changed.loc[changed.index[-1], "close"] *= 10
    after = add_technical_features(changed)
    pd.testing.assert_series_equal(
        before.iloc[:-1]["return_5d"].reset_index(drop=True),
        after.iloc[:-1]["return_5d"].reset_index(drop=True),
    )


def test_liquidity_flags_low_volume(canonical_prices):
    group = canonical_prices[canonical_prices["ticker"] == "ACI"].copy()
    group["volume"] = 1
    group = add_technical_features(group)
    result = add_liquidity_features(
        group,
        lookback=20,
        stale_price_sessions=5,
        low_volume_threshold=10_000,
        low_turnover_threshold=100_000,
        floor_flat_sessions=3,
    )
    assert bool(result.iloc[-1]["liquidity_flag"])


def test_consecutive_true_treats_nullable_values_as_false():
    values = pd.Series([True, True, pd.NA, True, False, True], dtype="boolean")
    result = _consecutive_true(values)
    assert result.tolist() == [1, 2, 0, 1, 0, 1]


def test_market_breadth_allows_an_all_missing_first_session():
    values = pd.Series([pd.NA, pd.NA], dtype="Float64")
    assert pd.isna(_breadth(values))
