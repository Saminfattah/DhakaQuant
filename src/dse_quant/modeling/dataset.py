from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

NON_FEATURE_COLUMNS = {
    "date",
    "ticker",
    "source",
    "ingested_at",
    "future_return",
    "target",
    "predicted_class",
}
RAW_PRICE_COLUMNS = {"open", "high", "low", "close", "volume", "trade_count"}


@dataclass(frozen=True)
class DatasetSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    feature_names: list[str]
    boundaries: dict[str, str]


def add_target(frame: pd.DataFrame, horizon: int, minimum_return: float) -> pd.DataFrame:
    ordered = frame.sort_values(["ticker", "date"]).copy()
    future_close = ordered.groupby("ticker")["close"].shift(-horizon)
    ordered["future_return"] = future_close / ordered["close"] - 1
    known = ordered["future_return"].notna()
    target = pd.Series(pd.NA, index=ordered.index, dtype="Int8")
    target.loc[known] = (
        ordered.loc[known, "future_return"].gt(minimum_return).astype("int8")
    )
    ordered["target"] = target
    return ordered


def feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = NON_FEATURE_COLUMNS | RAW_PRICE_COLUMNS
    return [
        column
        for column in frame.select_dtypes(include=[np.number, "bool"]).columns
        if column not in excluded
    ]


def chronological_split(
    frame: pd.DataFrame,
    *,
    validation_fraction: float,
    test_fraction: float,
    embargo_sessions: int,
    minimum_training_rows: int,
    minimum_validation_rows: int,
) -> DatasetSplit:
    labeled = frame.loc[frame["target"].notna()].sort_values(["date", "ticker"]).copy()
    dates = np.array(sorted(pd.to_datetime(labeled["date"]).unique()))
    if len(dates) < 30:
        raise ValueError("At least 30 trading sessions are required for chronological splitting.")
    test_count = max(1, int(len(dates) * test_fraction))
    validation_count = max(1, int(len(dates) * validation_fraction))
    test_start_index = len(dates) - test_count
    validation_start_index = test_start_index - embargo_sessions - validation_count
    train_end_index = validation_start_index - embargo_sessions
    if train_end_index <= 0:
        raise ValueError("Not enough sessions for train/validation/test split plus embargoes.")

    train_dates = dates[:train_end_index]
    validation_dates = dates[validation_start_index : validation_start_index + validation_count]
    test_dates = dates[test_start_index:]
    train = labeled[labeled["date"].isin(train_dates)]
    validation = labeled[labeled["date"].isin(validation_dates)]
    test = labeled[labeled["date"].isin(test_dates)]
    if len(train) < minimum_training_rows:
        raise ValueError(f"Training rows {len(train)} are below required {minimum_training_rows}.")
    if len(validation) < minimum_validation_rows:
        raise ValueError(
            f"Validation rows {len(validation)} are below required {minimum_validation_rows}."
        )
    names = feature_columns(labeled)
    if not names:
        raise ValueError("No numeric feature columns were generated.")
    boundaries = {
        "train_start": str(pd.Timestamp(train_dates[0]).date()),
        "train_end": str(pd.Timestamp(train_dates[-1]).date()),
        "validation_start": str(pd.Timestamp(validation_dates[0]).date()),
        "validation_end": str(pd.Timestamp(validation_dates[-1]).date()),
        "test_start": str(pd.Timestamp(test_dates[0]).date()),
        "test_end": str(pd.Timestamp(test_dates[-1]).date()),
    }
    return DatasetSplit(train, validation, test, names, boundaries)
