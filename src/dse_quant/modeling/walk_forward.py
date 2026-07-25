from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score

RISK_COLUMNS = (
    "liquidity_flag",
    "stale_price_flag",
    "floor_price_flag",
    "insufficient_history_flag",
)


@dataclass(frozen=True)
class YearFold:
    year: int
    train: pd.DataFrame
    validation: pd.DataFrame
    boundaries: dict[str, str]


def actionable_labeled_rows(
    frame: pd.DataFrame,
    *,
    exclude_risk_flagged_rows: bool,
) -> pd.DataFrame:
    eligible = frame["target"].notna()
    if exclude_risk_flagged_rows:
        missing = sorted(set(RISK_COLUMNS) - set(frame.columns))
        if missing:
            raise ValueError(f"Risk-filter columns are missing: {missing}")
        for column in RISK_COLUMNS:
            eligible &= ~frame[column].fillna(True).astype(bool)
    return frame.loc[eligible].sort_values(["date", "ticker"]).copy()


def make_year_fold(
    frame: pd.DataFrame,
    *,
    year: int,
    training_window_years: int,
    embargo_sessions: int,
    minimum_training_rows: int,
    minimum_validation_rows: int,
) -> YearFold:
    dates = pd.to_datetime(frame["date"])
    validation = frame.loc[dates.dt.year == year].copy()
    if len(validation) < minimum_validation_rows:
        raise ValueError(
            f"Walk-forward year {year} has {len(validation)} validation rows; "
            f"minimum is {minimum_validation_rows}."
        )
    validation_start = pd.to_datetime(validation["date"]).min()
    previous_sessions = np.array(
        sorted(pd.to_datetime(frame.loc[dates < validation_start, "date"]).unique())
    )
    if len(previous_sessions) <= embargo_sessions:
        raise ValueError(f"Not enough sessions before walk-forward year {year}.")
    train_end = pd.Timestamp(previous_sessions[-(embargo_sessions + 1)])
    train_start = validation_start - pd.DateOffset(years=training_window_years)
    train_dates = dates.between(train_start, train_end)
    train = frame.loc[train_dates].copy()
    if len(train) < minimum_training_rows:
        raise ValueError(
            f"Walk-forward year {year}, {training_window_years}-year window has "
            f"{len(train)} training rows; minimum is {minimum_training_rows}."
        )
    boundaries = {
        "train_start": str(pd.to_datetime(train["date"]).min().date()),
        "train_end": str(pd.to_datetime(train["date"]).max().date()),
        "validation_start": str(pd.to_datetime(validation["date"]).min().date()),
        "validation_end": str(pd.to_datetime(validation["date"]).max().date()),
    }
    return YearFold(year=year, train=train, validation=validation, boundaries=boundaries)


def inner_validation_split(
    train: pd.DataFrame,
    *,
    validation_fraction: float,
    embargo_sessions: int,
    minimum_training_rows: int,
    minimum_validation_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = np.array(sorted(pd.to_datetime(train["date"]).unique()))
    validation_sessions = max(20, int(len(dates) * validation_fraction))
    if validation_sessions + embargo_sessions >= len(dates):
        raise ValueError("Not enough sessions for inner validation and embargo.")
    validation_dates = dates[-validation_sessions:]
    train_dates = dates[: -(validation_sessions + embargo_sessions)]
    inner_train = train[train["date"].isin(train_dates)].copy()
    inner_validation = train[train["date"].isin(validation_dates)].copy()
    if len(inner_train) < minimum_training_rows:
        raise ValueError(
            f"Inner training rows {len(inner_train)} are below required "
            f"{minimum_training_rows}."
        )
    if len(inner_validation) < minimum_validation_rows:
        raise ValueError(
            f"Inner validation rows {len(inner_validation)} are below required "
            f"{minimum_validation_rows}."
        )
    return inner_train, inner_validation


def threshold_candidates(start: float, stop: float, step: float) -> list[float]:
    if not 0 <= start <= stop <= 1:
        raise ValueError("Threshold range must satisfy 0 <= start <= stop <= 1.")
    if step <= 0:
        raise ValueError("Threshold step must be positive.")
    count = int(np.floor((stop - start) / step + 1e-9))
    values = [round(start + index * step, 10) for index in range(count + 1)]
    if values[-1] < stop - 1e-9:
        values.append(stop)
    return values


def evaluate_thresholds(
    scored: pd.DataFrame,
    thresholds: list[float],
    *,
    minimum_call_rate: float,
    minimum_fold_call_rate: float,
) -> pd.DataFrame:
    required = {"fold", "target", "probability"}
    missing = sorted(required - set(scored.columns))
    if missing:
        raise ValueError(f"Scored walk-forward data is missing: {missing}")
    rows: list[dict[str, object]] = []
    truth = scored["target"].astype(int)
    for threshold in thresholds:
        predicted = scored["probability"].ge(threshold)
        fold_precision: list[float] = []
        fold_call_rates: list[float] = []
        for _, fold in scored.groupby("fold", observed=True):
            fold_prediction = fold["probability"].ge(threshold)
            fold_call_rates.append(float(fold_prediction.mean()))
            fold_precision.append(
                float(
                    precision_score(
                        fold["target"].astype(int),
                        fold_prediction,
                        zero_division=0,
                    )
                )
            )
        call_rate = float(predicted.mean())
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision_score(truth, predicted, zero_division=0)),
                "recall": float(recall_score(truth, predicted, zero_division=0)),
                "call_rate": call_rate,
                "positive_calls": int(predicted.sum()),
                "median_fold_precision": float(np.median(fold_precision)),
                "worst_fold_precision": float(np.min(fold_precision)),
                "minimum_fold_call_rate": float(np.min(fold_call_rates)),
                "eligible": bool(
                    call_rate >= minimum_call_rate
                    and min(fold_call_rates) >= minimum_fold_call_rate
                ),
            }
        )
    return pd.DataFrame(rows)


def select_threshold(table: pd.DataFrame) -> pd.Series:
    eligible = table[table["eligible"].astype(bool)].copy()
    if eligible.empty:
        raise ValueError(
            "No probability threshold satisfied the configured aggregate and per-fold "
            "minimum call rates."
        )
    ranked = eligible.sort_values(
        [
            "median_fold_precision",
            "worst_fold_precision",
            "precision",
            "threshold",
        ],
        ascending=[False, False, False, True],
    )
    return ranked.iloc[0]
