from __future__ import annotations

import pandas as pd

from dse_quant.modeling.walk_forward import (
    actionable_labeled_rows,
    evaluate_thresholds,
    make_year_fold,
    select_threshold,
    threshold_candidates,
)


def _walk_forward_frame() -> pd.DataFrame:
    dates = pd.bdate_range("2017-01-01", "2023-12-31")
    rows = []
    for ticker in ("ACI", "BATBC"):
        for index, day in enumerate(dates):
            rows.append(
                {
                    "date": day,
                    "ticker": ticker,
                    "target": index % 2,
                    "liquidity_flag": False,
                    "stale_price_flag": False,
                    "floor_price_flag": False,
                    "insufficient_history_flag": False,
                }
            )
    return pd.DataFrame(rows)


def test_actionable_rows_remove_risk_flags():
    frame = _walk_forward_frame().head(3).copy()
    frame.loc[frame.index[0], "liquidity_flag"] = True
    frame.loc[frame.index[1], "target"] = pd.NA
    result = actionable_labeled_rows(frame, exclude_risk_flagged_rows=True)
    assert result.index.tolist() == [frame.index[2]]


def test_year_fold_uses_only_prior_window_and_embargo():
    frame = _walk_forward_frame()
    fold = make_year_fold(
        frame,
        year=2023,
        training_window_years=5,
        embargo_sessions=3,
        minimum_training_rows=100,
        minimum_validation_rows=100,
    )
    assert pd.to_datetime(fold.train["date"]).max() < pd.to_datetime(
        fold.validation["date"]
    ).min()
    assert pd.to_datetime(fold.train["date"]).min().year >= 2018
    assert set(pd.to_datetime(fold.validation["date"]).dt.year) == {2023}


def test_threshold_selection_respects_call_rate_and_fold_robustness():
    scored = pd.DataFrame(
        {
            "fold": [2022] * 5 + [2023] * 5,
            "target": [1, 1, 0, 0, 0, 1, 0, 0, 0, 0],
            "probability": [0.9, 0.8, 0.7, 0.4, 0.2, 0.85, 0.6, 0.5, 0.3, 0.1],
        }
    )
    table = evaluate_thresholds(
        scored,
        [0.5, 0.8, 0.9],
        minimum_call_rate=0.2,
        minimum_fold_call_rate=0.2,
    )
    selected = select_threshold(table)
    assert selected["threshold"] == 0.8
    assert selected["eligible"]
    assert threshold_candidates(0.5, 0.6, 0.05) == [0.5, 0.55, 0.6]
