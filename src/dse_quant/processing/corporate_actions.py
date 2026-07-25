from __future__ import annotations

import pandas as pd


def build_join_boundary_report(
    frame: pd.DataFrame,
    *,
    sessions: int = 10,
    return_warning: float = 0.50,
) -> pd.DataFrame:
    """Flag discontinuities near transitions between data sources.

    This does not adjust prices. It exposes source-boundary jumps for human review because the
    Kaggle series may be adjusted while the official archive is normally raw.
    """
    ordered = frame.sort_values(["ticker", "date"]).copy()
    ordered["previous_source"] = ordered.groupby("ticker")["source"].shift(1)
    ordered["previous_close"] = ordered.groupby("ticker")["close"].shift(1)
    ordered["source_changed"] = ordered["source"] != ordered["previous_source"]
    ordered["boundary_return"] = ordered["close"] / ordered["previous_close"] - 1
    boundaries = ordered.loc[
        ordered["source_changed"] & ordered["previous_source"].notna(),
        ["ticker", "date", "previous_source", "source", "previous_close", "close", "boundary_return"],
    ].copy()
    boundaries["warning"] = boundaries["boundary_return"].abs() >= return_warning
    boundaries["inspection_sessions"] = sessions
    return boundaries.reset_index(drop=True)

