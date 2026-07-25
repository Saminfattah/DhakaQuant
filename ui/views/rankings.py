from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from ui.services.data_loader import load_json, load_signals


def _ranking_table(frame: pd.DataFrame, sort_column: str) -> None:
    columns = [
        "ticker",
        "latest_price",
        "probability_up",
        "probability_down",
        "signal",
        "reason_codes",
    ]
    st.dataframe(
        frame.sort_values(sort_column, ascending=False)[columns],
        hide_index=True,
        width="stretch",
        height=650,
        column_config={
            "latest_price": st.column_config.NumberColumn(format="৳%.2f"),
            "probability_up": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
            "probability_down": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
        },
    )


def render(root: Path) -> None:
    st.markdown('<div class="page-kicker">Probability review</div>', unsafe_allow_html=True)
    st.title("Rankings")
    metadata = load_json(root / "models/latest_metrics.json", default={})
    horizon = metadata.get("prediction_horizon")
    horizon_text = f" over the next {horizon} trading sessions" if horizon else ""
    st.info(
        "These rankings order DSE tickers by the model's estimated probability of moving "
        f"up or down{horizon_text}. Higher probability means stronger model confidence, "
        "not a larger expected return or a trade recommendation. Review the signal and "
        "reason codes before interpreting a ticker."
    )
    st.markdown(
        """
**Ranking tabs**

- **Upside:** All tickers, highest estimated probability of moving up first.
- **Downside:** All tickers, highest estimated probability of moving down first.
- **Avoided:** Tickers marked `AVOID` because a data-quality or market-risk rule was triggered.
- **Normal liquidity:** Tickers not flagged by the model's low-liquidity rule.
- **Stale data:** Tickers whose latest usable price data is old or has not changed recently.
"""
    )
    frame = load_signals(root)
    upside, downside, avoided, liquid, stale = st.tabs(
        ["Upside", "Downside", "Avoided", "Normal liquidity", "Stale data"]
    )
    with upside:
        _ranking_table(frame, "probability_up")
    with downside:
        _ranking_table(frame, "probability_down")
    with avoided:
        _ranking_table(frame[frame["signal"] == "AVOID"], "probability_up")
    with liquid:
        _ranking_table(frame[~frame["liquidity_flag"].fillna(False)], "probability_up")
    with stale:
        _ranking_table(frame[frame["stale_price_flag"].fillna(False)], "probability_up")
