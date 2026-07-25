from __future__ import annotations

from pathlib import Path

import streamlit as st

from ui.components.charts import indicator_chart, price_volume_chart
from ui.components.metrics import date_text, money, number, percent
from ui.services.data_loader import (
    filter_date_range,
    list_tickers,
    load_signals,
    load_ticker_features,
)


def render(root: Path) -> None:
    st.markdown('<div class="page-kicker">Historical context</div>', unsafe_allow_html=True)
    st.title("Ticker Explorer")
    tickers = list_tickers(root)
    if not tickers:
        raise FileNotFoundError("No canonical ticker data exists yet.")
    requested = st.session_state.get("selected_ticker")
    default_index = tickers.index(requested) if requested in tickers else 0
    ticker = st.selectbox("Ticker", tickers, index=default_index)
    st.session_state["selected_ticker"] = ticker
    period = st.segmented_control(
        "Date range",
        ["3M", "6M", "1Y", "3Y", "5Y", "All"],
        default="1Y",
    )

    history = load_ticker_features(root, ticker)
    if history.empty:
        st.info(f"No feature history is available for {ticker}.")
        return
    shown = filter_date_range(history, period or "1Y")
    latest = history.iloc[-1]
    signals = load_signals(root)
    signal_rows = signals[signals["ticker"] == ticker]
    signal = signal_rows.iloc[0] if len(signal_rows) else None

    columns = st.columns(6)
    columns[0].metric("Latest close", money(latest["close"]))
    columns[1].metric("Latest volume", number(latest["volume"]))
    columns[2].metric("P(Up)", percent(signal["probability_up"] if signal is not None else None))
    columns[3].metric("P(Down)", percent(signal["probability_down"] if signal is not None else None))
    columns[4].metric("Signal", signal["signal"] if signal is not None else "—")
    columns[5].metric("Observations", number(len(history)))

    detail = st.columns(3)
    detail[0].caption(f"Available: {date_text(history['date'].min())} to {date_text(history['date'].max())}")
    detail[1].caption(
        f"Prediction: {date_text(signal['prediction_date']) if signal is not None else '—'}"
    )
    flags = []
    if signal is not None:
        for field, label in (
            ("liquidity_flag", "Low liquidity"),
            ("stale_price_flag", "Stale"),
            ("floor_price_flag", "Floor proxy"),
        ):
            if bool(signal.get(field, False)):
                flags.append(label)
    detail[2].caption("Risk flags: " + (", ".join(flags) if flags else "None"))

    if signal is not None and signal.get("reason_codes"):
        chips = "".join(
            f'<span class="reason-chip">{code.replace("_", " ").title()}</span>'
            for code in str(signal["reason_codes"]).split(",")
            if code
        )
        st.markdown(chips, unsafe_allow_html=True)

    st.plotly_chart(price_volume_chart(shown), width="stretch")
    first, second = st.columns(2)
    with first:
        st.plotly_chart(indicator_chart(shown, ["rsi_14"], "RSI 14", 50), width="stretch")
        st.plotly_chart(
            indicator_chart(shown, ["volatility_20"], "Rolling volatility"),
            width="stretch",
        )
    with second:
        st.plotly_chart(
            indicator_chart(shown, ["macd", "macd_signal"], "MACD"),
            width="stretch",
        )
        st.plotly_chart(
            indicator_chart(shown, ["relative_volume_20"], "Relative volume", 1),
            width="stretch",
        )
