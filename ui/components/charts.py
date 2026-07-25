from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PALETTE = {
    "BUY WATCH": "#0f766e",
    "HOLD": "#64748b",
    "AVOID": "#d97706",
    "SELL REVIEW": "#b91c1c",
}


def signal_distribution(frame: pd.DataFrame) -> go.Figure:
    counts = frame["signal"].value_counts().rename_axis("signal").reset_index(name="count")
    figure = px.bar(
        counts,
        x="signal",
        y="count",
        color="signal",
        color_discrete_map=PALETTE,
        text_auto=True,
    )
    figure.update_layout(showlegend=False, xaxis_title=None, yaxis_title="Tickers", height=330)
    return figure


def probability_distribution(frame: pd.DataFrame) -> go.Figure:
    figure = px.histogram(
        frame,
        x="probability_up",
        nbins=20,
        color_discrete_sequence=["#0f766e"],
    )
    figure.update_layout(
        xaxis_title="Probability up",
        yaxis_title="Tickers",
        bargap=0.06,
        height=330,
    )
    figure.update_xaxes(tickformat=".0%")
    return figure


def price_volume_chart(frame: pd.DataFrame) -> go.Figure:
    data = downsample(frame)
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.72, 0.28],
    )
    figure.add_trace(
        go.Candlestick(
            x=data["date"],
            open=data["open"],
            high=data["high"],
            low=data["low"],
            close=data["close"],
            name="OHLC",
            increasing_line_color="#0f766e",
            decreasing_line_color="#b91c1c",
        ),
        row=1,
        col=1,
    )
    for column, label, color in (
        ("sma_20", "SMA 20", "#2563eb"),
        ("sma_50", "SMA 50", "#7c3aed"),
        ("sma_200", "SMA 200", "#d97706"),
    ):
        if column in data:
            figure.add_trace(
                go.Scatter(x=data["date"], y=data[column], name=label, line={"width": 1.3, "color": color}),
                row=1,
                col=1,
            )
    figure.add_trace(
        go.Bar(x=data["date"], y=data["volume"], name="Volume", marker_color="#64748b"),
        row=2,
        col=1,
    )
    figure.update_layout(height=620, xaxis_rangeslider_visible=False, legend_orientation="h")
    figure.update_yaxes(title_text="Price (BDT)", row=1, col=1)
    figure.update_yaxes(title_text="Volume", row=2, col=1)
    return figure


def indicator_chart(
    frame: pd.DataFrame, columns: list[str], title: str, threshold: float | None = None
) -> go.Figure:
    data = downsample(frame)
    figure = go.Figure()
    for column in columns:
        if column in data:
            figure.add_trace(go.Scatter(x=data["date"], y=data[column], name=column.replace("_", " ").title()))
    if threshold is not None:
        figure.add_hline(y=threshold, line_dash="dot", line_color="#94a3b8")
    figure.update_layout(height=300, title=title, legend_orientation="h", margin={"t": 45})
    return figure


def confusion_matrix_figure(matrix: list[list[int]]) -> go.Figure:
    figure = px.imshow(
        matrix,
        text_auto=True,
        labels={"x": "Predicted", "y": "Actual", "color": "Rows"},
        x=["Down", "Up"],
        y=["Down", "Up"],
        color_continuous_scale="Blues",
    )
    figure.update_layout(height=360)
    return figure


def calibration_figure(rows: list[dict]) -> go.Figure:
    frame = pd.DataFrame(rows)
    figure = go.Figure()
    if not frame.empty:
        figure.add_trace(
            go.Scatter(
                x=frame["predicted_probability"],
                y=frame["observed_rate"],
                mode="lines+markers",
                name="Model",
            )
        )
    figure.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect", line={"dash": "dot"})
    )
    figure.update_layout(
        height=360,
        xaxis_title="Predicted probability",
        yaxis_title="Observed frequency",
    )
    return figure


def downsample(frame: pd.DataFrame, maximum_rows: int = 3000) -> pd.DataFrame:
    if len(frame) <= maximum_rows:
        return frame
    step = max(1, len(frame) // maximum_rows)
    result = frame.iloc[::step].copy()
    if result.index[-1] != frame.index[-1]:
        result = pd.concat([result, frame.iloc[[-1]]])
    return result

