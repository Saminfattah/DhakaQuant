from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from ui.components.charts import probability_distribution, signal_distribution
from ui.components.metrics import date_text, local_date_text, number, percent, time_text
from ui.components.warnings import model_quality_gate, render_quality_warning
from ui.services.data_loader import (
    load_json,
    load_predictions,
    load_signals,
    parquet_metadata,
    tail_text,
)
from ui.services.settings_manager import load_yaml


def _latest_date(frame: pd.DataFrame, column: str):
    if column not in frame or frame.empty:
        return None
    values = pd.to_datetime(frame[column], errors="coerce")
    return values.max() if values.notna().any() else None


def _ranking_frame(
    signals: pd.DataFrame,
    *,
    probability_column: str,
    ascending: bool,
) -> pd.DataFrame:
    visible = signals.sort_values(probability_column, ascending=ascending).head(8).copy()
    visible["status_notes"] = (
        visible.get("reason_codes", pd.Series(index=visible.index, dtype="object"))
        .fillna("")
        .astype(str)
        .str.replace(",", ", ", regex=False)
        .replace("", "No data-quality flags")
    )
    columns = [
        column
        for column in (
            "ticker",
            "prediction_date",
            "latest_price",
            probability_column,
            "signal",
            "status_notes",
        )
        if column in visible
    ]
    return visible[columns]


def _ranking_column_config(probability_column: str) -> dict:
    return {
        "ticker": st.column_config.TextColumn("Ticker", width="small"),
        "prediction_date": st.column_config.DateColumn(
            "Prediction date",
            format="DD MMM YYYY",
            width="medium",
        ),
        "latest_price": st.column_config.NumberColumn(
            "Last price",
            format="৳ %.2f",
            width="small",
        ),
        probability_column: st.column_config.ProgressColumn(
            "P(Up)" if probability_column == "probability_up" else "P(Down)",
            format="percent",
            min_value=0,
            max_value=1,
            width="medium",
        ),
        "signal": st.column_config.TextColumn("Research signal", width="medium"),
        "status_notes": st.column_config.TextColumn(
            "Data-quality notes",
            help="Why a ticker may need extra review.",
            width="large",
        ),
    }


def render(root: Path) -> None:
    st.markdown('<div class="page-kicker">Market and model status</div>', unsafe_allow_html=True)
    st.title("Overview")
    st.caption(
        "A clear summary of data freshness, model coverage, quality, and current research signals."
    )

    signals = load_signals(root)
    predictions = load_predictions(root)
    metadata = load_json(root / "models/latest_metrics.json", default={})
    validation = load_json(root / "data/outputs/validation_report.json", default={})
    settings = load_yaml(root / "config/settings.yaml")
    price_meta = parquet_metadata(root / "data/processed/daily_prices.parquet")

    validation_metrics = metadata.get("metrics", {}).get("validation", {})
    test_metrics = metadata.get("metrics", {}).get("test", {})
    required = float(settings["signals"]["minimum_validation_precision"])
    actual = validation_metrics.get("precision")
    gate = model_quality_gate(
        actual,
        required,
        bool(settings["signals"].get("require_model_quality_for_buy", True)),
    )
    render_quality_warning(gate)

    log_path = root / "logs/dse_quant.log"
    pipeline_activity = (
        pd.Timestamp(log_path.stat().st_mtime, unit="s", tz="UTC")
        if log_path.exists()
        else None
    )
    prediction_date = _latest_date(predictions, "prediction_date")
    freshness_date = _latest_date(predictions, "data_freshness_date")

    st.subheader("Dates and freshness")
    with st.container(border=True, key="overview_dates"):
        date_columns = st.columns(4)
        date_columns[0].metric(
            "Market data through",
            date_text(validation.get("last_date")),
            help="Most recent trading date in the validated local price archive.",
        )
        date_columns[0].caption("Latest validated market session")
        date_columns[1].metric(
            "Predictions dated",
            date_text(prediction_date),
            help="Latest trading date represented by the current prediction output.",
        )
        date_columns[1].caption(
            f"Source data through {date_text(freshness_date)}"
        )
        date_columns[2].metric(
            "Model trained",
            local_date_text(metadata.get("created_at")),
            help="Training completion time shown in Dhaka time.",
        )
        date_columns[2].caption(
            f"{time_text(metadata.get('created_at'))} · Asia/Dhaka"
        )
        date_columns[3].metric(
            "Pipeline activity",
            local_date_text(pipeline_activity),
            help="Most recent entry written to the local pipeline log.",
        )
        date_columns[3].caption(f"{time_text(pipeline_activity)} · Asia/Dhaka")

    st.subheader("Coverage and active model")
    model_universe = metadata.get("model_universe", {})
    with st.container(key="overview_model_section"):
        coverage_panel, model_panel = st.columns([1.45, 1])
        with coverage_panel, st.container(border=True, key="overview_coverage"):
            coverage_columns = st.columns(2)
            coverage_columns[0].metric(
                "Price archive rows",
                number(price_meta.get("rows")),
                help="All rows in the canonical daily price archive.",
            )
            coverage_columns[1].metric(
                "Listed securities",
                number(validation.get("tickers")),
                help="DSE-listed securities retained in the validated price archive.",
            )
            coverage_columns[0].metric(
                "Model universe",
                number(model_universe.get("tickers_after_filter")),
                help="Instruments eligible for the model after fixed-income filtering.",
            )
            coverage_columns[1].metric(
                "Current predictions",
                number(len(predictions)),
                help="One current prediction per model-universe instrument.",
            )
        with model_panel, st.container(border=True, key="overview_model"):
            st.markdown("**Active model version**")
            st.code(metadata.get("model_version", "—"), language=None)
            model_columns = st.columns(2)
            model_columns[0].metric("Validation precision", percent(actual))
            model_columns[1].metric(
                "2026 holdout precision",
                percent(test_metrics.get("precision")),
            )
            walk_forward = metadata.get("metrics", {}).get("walk_forward", {})
            model_columns[0].metric(
                "Decision threshold",
                percent(metadata.get("probability_threshold")),
            )
            selected_window = walk_forward.get("selected_training_window_years")
            model_columns[1].metric(
                "Training window",
                f"{selected_window} years" if selected_window else "—",
            )
            st.caption(
                "Validation: 2022–2025 walk-forward folds · Holdout: 2026 · "
                f"Data validation: {'Passed' if validation.get('valid') else 'Review required'}"
            )

    st.subheader("Signal snapshot")
    st.caption(
        f"Latest research classification for {number(len(signals))} instruments. "
        "BUY WATCH remains subject to the model-quality gate above."
    )
    counts = signals["signal"].value_counts()
    with st.container(key="overview_signal_counts"):
        count_columns = st.columns(4)
        for column, label in zip(
            count_columns,
            ("BUY WATCH", "HOLD", "AVOID", "SELL REVIEW"),
            strict=True,
        ):
            column.metric(label, int(counts.get(label, 0)))

    with st.container(key="overview_charts"):
        chart_left, chart_right = st.columns(2)
        with chart_left:
            st.plotly_chart(signal_distribution(signals), width="stretch")
        with chart_right:
            st.plotly_chart(probability_distribution(predictions), width="stretch")

    with st.container(key="overview_rankings"):
        upside, downside = st.columns(2)
        with upside:
            st.subheader("Highest upside probability")
            st.dataframe(
                _ranking_frame(
                    signals,
                    probability_column="probability_up",
                    ascending=False,
                ),
                hide_index=True,
                width="stretch",
                height=330,
                column_config=_ranking_column_config("probability_up"),
            )
        with downside:
            st.subheader("Highest downside probability")
            st.dataframe(
                _ranking_frame(
                    signals,
                    probability_column="probability_down",
                    ascending=False,
                ),
                hide_index=True,
                width="stretch",
                height=330,
                column_config=_ranking_column_config("probability_down"),
            )

    with st.expander("Recent pipeline log"):
        st.code(tail_text(log_path, 30), language="text")
