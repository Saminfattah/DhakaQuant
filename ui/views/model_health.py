from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from ui.components.charts import calibration_figure, confusion_matrix_figure
from ui.components.metrics import datetime_text, number, percent
from ui.components.warnings import model_quality_gate, render_quality_warning
from ui.services.data_loader import load_json
from ui.services.settings_manager import load_yaml


def _metric_grid(metrics: dict) -> None:
    fields = [
        ("Precision", "precision", percent),
        ("Recall", "recall", percent),
        ("F1", "f1", percent),
        ("ROC AUC", "roc_auc", lambda value: number(value, 3)),
        ("PR AUC", "pr_auc", lambda value: number(value, 3)),
        ("Log loss", "log_loss", lambda value: number(value, 3)),
        ("Brier score", "brier_score", lambda value: number(value, 3)),
        ("Coverage", "prediction_coverage", percent),
        ("Positive call rate", "positive_call_rate", percent),
        ("Positive rate", "positive_rate", percent),
    ]
    columns = st.columns(5)
    for index, (label, key, formatter) in enumerate(fields):
        columns[index % len(columns)].metric(label, formatter(metrics.get(key)))


def render(root: Path) -> None:
    st.markdown('<div class="page-kicker">Transparent evaluation</div>', unsafe_allow_html=True)
    st.title("Model Health")
    metadata = load_json(root / "models/latest_metrics.json", default={})
    if not metadata:
        raise FileNotFoundError("No model metrics exist yet.")
    settings = load_yaml(root / "config/settings.yaml")
    metrics = metadata.get("metrics", {})
    validation = metrics.get("validation", {})
    test = metrics.get("test", {})
    required = float(settings["signals"]["minimum_validation_precision"])
    gate = model_quality_gate(
        validation.get("precision"),
        required,
        bool(settings["signals"].get("require_model_quality_for_buy", True)),
    )
    render_quality_warning(gate)

    summary = st.columns(6)
    summary[0].metric("Model", metadata.get("model_version", "—"))
    summary[1].metric("Created", datetime_text(metadata.get("created_at")))
    summary[2].metric("Horizon", f"{metadata.get('prediction_horizon', '—')} sessions")
    summary[3].metric("Target return", percent(metadata.get("minimum_return_threshold")))
    summary[4].metric("Class threshold", percent(metadata.get("probability_threshold")))
    summary[5].metric("Features", len(metadata.get("feature_names", [])))

    boundaries = metrics.get("boundaries", {})
    st.caption(
        "Train: {train_start} → {train_end} · Validation: {validation_start} → "
        "{validation_end} · Test: {test_start} → {test_end}".format(**boundaries)
        if boundaries
        else "Chronological boundaries unavailable."
    )

    validation_tab, test_tab = st.tabs(["Validation metrics", "Final test metrics"])
    with validation_tab:
        _metric_grid(validation)
        if validation.get("confusion_matrix"):
            st.plotly_chart(
                confusion_matrix_figure(validation["confusion_matrix"]),
                width="stretch",
            )
    with test_tab:
        _metric_grid(test)
        chart_left, chart_right = st.columns(2)
        with chart_left:
            if test.get("confusion_matrix"):
                st.plotly_chart(
                    confusion_matrix_figure(test["confusion_matrix"]),
                    width="stretch",
                )
        with chart_right:
            st.plotly_chart(
                calibration_figure(metrics.get("calibration", [])),
                width="stretch",
            )

    st.subheader("Segmented performance")
    segments = metrics.get("segments", {})
    by_year = pd.DataFrame.from_dict(segments.get("by_year", {}), orient="index")
    if not by_year.empty:
        by_year.index.name = "year"
        year_long = by_year.reset_index().melt(
            id_vars="year",
            value_vars=[column for column in ("precision", "recall", "f1") if column in by_year],
            var_name="metric",
            value_name="value",
        )
        st.plotly_chart(
            px.line(year_long, x="year", y="value", color="metric", markers=True),
            width="stretch",
        )
    liquidity = pd.DataFrame.from_dict(segments.get("by_liquidity", {}), orient="index")
    if not liquidity.empty:
        st.dataframe(liquidity, width="stretch")

    baseline = metrics.get("baselines", {})
    with st.expander("Baseline comparison"):
        st.json(baseline)

    walk_forward = metrics.get("walk_forward", {})
    if walk_forward:
        st.subheader("Walk-forward model selection")
        selection = st.columns(5)
        selection[0].metric(
            "Selected window",
            f"{walk_forward.get('selected_training_window_years', '—')} years",
        )
        selection[1].metric(
            "Selected threshold",
            percent(walk_forward.get("selected_threshold")),
        )
        selection[2].metric(
            "Minimum call rate",
            percent(walk_forward.get("minimum_call_rate")),
        )
        selection[3].metric(
            "Validation years",
            ", ".join(str(value) for value in walk_forward.get("validation_years", [])),
        )
        selection[4].metric("Holdout year", walk_forward.get("holdout_year", "—"))
        comparison = pd.DataFrame(walk_forward.get("window_comparison", []))
        if not comparison.empty:
            visible = [
                column
                for column in (
                    "training_window_years",
                    "selected_threshold",
                    "precision",
                    "recall",
                    "call_rate",
                    "median_fold_precision",
                    "worst_fold_precision",
                )
                if column in comparison
            ]
            st.dataframe(comparison[visible], width="stretch", hide_index=True)

    st.subheader("Feature importance")
    artifact_path = root / "models/latest.joblib"
    if artifact_path.exists():
        artifact = joblib.load(artifact_path)
        model = artifact.get("model")
        names = artifact.get("metadata", {}).get("feature_names", [])
        importance = getattr(model, "feature_importances_", None)
        if importance is not None and len(importance) == len(names):
            feature_frame = (
                pd.DataFrame({"feature": names, "importance": importance})
                .sort_values("importance", ascending=False)
                .head(25)
            )
            st.plotly_chart(
                px.bar(feature_frame, x="importance", y="feature", orientation="h"),
                width="stretch",
            )
        else:
            st.info("Feature importance is unavailable in this model artifact.")
