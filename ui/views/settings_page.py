from __future__ import annotations

import copy
from pathlib import Path

import streamlit as st

from ui.services.settings_manager import (
    atomic_save_yaml,
    load_yaml,
    restore_defaults,
    validate_ui_settings,
)


def render(root: Path) -> None:
    st.markdown('<div class="page-kicker">Validated local configuration</div>', unsafe_allow_html=True)
    st.title("Settings")
    st.caption("Credentials are never displayed or edited here.")
    settings_path = root / "config/settings.yaml"
    defaults_path = root / "config/default_settings.yaml"
    current = load_yaml(settings_path)
    updated = copy.deepcopy(current)

    project = updated["project"]
    ingestion = updated["ingestion"]
    features = updated["features"]
    model = updated["model"]
    parameters = model["parameters"]
    signals = updated["signals"]

    with st.form("settings_form"):
        st.subheader("Signals and target")
        columns = st.columns(4)
        model["prediction_horizon"] = columns[0].number_input(
            "Prediction horizon (sessions)", 1, 20, int(model["prediction_horizon"])
        )
        model["minimum_return_threshold"] = columns[1].number_input(
            "Minimum future return (%)",
            -100.0,
            100.0,
            float(model["minimum_return_threshold"]) * 100,
            0.1,
        ) / 100
        signals["buy_watch_threshold"] = columns[2].number_input(
            "BUY WATCH threshold (%)",
            0.0,
            100.0,
            float(signals["buy_watch_threshold"]) * 100,
            1.0,
        ) / 100
        signals["sell_review_threshold"] = columns[3].number_input(
            "SELL REVIEW threshold (%)",
            0.0,
            100.0,
            float(signals["sell_review_threshold"]) * 100,
            1.0,
        ) / 100

        columns = st.columns(3)
        signals["minimum_validation_precision"] = columns[0].number_input(
            "Required validation precision (%)",
            0.0,
            100.0,
            float(signals["minimum_validation_precision"]) * 100,
            1.0,
        ) / 100
        model["probability_threshold"] = columns[1].number_input(
            "Classification threshold (%)",
            0.0,
            100.0,
            float(model["probability_threshold"]) * 100,
            1.0,
        ) / 100
        project["random_seed"] = columns[2].number_input(
            "Random seed", 0, 2**31 - 1, int(project["random_seed"])
        )

        st.subheader("Feature risk controls")
        columns = st.columns(4)
        features["minimum_history"] = columns[0].number_input(
            "Minimum history", 10, 5000, int(features["minimum_history"])
        )
        features["stale_price_sessions"] = columns[1].number_input(
            "Stale sessions", 1, 100, int(features["stale_price_sessions"])
        )
        features["low_volume_threshold"] = columns[2].number_input(
            "Low-volume threshold", 0.0, 1e12, float(features["low_volume_threshold"])
        )
        features["low_turnover_threshold"] = columns[3].number_input(
            "Low-turnover threshold", 0.0, 1e15, float(features["low_turnover_threshold"])
        )

        st.subheader("Network and LightGBM")
        columns = st.columns(3)
        ingestion["request_timeout_seconds"] = columns[0].number_input(
            "Request timeout", 5, 300, int(ingestion["request_timeout_seconds"])
        )
        ingestion["request_retries"] = columns[1].number_input(
            "Request retries", 1, 10, int(ingestion["request_retries"])
        )
        ingestion["rate_limit_seconds"] = columns[2].number_input(
            "Rate limit (seconds)", 0.0, 60.0, float(ingestion["rate_limit_seconds"]), 0.1
        )
        columns = st.columns(3)
        parameters["n_estimators"] = columns[0].number_input(
            "Estimators", 10, 5000, int(parameters["n_estimators"])
        )
        parameters["learning_rate"] = columns[1].number_input(
            "Learning rate", 0.0001, 1.0, float(parameters["learning_rate"]), 0.005, format="%.4f"
        )
        parameters["num_leaves"] = columns[2].number_input(
            "Leaves", 2, 1024, int(parameters["num_leaves"])
        )
        confirmation = st.checkbox("I confirm these configuration changes.")
        submitted = st.form_submit_button("Validate and save", type="primary")

    if submitted:
        errors = validate_ui_settings(updated)
        if errors:
            for error in errors:
                st.error(error)
        elif not confirmation:
            st.warning("Confirm the changes before saving.")
        else:
            backup = atomic_save_yaml(updated, settings_path)
            st.success(f"Settings saved. Backup: {backup.name if backup else 'not required'}")
            st.info(
                "Signal threshold changes require `signals`; feature changes require "
                "`build-features`; model/target changes require `run-all`."
            )

    st.divider()
    controls = st.columns(3)
    if controls[0].button("Reload from file"):
        st.rerun()
    restore_confirmed = controls[1].checkbox("Confirm defaults", key="restore_confirm")
    if controls[2].button("Restore documented defaults", disabled=not restore_confirmed):
        backup = restore_defaults(defaults_path, settings_path)
        st.success(f"Defaults restored. Backup: {backup.name if backup else 'not required'}")
        st.rerun()

