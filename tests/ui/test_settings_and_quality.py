from __future__ import annotations

import yaml

from ui.components.warnings import model_quality_gate
from ui.services.settings_manager import atomic_save_yaml, load_yaml, validate_ui_settings


def valid_settings() -> dict:
    return {
        "project": {"random_seed": 42},
        "ingestion": {
            "request_timeout_seconds": 30,
            "request_retries": 4,
            "rate_limit_seconds": 1,
        },
        "features": {
            "minimum_history": 60,
            "stale_price_sessions": 5,
            "low_volume_threshold": 10000,
            "low_turnover_threshold": 100000,
        },
        "model": {
            "prediction_horizon": 3,
            "minimum_return_threshold": 0,
            "probability_threshold": 0.5,
            "parameters": {"n_estimators": 100, "learning_rate": 0.05, "num_leaves": 31},
        },
        "signals": {
            "buy_watch_threshold": 0.7,
            "sell_review_threshold": 0.6,
            "minimum_validation_precision": 0.65,
        },
    }


def test_quality_gate_threshold_boundary():
    assert model_quality_gate(0.65, 0.65).passed
    assert not model_quality_gate(0.649, 0.65).buy_watch_enabled
    assert model_quality_gate(0.1, 0.65, require_quality=False).buy_watch_enabled


def test_settings_validation_rejects_probability_over_one():
    values = valid_settings()
    values["signals"]["buy_watch_threshold"] = 1.5
    errors = validate_ui_settings(values)
    assert any("buy_watch_threshold" in error for error in errors)


def test_atomic_settings_save_creates_backup(tmp_path):
    path = tmp_path / "settings.yaml"
    original = valid_settings()
    path.write_text(yaml.safe_dump(original), encoding="utf-8")
    updated = valid_settings()
    updated["project"]["random_seed"] = 99
    backup = atomic_save_yaml(updated, path)
    assert backup is not None and backup.exists()
    assert load_yaml(path)["project"]["random_seed"] == 99
    assert load_yaml(backup)["project"]["random_seed"] == 42


def test_walk_forward_settings_reject_holdout_leakage():
    values = valid_settings()
    values["model"]["training_strategy"] = "walk_forward"
    values["model"]["walk_forward"] = {
        "training_windows_years": [5, 7],
        "validation_years": [2024, 2025],
        "holdout_year": 2025,
        "threshold_start": 0.45,
        "threshold_stop": 0.75,
        "threshold_step": 0.025,
        "minimum_call_rate": 0.05,
        "minimum_fold_call_rate": 0.02,
    }
    errors = validate_ui_settings(values)
    assert any("holdout_year cannot" in error for error in errors)
