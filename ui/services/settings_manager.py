from __future__ import annotations

import os
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise TypeError("Settings file must contain a YAML mapping.")
    return value


def validate_ui_settings(values: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    project = values.get("project", {})
    ingestion = values.get("ingestion", {})
    features = values.get("features", {})
    model = values.get("model", {})
    signals = values.get("signals", {})
    parameters = model.get("parameters", {})

    def bounded(path: str, value: Any, minimum: float, maximum: float) -> None:
        try:
            numeric = float(value)
            if not minimum <= numeric <= maximum:
                errors.append(f"{path} must be between {minimum} and {maximum}.")
        except (TypeError, ValueError):
            errors.append(f"{path} must be numeric.")

    bounded("model.prediction_horizon", model.get("prediction_horizon"), 1, 20)
    bounded("model.minimum_return_threshold", model.get("minimum_return_threshold"), -1, 1)
    bounded("model.probability_threshold", model.get("probability_threshold"), 0, 1)
    bounded("signals.buy_watch_threshold", signals.get("buy_watch_threshold"), 0, 1)
    bounded("signals.sell_review_threshold", signals.get("sell_review_threshold"), 0, 1)
    bounded("signals.minimum_validation_precision", signals.get("minimum_validation_precision"), 0, 1)
    bounded("features.minimum_history", features.get("minimum_history"), 10, 5000)
    bounded("features.stale_price_sessions", features.get("stale_price_sessions"), 1, 100)
    bounded("features.low_volume_threshold", features.get("low_volume_threshold"), 0, 1e12)
    bounded("features.low_turnover_threshold", features.get("low_turnover_threshold"), 0, 1e15)
    bounded("ingestion.request_timeout_seconds", ingestion.get("request_timeout_seconds"), 5, 300)
    bounded("ingestion.request_retries", ingestion.get("request_retries"), 1, 10)
    bounded("ingestion.rate_limit_seconds", ingestion.get("rate_limit_seconds"), 0, 60)
    bounded("project.random_seed", project.get("random_seed"), 0, 2**31 - 1)
    bounded("model.parameters.n_estimators", parameters.get("n_estimators"), 10, 5000)
    bounded("model.parameters.learning_rate", parameters.get("learning_rate"), 0.0001, 1)
    bounded("model.parameters.num_leaves", parameters.get("num_leaves"), 2, 1024)
    strategy = str(model.get("training_strategy", "chronological")).strip().lower()
    if strategy not in {"chronological", "walk_forward"}:
        errors.append("model.training_strategy must be chronological or walk_forward.")
    if strategy == "walk_forward":
        walk_forward = model.get("walk_forward", {})
        bounded(
            "model.walk_forward.threshold_start",
            walk_forward.get("threshold_start"),
            0,
            1,
        )
        bounded(
            "model.walk_forward.threshold_stop",
            walk_forward.get("threshold_stop"),
            0,
            1,
        )
        bounded(
            "model.walk_forward.threshold_step",
            walk_forward.get("threshold_step"),
            0.001,
            1,
        )
        bounded(
            "model.walk_forward.minimum_call_rate",
            walk_forward.get("minimum_call_rate"),
            0,
            1,
        )
        bounded(
            "model.walk_forward.minimum_fold_call_rate",
            walk_forward.get("minimum_fold_call_rate"),
            0,
            1,
        )
        windows = walk_forward.get("training_windows_years", [])
        years = walk_forward.get("validation_years", [])
        holdout = walk_forward.get("holdout_year")
        if not isinstance(windows, list) or not windows or any(
            not isinstance(value, int) or value < 1 for value in windows
        ):
            errors.append(
                "model.walk_forward.training_windows_years must contain positive integers."
            )
        if not isinstance(years, list) or not years or any(
            not isinstance(value, int) or value < 2000 for value in years
        ):
            errors.append(
                "model.walk_forward.validation_years must contain valid calendar years."
            )
        if not isinstance(holdout, int) or holdout < 2000:
            errors.append("model.walk_forward.holdout_year must be a valid calendar year.")
        elif isinstance(years, list) and holdout in years:
            errors.append(
                "model.walk_forward.holdout_year cannot also be a validation year."
            )
    return errors


def atomic_save_yaml(values: dict[str, Any], path: Path) -> Path | None:
    errors = validate_ui_settings(values)
    if errors:
        raise ValueError(" ".join(errors))
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = None
    if path.exists():
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        backup = path.with_name(f"{path.stem}.backup-{timestamp}{path.suffix}")
        shutil.copy2(path, backup)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}-", suffix=path.suffix, dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            yaml.safe_dump(values, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return backup


def restore_defaults(default_path: Path, settings_path: Path) -> Path | None:
    defaults = load_yaml(default_path)
    return atomic_save_yaml(defaults, settings_path)
