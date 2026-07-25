from __future__ import annotations

import gc
import json
import logging
import os
import tempfile
from datetime import UTC, datetime

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from dse_quant.config import Settings
from dse_quant.io_utils import atomic_write_dataframe, atomic_write_json, read_dataframe
from dse_quant.modeling.dataset import add_target, chronological_split, feature_columns
from dse_quant.modeling.evaluate import (
    baseline_metrics,
    calibration_table,
    classification_metrics,
    segmented_metrics,
)
from dse_quant.modeling.walk_forward import (
    actionable_labeled_rows,
    evaluate_thresholds,
    make_year_fold,
    select_threshold,
    threshold_candidates,
)

LOGGER = logging.getLogger(__name__)


def _features(frame: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    return frame[names].replace([np.inf, -np.inf], np.nan).astype("float32")


def _atomic_joblib(payload: dict, destination) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}-", suffix=destination.suffix, dir=destination.parent
    )
    os.close(descriptor)
    try:
        joblib.dump(payload, temporary_name)
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _model_parameters(cfg: dict, seed: int) -> dict:
    parameters = dict(cfg["parameters"])
    parameters["random_state"] = seed
    parameters["class_weight"] = cfg.get("class_weight")
    return parameters


def _fit_classifier(
    train: pd.DataFrame,
    feature_names: list[str],
    parameters: dict,
) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(**parameters)
    model.fit(
        _features(train, feature_names),
        train["target"].astype(int),
        callbacks=[lgb.log_evaluation(period=0)],
    )
    return model


def _save_model(
    settings: Settings,
    model: lgb.LGBMClassifier,
    metadata: dict,
) -> None:
    model_version = str(metadata["model_version"])
    artifact = {"model": model, "metadata": metadata}
    versioned = settings.path("models") / f"{model_version}.joblib"
    latest = settings.path("models") / "latest.joblib"
    _atomic_joblib(artifact, versioned)
    _atomic_joblib(artifact, latest)
    atomic_write_json(metadata, settings.path("models") / f"{model_version}.json")
    atomic_write_json(metadata, settings.path("models") / "latest_metrics.json")


def _base_metadata(
    settings: Settings,
    cfg: dict,
    *,
    feature_names: list[str],
    parameters: dict,
    threshold: float,
    metrics: dict,
    training_strategy: str,
) -> dict:
    created_at = datetime.now(UTC)
    metadata = {
        "model_version": created_at.strftime("lgbm-%Y%m%dT%H%M%SZ"),
        "created_at": created_at.isoformat(),
        "feature_names": feature_names,
        "prediction_horizon": int(cfg["prediction_horizon"]),
        "minimum_return_threshold": float(cfg["minimum_return_threshold"]),
        "probability_threshold": float(threshold),
        "random_seed": int(settings.section("project")["random_seed"]),
        "parameters": parameters,
        "training_strategy": training_strategy,
        "metrics": metrics,
    }
    universe_summary_path = settings.path("outputs") / "model_universe_summary.json"
    if universe_summary_path.exists():
        metadata["model_universe"] = json.loads(
            universe_summary_path.read_text(encoding="utf-8")
        )
    return metadata


def _train_chronological(settings: Settings, frame: pd.DataFrame, cfg: dict) -> dict:
    project_cfg = settings.section("project")
    labeled = add_target(
        frame,
        int(cfg["prediction_horizon"]),
        float(cfg["minimum_return_threshold"]),
    )
    split = chronological_split(
        labeled,
        validation_fraction=float(cfg["validation_fraction"]),
        test_fraction=float(cfg["test_fraction"]),
        embargo_sessions=int(cfg["embargo_sessions"]),
        minimum_training_rows=int(cfg["minimum_training_rows"]),
        minimum_validation_rows=int(cfg["minimum_validation_rows"]),
    )
    seed = int(project_cfg["random_seed"])
    parameters = _model_parameters(cfg, seed)
    model = lgb.LGBMClassifier(**parameters)
    model.fit(
        _features(split.train, split.feature_names),
        split.train["target"].astype(int),
        eval_X=_features(split.validation, split.feature_names),
        eval_y=split.validation["target"].astype(int),
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=0)],
    )
    threshold = float(cfg["probability_threshold"])
    validation_probability = model.predict_proba(
        _features(split.validation, split.feature_names)
    )[:, 1]
    test_probability = model.predict_proba(_features(split.test, split.feature_names))[:, 1]
    training_rate = float(split.train["target"].mean())
    metrics = {
        "boundaries": split.boundaries,
        "validation": classification_metrics(
            split.validation["target"], validation_probability, threshold
        ),
        "test": classification_metrics(split.test["target"], test_probability, threshold),
        "calibration": calibration_table(split.test["target"], test_probability),
        "segments": segmented_metrics(split.test, test_probability, threshold),
        "baselines": baseline_metrics(split.test["target"], training_rate),
    }
    metadata = _base_metadata(
        settings,
        cfg,
        feature_names=split.feature_names,
        parameters=parameters,
        threshold=threshold,
        metrics=metrics,
        training_strategy="chronological",
    )
    _save_model(settings, model, metadata)
    LOGGER.info(
        "Saved model %s; validation precision=%s",
        metadata["model_version"],
        metrics["validation"]["precision"],
    )
    return metadata


def _fold_metrics(scored: pd.DataFrame, threshold: float) -> dict[str, dict]:
    return {
        str(fold): classification_metrics(
            group["target"],
            group["probability"].to_numpy(),
            threshold,
        )
        for fold, group in scored.groupby("fold", observed=True)
    }


def _score_window(
    eligible: pd.DataFrame,
    *,
    window_years: int,
    validation_years: list[int],
    feature_names: list[str],
    parameters: dict,
    cfg: dict,
    walk_cfg: dict,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    scored_folds: list[pd.DataFrame] = []
    fold_boundaries: dict[str, dict[str, str]] = {}
    for year in validation_years:
        fold = make_year_fold(
            eligible,
            year=year,
            training_window_years=window_years,
            embargo_sessions=int(cfg["embargo_sessions"]),
            minimum_training_rows=int(cfg["minimum_training_rows"]),
            minimum_validation_rows=int(cfg["minimum_validation_rows"]),
        )
        LOGGER.info(
            "Walk-forward fit: %s-year window, validation %s, %s train rows, "
            "%s validation rows",
            window_years,
            year,
            len(fold.train),
            len(fold.validation),
        )
        model = _fit_classifier(fold.train, feature_names, parameters)
        probability = model.predict_proba(_features(fold.validation, feature_names))[:, 1]
        scored = fold.validation[
            ["date", "ticker", "target", "liquidity_flag"]
        ].copy()
        scored["probability"] = probability
        scored["fold"] = year
        scored_folds.append(scored)
        fold_boundaries[str(year)] = fold.boundaries
        del model, fold, probability
        gc.collect()

    out_of_fold = pd.concat(scored_folds, ignore_index=True)
    thresholds = threshold_candidates(
        float(walk_cfg["threshold_start"]),
        float(walk_cfg["threshold_stop"]),
        float(walk_cfg["threshold_step"]),
    )
    table = evaluate_thresholds(
        out_of_fold,
        thresholds,
        minimum_call_rate=float(walk_cfg["minimum_call_rate"]),
        minimum_fold_call_rate=float(walk_cfg["minimum_fold_call_rate"]),
    )
    selected = select_threshold(table)
    threshold = float(selected["threshold"])
    summary = {
        "training_window_years": window_years,
        "selected_threshold": threshold,
        "precision": float(selected["precision"]),
        "recall": float(selected["recall"]),
        "call_rate": float(selected["call_rate"]),
        "median_fold_precision": float(selected["median_fold_precision"]),
        "worst_fold_precision": float(selected["worst_fold_precision"]),
        "minimum_fold_call_rate": float(selected["minimum_fold_call_rate"]),
        "folds": _fold_metrics(out_of_fold, threshold),
        "fold_boundaries": fold_boundaries,
    }
    table.insert(0, "training_window_years", window_years)
    table["selected"] = table["threshold"].eq(threshold)
    return summary, table, out_of_fold


def _select_window(window_summaries: list[dict]) -> dict:
    return max(
        window_summaries,
        key=lambda summary: (
            summary["median_fold_precision"],
            summary["worst_fold_precision"],
            summary["precision"],
            -summary["training_window_years"],
        ),
    )


def _train_walk_forward(settings: Settings, frame: pd.DataFrame, cfg: dict) -> dict:
    walk_cfg = dict(cfg["walk_forward"])
    seed = int(settings.section("project")["random_seed"])
    parameters = _model_parameters(cfg, seed)
    targeted = add_target(
        frame,
        int(cfg["prediction_horizon"]),
        float(cfg["minimum_return_threshold"]),
    )
    eligible = actionable_labeled_rows(
        targeted,
        exclude_risk_flagged_rows=bool(walk_cfg.get("exclude_risk_flagged_rows", True)),
    )
    names = feature_columns(eligible)
    if not names:
        raise ValueError("No numeric model features were generated.")

    validation_years = [int(year) for year in walk_cfg["validation_years"]]
    holdout_year = int(walk_cfg["holdout_year"])
    if holdout_year in validation_years:
        raise ValueError("The walk-forward holdout year cannot also be a validation year.")
    windows = [int(value) for value in walk_cfg["training_windows_years"]]
    window_summaries: list[dict] = []
    selection_tables: list[pd.DataFrame] = []
    out_of_fold_by_window: dict[int, pd.DataFrame] = {}

    for window_years in windows:
        summary, table, out_of_fold = _score_window(
            eligible,
            window_years=window_years,
            validation_years=validation_years,
            feature_names=names,
            parameters=parameters,
            cfg=cfg,
            walk_cfg=walk_cfg,
        )
        window_summaries.append(summary)
        selection_tables.append(table)
        out_of_fold_by_window[window_years] = out_of_fold
        LOGGER.info(
            "Walk-forward window result: %s-year window, threshold=%.3f, "
            "precision=%.4f, call_rate=%.4f, median_fold_precision=%.4f, "
            "worst_fold_precision=%.4f",
            window_years,
            summary["selected_threshold"],
            summary["precision"],
            summary["call_rate"],
            summary["median_fold_precision"],
            summary["worst_fold_precision"],
        )

    selected = _select_window(window_summaries)
    selected_window = int(selected["training_window_years"])
    selected_threshold = float(selected["selected_threshold"])
    selected_oof = out_of_fold_by_window[selected_window]
    LOGGER.info(
        "Selected %s-year training window and %.3f threshold using walk-forward folds only",
        selected_window,
        selected_threshold,
    )

    holdout = make_year_fold(
        eligible,
        year=holdout_year,
        training_window_years=selected_window,
        embargo_sessions=int(cfg["embargo_sessions"]),
        minimum_training_rows=int(cfg["minimum_training_rows"]),
        minimum_validation_rows=int(cfg["minimum_validation_rows"]),
    )
    holdout_model = _fit_classifier(holdout.train, names, parameters)
    holdout_probability = holdout_model.predict_proba(
        _features(holdout.validation, names)
    )[:, 1]
    validation_metrics = classification_metrics(
        selected_oof["target"],
        selected_oof["probability"].to_numpy(),
        selected_threshold,
    )
    test_metrics = classification_metrics(
        holdout.validation["target"],
        holdout_probability,
        selected_threshold,
    )

    latest_labeled_date = pd.to_datetime(eligible["date"]).max()
    deployment_start = latest_labeled_date - pd.DateOffset(years=selected_window)
    deployment = eligible[
        pd.to_datetime(eligible["date"]).between(deployment_start, latest_labeled_date)
    ].copy()
    if len(deployment) < int(cfg["minimum_training_rows"]):
        raise ValueError(
            f"Final deployment rows {len(deployment)} are below required "
            f"{cfg['minimum_training_rows']}."
        )
    del holdout_model
    gc.collect()
    final_model = _fit_classifier(deployment, names, parameters)

    boundaries = {
        "train_start": str(pd.to_datetime(deployment["date"]).min().date()),
        "train_end": str(pd.to_datetime(deployment["date"]).max().date()),
        "validation_start": f"{min(validation_years)}-01-01",
        "validation_end": f"{max(validation_years)}-12-31",
        "test_start": holdout.boundaries["validation_start"],
        "test_end": holdout.boundaries["validation_end"],
    }
    metrics = {
        "boundaries": boundaries,
        "validation": validation_metrics,
        "test": test_metrics,
        "calibration": calibration_table(holdout.validation["target"], holdout_probability),
        "segments": segmented_metrics(
            holdout.validation,
            holdout_probability,
            selected_threshold,
        ),
        "baselines": baseline_metrics(
            holdout.validation["target"],
            float(holdout.train["target"].mean()),
        ),
        "walk_forward": {
            "selection_uses_holdout": False,
            "validation_years": validation_years,
            "holdout_year": holdout_year,
            "minimum_call_rate": float(walk_cfg["minimum_call_rate"]),
            "minimum_fold_call_rate": float(walk_cfg["minimum_fold_call_rate"]),
            "selected_training_window_years": selected_window,
            "selected_threshold": selected_threshold,
            "selected_fold_metrics": selected["folds"],
            "window_comparison": window_summaries,
        },
        "deployment": {
            "training_window_years": selected_window,
            "rows": len(deployment),
            "tickers": int(deployment["ticker"].nunique()),
            "train_start": boundaries["train_start"],
            "train_end": boundaries["train_end"],
            "risk_flagged_rows_excluded": bool(
                walk_cfg.get("exclude_risk_flagged_rows", True)
            ),
        },
    }
    metadata = _base_metadata(
        settings,
        cfg,
        feature_names=names,
        parameters=parameters,
        threshold=selected_threshold,
        metrics=metrics,
        training_strategy="walk_forward_recent_window",
    )
    _save_model(settings, final_model, metadata)
    selection_report = pd.concat(selection_tables, ignore_index=True)
    atomic_write_dataframe(
        selection_report,
        settings.path("outputs") / "model_selection_report.csv",
    )
    LOGGER.info(
        "Saved walk-forward model %s; OOF precision=%.4f, holdout precision=%.4f, "
        "call threshold=%.3f, deployment rows=%s",
        metadata["model_version"],
        validation_metrics["precision"],
        test_metrics["precision"],
        selected_threshold,
        len(deployment),
    )
    return metadata


def train_model(settings: Settings) -> dict:
    cfg = settings.section("model")
    frame = read_dataframe(settings.path("features") / "daily_features.parquet")
    strategy = str(cfg.get("training_strategy", "chronological")).strip().lower()
    if strategy == "walk_forward":
        return _train_walk_forward(settings, frame, cfg)
    if strategy != "chronological":
        raise ValueError(f"Unsupported model.training_strategy: {strategy}")
    return _train_chronological(settings, frame, cfg)
