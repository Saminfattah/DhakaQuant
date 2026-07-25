from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _safe_metric(function, *args, **kwargs):
    try:
        value = function(*args, **kwargs)
        return float(value)
    except ValueError:
        return None


def classification_metrics(
    target: pd.Series | np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> dict:
    truth = np.asarray(target, dtype=int)
    raw_probability = np.asarray(probability, dtype=float)
    coverage = float(np.isfinite(raw_probability).mean()) if len(raw_probability) else 0.0
    probability = np.clip(raw_probability, 1e-7, 1 - 1e-7)
    predicted = (probability >= threshold).astype(int)
    matrix = confusion_matrix(truth, predicted, labels=[0, 1])
    return {
        "rows": len(truth),
        "prediction_coverage": coverage,
        "positive_calls": int(predicted.sum()),
        "positive_call_rate": float(predicted.mean()) if len(predicted) else 0.0,
        "positive_rate": float(truth.mean()) if len(truth) else None,
        "precision": _safe_metric(precision_score, truth, predicted, zero_division=0),
        "recall": _safe_metric(recall_score, truth, predicted, zero_division=0),
        "f1": _safe_metric(f1_score, truth, predicted, zero_division=0),
        "roc_auc": _safe_metric(roc_auc_score, truth, probability),
        "pr_auc": _safe_metric(average_precision_score, truth, probability),
        "log_loss": _safe_metric(log_loss, truth, probability, labels=[0, 1]),
        "brier_score": _safe_metric(brier_score_loss, truth, probability),
        "confusion_matrix": matrix.tolist(),
    }


def calibration_table(target: pd.Series, probability: np.ndarray) -> list[dict]:
    table = pd.DataFrame({"target": target.to_numpy(), "probability": probability})
    table["bucket"] = pd.cut(
        table["probability"], bins=np.linspace(0, 1, 11), include_lowest=True, duplicates="drop"
    )
    grouped = table.groupby("bucket", observed=True).agg(
        rows=("target", "size"),
        predicted_probability=("probability", "mean"),
        observed_rate=("target", "mean"),
    )
    return [
        {
            "bucket": str(index),
            "rows": int(row["rows"]),
            "predicted_probability": float(row["predicted_probability"]),
            "observed_rate": float(row["observed_rate"]),
        }
        for index, row in grouped.iterrows()
    ]


def segmented_metrics(
    frame: pd.DataFrame, probability: np.ndarray, threshold: float
) -> dict[str, dict]:
    scored = frame[["date", "target", "liquidity_flag"]].copy()
    scored["probability"] = probability
    scored["year"] = pd.to_datetime(scored["date"]).dt.year
    by_year = {
        str(year): classification_metrics(group["target"], group["probability"].to_numpy(), threshold)
        for year, group in scored.groupby("year")
    }
    by_liquidity = {
        ("low_liquidity" if bool(flag) else "normal_liquidity"): classification_metrics(
            group["target"], group["probability"].to_numpy(), threshold
        )
        for flag, group in scored.groupby("liquidity_flag")
    }
    return {"by_year": by_year, "by_liquidity": by_liquidity}


def baseline_metrics(target: pd.Series, training_positive_rate: float) -> dict:
    truth = target.to_numpy(dtype=int)
    frequency = np.full(len(truth), min(max(training_positive_rate, 1e-7), 1 - 1e-7))
    always_up = np.ones(len(truth), dtype=int)
    return {
        "always_up_precision": float(precision_score(truth, always_up, zero_division=0)),
        "always_up_accuracy": float((truth == always_up).mean()),
        "historical_frequency": float(training_positive_rate),
        "historical_frequency_log_loss": float(log_loss(truth, frequency, labels=[0, 1])),
        "historical_frequency_brier": float(brier_score_loss(truth, frequency)),
    }
