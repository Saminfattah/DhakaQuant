from __future__ import annotations

import logging
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd

from dse_quant.config import Settings
from dse_quant.io_utils import atomic_write_dataframe, read_dataframe

LOGGER = logging.getLogger(__name__)


def _prediction_reason_codes(row: pd.Series) -> str:
    reasons: list[str] = []
    if bool(row.get("insufficient_history_flag", False)):
        reasons.append("INSUFFICIENT_HISTORY")
    if bool(row.get("liquidity_flag", False)):
        reasons.append("LOW_LIQUIDITY")
    if bool(row.get("low_volume_flag", False)):
        reasons.append("LOW_VOLUME")
    if bool(row.get("stale_price_flag", False)):
        reasons.append("STALE_PRICE")
    if bool(row.get("data_not_fresh_flag", False)):
        reasons.append("DATA_NOT_FRESH")
    if bool(row.get("floor_price_flag", False)):
        reasons.append("POSSIBLE_FLOOR_PRICE")
    return ",".join(reasons)


def generate_predictions(settings: Settings) -> pd.DataFrame:
    artifact_path = settings.path("models") / "latest.joblib"
    if not artifact_path.exists():
        raise FileNotFoundError("No trained model found. Run the train command first.")
    artifact = joblib.load(artifact_path)
    model = artifact["model"]
    metadata = artifact["metadata"]
    feature_names = list(metadata["feature_names"])
    frame = read_dataframe(settings.path("features") / "daily_features.parquet")
    ordered = frame.sort_values(["ticker", "date"])
    latest = ordered.groupby("ticker", as_index=False).tail(1).copy()
    missing = sorted(set(feature_names) - set(latest.columns))
    if missing:
        raise ValueError(f"Feature dataset is incompatible with model; missing: {missing}")

    matrix = latest[feature_names].replace([np.inf, -np.inf], np.nan).astype(float)
    probability_up = model.predict_proba(matrix)[:, 1]
    global_freshness = pd.to_datetime(frame["date"]).max()
    latest["probability_up"] = probability_up
    latest["probability_down"] = 1 - probability_up
    latest["predicted_class"] = (
        probability_up >= float(metadata["probability_threshold"])
    ).astype(int)
    latest["model_version"] = metadata["model_version"]
    latest["prediction_date"] = pd.to_datetime(latest["date"])
    latest["data_freshness_date"] = global_freshness
    latest["latest_price"] = latest["close"]
    latest["data_not_fresh_flag"] = latest["prediction_date"] < global_freshness
    latest["stale_price_flag"] = latest["stale_price_flag"] | latest["data_not_fresh_flag"]
    latest["reason_codes"] = latest.apply(_prediction_reason_codes, axis=1)
    columns = [
        "ticker",
        "prediction_date",
        "latest_price",
        "probability_up",
        "probability_down",
        "predicted_class",
        "model_version",
        "data_freshness_date",
        "liquidity_flag",
        "stale_price_flag",
        "floor_price_flag",
        "insufficient_history_flag",
        "reason_codes",
    ]
    predictions = latest[columns].sort_values("probability_up", ascending=False).reset_index(drop=True)
    output_dir = settings.path("outputs")
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    atomic_write_dataframe(predictions, output_dir / "latest_predictions.csv")
    atomic_write_dataframe(predictions, output_dir / "latest_predictions.parquet")
    atomic_write_dataframe(predictions, output_dir / f"predictions_{timestamp}.parquet")
    LOGGER.info("Generated predictions for %s tickers", len(predictions))
    return predictions
