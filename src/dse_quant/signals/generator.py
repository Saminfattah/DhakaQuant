from __future__ import annotations

import logging

import joblib
import pandas as pd

from dse_quant.config import Settings
from dse_quant.io_utils import atomic_write_dataframe, read_dataframe

LOGGER = logging.getLogger(__name__)


def _label(row: pd.Series, cfg: dict, model_quality_ok: bool) -> tuple[str, str]:
    reasons = [code for code in str(row.get("reason_codes", "")).split(",") if code]
    blocked = bool(
        row.get("insufficient_history_flag")
        or row.get("liquidity_flag")
        or row.get("stale_price_flag")
        or row.get("floor_price_flag")
    )
    if blocked:
        return "AVOID", ",".join(reasons)
    if row["probability_down"] >= float(cfg["sell_review_threshold"]):
        reasons.append("HIGH_DOWNSIDE_PROBABILITY")
        return "SELL REVIEW", ",".join(dict.fromkeys(reasons))
    if row["probability_up"] >= float(cfg["buy_watch_threshold"]):
        if bool(cfg.get("require_model_quality_for_buy", True)) and not model_quality_ok:
            reasons.append("MODEL_BELOW_REQUIRED_PRECISION")
            return "HOLD", ",".join(dict.fromkeys(reasons))
        reasons.append("HIGH_UPSIDE_PROBABILITY")
        return "BUY WATCH", ",".join(dict.fromkeys(reasons))
    return "HOLD", ",".join(reasons)


def generate_signals(settings: Settings) -> pd.DataFrame:
    cfg = settings.section("signals")
    predictions = read_dataframe(settings.path("outputs") / "latest_predictions.parquet")
    artifact = joblib.load(settings.path("models") / "latest.joblib")
    validation_precision = (
        artifact["metadata"].get("metrics", {}).get("validation", {}).get("precision")
    )
    model_quality_ok = (
        validation_precision is not None
        and float(validation_precision) >= float(cfg["minimum_validation_precision"])
    )
    decisions = predictions.apply(lambda row: _label(row, cfg, model_quality_ok), axis=1)
    signals = predictions.copy()
    signals["signal"] = [decision[0] for decision in decisions]
    signals["reason_codes"] = [decision[1] for decision in decisions]
    signals["validation_precision"] = validation_precision
    columns = [
        "ticker",
        "prediction_date",
        "latest_price",
        "probability_up",
        "probability_down",
        "signal",
        "model_version",
        "data_freshness_date",
        "validation_precision",
        "liquidity_flag",
        "stale_price_flag",
        "floor_price_flag",
        "reason_codes",
    ]
    signals = signals[columns]
    output_dir = settings.path("outputs")
    atomic_write_dataframe(signals, output_dir / "latest_signals.csv")
    atomic_write_dataframe(signals, output_dir / "latest_signals.parquet")
    atomic_write_dataframe(
        signals.sort_values("probability_up", ascending=False),
        output_dir / "upside_ranking.csv",
    )
    atomic_write_dataframe(
        signals.sort_values("probability_down", ascending=False),
        output_dir / "downside_ranking.csv",
    )
    LOGGER.info("Generated research signals: %s", signals["signal"].value_counts().to_dict())
    return signals

