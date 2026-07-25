from __future__ import annotations

import numpy as np
import pandas as pd

from dse_quant.config import Settings
from dse_quant.features.pipeline import build_features
from dse_quant.io_utils import atomic_write_dataframe
from dse_quant.modeling.predict import generate_predictions
from dse_quant.modeling.train import train_model
from dse_quant.signals.generator import generate_signals


def test_synthetic_pipeline_end_to_end(tmp_path):
    values = {
        "project": {"timezone": "Asia/Dhaka", "random_seed": 42},
        "paths": {
            "raw": "data/raw",
            "processed": "data/processed",
            "features": "data/features",
            "outputs": "data/outputs",
            "models": "models",
            "logs": "logs",
        },
        "features": {
            "minimum_history": 30,
            "stale_price_sessions": 5,
            "low_volume_lookback": 20,
            "low_volume_threshold": 100,
            "low_turnover_threshold": 1000,
            "floor_flat_sessions": 3,
        },
        "model": {
            "prediction_horizon": 3,
            "minimum_return_threshold": 0.0,
            "validation_fraction": 0.15,
            "test_fraction": 0.15,
            "embargo_sessions": 3,
            "probability_threshold": 0.50,
            "class_weight": "balanced",
            "minimum_training_rows": 100,
            "minimum_validation_rows": 20,
            "parameters": {
                "objective": "binary",
                "n_estimators": 25,
                "learning_rate": 0.1,
                "num_leaves": 15,
                "verbosity": -1,
                "n_jobs": 1,
            },
        },
        "signals": {
            "buy_watch_threshold": 0.70,
            "sell_review_threshold": 0.60,
            "minimum_validation_precision": 0.0,
            "require_model_quality_for_buy": True,
        },
    }
    settings = Settings(root=tmp_path, values=values)
    settings.ensure_directories()
    dates = pd.bdate_range("2024-01-01", periods=220)
    rows = []
    for ticker_index, ticker in enumerate(("ACI", "BATBC", "SQURPHARMA", "GP")):
        for index, day in enumerate(dates):
            close = 100 + ticker_index * 20 + np.sin(index / 5) * 5 + index * 0.03
            rows.append(
                {
                    "date": day,
                    "ticker": ticker,
                    "open": close * 0.998,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": 20_000 + (index % 30) * 500 + ticker_index * 100,
                    "trade_count": 100 + index % 20,
                    "source": "synthetic",
                    "ingested_at": pd.Timestamp("2026-01-01", tz="UTC"),
                }
            )
    prices = pd.DataFrame(rows)
    atomic_write_dataframe(prices, settings.path("processed") / "daily_prices.parquet")

    features = build_features(settings)
    metadata = train_model(settings)
    predictions = generate_predictions(settings)
    signals = generate_signals(settings)

    assert len(features) == len(prices)
    assert metadata["metrics"]["validation"]["rows"] > 0
    assert set(predictions["ticker"]) == {"ACI", "BATBC", "SQURPHARMA", "GP"}
    assert predictions["probability_up"].between(0, 1).all()
    assert signals["signal"].isin({"BUY WATCH", "HOLD", "AVOID", "SELL REVIEW"}).all()
    assert (settings.path("outputs") / "latest_signals.csv").exists()

