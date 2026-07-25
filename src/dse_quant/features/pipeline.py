from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from dse_quant.config import Settings
from dse_quant.features.liquidity import add_liquidity_features
from dse_quant.features.technical import add_technical_features
from dse_quant.io_utils import atomic_write_dataframe, atomic_write_json, read_dataframe
from dse_quant.processing.instrument_universe import (
    DEFAULT_CANDIDATE_PATTERNS,
    filter_model_universe,
)

LOGGER = logging.getLogger(__name__)


def _breadth(values: pd.Series) -> float:
    valid = values.dropna()
    return float((valid > 0).mean()) if len(valid) else np.nan


def _market_features(frame: pd.DataFrame) -> pd.DataFrame:
    daily = frame.groupby("date", as_index=False).agg(
        market_return_1d=("return_1d", "mean"),
        market_breadth=("return_1d", _breadth),
        market_median_volume=("volume", "median"),
    )
    daily["market_return_5d"] = daily["market_return_1d"].rolling(5, min_periods=5).sum()
    return daily


def build_features(settings: Settings) -> pd.DataFrame:
    source = read_dataframe(settings.path("processed") / "daily_prices.parquet")
    cfg = settings.section("features")
    universe_cfg = settings.section("model_universe")
    if bool(universe_cfg.get("exclude_fixed_income", False)):
        classification = read_dataframe(
            settings.path("processed") / "instrument_classification.parquet"
        )
        patterns = tuple(
            universe_cfg.get("fixed_income_candidate_patterns", DEFAULT_CANDIDATE_PATTERNS)
        )
        source, universe_audit, universe_summary = filter_model_universe(
            source,
            classification,
            candidate_patterns=patterns,
        )
        atomic_write_dataframe(
            universe_audit,
            settings.path("outputs") / "model_universe_audit.csv",
        )
        atomic_write_json(
            universe_summary,
            settings.path("outputs") / "model_universe_summary.json",
        )
        LOGGER.info("Model-universe filter summary: %s", universe_summary)
    ordered = source.sort_values(["ticker", "date"]).copy()
    technical = pd.concat(
        [add_technical_features(group) for _, group in ordered.groupby("ticker", sort=False)],
        ignore_index=True,
    )
    enriched = pd.concat(
        [
            add_liquidity_features(
                group,
                lookback=int(cfg["low_volume_lookback"]),
                stale_price_sessions=int(cfg["stale_price_sessions"]),
                low_volume_threshold=float(cfg["low_volume_threshold"]),
                low_turnover_threshold=float(cfg["low_turnover_threshold"]),
                floor_flat_sessions=int(cfg["floor_flat_sessions"]),
            )
            for _, group in technical.groupby("ticker", sort=False)
        ],
        ignore_index=True,
    )
    enriched = enriched.merge(_market_features(enriched), on="date", how="left", validate="many_to_one")
    enriched["day_of_week"] = pd.to_datetime(enriched["date"]).dt.dayofweek
    enriched["month"] = pd.to_datetime(enriched["date"]).dt.month
    enriched["history_count"] = enriched.groupby("ticker").cumcount() + 1
    enriched["insufficient_history_flag"] = enriched["history_count"] < int(cfg["minimum_history"])
    numeric = enriched.select_dtypes(include=[np.number]).columns
    enriched[numeric] = enriched[numeric].replace([np.inf, -np.inf], np.nan)
    output = settings.path("features") / "daily_features.parquet"
    atomic_write_dataframe(enriched, output)
    LOGGER.info("Feature dataset saved: %s rows, %s columns", len(enriched), len(enriched.columns))
    return enriched
