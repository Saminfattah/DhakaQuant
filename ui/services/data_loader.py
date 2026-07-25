from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import streamlit as st

from ui.services.settings_manager import load_yaml


class UIDataError(RuntimeError):
    """A concise, user-facing local artifact error."""


@dataclass(frozen=True)
class FileSignature:
    path: str
    modified_ns: int
    size: int

    def as_tuple(self) -> tuple[str, int, int]:
        return (self.path, self.modified_ns, self.size)


def file_signature(path: Path) -> FileSignature:
    if not path.exists():
        raise FileNotFoundError(path)
    stat = path.stat()
    return FileSignature(str(path.resolve()), stat.st_mtime_ns, stat.st_size)


def preferred_artifact(parquet_path: Path, csv_path: Path) -> Path | None:
    if parquet_path.exists():
        return parquet_path
    if csv_path.exists():
        return csv_path
    return None


def require_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise UIDataError(f"{label} is missing required columns: {', '.join(missing)}")


@st.cache_data(show_spinner=False)
def _cached_frame(
    signature: tuple[str, int, int],
    columns: tuple[str, ...] | None,
    filters: tuple[tuple[str, str, str], ...] | None,
) -> pd.DataFrame:
    path = Path(signature[0])
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(
            path,
            columns=list(columns) if columns else None,
            filters=list(filters) if filters else None,
        )
    frame = pd.read_csv(path, usecols=list(columns) if columns else None)
    if filters:
        for column, operator, value in filters:
            if operator != "==":
                raise UIDataError(f"Unsupported CSV filter: {operator}")
            frame = frame[frame[column].astype(str) == value]
    return frame


def load_frame(
    path: Path,
    *,
    required: Iterable[str] = (),
    label: str = "Artifact",
    columns: Iterable[str] | None = None,
    filters: Iterable[tuple[str, str, str]] | None = None,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = _cached_frame(
        file_signature(path).as_tuple(),
        tuple(columns) if columns else None,
        tuple(filters) if filters else None,
    ).copy()
    for column in ("date", "prediction_date", "data_freshness_date", "ingested_at"):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    require_columns(frame, required, label)
    return frame


@st.cache_data(show_spinner=False)
def _cached_json(signature: tuple[str, int, int]) -> dict[str, Any] | list[Any]:
    return json.loads(Path(signature[0]).read_text(encoding="utf-8"))


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return _cached_json(file_signature(path).as_tuple())
    except (OSError, json.JSONDecodeError) as exc:
        raise UIDataError(f"Could not read {path.name}: {exc}") from exc


def load_signals(root: Path) -> pd.DataFrame:
    path = preferred_artifact(
        root / "data/outputs/latest_signals.parquet",
        root / "data/outputs/latest_signals.csv",
    )
    if path is None:
        raise FileNotFoundError("No signal output exists yet.")
    required = {
        "ticker",
        "probability_up",
        "probability_down",
        "signal",
        "model_version",
        "reason_codes",
    }
    frame = load_frame(path, required=required, label="Signal output")
    metadata = load_json(root / "models/latest_metrics.json", default={})
    settings_path = root / "config/settings.yaml"
    if settings_path.exists():
        settings = load_yaml(settings_path)
        signal_settings = settings.get("signals", {})
        required_precision = float(signal_settings.get("minimum_validation_precision", 1))
        require_quality = bool(signal_settings.get("require_model_quality_for_buy", True))
        actual_precision = (
            metadata.get("metrics", {}).get("validation", {}).get("precision")
            if isinstance(metadata, dict)
            else None
        )
        frame = enforce_model_quality_gate(
            frame,
            actual_precision=actual_precision,
            required_precision=required_precision,
            require_quality=require_quality,
        )
    return frame


def enforce_model_quality_gate(
    frame: pd.DataFrame,
    *,
    actual_precision: float | None,
    required_precision: float,
    require_quality: bool,
) -> pd.DataFrame:
    result = frame.copy()
    quality_passed = actual_precision is not None and actual_precision >= required_precision
    if require_quality and not quality_passed:
        blocked = result["signal"].eq("BUY WATCH")
        result.loc[blocked, "signal"] = "HOLD"
        existing = result.loc[blocked, "reason_codes"].fillna("").astype(str)
        result.loc[blocked, "reason_codes"] = existing.apply(
            lambda value: ",".join(
                dict.fromkeys([code for code in value.split(",") if code] + ["MODEL_BELOW_REQUIRED_PRECISION"])
            )
        )
    return result


def load_predictions(root: Path) -> pd.DataFrame:
    path = preferred_artifact(
        root / "data/outputs/latest_predictions.parquet",
        root / "data/outputs/latest_predictions.csv",
    )
    if path is None:
        raise FileNotFoundError("No prediction output exists yet.")
    return load_frame(
        path,
        required={"ticker", "probability_up", "probability_down", "model_version"},
        label="Prediction output",
    )


def parquet_metadata(path: Path) -> dict[str, Any]:
    signature = file_signature(path)
    return _cached_parquet_metadata(signature.as_tuple())


@st.cache_data(show_spinner=False)
def _cached_parquet_metadata(signature: tuple[str, int, int]) -> dict[str, Any]:
    metadata = pq.ParquetFile(signature[0]).metadata
    return {"rows": metadata.num_rows, "columns": metadata.num_columns}


def list_tickers(root: Path) -> list[str]:
    feature_path = root / "data/features/daily_features.parquet"
    path = (
        feature_path
        if feature_path.exists()
        else root / "data/processed/daily_prices.parquet"
    )
    if not path.exists():
        return []
    frame = load_frame(path, columns=["ticker"])
    return sorted(frame["ticker"].dropna().astype(str).unique().tolist())


TICKER_FEATURE_COLUMNS = (
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "sma_20",
    "sma_50",
    "sma_200",
    "rsi_14",
    "macd",
    "macd_signal",
    "volatility_20",
    "relative_volume_20",
)


def load_ticker_features(root: Path, ticker: str) -> pd.DataFrame:
    path = root / "data/features/daily_features.parquet"
    if not path.exists():
        raise FileNotFoundError("Feature data does not exist. Run build-features first.")
    return load_frame(
        path,
        required={"date", "ticker", "close", "volume"},
        label=f"{ticker} features",
        columns=TICKER_FEATURE_COLUMNS,
        filters=[("ticker", "==", ticker.upper())],
    ).sort_values("date")


def filter_date_range(frame: pd.DataFrame, period: str) -> pd.DataFrame:
    if frame.empty or period == "All":
        return frame
    latest = pd.to_datetime(frame["date"]).max()
    offsets = {
        "3M": pd.DateOffset(months=3),
        "6M": pd.DateOffset(months=6),
        "1Y": pd.DateOffset(years=1),
        "3Y": pd.DateOffset(years=3),
        "5Y": pd.DateOffset(years=5),
    }
    if period not in offsets:
        raise ValueError(f"Unsupported date range: {period}")
    return frame[pd.to_datetime(frame["date"]) >= latest - offsets[period]]


def filter_signals(
    frame: pd.DataFrame,
    *,
    search: str = "",
    signals: Iterable[str] = (),
    minimum_up: float = 0,
    minimum_down: float = 0,
    liquidity: str = "All",
    stale: str = "All",
    floor: str = "All",
    model_versions: Iterable[str] = (),
) -> pd.DataFrame:
    result = frame.copy()
    if search.strip():
        result = result[result["ticker"].str.contains(search.strip(), case=False, na=False)]
    selected_signals = list(signals)
    if selected_signals:
        result = result[result["signal"].isin(selected_signals)]
    result = result[
        result["probability_up"].ge(minimum_up)
        & result["probability_down"].ge(minimum_down)
    ]
    for column, selection in (
        ("liquidity_flag", liquidity),
        ("stale_price_flag", stale),
        ("floor_price_flag", floor),
    ):
        if selection != "All" and column in result:
            result = result[result[column].fillna(False).astype(bool) == (selection == "Flagged")]
    selected_versions = list(model_versions)
    if selected_versions:
        result = result[result["model_version"].isin(selected_versions)]
    return result


def tail_text(path: Path, lines: int = 40) -> str:
    if not path.exists():
        return "No log file exists yet."
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        content = handle.readlines()
    return "".join(content[-lines:])
