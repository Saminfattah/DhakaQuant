from __future__ import annotations

import logging
import shutil
from pathlib import Path

import kagglehub
import pandas as pd

from dse_quant.processing.cleaning import normalize_price_frame

LOGGER = logging.getLogger(__name__)
REQUIRED_ALIASES = {"date", "ticker", "scrip", "trading code", "open", "openp*", "high", "low", "close", "closep*", "volume"}


def _candidate_score(path: Path) -> tuple[int, int]:
    try:
        sample = pd.read_csv(path, nrows=5)
    except (OSError, UnicodeError, ValueError, pd.errors.ParserError):
        return (0, 0)
    columns = {str(column).strip().lower() for column in sample.columns}
    recognized = len(columns & REQUIRED_ALIASES)
    return (recognized, path.stat().st_size)


def select_price_csv(download_dir: Path) -> Path:
    candidates = list(download_dir.rglob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No CSV files found in Kaggle download: {download_dir}")
    ranked = sorted(candidates, key=_candidate_score, reverse=True)
    if _candidate_score(ranked[0])[0] < 6:
        raise ValueError("No Kaggle CSV has the required DSE price columns.")
    return ranked[0]


def download_kaggle_history(dataset: str, cache_dir: Path, reject_invalid: bool = True) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_csv = cache_dir / "kaggle_daily_prices.csv"
    if local_csv.exists():
        LOGGER.info("Using cached Kaggle history: %s", local_csv)
    else:
        LOGGER.info("Downloading Kaggle dataset %s", dataset)
        downloaded = Path(kagglehub.dataset_download(dataset))
        selected = select_price_csv(downloaded)
        shutil.copy2(selected, local_csv)
    frame = pd.read_csv(local_csv, low_memory=False)
    return normalize_price_frame(frame, source="kaggle", reject_invalid=reject_invalid)
