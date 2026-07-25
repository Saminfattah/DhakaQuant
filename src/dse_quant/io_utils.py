from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd


def _atomic_target(path: Path) -> tuple[Path, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=path.suffix, dir=path.parent)
    os.close(handle)
    return Path(name), name


def atomic_write_dataframe(frame: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    temporary, _ = _atomic_target(path)
    try:
        if path.suffix.lower() == ".parquet":
            frame.to_parquet(temporary, index=index)
        elif path.suffix.lower() == ".csv":
            frame.to_csv(temporary, index=index)
        else:
            raise ValueError(f"Unsupported dataframe format: {path.suffix}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(payload: Any, path: Path) -> None:
    temporary, _ = _atomic_target(path)
    try:
        temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        for column in ("date", "prediction_date", "data_freshness_date", "ingested_at"):
            if column in frame:
                frame[column] = pd.to_datetime(frame[column], errors="coerce")
        return frame
    raise ValueError(f"Unsupported dataframe format: {path.suffix}")
