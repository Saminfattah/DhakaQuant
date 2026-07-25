from __future__ import annotations

import pandas as pd

from dse_quant.io_utils import atomic_write_dataframe, read_dataframe


def test_csv_and_parquet_roundtrip(tmp_path):
    frame = pd.DataFrame(
        {
            "ticker": ["ACI"],
            "date": [pd.Timestamp("2026-01-01")],
            "probability_up": [0.75],
        }
    )
    csv_path = tmp_path / "sample.csv"
    parquet_path = tmp_path / "sample.parquet"
    atomic_write_dataframe(frame, csv_path)
    atomic_write_dataframe(frame, parquet_path)
    assert read_dataframe(csv_path).columns.tolist() == frame.columns.tolist()
    assert read_dataframe(parquet_path).columns.tolist() == frame.columns.tolist()

