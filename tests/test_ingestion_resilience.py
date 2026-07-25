from __future__ import annotations

import json
import logging
from datetime import timedelta

import pandas as pd
import requests

from dse_quant.config import Settings
from dse_quant.ingestion.pipeline import run_ingestion
from dse_quant.io_utils import atomic_write_dataframe


def test_archive_connection_failure_is_a_recoverable_warning(
    tmp_path,
    canonical_prices,
    monkeypatch,
    caplog,
):
    values = {
        "project": {"timezone": "Asia/Dhaka"},
        "paths": {
            "raw": "data/raw",
            "processed": "data/processed",
            "features": "data/features",
            "outputs": "data/outputs",
            "models": "models",
            "logs": "logs",
        },
        "ingestion": {
            "kaggle_dataset": "unused",
            "dse_archive_url": "https://example.test/archive",
            "company_listing_url": "https://example.test/listing",
            "filter_to_current_listings": True,
            "minimum_listed_tickers": 1,
            "allow_listing_cache_fallback": True,
            "archive_start_override": None,
            "chunk_days": 30,
            "request_timeout_seconds": 1,
            "request_retries": 1,
            "retry_min_seconds": 0,
            "retry_max_seconds": 0,
            "rate_limit_seconds": 0,
            "user_agent": "test",
            "write_csv_copy": False,
        },
        "validation": {
            "reject_invalid_rows": True,
            "minimum_rows_per_recent_session": 1,
            "join_boundary_sessions": 2,
            "join_return_warning": 0.50,
        },
        "model_universe": {"exclude_fixed_income": False},
    }
    settings = Settings(root=tmp_path, values=values)
    settings.ensure_directories()
    atomic_write_dataframe(
        canonical_prices,
        settings.path("processed") / "daily_prices_all.parquet",
    )
    listed = pd.DataFrame(
        {
            "ticker": ["ACI", "BATBC"],
            "detail_url": [
                "https://example.test/company?name=ACI",
                "https://example.test/company?name=BATBC",
            ],
        }
    )

    def fail_archive(*args, **kwargs):
        raise requests.ConnectionError("[WinError 10013] socket access denied")

    monkeypatch.setattr(
        "dse_quant.ingestion.pipeline.DSEArchiveClient.fetch",
        fail_archive,
    )
    monkeypatch.setattr(
        "dse_quant.ingestion.pipeline.ListedUniverseClient.fetch",
        lambda *args, **kwargs: listed,
    )
    end_date = pd.to_datetime(canonical_prices["date"]).max().date() + timedelta(days=1)

    with caplog.at_level(logging.WARNING):
        result = run_ingestion(settings, end_date=end_date)

    failures = json.loads(
        (settings.path("raw") / "failed_ranges.json").read_text(encoding="utf-8")
    )
    assert len(result) == len(canonical_prices)
    assert failures[0]["error_type"] == "ConnectionError"
    assert failures[0]["recoverable"] is True
    pipeline_records = [
        record
        for record in caplog.records
        if record.name == "dse_quant.ingestion.pipeline"
    ]
    assert any("retaining existing history" in record.message for record in pipeline_records)
    assert not any(record.levelno >= logging.ERROR for record in pipeline_records)
