from __future__ import annotations

import argparse
import logging
import sys
from datetime import date

import pandas as pd

from dse_quant.config import load_settings
from dse_quant.features.pipeline import build_features
from dse_quant.ingestion.pipeline import run_ingestion
from dse_quant.io_utils import atomic_write_json, read_dataframe
from dse_quant.logging_config import configure_logging
from dse_quant.modeling.predict import generate_predictions
from dse_quant.modeling.train import train_model
from dse_quant.processing.validation import validate_daily_prices
from dse_quant.signals.generator import generate_signals
from dse_quant.tls import configure_tls

LOGGER = logging.getLogger(__name__)


def _parse_date(value: str | None) -> date | None:
    return pd.Timestamp(value).date() if value else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dse-quant",
        description="Local-only DSE daily data, predictions, and research signals.",
    )
    parser.add_argument("--config", help="Path to settings YAML.")
    parser.add_argument("--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)
    ingest = subparsers.add_parser("ingest", help="Update canonical daily prices.")
    ingest.add_argument("--end-date", help="Optional inclusive YYYY-MM-DD end date.")
    subparsers.add_parser("validate", help="Validate canonical daily prices.")
    subparsers.add_parser("build-features", help="Build leakage-safe feature data.")
    subparsers.add_parser("train", help="Train and evaluate a chronological LightGBM model.")
    subparsers.add_parser("predict", help="Generate latest predictions from the approved model.")
    subparsers.add_parser("signals", help="Generate research labels and rankings.")
    run_all = subparsers.add_parser("run-all", help="Ingest, build, train, predict, and signal.")
    run_all.add_argument("--end-date", help="Optional inclusive YYYY-MM-DD end date.")
    daily = subparsers.add_parser(
        "daily-run", help="Update, rebuild features, predict, and signal without retraining."
    )
    daily.add_argument("--end-date", help="Optional inclusive YYYY-MM-DD end date.")
    return parser


def execute(args: argparse.Namespace) -> None:
    settings = load_settings(args.config)
    configure_logging(settings.path("logs"), args.verbose)
    configure_tls(bool(settings.section("ingestion").get("use_system_ca_store", True)))
    command = args.command
    end_date = _parse_date(getattr(args, "end_date", None))

    if command == "ingest":
        run_ingestion(settings, end_date)
    elif command == "validate":
        frame = read_dataframe(settings.path("processed") / "daily_prices.parquet")
        cfg = settings.section("validation")
        report = validate_daily_prices(
            frame, int(cfg.get("minimum_rows_per_recent_session", 50))
        )
        atomic_write_json(report.to_dict(), settings.path("outputs") / "validation_report.json")
        if not report.valid:
            raise ValueError(f"Canonical data failed validation: {report.to_dict()}")
        LOGGER.info("Validation passed: %s", report.to_dict())
    elif command == "build-features":
        build_features(settings)
    elif command == "train":
        train_model(settings)
    elif command == "predict":
        generate_predictions(settings)
    elif command == "signals":
        generate_signals(settings)
    elif command == "run-all":
        run_ingestion(settings, end_date)
        build_features(settings)
        train_model(settings)
        generate_predictions(settings)
        generate_signals(settings)
    elif command == "daily-run":
        run_ingestion(settings, end_date)
        build_features(settings)
        generate_predictions(settings)
        generate_signals(settings)
    else:
        raise ValueError(f"Unknown command: {command}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        execute(args)
        return 0
    except Exception:
        logging.basicConfig(level=logging.ERROR)
        LOGGER.exception("Command failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
