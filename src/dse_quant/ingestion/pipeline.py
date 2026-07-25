from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from dse_quant.config import Settings
from dse_quant.ingestion.dse_archive import DSEArchiveClient, DSEArchiveError
from dse_quant.ingestion.kaggle_source import download_kaggle_history
from dse_quant.ingestion.listed_universe import (
    ListedUniverseClient,
    ListedUniverseError,
    filter_to_listed_universe,
    validate_listed_universe,
)
from dse_quant.io_utils import atomic_write_dataframe, atomic_write_json, read_dataframe
from dse_quant.processing.cleaning import merge_price_frames
from dse_quant.processing.corporate_actions import build_join_boundary_report
from dse_quant.processing.instrument_universe import update_instrument_classification
from dse_quant.processing.validation import validate_daily_prices

LOGGER = logging.getLogger(__name__)


def date_chunks(start: date, end: date, chunk_days: int):
    current = start
    while current <= end:
        chunk_end = min(current + timedelta(days=chunk_days - 1), end)
        yield current, chunk_end
        current = chunk_end + timedelta(days=1)


def run_ingestion(settings: Settings, end_date: date | None = None) -> pd.DataFrame:
    cfg = settings.section("ingestion")
    validation_cfg = settings.section("validation")
    canonical_path = settings.path("processed") / "daily_prices.parquet"
    unfiltered_path = settings.path("processed") / "daily_prices_all.parquet"
    listed_path = settings.path("processed") / "listed_universe.parquet"
    csv_path = settings.path("processed") / "daily_prices.csv"
    failure_path = settings.path("raw") / "failed_ranges.json"

    if unfiltered_path.exists():
        base = read_dataframe(unfiltered_path)
        LOGGER.info("Loaded %s existing unfiltered history rows", len(base))
    elif canonical_path.exists():
        base = read_dataframe(canonical_path)
        LOGGER.info("Loaded %s existing canonical rows for unfiltered-history migration", len(base))
    else:
        base = download_kaggle_history(
            str(cfg["kaggle_dataset"]),
            settings.path("raw") / "kaggle",
            bool(validation_cfg.get("reject_invalid_rows", True)),
        )

    newest = pd.to_datetime(base["date"]).max().date()
    override = cfg.get("archive_start_override")
    start = pd.Timestamp(override).date() if override else newest + timedelta(days=1)
    timezone_name = str(settings.section("project").get("timezone", "Asia/Dhaka"))
    end = end_date or datetime.now(ZoneInfo(timezone_name)).date()
    additions: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []

    if start <= end:
        client = DSEArchiveClient(
            url=str(cfg["dse_archive_url"]),
            timeout=int(cfg["request_timeout_seconds"]),
            retries=int(cfg["request_retries"]),
            retry_min_seconds=float(cfg["retry_min_seconds"]),
            retry_max_seconds=float(cfg["retry_max_seconds"]),
            rate_limit_seconds=float(cfg["rate_limit_seconds"]),
            user_agent=str(cfg["user_agent"]),
        )
        for chunk_start, chunk_end in date_chunks(start, end, int(cfg["chunk_days"])):
            try:
                additions.append(
                    client.fetch(
                        chunk_start,
                        chunk_end,
                        bool(validation_cfg.get("reject_invalid_rows", True)),
                    )
                )
            except requests.exceptions.SSLError as exc:
                failure = {
                    "start": str(chunk_start),
                    "end": str(chunk_end),
                    "error_type": type(exc).__name__,
                    "recoverable": False,
                    "error": str(exc),
                }
                failures.append(failure)
                atomic_write_json(failures, failure_path)
                raise DSEArchiveError(
                    "DSE HTTPS certificate validation failed using the operating-system trust "
                    "store. The canonical dataset was not replaced. Check the Windows date/time, "
                    "proxy/antivirus HTTPS inspection, and DSE certificate availability."
                ) from exc
            except requests.exceptions.RequestException as exc:
                LOGGER.warning(
                    "DSE archive connection failed for %s to %s; retaining existing history "
                    "and continuing with cached data. %s: %s",
                    chunk_start,
                    chunk_end,
                    type(exc).__name__,
                    exc,
                )
                failures.append(
                    {
                        "start": str(chunk_start),
                        "end": str(chunk_end),
                        "error_type": type(exc).__name__,
                        "recoverable": True,
                        "error": str(exc),
                    }
                )
            except DSEArchiveError as exc:
                LOGGER.error(
                    "DSE archive response could not be processed for %s to %s; "
                    "retaining existing history. %s",
                    chunk_start,
                    chunk_end,
                    exc,
                )
                failures.append(
                    {
                        "start": str(chunk_start),
                        "end": str(chunk_end),
                        "error_type": type(exc).__name__,
                        "recoverable": True,
                        "error": str(exc),
                    }
                )
            except Exception as exc:
                LOGGER.exception("DSE archive range failed: %s to %s", chunk_start, chunk_end)
                failures.append(
                    {
                        "start": str(chunk_start),
                        "end": str(chunk_end),
                        "error_type": type(exc).__name__,
                        "recoverable": False,
                        "error": str(exc),
                    }
                )

    combined_all = merge_price_frames([base, *additions])
    minimum_recent_rows = int(validation_cfg.get("minimum_rows_per_recent_session", 50))
    unfiltered_report = validate_daily_prices(combined_all, minimum_recent_rows)
    if not unfiltered_report.valid:
        raise ValueError(
            f"Refusing to replace unfiltered daily data: {unfiltered_report.to_dict()}"
        )

    if bool(cfg.get("filter_to_current_listings", True)):
        minimum_listed = int(cfg.get("minimum_listed_tickers", 200))
        client = ListedUniverseClient(
            url=str(cfg["company_listing_url"]),
            timeout=int(cfg["request_timeout_seconds"]),
            retries=int(cfg["request_retries"]),
            retry_min_seconds=float(cfg["retry_min_seconds"]),
            retry_max_seconds=float(cfg["retry_max_seconds"]),
            user_agent=str(cfg["user_agent"]),
            minimum_tickers=minimum_listed,
        )
        try:
            listed_universe = client.fetch()
            atomic_write_dataframe(listed_universe, listed_path)
            listing_source = "live"
        except Exception as exc:
            if not bool(cfg.get("allow_listing_cache_fallback", True)) or not listed_path.exists():
                raise ListedUniverseError(
                    "Could not retrieve a valid DSE listed universe and no permitted cache "
                    "was available. The canonical dataset was not replaced."
                ) from exc
            LOGGER.warning(
                "Live DSE company-listing fetch failed; using cached universe at %s: %s",
                listed_path,
                exc,
            )
            listed_universe = validate_listed_universe(
                read_dataframe(listed_path), minimum_listed
            )
            listing_source = "cache"
        combined, universe_audit = filter_to_listed_universe(combined_all, listed_universe)
        universe_summary = {
            "listing_source": listing_source,
            "company_listing_url": str(cfg["company_listing_url"]),
            "listed_tickers": int(listed_universe["ticker"].nunique()),
            "history_tickers_before_filter": int(combined_all["ticker"].nunique()),
            "history_tickers_retained": int(combined["ticker"].nunique()),
            "history_tickers_excluded": int(
                (universe_audit["status"] == "excluded_not_currently_listed").sum()
            ),
            "listed_tickers_without_history": int(
                (universe_audit["status"] == "listed_without_history").sum()
            ),
            "rows_before_filter": len(combined_all),
            "rows_after_filter": len(combined),
        }
    else:
        combined = combined_all
        universe_audit = pd.DataFrame()
        universe_summary = {
            "listing_source": "disabled",
            "history_tickers_before_filter": int(combined_all["ticker"].nunique()),
            "history_tickers_retained": int(combined_all["ticker"].nunique()),
            "history_tickers_excluded": 0,
            "rows_before_filter": len(combined_all),
            "rows_after_filter": len(combined_all),
        }

    report = validate_daily_prices(
        combined, minimum_recent_rows
    )
    if not report.valid:
        raise ValueError(f"Refusing to replace canonical data: {report.to_dict()}")
    if bool(settings.section("model_universe").get("exclude_fixed_income", False)):
        if not bool(cfg.get("filter_to_current_listings", True)):
            raise ValueError(
                "Fixed-income classification requires ingestion.filter_to_current_listings."
            )
        update_instrument_classification(settings, listed_universe, combined["ticker"])

    boundary = build_join_boundary_report(
        combined,
        sessions=int(validation_cfg.get("join_boundary_sessions", 10)),
        return_warning=float(validation_cfg.get("join_return_warning", 0.50)),
    )
    atomic_write_dataframe(boundary, settings.path("outputs") / "join_boundary_report.csv")
    atomic_write_json(report.to_dict(), settings.path("outputs") / "validation_report.json")
    atomic_write_json(
        unfiltered_report.to_dict(),
        settings.path("outputs") / "unfiltered_validation_report.json",
    )
    atomic_write_json(
        universe_summary,
        settings.path("outputs") / "universe_filter_summary.json",
    )
    if not universe_audit.empty:
        atomic_write_dataframe(
            universe_audit,
            settings.path("outputs") / "universe_filter_audit.csv",
        )
    atomic_write_json(failures, failure_path)
    atomic_write_dataframe(combined_all, unfiltered_path)
    atomic_write_dataframe(combined, canonical_path)
    if bool(cfg.get("write_csv_copy", True)):
        atomic_write_dataframe(combined, csv_path)
    LOGGER.info(
        "Canonical daily dataset saved: %s rows, %s tickers, through %s",
        len(combined),
        combined["ticker"].nunique(),
        combined["date"].max(),
    )
    if bool(cfg.get("filter_to_current_listings", True)):
        LOGGER.info("DSE listed-universe filter summary: %s", universe_summary)
    if failures:
        LOGGER.warning(
            "Ingestion completed with %s archive range warning(s). No failed range replaced "
            "existing data; canonical prices remain available through %s.",
            len(failures),
            combined["date"].max(),
        )
    return combined
