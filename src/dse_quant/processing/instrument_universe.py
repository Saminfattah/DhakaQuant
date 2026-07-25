from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime

import pandas as pd
import requests
from lxml import html as lxml_html
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from dse_quant.config import Settings
from dse_quant.io_utils import atomic_write_dataframe, read_dataframe

LOGGER = logging.getLogger(__name__)
DEFAULT_CANDIDATE_PATTERNS = (
    r"^TB(?:2|5|10|15|20)Y\d+$",
    r"BOND",
    r"SUKUK",
    r"PB$",
    r"^DEB",
)


class InstrumentClassificationError(RuntimeError):
    pass


def fixed_income_candidates(
    tickers: pd.Series | list[str] | set[str],
    patterns: list[str] | tuple[str, ...] = DEFAULT_CANDIDATE_PATTERNS,
) -> set[str]:
    compiled = [re.compile(pattern, flags=re.IGNORECASE) for pattern in patterns]
    normalized = {str(ticker).strip().upper() for ticker in tickers}
    return {
        ticker
        for ticker in normalized
        if any(pattern.search(ticker) for pattern in compiled)
    }


def _extract_field(text: str, pattern: str, field: str) -> str:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        raise InstrumentClassificationError(
            f"DSE security detail page is missing the {field} field."
        )
    return " ".join(match.group(1).split())


def parse_instrument_profile(
    page_html: str,
    *,
    ticker: str,
    detail_url: str,
    classified_at: datetime | None = None,
) -> dict[str, object]:
    try:
        root = lxml_html.fromstring(page_html)
    except (ValueError, TypeError) as exc:
        raise InstrumentClassificationError("DSE returned invalid security-detail HTML.") from exc
    text = " ".join(root.text_content().split())
    instrument_pattern = (
        r"Type of Instrument\s+(.*?)\s+"
        r"(?:Face/par Value|Total No\. of Outstanding Securities)"
    )
    instrument_match = re.search(instrument_pattern, text, flags=re.IGNORECASE)
    if not instrument_match:
        raise InstrumentClassificationError(
            "DSE security detail page is missing the Type of Instrument field."
        )
    instrument_type = " ".join(instrument_match.group(1).split())
    security_details = text[instrument_match.end() :]
    sector = _extract_field(
        security_details,
        r"Sector\s+(.*?)\s+(?:Closing Price Graph|Face/par Value|Market Lot|Tenure)",
        "Sector",
    )
    classification_text = f"{instrument_type} {sector}".lower()
    is_fixed_income = any(
        marker in classification_text
        for marker in ("bond", "debt", "sukuk", "g-sec", "debenture")
    ) and instrument_type.strip().lower() != "equity"
    return {
        "ticker": str(ticker).strip().upper(),
        "instrument_type": instrument_type,
        "sector": sector,
        "is_fixed_income": is_fixed_income,
        "detail_url": detail_url,
        "classification_source": "official_dse_security_detail",
        "classified_at": pd.Timestamp(classified_at or datetime.now(UTC)),
    }


class DSEInstrumentProfileClient:
    def __init__(
        self,
        *,
        timeout: int,
        retries: int,
        retry_min_seconds: float,
        retry_max_seconds: float,
        user_agent: str,
        workers: int = 4,
    ) -> None:
        self.timeout = timeout
        self.user_agent = user_agent
        self.workers = max(1, workers)
        self._fetch_with_retry = retry(
            stop=stop_after_attempt(retries),
            wait=wait_exponential(min=retry_min_seconds, max=retry_max_seconds),
            retry=retry_if_exception_type(
                (requests.RequestException, InstrumentClassificationError)
            ),
            reraise=True,
        )(self._fetch_once)

    def _fetch_once(self, ticker: str, detail_url: str) -> dict[str, object]:
        response = requests.get(
            detail_url,
            headers={"User-Agent": self.user_agent},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return parse_instrument_profile(
            response.text,
            ticker=ticker,
            detail_url=detail_url,
        )

    def fetch_many(self, securities: pd.DataFrame) -> pd.DataFrame:
        if securities.empty:
            return pd.DataFrame()
        LOGGER.info("Classifying %s new DSE fixed-income candidates", len(securities))
        profiles: list[dict[str, object]] = []
        failures: list[str] = []
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {
                executor.submit(self._fetch_with_retry, row.ticker, row.detail_url): row.ticker
                for row in securities.itertuples(index=False)
            }
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    profiles.append(future.result())
                except (requests.RequestException, InstrumentClassificationError) as exc:
                    failures.append(f"{ticker}: {exc}")
        if failures:
            raise InstrumentClassificationError(
                "Could not verify DSE instrument classifications: " + "; ".join(failures)
            )
        return pd.DataFrame(profiles).sort_values("ticker").reset_index(drop=True)


def update_instrument_classification(
    settings: Settings,
    listed_universe: pd.DataFrame,
    available_tickers: pd.Series,
) -> pd.DataFrame:
    cfg = settings.section("model_universe")
    if not bool(cfg.get("exclude_fixed_income", False)):
        return pd.DataFrame()

    patterns = tuple(cfg.get("fixed_income_candidate_patterns", DEFAULT_CANDIDATE_PATTERNS))
    candidates = fixed_income_candidates(available_tickers, patterns)
    classification_path = settings.path("processed") / "instrument_classification.parquet"
    if classification_path.exists():
        cached = read_dataframe(classification_path)
    else:
        cached = pd.DataFrame()
    if not cached.empty:
        required = {"ticker", "instrument_type", "sector", "is_fixed_income"}
        if not required <= set(cached.columns):
            cached = pd.DataFrame()
        else:
            valid_cache = (
                cached["instrument_type"].notna()
                & cached["sector"].notna()
                & cached["sector"].astype(str).str.len().between(1, 100)
            )
            cached = cached.loc[valid_cache].copy()
    known = set(cached["ticker"].astype(str)) if "ticker" in cached else set()
    missing = sorted(candidates - known)

    if missing:
        details = listed_universe[listed_universe["ticker"].isin(missing)][
            ["ticker", "detail_url"]
        ].drop_duplicates("ticker")
        unavailable = sorted(set(missing) - set(details["ticker"]))
        if unavailable:
            raise InstrumentClassificationError(
                "The official DSE listing has no security-detail link for candidates: "
                + ", ".join(unavailable)
            )
        ingestion_cfg = settings.section("ingestion")
        client = DSEInstrumentProfileClient(
            timeout=int(ingestion_cfg["request_timeout_seconds"]),
            retries=int(ingestion_cfg["request_retries"]),
            retry_min_seconds=float(ingestion_cfg["retry_min_seconds"]),
            retry_max_seconds=float(ingestion_cfg["retry_max_seconds"]),
            user_agent=str(ingestion_cfg["user_agent"]),
            workers=int(cfg.get("classification_workers", 4)),
        )
        fetched = client.fetch_many(details)
        classification = pd.concat([cached, fetched], ignore_index=True)
    else:
        classification = cached

    classification = (
        classification.drop_duplicates("ticker", keep="last")
        .sort_values("ticker")
        .reset_index(drop=True)
    )
    unresolved = candidates - set(classification["ticker"])
    if unresolved:
        raise InstrumentClassificationError(
            "Missing official DSE classifications for: " + ", ".join(sorted(unresolved))
        )
    atomic_write_dataframe(classification, classification_path)
    fixed_count = int(
        classification[
            classification["ticker"].isin(candidates)
            & classification["is_fixed_income"].fillna(False).astype(bool)
        ]["ticker"].nunique()
    )
    LOGGER.info(
        "Official DSE classification identifies %s fixed-income model exclusions",
        fixed_count,
    )
    return classification


def filter_model_universe(
    prices: pd.DataFrame,
    classification: pd.DataFrame,
    *,
    candidate_patterns: list[str] | tuple[str, ...] = DEFAULT_CANDIDATE_PATTERNS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    candidates = fixed_income_candidates(prices["ticker"], candidate_patterns)
    known = set(classification["ticker"].astype(str))
    unresolved = candidates - known
    if unresolved:
        raise InstrumentClassificationError(
            "Run ingestion to classify new fixed-income candidates before building features: "
            + ", ".join(sorted(unresolved))
        )

    fixed = set(
        classification.loc[
            classification["is_fixed_income"].fillna(False).astype(bool), "ticker"
        ].astype(str)
    )
    filtered = prices[~prices["ticker"].isin(fixed)].copy()
    history = (
        prices.groupby("ticker", observed=True)
        .agg(
            history_rows=("date", "size"),
            first_history_date=("date", "min"),
            last_history_date=("date", "max"),
        )
        .reset_index()
    )
    profile_columns = [
        "ticker",
        "instrument_type",
        "sector",
        "is_fixed_income",
        "detail_url",
        "classification_source",
        "classified_at",
    ]
    profiles = classification[
        [column for column in profile_columns if column in classification]
    ].copy()
    audit = history.merge(profiles, on="ticker", how="left")
    audit["is_fixed_income"] = audit["ticker"].isin(fixed)
    audit["model_status"] = audit["is_fixed_income"].map(
        {True: "excluded_fixed_income", False: "included"}
    )
    audit = audit.sort_values(["model_status", "ticker"]).reset_index(drop=True)
    summary = {
        "classification_source": "official DSE company listing and security-detail pages",
        "tickers_before_filter": int(prices["ticker"].nunique()),
        "tickers_after_filter": int(filtered["ticker"].nunique()),
        "fixed_income_tickers_excluded": len(set(prices["ticker"]) & fixed),
        "rows_before_filter": len(prices),
        "rows_after_filter": len(filtered),
        "excluded_tickers": sorted(set(prices["ticker"]) & fixed),
    }
    return filtered, audit, summary
