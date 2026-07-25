from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from html.parser import HTMLParser
from urllib.parse import parse_qs, urljoin, urlparse

import pandas as pd
import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

LOGGER = logging.getLogger(__name__)
TICKER_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9&()._-]*$")


class ListedUniverseError(RuntimeError):
    pass


def normalize_ticker(value: object) -> str:
    return str(value).strip().upper()


class _CompanyLinkParser(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.links: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        href = dict(attrs).get("href")
        if not href:
            return
        absolute_url = urljoin(self.base_url, href)
        parsed = urlparse(absolute_url)
        if not parsed.path.lower().endswith("/displaycompany.php"):
            return
        ticker = normalize_ticker(parse_qs(parsed.query).get("name", [""])[0])
        if ticker and TICKER_PATTERN.fullmatch(ticker):
            self.links.append((ticker, absolute_url))


def parse_listed_tickers(
    html: str,
    *,
    source_url: str,
    fetched_at: datetime | None = None,
) -> pd.DataFrame:
    parser = _CompanyLinkParser(source_url)
    parser.feed(html)
    if not parser.links:
        raise ListedUniverseError(
            "The DSE company-listing page contained no recognizable company ticker links."
        )

    timestamp = fetched_at or datetime.now(UTC)
    frame = pd.DataFrame(parser.links, columns=["ticker", "detail_url"])
    frame = frame.drop_duplicates("ticker", keep="last").sort_values("ticker").reset_index(drop=True)
    frame["source_url"] = source_url
    frame["fetched_at"] = pd.Timestamp(timestamp)
    return frame


class ListedUniverseClient:
    def __init__(
        self,
        *,
        url: str,
        timeout: int = 30,
        retries: int = 4,
        retry_min_seconds: float = 2,
        retry_max_seconds: float = 30,
        user_agent: str,
        minimum_tickers: int = 200,
    ) -> None:
        self.url = url
        self.timeout = timeout
        self.minimum_tickers = minimum_tickers
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self._fetch_with_retry = retry(
            stop=stop_after_attempt(retries),
            wait=wait_exponential(min=retry_min_seconds, max=retry_max_seconds),
            retry=retry_if_exception_type((requests.RequestException, ListedUniverseError)),
            reraise=True,
        )(self._fetch_once)

    def _fetch_once(self) -> pd.DataFrame:
        response = self.session.get(self.url, timeout=self.timeout)
        response.raise_for_status()
        frame = parse_listed_tickers(response.text, source_url=self.url)
        if len(frame) < self.minimum_tickers:
            raise ListedUniverseError(
                "DSE company-listing extraction returned only "
                f"{len(frame)} tickers; expected at least {self.minimum_tickers}. "
                "The page layout may have changed."
            )
        return frame

    def fetch(self) -> pd.DataFrame:
        LOGGER.info("Fetching current listed universe from %s", self.url)
        result = self._fetch_with_retry()
        LOGGER.info("Fetched %s unique DSE-listed tickers", len(result))
        return result


def validate_listed_universe(frame: pd.DataFrame, minimum_tickers: int) -> pd.DataFrame:
    if "ticker" not in frame:
        raise ListedUniverseError("Cached listed-universe data has no ticker column.")
    normalized = frame.copy()
    normalized["ticker"] = normalized["ticker"].map(normalize_ticker)
    normalized = normalized[
        normalized["ticker"].map(lambda value: bool(TICKER_PATTERN.fullmatch(value)))
    ]
    normalized = normalized.drop_duplicates("ticker", keep="last").sort_values("ticker")
    normalized = normalized.reset_index(drop=True)
    if len(normalized) < minimum_tickers:
        raise ListedUniverseError(
            f"Listed-universe data has {len(normalized)} tickers; "
            f"expected at least {minimum_tickers}."
        )
    return normalized


def filter_to_listed_universe(
    prices: pd.DataFrame,
    listed_universe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    listed = set(listed_universe["ticker"].map(normalize_ticker))
    normalized_prices = prices.copy()
    normalized_prices["ticker"] = normalized_prices["ticker"].map(normalize_ticker)
    history = (
        normalized_prices.groupby("ticker", observed=True)
        .agg(
            history_rows=("date", "size"),
            first_history_date=("date", "min"),
            last_history_date=("date", "max"),
        )
        .reset_index()
    )
    history_tickers = set(history["ticker"])

    audit_tickers = sorted(listed | history_tickers)
    audit = pd.DataFrame({"ticker": audit_tickers})
    audit["status"] = audit["ticker"].map(
        lambda ticker: (
            "included"
            if ticker in listed and ticker in history_tickers
            else "listed_without_history"
            if ticker in listed
            else "excluded_not_currently_listed"
        )
    )
    audit = audit.merge(history, on="ticker", how="left")
    audit["history_rows"] = audit["history_rows"].astype("Int64")
    audit["listed_on_dse_page"] = audit["ticker"].isin(listed)

    filtered = normalized_prices[normalized_prices["ticker"].isin(listed)].copy()
    filtered = filtered.sort_values(["ticker", "date"]).reset_index(drop=True)
    if filtered.empty:
        raise ListedUniverseError(
            "No local price tickers matched the current DSE company-listing page."
        )
    return filtered, audit
