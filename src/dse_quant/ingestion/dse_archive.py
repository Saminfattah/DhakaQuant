from __future__ import annotations

import logging
import time
from datetime import date
from io import StringIO

import pandas as pd
import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from dse_quant.processing.cleaning import normalize_column_name, normalize_price_frame

LOGGER = logging.getLogger(__name__)


class DSEArchiveError(RuntimeError):
    pass


class DSEArchiveClient:
    def __init__(
        self,
        *,
        url: str,
        timeout: int = 30,
        retries: int = 4,
        retry_min_seconds: float = 2,
        retry_max_seconds: float = 30,
        rate_limit_seconds: float = 1,
        user_agent: str,
    ) -> None:
        self.url = url
        self.timeout = timeout
        self.rate_limit_seconds = rate_limit_seconds
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self._fetch_with_retry = retry(
            stop=stop_after_attempt(retries),
            wait=wait_exponential(min=retry_min_seconds, max=retry_max_seconds),
            retry=retry_if_exception_type((requests.RequestException, DSEArchiveError)),
            reraise=True,
        )(self._fetch_once)

    def _fetch_once(self, start: date, end: date) -> pd.DataFrame:
        params = {
            "startDate": start.isoformat(),
            "endDate": end.isoformat(),
            "inst": "All Instrument",
            "archive": "data",
        }
        response = self.session.get(self.url, params=params, timeout=self.timeout)
        response.raise_for_status()
        try:
            tables = pd.read_html(StringIO(response.text))
        except ValueError as exc:
            raise DSEArchiveError("DSE response contained no HTML tables.") from exc
        candidates = []
        for table in tables:
            normalized = {normalize_column_name(column) for column in table.columns}
            if {"date", "ticker", "open", "high", "low", "close", "volume"} <= normalized:
                candidates.append(table)
        if not candidates:
            raise DSEArchiveError(
                "DSE archive layout changed or the price table is missing required columns."
            )
        return max(candidates, key=len)

    def fetch(self, start: date, end: date, reject_invalid: bool = True) -> pd.DataFrame:
        LOGGER.info("Fetching official DSE archive %s to %s", start, end)
        raw = self._fetch_with_retry(start, end)
        result = normalize_price_frame(raw, source="dse_archive", reject_invalid=reject_invalid)
        if result.empty:
            LOGGER.warning("DSE archive returned no valid rows for %s to %s", start, end)
        if self.rate_limit_seconds:
            time.sleep(self.rate_limit_seconds)
        return result
