from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest
import responses

from dse_quant.ingestion.listed_universe import (
    ListedUniverseClient,
    ListedUniverseError,
    filter_to_listed_universe,
    parse_listed_tickers,
)

SAMPLE_HTML = """
<html><body>
  <a href="displayCompany.php?name=ACI">ACI 200.00</a>
  <a href="/displayCompany.php?name=BATBC">BATBC</a>
  <a href="displayCompany.php?name=KAY%26QUE">KAY&amp;QUE</a>
  <a href="displayCompany.php?name=ACI">ACI</a>
  <a href="other.php?name=IGNORE">Ignore</a>
</body></html>
"""


def test_parse_listed_tickers_deduplicates_and_decodes_symbols():
    fetched_at = datetime(2026, 7, 25, tzinfo=UTC)
    result = parse_listed_tickers(
        SAMPLE_HTML,
        source_url="https://www.dsebd.org/company_listing.php",
        fetched_at=fetched_at,
    )
    assert result["ticker"].tolist() == ["ACI", "BATBC", "KAY&QUE"]
    assert result["ticker"].is_unique
    assert result["fetched_at"].eq(pd.Timestamp(fetched_at)).all()
    assert result.loc[result["ticker"] == "BATBC", "detail_url"].iloc[0].startswith(
        "https://www.dsebd.org/"
    )


@responses.activate
def test_listed_universe_client_rejects_suspiciously_small_extraction():
    url = "https://example.test/company_listing.php"
    responses.get(url, body=SAMPLE_HTML, status=200)
    client = ListedUniverseClient(
        url=url,
        timeout=1,
        retries=1,
        retry_min_seconds=0,
        retry_max_seconds=0,
        user_agent="test",
        minimum_tickers=4,
    )
    with pytest.raises(ListedUniverseError, match="only 3 tickers"):
        client.fetch()


def test_filter_to_listed_universe_returns_filtered_prices_and_audit():
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-01"]),
            "ticker": ["aci", "ACI", "OLD"],
            "close": [100.0, 101.0, 50.0],
        }
    )
    listed = pd.DataFrame({"ticker": ["ACI", "NEW"]})
    filtered, audit = filter_to_listed_universe(prices, listed)

    assert filtered["ticker"].unique().tolist() == ["ACI"]
    statuses = audit.set_index("ticker")["status"].to_dict()
    assert statuses == {
        "ACI": "included",
        "NEW": "listed_without_history",
        "OLD": "excluded_not_currently_listed",
    }
    assert audit.set_index("ticker").loc["ACI", "history_rows"] == 2
