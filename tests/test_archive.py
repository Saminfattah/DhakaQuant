from __future__ import annotations

from datetime import date

import responses

from dse_quant.ingestion.dse_archive import DSEArchiveClient


@responses.activate
def test_archive_parses_expected_table():
    html = """
    <table>
      <tr><th>DATE</th><th>TRADING CODE</th><th>OPENP*</th><th>HIGH</th>
      <th>LOW</th><th>CLOSEP*</th><th>VOLUME</th></tr>
      <tr><td>2026-01-01</td><td>ACI</td><td>100</td><td>102</td>
      <td>99</td><td>101</td><td>10000</td></tr>
    </table>
    """
    responses.get("https://example.test/archive", body=html, status=200)
    client = DSEArchiveClient(
        url="https://example.test/archive",
        timeout=1,
        retries=1,
        retry_min_seconds=0,
        retry_max_seconds=0,
        rate_limit_seconds=0,
        user_agent="test",
    )
    result = client.fetch(date(2026, 1, 1), date(2026, 1, 1))
    assert len(result) == 1
    assert result.iloc[0]["ticker"] == "ACI"
    assert result.iloc[0]["source"] == "dse_archive"
