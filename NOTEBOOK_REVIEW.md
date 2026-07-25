# Source notebook review

Reviewed notebook:

`sample/Complete_DSEX_2000_2026_extraction_no_BDSHARE.ipynb`

## Logic retained

- Kaggle dataset `muhammedalif/dsc-prices` provides the historical baseline.
- The official DSE day-end archive fills the gap after the newest locally available date.
- Archive requests are divided into manageable date ranges.
- Source columns are normalized to a common daily OHLCV representation.
- Ticker/date duplicates retain the newest observation.
- A local canonical dataset supports incremental runs.

The saved notebook output demonstrates that this approach assembled 1,594,650 observations
from 2000-01-01 through 2026-07-20 when it was run.

## Logic corrected

- TLS certificate verification remains enabled.
- Requests use explicit timeouts, retries, exponential backoff, rate limiting, and a stable
  user agent.
- The Kaggle CSV is selected by inspecting candidate schemas instead of taking the first file.
- The pipeline is independent of notebook cell order and Google Colab.
- Previously collected DSE ranges cannot be lost on a later incremental run.
- Invalid dates, missing identifiers, negative values, malformed numerics, duplicate keys, and
  inconsistent OHLC rows are detected.
- Failed DSE ranges are recorded in `data/raw/failed_ranges.json`.
- Canonical files are written atomically only after validation.
- The last valid canonical dataset remains intact when processing or validation fails.
- A source-boundary report highlights discontinuities between Kaggle and official DSE records.

## Data assumptions requiring human review

- The Kaggle dataset's corporate-action adjustment policy is not established by the notebook.
- Official archive prices are treated as raw day-end observations.
- Large price changes where the source changes may represent adjustment differences rather than
  real returns. The pipeline reports these but does not automatically adjust them.
- `trade_count` remains nullable when a source does not expose it.
- HTML scraping can stop working if DSE changes its archive page. A schema change causes a clear
  failure rather than silently ingesting the wrong table.
- Liquidity, stale-price, and floor-price flags are conservative research proxies, not official
  DSE classifications.

