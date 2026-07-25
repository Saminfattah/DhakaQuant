# DhakaQuant

DhakaQuant is a local-only quantitative research pipeline for Dhaka Stock Exchange daily prices,
leakage-safe features, LightGBM predictions, and human-review signals.

It does not use `bdshare`, Supabase, PostgreSQL, a web application, broker APIs, or automated
execution. Outputs remain on the local computer as CSV, Parquet, JSON, and model artifact files.

This is research software, not financial advice. Predictions and backtests cannot guarantee
future returns.

## What it does

1. Downloads the historical `muhammedalif/dsc-prices` Kaggle dataset.
2. Fetches missing dates from the official DSE day-end archive.
3. Retrieves the current official DSE company listing and keeps only tickers present on that page.
4. Uses the linked official DSE security details to identify and exclude fixed-income instruments
   from the model universe.
5. Validates and atomically maintains a canonical daily-price dataset.
6. Builds technical, momentum, volatility, liquidity, market, and calendar features.
7. Creates a configurable forward three-session classification target.
8. Compares recent 5-, 7-, and 10-year LightGBM windows with annual walk-forward validation.
9. Selects a probability threshold subject to aggregate and per-fold minimum call coverage.
10. Evaluates the selected design once on a separate recent holdout, then refits on current data.
11. Saves model metrics, calibration, baselines, and segmented evaluation.
12. Produces current probability estimates and research-only signals.

See [NOTEBOOK_REVIEW.md](NOTEBOOK_REVIEW.md) for the assessment of the source notebook and the
changes made to its extraction logic.

## Requirements

- Windows, macOS, or Linux
- Python 3.11 or newer
- Internet access for the initial Kaggle download and DSE archive updates
- Enough free disk space for roughly 1.6 million or more daily observations and feature columns

No database server is required.

## DhakaQuant research dashboard

The optional Streamlit interface reads the same local Parquet, CSV, JSON, model, configuration,
and log files produced by the worker. It does not upload data or start a public service.

Launch it from PowerShell:

```powershell
cd path\to\DhakaQuant
.\run_ui.ps1
```

If PowerShell script execution is unavailable, run:

```powershell
.\.venv\Scripts\python.exe -m streamlit run ui\app.py --server.address 127.0.0.1
```

Open `http://127.0.0.1:8501`.

### Streamlit Community Cloud

Deploy with these coordinates:

- Repository: `Saminfattah/DhakaQuant`
- Branch: `main`
- Main file path: `ui/app.py`
- Python: `3.14`

The deployment-specific dependencies are pinned in `ui/requirements.txt`. The root
`requirements.txt` remains the complete dependency set for running ingestion, feature building,
training, prediction, and signals locally. The hosted UI uses a newer PyArrow release with a
prebuilt Python 3.14 Linux wheel, while the local worker remains tested on Python 3.12.

Dashboard areas:

- **Overview** — freshness, model gate, signal counts, probability distributions, and recent logs.
- **Signals** — search, risk filters, probability filters, pagination, and CSV download.
- **Rankings** — upside, downside, avoided, liquidity, and stale-data views.
- **Ticker Explorer** — OHLC, volume, moving averages, RSI, MACD, volatility, and relative volume.
- **Model Health** — chronological metrics, calibration, confusion matrices, baselines, and importance.
- **Data Quality** — validation, failed ranges, and Kaggle-to-DSE boundary warnings.
- **Pipeline Control** — fixed, safe CLI actions with background state and logs.
- **Settings** — validated configuration edits with confirmation and automatic backups.

Pipeline actions never run when the browser refreshes. `train` and `run-all` require explicit
confirmation. Closing the browser does not stop a running worker process or delete local data.

## Windows PowerShell installation

Clone the repository, then create the local environment:

```powershell
git clone https://github.com/Saminfattah/DhakaQuant.git
cd DhakaQuant
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

If PowerShell blocks activation, use the virtual environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

The workspace-provided virtual environment may not include `Activate.ps1`. Activation is optional;
run every command through `.\.venv\Scripts\python.exe` or
`.\.venv\Scripts\dse-quant.exe` in that case.

## Kaggle authentication

`kagglehub` may use an existing Kaggle login or cached credentials. If credentials are required,
copy `.env.example` to `.env` and set `KAGGLE_USERNAME` and `KAGGLE_KEY` in the environment before
running ingestion. Do not commit credentials.

PowerShell example for the current terminal:

```powershell
$env:KAGGLE_USERNAME = "your_username"
$env:KAGGLE_KEY = "your_key"
```

## Configuration

Edit `config/settings.yaml`.

Important settings include:

- Kaggle dataset, DSE archive URL, and DSE company-listing URL
- Current-listing filter, minimum extracted ticker count, and cache fallback
- Fixed-income model exclusion and DSE classification candidate patterns
- Request chunk size, retries, timeout, and rate limit
- Price-validation thresholds
- Feature history and liquidity thresholds
- Prediction horizon and minimum future return
- Walk-forward years, recent training-window candidates, holdout year, and embargo
- Threshold search range and aggregate/per-fold minimum call rates
- LightGBM parameters and random seed
- BUY WATCH, SELL REVIEW, and minimum model-quality thresholds

Strategy thresholds are configuration values; they are not scattered as constants throughout the
code.

## Commands

Run commands from `dse_quant_worker` with its virtual environment active.

### Update daily prices

```powershell
python -m dse_quant.cli ingest
```

For a reproducible historical cutoff:

```powershell
python -m dse_quant.cli ingest --end-date 2026-07-20
```

The initial run downloads Kaggle history. Later runs load the preserved unfiltered local history
and request only dates after its newest observation. Every ingestion refreshes the official DSE
company listing, then writes the user-facing canonical dataset with only matching tickers. If the
listing page is temporarily unavailable, a previously validated cached listing may be used.

### Validate the canonical dataset

```powershell
python -m dse_quant.cli validate
```

### Build features

```powershell
python -m dse_quant.cli build-features
```

### Train and evaluate

```powershell
python -m dse_quant.cli train
```

Training uses annual walk-forward folds by default. The configured recent-window candidates and
probability thresholds are selected using validation folds only; the holdout year is never used
for model selection. Low-liquidity, stale, floor-price, and insufficient-history rows are excluded
from fitting and evaluation because they cannot produce actionable BUY signals. After evaluation,
the deployment estimator is refit on all eligible labeled observations in the selected recent
window.

The target is calculated independently within each ticker using the configured number of future
trading observations. Rows without a known future target are excluded from training, but retained
for current prediction.

### Generate predictions

```powershell
python -m dse_quant.cli predict
```

### Generate signals

```powershell
python -m dse_quant.cli signals
```

### Complete first-time workflow

```powershell
python -m dse_quant.cli run-all
```

### Routine daily workflow without retraining

```powershell
python -m dse_quant.cli daily-run
```

`daily-run` updates data, rebuilds deterministic features, loads `models/latest.joblib`, produces
predictions, and regenerates signals. It intentionally does not retrain the approved model.

Every command returns a nonzero process status when it fails and writes details to
`logs/dse_quant.log`.

## Outputs

### Canonical data

```text
data/processed/daily_prices.parquet
data/processed/daily_prices.csv
data/processed/daily_prices_all.parquet
data/processed/listed_universe.parquet
data/processed/instrument_classification.parquet
```

Canonical columns:

```text
date, ticker, open, high, low, close, volume, trade_count, source, ingested_at
```

`daily_prices.parquet` and its CSV copy contain only tickers found on the current official DSE
company-listing page. `daily_prices_all.parquet` is an internal retention file for delisted or
temporarily absent instruments, so a future listing change does not erase historical source data.
Fixed-income rows remain in the canonical price archive but are removed before feature generation,
model training, prediction, and signal generation.

### Quality reports

```text
data/outputs/validation_report.json
data/outputs/unfiltered_validation_report.json
data/outputs/join_boundary_report.csv
data/outputs/universe_filter_summary.json
data/outputs/universe_filter_audit.csv
data/outputs/model_universe_summary.json
data/outputs/model_universe_audit.csv
data/raw/failed_ranges.json
```

Review `join_boundary_report.csv` before model training. A suspicious price jump where the source
changes can indicate that Kaggle prices and official DSE prices use different corporate-action
adjustment policies.

### Features and models

```text
data/features/daily_features.parquet
models/latest.joblib
models/latest_metrics.json
models/lgbm-<timestamp>.joblib
models/lgbm-<timestamp>.json
data/outputs/model_selection_report.csv
```

The model artifact includes the fitted estimator, exact ordered feature list, model version,
configuration, walk-forward fold results, selected window and threshold, untouched holdout
metrics, deployment-refit boundaries, and evaluation metrics.

### Predictions and signals

```text
data/outputs/latest_predictions.csv
data/outputs/latest_predictions.parquet
data/outputs/predictions_<timestamp>.parquet
data/outputs/latest_signals.csv
data/outputs/latest_signals.parquet
data/outputs/upside_ranking.csv
data/outputs/downside_ranking.csv
```

Signals:

- `BUY WATCH`: sufficient upside probability, acceptable model precision, and no blocking risk.
- `SELL REVIEW`: sufficient downside probability.
- `HOLD`: no strong eligible edge or the model-quality gate blocks an upside label.
- `AVOID`: insufficient history, stale data, possible floor-price behavior, or weak liquidity.

These are decision-support labels only.

## Tests

```powershell
python -m pytest
```

The suite covers normalization, invalid rows, duplicate precedence, incremental retention, archive
parsing with mocked HTTP, OHLC validation, target alignment, future-leakage boundaries, liquidity
flags, signal thresholds, reproducible LightGBM behavior, and CSV/Parquet writes.

## Scheduling a daily run on Windows

First verify this command works:

```powershell
.\.venv\Scripts\python.exe -m dse_quant.cli daily-run
```

In Windows Task Scheduler:

1. Create a basic task scheduled after the DSE day-end archive is expected to be updated.
2. Set **Program/script** to the full path of `.venv\Scripts\python.exe`.
3. Set **Arguments** to `-m dse_quant.cli daily-run`.
4. Set **Start in** to the full cloned `DhakaQuant` directory.
5. Configure retries for temporary network failures.

Do not schedule model retraining automatically until the evaluation and join-boundary reports have
been reviewed. Retraining cadence should be a deliberate research decision.

## Troubleshooting

### Kaggle authentication failure

Confirm the Kaggle account can access the dataset and that the environment credentials are valid.
The downloaded source CSV is cached under `data/raw/kaggle`.

### DSE archive returns no recognizable table

Check `logs/dse_quant.log` and `data/raw/failed_ranges.json`. The DSE archive HTML may have changed,
or it may not have data for the requested period. The previous canonical dataset remains safe.

Temporary connection failures, including Windows socket error `10013`, are recorded as recoverable
warnings. The worker retains existing history, may use the validated cached listing, and clearly
reports the last available market date instead of printing a fatal traceback. Check Windows
Firewall, antivirus, proxy policy, and whether the process is running in a network-restricted
environment before the next update.

The default configuration validates HTTPS through the operating-system certificate store. This
avoids certificate-chain failures on Windows without disabling TLS verification. If validation
still fails, check the Windows date/time and any antivirus or proxy performing HTTPS inspection.

### DSE company-listing extraction fails

The filter refuses a suspiciously small or unrecognizable listing so that a DSE layout change
cannot wipe the canonical dataset. If a valid cached universe exists and
`allow_listing_cache_fallback` is enabled, it is used and recorded in
`data/outputs/universe_filter_summary.json`. Otherwise, the previous canonical dataset remains
unchanged.

### Validation refuses to save

Inspect `data/outputs/validation_report.json`. Invalid data never replaces the canonical file.

### Training says there are too few rows

Complete ingestion and feature generation first. For a deliberately small development sample,
temporarily lower `minimum_training_rows` and `minimum_validation_rows`; restore research-grade
values before interpreting metrics.

### Parquet support error

Reinstall `pyarrow` inside the active virtual environment:

```powershell
python -m pip install --upgrade pyarrow
```

## Limitations

- The official DSE HTML archive is not a stable formal API.
- The Kaggle dataset is a third-party historical source.
- Corporate-action adjustment consistency is not guaranteed.
- Daily OHLCV cannot reproduce intraday execution conditions.
- Liquidity and floor-price indicators are heuristics.
- Survivorship, stale prices, regime changes, and data errors can distort evaluation.
- Historical precision does not guarantee future precision.
- The pipeline never places trades and does not replace independent human review.
