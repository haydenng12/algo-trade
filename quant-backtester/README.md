# Quant Backtester

A modular, research-oriented historical backtesting platform for systematic equity strategies.

## Stage 1 — Foundation

Implemented now:

- Provider-agnostic OHLCV data cleaning and validation
- Deterministic CSV loading
- Optional Yahoo Finance daily-data loading
- Normalized timestamps, sorting, duplicate handling, numeric coercion, and missing-row handling
- OHLC/price/volume sanity checks
- Corporate-action-aware return convention: prefer `Adj Close` when available
- Simple returns and log returns
- Compact data-inspection summaries
- Basic price-history plotting
- Automated tests for data cleaning, validation, CSV loading, and return calculations

### Data schema

Each cleaned single-symbol frame uses a unique, increasing `pandas.DatetimeIndex` named `Date` and requires:

`Open`, `High`, `Low`, `Close`, `Volume`

`Adj Close` is optional. When present, historical return calculations use it by default so splits/dividends do not create false economic jumps.

### Missing-data convention

Rows missing required OHLCV values are dropped. Prices are **not** forward-filled or interpolated in Stage 1 because silently inventing market prices can contaminate later research.

## Setup

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run tests

```bash
pytest
```

## Load a CSV

```python
from data.loader import MarketDataLoader
from features.returns import simple_returns

data, report = MarketDataLoader.from_csv("path/to/SPY.csv")
returns = simple_returns(data)
```

## Download historical daily data

```python
from data.loader import MarketDataLoader

data, report = MarketDataLoader.from_yahoo(
    "SPY",
    start="2015-01-01",
    end="2026-01-01",
)
```

## Inspect a CSV

```bash
python examples/stage1_demo.py path/to/SPY.csv --symbol SPY --plot
```

## Stage 1 exit criterion

Stage 1 is complete when historical data can be loaded into the standardized schema reliably, validated before use, and converted into exact tested return series for downstream modules.

The next stage will introduce the strategy interface, moving-average strategy, portfolio state, next-period execution, equity accounting, and trade history.
