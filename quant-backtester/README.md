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

## Stage 2 — Core Backtester MVP

Implemented:

- Abstract `Strategy` interface: strategies generate signals only and never touch cash or execute orders
- Long/flat moving-average momentum strategy (`fast SMA > slow SMA`)
- Portfolio state with cash, quantity, entry price/date, and mark-to-market equity
- Single-symbol daily backtest engine
- Explicit anti-look-ahead timing: signal formed on day `t` executes at day `t+1` Open
- Stage-2 sizing convention: signal `1` invests all available cash; signal `0` exits completely
- Fractional shares enabled for exact deterministic accounting
- Daily equity curve containing signal, executed signal, cash, quantity, market value, and equity
- Completed round-trip trade log with entry/exit dates and prices, quantity, gross/net P&L, return, and holding period
- `BacktestResult` object shared with later analytics code
- Deterministic tests for strategy rules, execution timing, portfolio reconciliation, trade P&L, and open-position marking

Stage 2 intentionally assumes zero transaction costs and zero slippage. Configurable sizing, cash constraints, costs, and slippage belong to Stage 4 per the project roadmap.

### Run a Stage 2 backtest

```bash
python examples/stage2_demo.py SPY --start 2015-01-01 --end 2026-01-01 --fast 20 --slow 50
```

### Stage 2 timing convention

If the moving-average comparison produces a long signal using Tuesday's completed prices, the engine does **not** buy on Tuesday. It may buy at Wednesday's Open. This one-period delay is enforced inside the engine and covered by an automated test.

### Stage 2 exit criterion

Stage 2 is complete when the moving-average strategy can run end-to-end with reconciled cash/position accounting, an equity curve and completed trade history, while tests prove that no day-`t` signal can change holdings before day `t+1`.
