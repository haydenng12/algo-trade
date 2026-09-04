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

## Stage 3 — Performance Analytics

Implemented:

- Total return from initial capital to ending equity
- CAGR using the actual elapsed calendar time of the backtest
- Period-to-period portfolio returns derived from the equity curve
- Annualized volatility using daily sample standard deviation × `sqrt(252)`
- Annualized Sharpe ratio with an optional annual risk-free rate converted to a daily equivalent
- Annualized Sortino ratio using downside deviation below a configurable periodic target
- Drawdown series and maximum drawdown from the running equity high-water mark
- Completed-trade analytics: number of trades, win rate, profit factor, average trade return, average winner, average loser, best/worst trade, and average holding period
- Unified `performance_summary(...)` for later dashboard/report consumption
- Hand-checkable automated tests for each core formula and an end-to-end summary reconciliation test

### Stage 3 metric conventions

Portfolio metrics are calculated from the daily **equity curve**, not directly from the asset's raw returns. This matters because the strategy may be in cash on some days. Volatility and Sharpe use 252 trading periods per year. CAGR instead uses actual elapsed calendar time because it describes compound growth over the real backtest duration.

Maximum drawdown is reported as a negative percentage. For example, `-0.20` means the portfolio fell 20% from a previous equity peak before recovering or reaching the end of the test.

Trade statistics use **completed trades only**. An open position at the final backtest date remains reflected in ending equity and portfolio-level metrics, but it is not invented into a closed trade for win-rate or profit-factor calculations.

### Run Stage 3 analytics

```bash
python examples/stage3_demo.py SPY --start 2015-01-01 --end 2026-01-01 --fast 20 --slow 50
```

Optionally provide an annual effective risk-free rate, e.g. 4%:

```bash
python examples/stage3_demo.py SPY --risk-free 0.04
```

### Stage 3 exit criterion

Stage 3 is complete when all performance metrics are independently tested against known calculations and a complete backtest result can be converted into a consistent portfolio/trade performance summary. Transaction costs, slippage, modular sizing, and stronger cash/execution constraints remain Stage 4 work.


## Stage 4 — Realistic Trading

Stage 4 adds configurable execution realism while preserving the Stage 1–3 behavior when all frictions are zero.

### Added
- Transaction-cost model in basis points of executed notional.
- Symmetric adverse slippage model: buys execute above the observed Open and sells below it.
- Fixed-fraction position sizing with a reusable equal-weight sizing utility for later multi-asset work.
- Cash-aware buy resizing so fees/slippage cannot drive cash negative.
- Explicit configurable signal-to-execution delay (minimum one trading period to prevent look-ahead).
- Daily tracking of traded notional, fees, slippage, and cumulative execution friction.
- Trade-level reconciliation of gross P&L, transaction costs, slippage costs, total costs, and net P&L.

### Run the Stage 4 demo
From the repository root, run the example as a module:

```bash
python -m examples.stage4_demo SPY --start 2015-01-01 --end 2026-01-01 --fast 20 --slow 50 --fees-bps 5 --slippage-bps 5 --position-fraction 1.0 --delay 1
```

### Verify

```bash
pytest -v
```

Expected Stage 4 suite: 36 passing tests.

## Stage 5 — Research Framework

Stage 5 turns the engine into a controlled research platform rather than a single historical backtest.

Implemented modules:

- `research/benchmark.py` — frictionless buy-and-hold benchmark over the exact comparison period.
- `research/split.py` — chronological train/test and date-based splits; time-series data is never shuffled.
- `research/grid.py` — moving-average parameter grids, training-only parameter selection, and untouched out-of-sample evaluation.
- `research/oos.py` — the single shared OOS execution path used by OOS evaluation, walk-forward testing, and cost sensitivity; it warms indicators with past prices while starting portfolio accounting fresh at the evaluation boundary.
- `research/walk_forward.py` — rolling or expanding walk-forward testing. Each fold tunes on past data and evaluates only on the immediately following test window.
- `research/cost_sensitivity.py` — repeated OOS backtests under configurable friction assumptions using the same historical lookback context as the primary OOS run.
- `examples/stage5_demo.py` — real-data demonstration combining parameter selection, OOS evaluation, benchmark comparison, walk-forward folds, and cost sensitivity.

### Stage 5 research convention

Parameter choices are made using training data only. Test data is not used to choose SMA windows. For OOS evaluation, historical training prices may provide moving-average lookback context, but the test-period backtest starts flat with fresh capital. Cost-sensitivity runs use that exact same OOS signal-preparation path, so a sensitivity row with identical fees/slippage must reconcile with the primary OOS result. Walk-forward folds also use the shared OOS runner, start each fold flat, and chain each fold's ending capital into the next fold.

Run the complete test suite:

```bash
pytest -v
```

Run the Stage 5 research demo:

```bash
python -m examples.stage5_demo SPY --start 2010-01-01 --end 2026-01-01 --fees-bps 5 --slippage-bps 5
```
