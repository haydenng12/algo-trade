from __future__ import annotations

import argparse

from backtest.engine import BacktestEngine
from data.loader import MarketDataLoader
from strategies.moving_average import MovingAverageStrategy


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Stage 2 moving-average backtest.")
    parser.add_argument("symbol", nargs="?", default="SPY")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2026-01-01")
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--fast", type=int, default=20)
    parser.add_argument("--slow", type=int, default=50)
    args = parser.parse_args()

    data, report = MarketDataLoader.from_yahoo(args.symbol, args.start, args.end)
    strategy = MovingAverageStrategy(args.fast, args.slow)
    result = BacktestEngine(args.capital).run(args.symbol, data, strategy)

    print("Cleaning report:", report)
    print("\nEquity curve tail:")
    print(result.equity_curve.tail())
    print("\nCompleted trades:", len(result.trades))
    print(result.trades.tail())
    print(f"\nStarting capital: ${result.initial_capital:,.2f}")
    print(f"Ending equity:    ${result.ending_equity:,.2f}")


if __name__ == "__main__":
    main()
