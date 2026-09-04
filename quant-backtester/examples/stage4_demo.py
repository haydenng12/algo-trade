from __future__ import annotations

import argparse
import math

from analytics.performance import performance_summary
from backtest.engine import BacktestEngine
from data.loader import MarketDataLoader
from strategies.moving_average import MovingAverageStrategy


def _pct(value: float) -> str:
    return "N/A" if not math.isfinite(value) else f"{value:.2%}"


def _num(value: float) -> str:
    return "N/A" if not math.isfinite(value) else f"{value:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 4 backtest with execution frictions.")
    parser.add_argument("symbol", nargs="?", default="SPY")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2026-01-01")
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--fast", type=int, default=20)
    parser.add_argument("--slow", type=int, default=50)
    parser.add_argument("--fees-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--position-fraction", type=float, default=1.0)
    parser.add_argument("--delay", type=int, default=1)
    args = parser.parse_args()

    data, report = MarketDataLoader.from_yahoo(args.symbol, args.start, args.end)
    result = BacktestEngine(
        initial_capital=args.capital,
        transaction_cost_bps=args.fees_bps,
        slippage_bps=args.slippage_bps,
        position_fraction=args.position_fraction,
        execution_delay=args.delay,
    ).run(args.symbol, data, MovingAverageStrategy(args.fast, args.slow))
    metrics = performance_summary(result)

    total_fees = result.equity_curve["transaction_costs"].sum()
    total_slippage = result.equity_curve["slippage_costs"].sum()

    print("Cleaning report:", report)
    print(f"\n{args.symbol.upper()} Stage 4 Performance")
    print(f"Execution delay:           {args.delay} trading day(s)")
    print(f"Transaction cost:          {args.fees_bps:.2f} bps")
    print(f"Slippage:                  {args.slippage_bps:.2f} bps")
    print(f"Position fraction:         {args.position_fraction:.0%}")
    print(f"Starting capital:          ${result.initial_capital:,.2f}")
    print(f"Ending equity:             ${result.ending_equity:,.2f}")
    print(f"Total transaction fees:   ${total_fees:,.2f}")
    print(f"Estimated slippage cost:  ${total_slippage:,.2f}")
    print(f"Total return:              {_pct(float(metrics['total_return']))}")
    print(f"CAGR:                      {_pct(float(metrics['cagr']))}")
    print(f"Annualized volatility:     {_pct(float(metrics['annualized_volatility']))}")
    print(f"Sharpe ratio:              {_num(float(metrics['sharpe_ratio']))}")
    print(f"Sortino ratio:             {_num(float(metrics['sortino_ratio']))}")
    print(f"Maximum drawdown:          {_pct(float(metrics['maximum_drawdown']))}")
    print(f"Completed trades:          {metrics['number_of_trades']}")
    print(f"Win rate:                  {_pct(float(metrics['win_rate']))}")
    print(f"Profit factor:             {_num(float(metrics['profit_factor']))}")


if __name__ == "__main__":
    main()
