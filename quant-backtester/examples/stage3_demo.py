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
    parser = argparse.ArgumentParser(description="Run Stage 3 backtest analytics.")
    parser.add_argument("symbol", nargs="?", default="SPY")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2026-01-01")
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--fast", type=int, default=20)
    parser.add_argument("--slow", type=int, default=50)
    parser.add_argument("--risk-free", type=float, default=0.0)
    args = parser.parse_args()

    data, report = MarketDataLoader.from_yahoo(args.symbol, args.start, args.end)
    result = BacktestEngine(args.capital).run(
        args.symbol,
        data,
        MovingAverageStrategy(args.fast, args.slow),
    )
    metrics = performance_summary(result, risk_free_rate=args.risk_free)

    print("Cleaning report:", report)
    print(f"\n{args.symbol.upper()} Stage 3 Performance")
    print(f"Starting capital:          ${result.initial_capital:,.2f}")
    print(f"Ending equity:             ${result.ending_equity:,.2f}")
    print(f"Total return:              {_pct(float(metrics['total_return']))}")
    print(f"CAGR:                      {_pct(float(metrics['cagr']))}")
    print(f"Annualized volatility:     {_pct(float(metrics['annualized_volatility']))}")
    print(f"Sharpe ratio:              {_num(float(metrics['sharpe_ratio']))}")
    print(f"Sortino ratio:             {_num(float(metrics['sortino_ratio']))}")
    print(f"Maximum drawdown:          {_pct(float(metrics['maximum_drawdown']))}")
    print(f"Completed trades:          {metrics['number_of_trades']}")
    print(f"Win rate:                  {_pct(float(metrics['win_rate']))}")
    print(f"Profit factor:             {_num(float(metrics['profit_factor']))}")
    print(f"Average trade return:      {_pct(float(metrics['average_trade_return']))}")
    print(f"Average holding period:    {_num(float(metrics['average_holding_period_days']))} days")


if __name__ == "__main__":
    main()
