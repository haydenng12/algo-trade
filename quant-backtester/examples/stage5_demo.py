from __future__ import annotations

import argparse
import math

from analytics.performance import performance_summary
from data.loader import MarketDataLoader
from research.benchmark import benchmark_summary
from research.cost_sensitivity import cost_sensitivity
from research.grid import evaluate_oos, grid_search_moving_average, select_best_parameters
from research.split import chronological_split
from research.walk_forward import walk_forward_moving_average


def pct(x):
    return "N/A" if not math.isfinite(float(x)) else f"{float(x):.2%}"


def num(x):
    return "N/A" if not math.isfinite(float(x)) else f"{float(x):.2f}"


def main():
    p = argparse.ArgumentParser(description="Stage 5 research-framework demo")
    p.add_argument("symbol", nargs="?", default="SPY")
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--end", default="2026-01-01")
    p.add_argument("--capital", type=float, default=100_000)
    p.add_argument("--train-fraction", type=float, default=0.7)
    p.add_argument("--fees-bps", type=float, default=5)
    p.add_argument("--slippage-bps", type=float, default=5)
    args = p.parse_args()

    data, report = MarketDataLoader.from_yahoo(args.symbol, args.start, args.end)
    train, test = chronological_split(data, args.train_fraction)
    engine_kwargs = {
        "initial_capital": args.capital,
        "transaction_cost_bps": args.fees_bps,
        "slippage_bps": args.slippage_bps,
        "position_fraction": 1.0,
        "execution_delay": 1,
    }

    fast_grid = [10, 20, 30, 40]
    slow_grid = [50, 100, 150, 200]
    grid = grid_search_moving_average(
        args.symbol, train, fast_grid, slow_grid,
        selection_metric="sharpe_ratio", engine_kwargs=engine_kwargs,
    )
    fast, slow = select_best_parameters(grid)
    oos = evaluate_oos(args.symbol, train, test, fast, slow, engine_kwargs=engine_kwargs)
    oos_metrics = performance_summary(oos)
    bench = benchmark_summary(test, args.capital)

    # Walk-forward windows are expressed in trading rows: ~3y train, ~1y test.
    wf = walk_forward_moving_average(
        args.symbol, data, fast_grid, slow_grid,
        train_size=756, test_size=252, expanding=False,
        selection_metric="sharpe_ratio", engine_kwargs=engine_kwargs,
    )
    costs = cost_sensitivity(
        args.symbol, train, test, fast, slow, [0, 5, 10, 25],
        initial_capital=args.capital,
    )

    print("Cleaning report:", report)
    print(f"\nTraining period: {train.index[0].date()} -> {train.index[-1].date()} ({len(train)} rows)")
    print(f"Test period:     {test.index[0].date()} -> {test.index[-1].date()} ({len(test)} rows)")
    print(f"Selected on TRAINING data only: fast={fast}, slow={slow}")
    print("\nTop 5 training parameter combinations:")
    print(grid[["fast_window", "slow_window", "total_return", "sharpe_ratio", "maximum_drawdown"]].head().to_string(index=False))

    print("\nOUT-OF-SAMPLE STRATEGY VS BUY-AND-HOLD")
    print(f"Strategy total return:     {pct(oos_metrics['total_return'])}")
    print(f"Benchmark total return:    {pct(bench['total_return'])}")
    print(f"Strategy CAGR:             {pct(oos_metrics['cagr'])}")
    print(f"Benchmark CAGR:            {pct(bench['cagr'])}")
    print(f"Strategy Sharpe:           {num(oos_metrics['sharpe_ratio'])}")
    print(f"Benchmark Sharpe:          {num(bench['sharpe_ratio'])}")
    print(f"Strategy max drawdown:     {pct(oos_metrics['maximum_drawdown'])}")
    print(f"Benchmark max drawdown:    {pct(bench['maximum_drawdown'])}")

    print("\nWALK-FORWARD FOLDS")
    print(wf.folds.to_string(index=False))
    print(f"Walk-forward ending equity: ${wf.ending_equity:,.2f}")

    print("\nCOST SENSITIVITY ON OOS PERIOD")
    print(costs[["cost_bps_each", "ending_equity", "total_return", "sharpe_ratio", "maximum_drawdown"]].to_string(index=False))

    if args.fees_bps == args.slippage_bps and float(args.fees_bps) in set(costs["cost_bps_each"]):
        matching = costs.loc[costs["cost_bps_each"] == float(args.fees_bps)].iloc[0]
        reconciles = math.isclose(
            float(matching["ending_equity"]),
            float(oos.ending_equity),
            rel_tol=1e-12,
            abs_tol=1e-8,
        )
        print(
            f"\nOOS / cost-sensitivity reconciliation at {args.fees_bps:.2f} bps each: "
            f"{'PASS' if reconciles else 'FAIL'}"
        )
        print(f"Primary OOS ending equity:      ${oos.ending_equity:,.2f}")
        print(f"Sensitivity ending equity:      ${float(matching['ending_equity']):,.2f}")


if __name__ == "__main__":
    main()
