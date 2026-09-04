from __future__ import annotations

import itertools

import pandas as pd

from analytics.performance import performance_summary
from backtest.engine import BacktestEngine
from dashboard.runner import StrategyConfig, run_strategy
from strategies.moving_average import MovingAverageStrategy


def cost_sensitivity_table(
    frames: dict[str, pd.DataFrame],
    primary_symbol: str,
    config: StrategyConfig,
    cost_levels: list[float],
    *,
    initial_capital: float = 100_000.0,
) -> pd.DataFrame:
    rows = []
    for bps in cost_levels:
        run = run_strategy(
            frames,
            primary_symbol,
            config,
            initial_capital=initial_capital,
            fees_bps=float(bps),
            slippage_bps=float(bps),
        )
        rows.append({
            "cost_bps_each": float(bps),
            "ending_equity": float(run.equity.iloc[-1]),
            "total_return": float(run.summary["total_return"]),
            "sharpe_ratio": float(run.summary["sharpe_ratio"]),
            "maximum_drawdown": float(run.summary["maximum_drawdown"]),
        })
    return pd.DataFrame(rows)


def moving_average_parameter_grid(
    symbol: str,
    data: pd.DataFrame,
    fast_windows: list[int],
    slow_windows: list[int],
    *,
    initial_capital: float = 100_000.0,
    fees_bps: float = 5.0,
    slippage_bps: float = 5.0,
) -> pd.DataFrame:
    rows = []
    engine_kwargs = dict(
        initial_capital=initial_capital,
        transaction_cost_bps=fees_bps,
        slippage_bps=slippage_bps,
        execution_delay=1,
    )
    for fast, slow in itertools.product(fast_windows, slow_windows):
        if fast >= slow:
            continue
        result = BacktestEngine(**engine_kwargs).run(symbol, data, MovingAverageStrategy(fast, slow))
        summary = performance_summary(result)
        rows.append({
            "fast_window": fast,
            "slow_window": slow,
            "total_return": float(summary["total_return"]),
            "cagr": float(summary["cagr"]),
            "sharpe_ratio": float(summary["sharpe_ratio"]),
            "maximum_drawdown": float(summary["maximum_drawdown"]),
        })
    if not rows:
        raise ValueError("Parameter grid contains no valid fast < slow combinations.")
    return pd.DataFrame(rows).sort_values("sharpe_ratio", ascending=False).reset_index(drop=True)
