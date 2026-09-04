from __future__ import annotations

from itertools import product

import pandas as pd

from analytics.performance import performance_summary
from backtest.engine import BacktestEngine
from research.oos import run_oos_moving_average
from strategies.moving_average import MovingAverageStrategy


def _engine(config: dict) -> BacktestEngine:
    return BacktestEngine(**config)


def grid_search_moving_average(
    symbol: str,
    train_data: pd.DataFrame,
    fast_windows: list[int] | tuple[int, ...],
    slow_windows: list[int] | tuple[int, ...],
    *,
    selection_metric: str = "sharpe_ratio",
    engine_kwargs: dict | None = None,
) -> pd.DataFrame:
    """Evaluate every valid SMA pair on training data only."""
    config = dict(engine_kwargs or {})
    rows = []
    for fast, slow in product(fast_windows, slow_windows):
        if fast >= slow:
            continue
        result = _engine(config).run(symbol, train_data, MovingAverageStrategy(fast, slow))
        metrics = performance_summary(result)
        rows.append({"fast_window": fast, "slow_window": slow, **metrics})
    if not rows:
        raise ValueError("Parameter grid contains no valid fast < slow combinations.")
    frame = pd.DataFrame(rows)
    if selection_metric not in frame.columns:
        raise KeyError(f"Unknown selection metric: {selection_metric}")
    frame["selection_score"] = frame[selection_metric].astype(float)
    return frame.sort_values("selection_score", ascending=False, na_position="last").reset_index(drop=True)


def evaluate_oos(
    symbol: str,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    fast_window: int,
    slow_window: int,
    *,
    engine_kwargs: dict | None = None,
):
    """Evaluate frozen parameters on test data with past-only lookback context."""
    return run_oos_moving_average(
        symbol,
        train_data,
        test_data,
        fast_window,
        slow_window,
        engine_kwargs=engine_kwargs,
    )


def select_best_parameters(grid: pd.DataFrame) -> tuple[int, int]:
    if grid.empty:
        raise ValueError("Grid results cannot be empty.")
    row = grid.iloc[0]
    return int(row["fast_window"]), int(row["slow_window"])
