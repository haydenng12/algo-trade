from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from analytics.performance import performance_summary
from research.grid import evaluate_oos, grid_search_moving_average, select_best_parameters


@dataclass(frozen=True)
class WalkForwardResult:
    folds: pd.DataFrame
    equity_curve: pd.Series

    @property
    def ending_equity(self) -> float:
        return float(self.equity_curve.iloc[-1])


def walk_forward_moving_average(
    symbol: str,
    data: pd.DataFrame,
    fast_windows: list[int] | tuple[int, ...],
    slow_windows: list[int] | tuple[int, ...],
    *,
    train_size: int,
    test_size: int,
    expanding: bool = False,
    selection_metric: str = "sharpe_ratio",
    engine_kwargs: dict | None = None,
) -> WalkForwardResult:
    """Repeatedly tune on past data and test only on the immediately following window."""
    if train_size < 2 or test_size < 1:
        raise ValueError("train_size must be >=2 and test_size >=1.")
    if len(data) < train_size + test_size:
        raise ValueError("Not enough data for one walk-forward fold.")

    config = dict(engine_kwargs or {})
    base_capital = float(config.get("initial_capital", 100_000.0))
    fold_rows = []
    pieces = []
    capital = base_capital
    test_start = train_size
    fold = 1

    while test_start + test_size <= len(data):
        train_start = 0 if expanding else test_start - train_size
        train = data.iloc[train_start:test_start]
        test = data.iloc[test_start:test_start + test_size]

        grid = grid_search_moving_average(
            symbol, train, fast_windows, slow_windows,
            selection_metric=selection_metric, engine_kwargs=config,
        )
        fast, slow = select_best_parameters(grid)
        fold_config = dict(config)
        fold_config["initial_capital"] = capital
        result = evaluate_oos(symbol, train, test, fast, slow, engine_kwargs=fold_config)
        metrics = performance_summary(result)
        fold_rows.append({
            "fold": fold,
            "train_start": train.index[0], "train_end": train.index[-1],
            "test_start": test.index[0], "test_end": test.index[-1],
            "fast_window": fast, "slow_window": slow,
            "train_selection_score": float(grid.iloc[0]["selection_score"]),
            "test_total_return": float(metrics["total_return"]),
            "test_sharpe_ratio": float(metrics["sharpe_ratio"]),
            "test_maximum_drawdown": float(metrics["maximum_drawdown"]),
            "ending_equity": result.ending_equity,
        })
        piece = result.equity_curve["equity"].copy()
        pieces.append(piece)
        capital = result.ending_equity
        test_start += test_size
        fold += 1

    equity = pd.concat(pieces)
    equity = equity[~equity.index.duplicated(keep="last")]
    equity.name = "walk_forward_equity"
    return WalkForwardResult(pd.DataFrame(fold_rows), equity)
