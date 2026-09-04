from __future__ import annotations

import pandas as pd

from backtest.engine import BacktestEngine
from backtest.results import BacktestResult
from strategies.base import Strategy
from strategies.moving_average import MovingAverageStrategy


class SignalStrategy(Strategy):
    """Adapter for executing a precomputed signal series on evaluation data."""

    def __init__(self, signals: pd.Series):
        self.signals = signals.astype(int)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        return self.signals.reindex(data.index)


def moving_average_oos_signals(
    context_data: pd.DataFrame,
    evaluation_data: pd.DataFrame,
    fast_window: int,
    slow_window: int,
) -> pd.Series:
    """Generate evaluation-period SMA signals using past context only.

    ``context_data`` may contain observations strictly before the evaluation
    period. Those rows are used only to warm up indicators. Returned signals
    are restricted to ``evaluation_data`` dates, so no context-period portfolio
    state can leak into the evaluation backtest.
    """
    if context_data.empty:
        raise ValueError("context_data cannot be empty.")
    if evaluation_data.empty:
        raise ValueError("evaluation_data cannot be empty.")
    if context_data.index.max() >= evaluation_data.index.min():
        raise ValueError("context_data must end before evaluation_data begins.")

    combined = pd.concat([context_data, evaluation_data])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    signals = MovingAverageStrategy(fast_window, slow_window).generate_signals(combined)
    evaluation_signals = signals.reindex(evaluation_data.index)
    if evaluation_signals.isna().any():
        raise ValueError("Failed to produce complete evaluation-period signals.")
    return evaluation_signals.astype(int)


def run_oos_moving_average(
    symbol: str,
    context_data: pd.DataFrame,
    evaluation_data: pd.DataFrame,
    fast_window: int,
    slow_window: int,
    *,
    engine_kwargs: dict | None = None,
) -> BacktestResult:
    """Run a fresh-capital OOS backtest with legitimate pre-test lookback.

    Prices before the evaluation period may warm up indicators, but the engine
    itself receives only evaluation-period rows. Cash, positions, P&L, and trade
    history therefore always start fresh at the evaluation boundary.
    """
    signals = moving_average_oos_signals(
        context_data,
        evaluation_data,
        fast_window,
        slow_window,
    )
    engine = BacktestEngine(**dict(engine_kwargs or {}))
    return engine.run(symbol, evaluation_data, SignalStrategy(signals))
