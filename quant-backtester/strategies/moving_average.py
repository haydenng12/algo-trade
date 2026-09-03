from __future__ import annotations

import pandas as pd

from strategies.base import Strategy


class MovingAverageStrategy(Strategy):
    """Long/flat moving-average momentum strategy.

    Signal = 1 when fast SMA > slow SMA, otherwise 0. The signal is known only
    after the row's closing information is available; execution timing belongs
    to the backtest engine, not the strategy.
    """

    def __init__(self, fast_window: int = 20, slow_window: int = 50, price_column: str = "Adj Close"):
        if fast_window <= 0 or slow_window <= 0:
            raise ValueError("Moving-average windows must be positive integers.")
        if fast_window >= slow_window:
            raise ValueError("fast_window must be smaller than slow_window.")
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.price_column = price_column

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        column = self.price_column
        if column not in data.columns:
            if column == "Adj Close" and "Close" in data.columns:
                column = "Close"
            else:
                raise KeyError(f"Price column {self.price_column!r} not found in market data.")

        price = data[column].astype(float)
        fast = price.rolling(self.fast_window, min_periods=self.fast_window).mean()
        slow = price.rolling(self.slow_window, min_periods=self.slow_window).mean()

        signal = (fast > slow).astype(int)
        signal.loc[slow.isna()] = 0
        signal.name = "signal"
        return signal
