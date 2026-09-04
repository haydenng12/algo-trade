from __future__ import annotations

import pandas as pd

from strategies.base import Strategy


class ZScoreMeanReversionStrategy(Strategy):
    """Long/flat mean-reversion strategy based on a rolling price Z-score.

    A Z-score measures how many rolling standard deviations price is from its
    rolling mean. The strategy enters when price is unusually low and exits
    after it reverts toward the mean.
    """

    def __init__(
        self,
        lookback: int = 20,
        entry_z: float = -1.5,
        exit_z: float = 0.0,
        price_column: str = "Adj Close",
    ):
        if lookback < 2:
            raise ValueError("lookback must be at least 2.")
        if entry_z >= exit_z:
            raise ValueError("entry_z must be smaller than exit_z.")
        self.lookback = int(lookback)
        self.entry_z = float(entry_z)
        self.exit_z = float(exit_z)
        self.price_column = price_column

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        column = self.price_column
        if column not in data.columns:
            if column == "Adj Close" and "Close" in data.columns:
                column = "Close"
            else:
                raise KeyError(f"Price column {self.price_column!r} not found in market data.")

        price = data[column].astype(float)
        mean = price.rolling(self.lookback, min_periods=self.lookback).mean()
        std = price.rolling(self.lookback, min_periods=self.lookback).std(ddof=0)
        zscore = (price - mean) / std.replace(0.0, pd.NA)

        state = 0
        signals: list[int] = []
        for z in zscore:
            if pd.isna(z):
                state = 0
            elif state == 0 and z <= self.entry_z:
                state = 1
            elif state == 1 and z >= self.exit_z:
                state = 0
            signals.append(state)

        return pd.Series(signals, index=data.index, name="signal", dtype=int)
