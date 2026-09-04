from __future__ import annotations

import pandas as pd

from strategies.base import Strategy


class RSIMeanReversionStrategy(Strategy):
    """Long/flat RSI mean-reversion strategy using Wilder-style smoothing."""

    def __init__(
        self,
        lookback: int = 14,
        oversold: float = 30.0,
        exit_level: float = 50.0,
        price_column: str = "Adj Close",
    ):
        if lookback < 2:
            raise ValueError("lookback must be at least 2.")
        if not 0 <= oversold < exit_level <= 100:
            raise ValueError("Require 0 <= oversold < exit_level <= 100.")
        self.lookback = int(lookback)
        self.oversold = float(oversold)
        self.exit_level = float(exit_level)
        self.price_column = price_column

    def rsi(self, data: pd.DataFrame) -> pd.Series:
        column = self.price_column
        if column not in data.columns:
            if column == "Adj Close" and "Close" in data.columns:
                column = "Close"
            else:
                raise KeyError(f"Price column {self.price_column!r} not found in market data.")

        price = data[column].astype(float)
        delta = price.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        alpha = 1.0 / self.lookback
        avg_gain = gain.ewm(alpha=alpha, adjust=False, min_periods=self.lookback).mean()
        avg_loss = loss.ewm(alpha=alpha, adjust=False, min_periods=self.lookback).mean()
        rs = avg_gain / avg_loss.replace(0.0, pd.NA)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        # Purely rising windows have zero losses and therefore RSI=100.
        rsi = rsi.mask((avg_loss == 0) & avg_gain.notna(), 100.0)
        # Flat windows have neither gains nor losses; treat them as neutral.
        rsi = rsi.mask((avg_loss == 0) & (avg_gain == 0), 50.0)
        return rsi.rename("rsi")

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        rsi = self.rsi(data)
        state = 0
        signals: list[int] = []
        for value in rsi:
            if pd.isna(value):
                state = 0
            elif state == 0 and value <= self.oversold:
                state = 1
            elif state == 1 and value >= self.exit_level:
                state = 0
            signals.append(state)
        return pd.Series(signals, index=data.index, name="signal", dtype=int)
