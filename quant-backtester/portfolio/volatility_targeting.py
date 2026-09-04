from __future__ import annotations

import numpy as np
import pandas as pd


class VolatilityTargetSizer:
    """Convert realized volatility into a long-only target exposure fraction.

    target_fraction = target annual volatility / estimated annual volatility,
    capped at ``max_fraction``. With max_fraction <= 1 this never uses leverage.
    """

    def __init__(self, target_volatility: float = 0.10, lookback: int = 20, max_fraction: float = 1.0):
        if target_volatility <= 0:
            raise ValueError("target_volatility must be positive.")
        if lookback < 2:
            raise ValueError("lookback must be at least 2.")
        if not 0 < max_fraction <= 1:
            raise ValueError("max_fraction must be in (0, 1].")
        self.target_volatility = float(target_volatility)
        self.lookback = int(lookback)
        self.max_fraction = float(max_fraction)

    def target_fractions(self, prices: pd.Series) -> pd.Series:
        prices = prices.astype(float)
        returns = prices.pct_change()
        realized = returns.rolling(self.lookback, min_periods=self.lookback).std(ddof=1) * np.sqrt(252.0)
        fraction = self.target_volatility / realized.replace(0.0, np.nan)
        fraction = fraction.clip(lower=0.0, upper=self.max_fraction).fillna(0.0)
        return fraction.rename("target_fraction")

    def apply_to_signal(self, prices: pd.Series, signal: pd.Series) -> pd.Series:
        fraction = self.target_fractions(prices)
        aligned_signal = signal.reindex(prices.index).fillna(0).clip(lower=0, upper=1)
        return (fraction * aligned_signal).rename("target_weight")
