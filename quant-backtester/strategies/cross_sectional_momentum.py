from __future__ import annotations

import pandas as pd


class CrossSectionalMomentumStrategy:
    """Long-only relative-strength strategy across several assets.

    At each rebalance date, rank assets by lookback total return and equally
    weight the top ``top_n``. Target weights remain unchanged between
    rebalances. Execution is deliberately left to the multi-asset engine, which
    applies the usual next-period delay.
    """

    def __init__(self, lookback: int = 126, top_n: int = 2, rebalance_every: int = 21):
        if lookback < 2:
            raise ValueError("lookback must be at least 2.")
        if top_n < 1:
            raise ValueError("top_n must be at least 1.")
        if rebalance_every < 1:
            raise ValueError("rebalance_every must be at least 1.")
        self.lookback = int(lookback)
        self.top_n = int(top_n)
        self.rebalance_every = int(rebalance_every)

    def generate_target_weights(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        if len(data) < 2:
            raise ValueError("Cross-sectional momentum requires at least two assets.")
        normalized = {symbol.upper().strip(): frame for symbol, frame in data.items()}
        if any(not symbol for symbol in normalized):
            raise ValueError("symbols cannot be blank.")
        if len(normalized) != len(data):
            raise ValueError("symbols must be unique after normalization.")
        symbols = sorted(normalized)
        if self.top_n > len(symbols):
            raise ValueError("top_n cannot exceed the number of assets.")

        common = None
        for frame in normalized.values():
            common = frame.index if common is None else common.intersection(frame.index)
        if common is None or len(common) == 0:
            raise ValueError("Assets have no common dates.")
        common = pd.DatetimeIndex(common).sort_values()

        prices = pd.DataFrame({
            symbol: normalized[symbol]["Adj Close" if "Adj Close" in normalized[symbol].columns else "Close"].reindex(common).astype(float)
            for symbol in symbols
        })
        momentum = prices / prices.shift(self.lookback) - 1.0
        weights = pd.DataFrame(0.0, index=common, columns=symbols)
        current = pd.Series(0.0, index=symbols)

        for i, date in enumerate(common):
            if i >= self.lookback and (i - self.lookback) % self.rebalance_every == 0:
                scores = momentum.loc[date].dropna()
                if len(scores) >= self.top_n:
                    winners = scores.nlargest(self.top_n).index
                    current = pd.Series(0.0, index=symbols)
                    current.loc[winners] = 1.0 / self.top_n
            weights.loc[date] = current

        return weights
