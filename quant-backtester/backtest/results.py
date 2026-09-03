from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class BacktestResult:
    symbol: str
    initial_capital: float
    signals: pd.Series
    equity_curve: pd.DataFrame
    trades: pd.DataFrame

    @property
    def ending_equity(self) -> float:
        return float(self.equity_curve["equity"].iloc[-1])
