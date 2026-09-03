from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    """Common contract for all trading strategies.

    Strategies express desired direction only. They do not execute orders,
    modify cash, or know portfolio state.
    """

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Return a time-indexed signal series using values in {-1, 0, +1}."""
        raise NotImplementedError
