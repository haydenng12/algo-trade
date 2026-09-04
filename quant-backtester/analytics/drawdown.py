from __future__ import annotations

import pandas as pd


def _validate_equity(equity: pd.Series) -> pd.Series:
    values = equity.astype(float)
    if values.empty:
        raise ValueError("Equity series cannot be empty.")
    if values.isna().any():
        raise ValueError("Equity series cannot contain missing values.")
    if (values <= 0).any():
        raise ValueError("Equity values must be strictly positive.")
    return values


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Return percentage drawdown from the running equity peak.

    A value of -0.20 means equity is 20% below its previous high-water mark.
    """
    values = _validate_equity(equity)
    running_peak = values.cummax()
    drawdown = values / running_peak - 1.0
    drawdown.name = "drawdown"
    return drawdown


def maximum_drawdown(equity: pd.Series) -> float:
    """Return the most negative peak-to-trough drawdown."""
    return float(drawdown_series(equity).min())
