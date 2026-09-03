from __future__ import annotations

import numpy as np
import pandas as pd


def select_return_price(data: pd.DataFrame, *, prefer_adjusted: bool = True) -> pd.Series:
    """Choose the price series used for historical return calculations."""
    if prefer_adjusted and "Adj Close" in data.columns:
        price = data["Adj Close"]
    elif "Close" in data.columns:
        price = data["Close"]
    else:
        raise KeyError("Data must contain 'Close' or 'Adj Close'.")
    return price.astype(float)


def simple_returns(
    data: pd.DataFrame | pd.Series,
    *,
    prefer_adjusted: bool = True,
    name: str = "simple_return",
) -> pd.Series:
    """Compute R_t = P_t / P_(t-1) - 1 without filling the first NaN."""
    price = select_return_price(data, prefer_adjusted=prefer_adjusted) if isinstance(data, pd.DataFrame) else data.astype(float)
    result = price.pct_change(fill_method=None)
    result.name = name
    return result


def log_returns(
    data: pd.DataFrame | pd.Series,
    *,
    prefer_adjusted: bool = True,
    name: str = "log_return",
) -> pd.Series:
    """Compute ln(P_t / P_(t-1)); included now for a stable feature API."""
    price = select_return_price(data, prefer_adjusted=prefer_adjusted) if isinstance(data, pd.DataFrame) else data.astype(float)
    result = np.log(price / price.shift(1))
    result.name = name
    return result
