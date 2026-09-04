from __future__ import annotations

import pandas as pd

from analytics.drawdown import maximum_drawdown
from analytics.performance import cagr, equity_returns, total_return
from analytics.risk import annualized_volatility, sharpe_ratio


def _price(data: pd.DataFrame, price_column: str = "Adj Close") -> pd.Series:
    column = price_column if price_column in data.columns else "Close"
    if column not in data.columns:
        raise KeyError("Benchmark requires Adj Close or Close.")
    price = data[column].astype(float)
    if price.empty or price.isna().any() or (price <= 0).any():
        raise ValueError("Benchmark prices must be positive and non-missing.")
    return price


def buy_and_hold_equity(data: pd.DataFrame, initial_capital: float = 100_000.0, price_column: str = "Adj Close") -> pd.Series:
    """Frictionless buy-and-hold benchmark over the exact supplied period."""
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive.")
    price = _price(data, price_column)
    equity = initial_capital * price / price.iloc[0]
    equity.name = "benchmark_equity"
    return equity


def benchmark_summary(data: pd.DataFrame, initial_capital: float = 100_000.0, risk_free_rate: float = 0.0) -> dict[str, float]:
    equity = buy_and_hold_equity(data, initial_capital)
    years = (equity.index[-1] - equity.index[0]).total_seconds() / 86_400 / 365.25
    returns = equity_returns(equity)
    return {
        "ending_equity": float(equity.iloc[-1]),
        "total_return": total_return(initial_capital, float(equity.iloc[-1])),
        "cagr": cagr(initial_capital, float(equity.iloc[-1]), years) if years > 0 else float("nan"),
        "annualized_volatility": annualized_volatility(returns),
        "sharpe_ratio": sharpe_ratio(returns, risk_free_rate=risk_free_rate),
        "maximum_drawdown": maximum_drawdown(equity),
    }
