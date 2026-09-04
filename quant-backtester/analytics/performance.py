from __future__ import annotations

import math

import pandas as pd

from analytics.drawdown import maximum_drawdown
from analytics.risk import annualized_volatility, sharpe_ratio, sortino_ratio
from analytics.trades import trade_summary
from backtest.results import BacktestResult


def total_return(initial_value: float, ending_value: float) -> float:
    """Total percentage growth from starting value to ending value."""
    if initial_value <= 0:
        raise ValueError("initial_value must be positive.")
    if ending_value <= 0:
        raise ValueError("ending_value must be positive.")
    return float(ending_value / initial_value - 1.0)


def cagr(initial_value: float, ending_value: float, years: float) -> float:
    """Compound annual growth rate over a positive number of years."""
    if initial_value <= 0:
        raise ValueError("initial_value must be positive.")
    if ending_value <= 0:
        raise ValueError("ending_value must be positive.")
    if years <= 0:
        raise ValueError("years must be positive.")
    return float((ending_value / initial_value) ** (1.0 / years) - 1.0)


def equity_returns(equity: pd.Series) -> pd.Series:
    """Simple period-to-period returns from a positive equity curve."""
    values = equity.astype(float)
    if values.empty:
        raise ValueError("Equity series cannot be empty.")
    if values.isna().any():
        raise ValueError("Equity series cannot contain missing values.")
    if (values <= 0).any():
        raise ValueError("Equity values must be strictly positive.")
    out = values.pct_change()
    out.name = "portfolio_return"
    return out


def _elapsed_years(index: pd.Index) -> float:
    if len(index) < 2:
        return float("nan")
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("Equity curve must use a DatetimeIndex to calculate CAGR.")
    days = (index[-1] - index[0]).total_seconds() / 86_400.0
    if days <= 0:
        return float("nan")
    return days / 365.25


def performance_summary(
    result: BacktestResult,
    risk_free_rate: float = 0.0,
) -> dict[str, float | int]:
    """Compute Stage 3 portfolio and completed-trade analytics."""
    equity = result.equity_curve["equity"].astype(float)
    returns = equity_returns(equity)
    years = _elapsed_years(equity.index)

    summary: dict[str, float | int] = {
        "total_return": total_return(result.initial_capital, result.ending_equity),
        "cagr": (
            cagr(result.initial_capital, result.ending_equity, years)
            if math.isfinite(years) and years > 0
            else float("nan")
        ),
        "annualized_volatility": annualized_volatility(returns),
        "sharpe_ratio": sharpe_ratio(returns, risk_free_rate=risk_free_rate),
        "sortino_ratio": sortino_ratio(returns, risk_free_rate=risk_free_rate),
        "maximum_drawdown": maximum_drawdown(equity),
    }
    summary.update(trade_summary(result.trades))
    return summary
