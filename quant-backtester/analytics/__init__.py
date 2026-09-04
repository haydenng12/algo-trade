"""Performance analytics for backtest results."""

from analytics.drawdown import drawdown_series, maximum_drawdown
from analytics.performance import cagr, performance_summary, total_return
from analytics.risk import annualized_volatility, sharpe_ratio, sortino_ratio
from analytics.trades import trade_summary

__all__ = [
    "total_return",
    "cagr",
    "annualized_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "drawdown_series",
    "maximum_drawdown",
    "trade_summary",
    "performance_summary",
]
