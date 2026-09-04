from __future__ import annotations

import pandas as pd

from analytics.drawdown import maximum_drawdown
from analytics.risk import annualized_volatility, sharpe_ratio, sortino_ratio
from analytics.performance import cagr, total_return
from backtest.engine import BacktestEngine
from backtest.multi_asset import MultiAssetTargetWeightEngine, MultiAssetBacktestResult
from strategies.base import Strategy


def run_single_asset_strategy(
    symbol: str,
    data: pd.DataFrame,
    strategy: Strategy,
    *,
    engine_kwargs: dict | None = None,
):
    return BacktestEngine(**dict(engine_kwargs or {})).run(symbol, data, strategy)


def multi_asset_performance_summary(result: MultiAssetBacktestResult, risk_free_rate: float = 0.0) -> dict[str, float]:
    equity = result.equity_curve["equity"].astype(float)
    returns = equity.pct_change()
    elapsed_days = (equity.index[-1] - equity.index[0]).total_seconds() / 86_400.0
    years = elapsed_days / 365.25
    return {
        "total_return": total_return(result.initial_capital, result.ending_equity),
        "cagr": cagr(result.initial_capital, result.ending_equity, years),
        "annualized_volatility": annualized_volatility(returns),
        "sharpe_ratio": sharpe_ratio(returns, risk_free_rate=risk_free_rate),
        "sortino_ratio": sortino_ratio(returns, risk_free_rate=risk_free_rate),
        "maximum_drawdown": maximum_drawdown(equity),
    }
