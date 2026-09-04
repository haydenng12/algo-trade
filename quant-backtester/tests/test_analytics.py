import math

import numpy as np
import pandas as pd
import pytest

from analytics.drawdown import drawdown_series, maximum_drawdown
from analytics.performance import cagr, equity_returns, performance_summary, total_return
from analytics.risk import annualized_volatility, sharpe_ratio, sortino_ratio
from analytics.trades import trade_summary
from backtest.engine import BacktestEngine
from strategies.base import Strategy


class FixedSignalStrategy(Strategy):
    def __init__(self, signals):
        self._signals = signals

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(self._signals, index=data.index, name="signal", dtype=int)


def test_total_return_known_values():
    assert total_return(100.0, 125.0) == pytest.approx(0.25)


def test_cagr_known_two_year_doubling():
    expected = math.sqrt(2.0) - 1.0
    assert cagr(100.0, 200.0, 2.0) == pytest.approx(expected)


def test_equity_returns_are_simple_returns():
    equity = pd.Series([100.0, 110.0, 99.0])
    returns = equity_returns(equity)
    assert math.isnan(returns.iloc[0])
    assert returns.iloc[1] == pytest.approx(0.10)
    assert returns.iloc[2] == pytest.approx(-0.10)


def test_annualized_volatility_matches_sample_standard_deviation():
    returns = pd.Series([0.01, -0.01, 0.02, -0.02])
    expected = returns.std(ddof=1) * math.sqrt(252)
    assert annualized_volatility(returns) == pytest.approx(expected)


def test_sharpe_matches_hand_formula_with_zero_risk_free_rate():
    returns = pd.Series([0.01, -0.005, 0.02, 0.0])
    expected = returns.mean() / returns.std(ddof=1) * math.sqrt(252)
    assert sharpe_ratio(returns) == pytest.approx(expected)


def test_sortino_penalizes_only_returns_below_target():
    returns = pd.Series([0.02, -0.01, 0.01, -0.03])
    downside = np.minimum(returns.to_numpy(), 0.0)
    downside_dev = np.sqrt(np.mean(np.square(downside)))
    expected = returns.mean() * 252 / (downside_dev * math.sqrt(252))
    assert sortino_ratio(returns) == pytest.approx(expected)


def test_drawdown_path_and_maximum_drawdown():
    equity = pd.Series([100.0, 120.0, 90.0, 108.0, 130.0])
    drawdown = drawdown_series(equity)
    assert drawdown.tolist() == pytest.approx([0.0, 0.0, -0.25, -0.10, 0.0])
    assert maximum_drawdown(equity) == pytest.approx(-0.25)


def test_trade_summary_known_wins_losses_and_profit_factor():
    trades = pd.DataFrame(
        {
            "net_pnl": [100.0, -50.0, 25.0],
            "return_pct": [0.10, -0.05, 0.025],
            "holding_period_days": [5, 3, 4],
        }
    )
    summary = trade_summary(trades)
    assert summary["number_of_trades"] == 3
    assert summary["win_rate"] == pytest.approx(2 / 3)
    assert summary["profit_factor"] == pytest.approx(125 / 50)
    assert summary["average_trade_return"] == pytest.approx(0.025)
    assert summary["average_winner"] == pytest.approx(0.0625)
    assert summary["average_loser"] == pytest.approx(-0.05)
    assert summary["best_trade"] == pytest.approx(0.10)
    assert summary["worst_trade"] == pytest.approx(-0.05)
    assert summary["average_holding_period_days"] == pytest.approx(4.0)


def test_empty_trade_summary_is_explicit_not_zero_performance():
    trades = pd.DataFrame(columns=["net_pnl", "return_pct", "holding_period_days"])
    summary = trade_summary(trades)
    assert summary["number_of_trades"] == 0
    assert math.isnan(summary["win_rate"])
    assert math.isnan(summary["profit_factor"])


def test_profit_factor_is_infinite_when_there_are_wins_but_no_losses():
    trades = pd.DataFrame(
        {
            "net_pnl": [10.0, 20.0],
            "return_pct": [0.01, 0.02],
            "holding_period_days": [1, 2],
        }
    )
    assert math.isinf(trade_summary(trades)["profit_factor"])


def test_full_performance_summary_reconciles_with_backtest_result():
    dates = pd.date_range("2025-01-01", periods=5, freq="365D")
    data = pd.DataFrame(
        {
            "Open": [100.0, 100.0, 110.0, 120.0, 120.0],
            "High": [101.0, 111.0, 121.0, 121.0, 121.0],
            "Low": [99.0, 99.0, 109.0, 119.0, 119.0],
            "Close": [100.0, 110.0, 120.0, 120.0, 120.0],
            "Volume": [1000] * 5,
        },
        index=pd.DatetimeIndex(dates, name="Date"),
    )
    result = BacktestEngine(initial_capital=1_000.0).run(
        "TEST", data, FixedSignalStrategy([1, 1, 0, 0, 0])
    )
    summary = performance_summary(result)

    assert summary["total_return"] == pytest.approx(result.ending_equity / 1_000.0 - 1.0)
    assert summary["maximum_drawdown"] <= 0.0
    assert summary["number_of_trades"] == 1
    assert summary["win_rate"] == pytest.approx(1.0)
