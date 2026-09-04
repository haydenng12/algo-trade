from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dashboard.research import cost_sensitivity_table, moving_average_parameter_grid
from dashboard.runner import StrategyConfig, equal_weight_buy_and_hold_equity, run_strategy


def frame_from_prices(prices, start="2020-01-01"):
    idx = pd.bdate_range(start, periods=len(prices))
    close = pd.Series(prices, index=idx, dtype=float)
    return pd.DataFrame({
        "Open": close,
        "High": close * 1.01,
        "Low": close * 0.99,
        "Close": close,
        "Adj Close": close,
        "Volume": 1_000_000,
    }, index=idx)


def test_equal_weight_benchmark_uses_same_universe():
    a = frame_from_prices([100, 110, 120])
    b = frame_from_prices([100, 90, 80])
    equity = equal_weight_buy_and_hold_equity({"A": a, "B": b}, 100_000)
    assert equity.iloc[0] == pytest.approx(100_000)
    assert equity.iloc[-1] == pytest.approx(100_000)  # +20% and -20% average to flat


def test_dashboard_single_asset_run_returns_benchmark_and_drawdown():
    prices = np.linspace(100, 150, 90)
    frame = frame_from_prices(prices)
    run = run_strategy(
        {"SPY": frame}, "SPY", StrategyConfig("Moving Average Trend", {"fast_window": 5, "slow_window": 20}),
        initial_capital=100_000, fees_bps=5, slippage_bps=5,
    )
    assert len(run.equity) == len(frame)
    assert len(run.benchmark_equity) == len(frame)
    assert run.trades is not None
    assert run.drawdown.index.equals(frame.index)


def test_dashboard_cross_sectional_uses_equal_weight_universe_benchmark():
    n = 180
    frames = {
        "SPY": frame_from_prices(np.linspace(100, 150, n)),
        "QQQ": frame_from_prices(np.linspace(100, 190, n)),
        "IWM": frame_from_prices(np.linspace(100, 120, n)),
    }
    run = run_strategy(
        frames, "SPY",
        StrategyConfig("Cross-Sectional Momentum", {"lookback": 20, "top_n": 1, "rebalance_every": 10}),
    )
    assert run.benchmark_name == "Equal-Weight Universe Buy & Hold"
    assert run.target_weights is not None
    assert set(run.target_weights.columns) == {"IWM", "QQQ", "SPY"}


def test_cost_sensitivity_reuses_same_strategy_and_higher_costs_reduce_equity():
    prices = 100 + np.sin(np.arange(160) / 5) * 8 + np.arange(160) * 0.15
    frame = frame_from_prices(prices)
    table = cost_sensitivity_table(
        {"SPY": frame}, "SPY",
        StrategyConfig("Moving Average Trend", {"fast_window": 5, "slow_window": 15}),
        [0, 25],
    )
    assert table.loc[0, "ending_equity"] >= table.loc[1, "ending_equity"]


def test_parameter_grid_filters_invalid_pairs_and_sorts_by_sharpe():
    frame = frame_from_prices(100 + np.sin(np.arange(120) / 6) * 5 + np.arange(120) * 0.1)
    grid = moving_average_parameter_grid("SPY", frame, [10, 50], [20, 40])
    assert (grid["fast_window"] < grid["slow_window"]).all()
    assert grid["sharpe_ratio"].is_monotonic_decreasing

from dashboard.presentation import benchmark_comparison_table, configuration_rows, interpretation_text


def test_benchmark_comparison_table_formats_finance_values_for_display():
    summary = {
        "total_return": 1.0885,
        "cagr": 0.0693,
        "annualized_volatility": 0.1131,
        "sharpe_ratio": 0.65,
        "maximum_drawdown": -0.292,
    }
    benchmark = {
        "total_return": 2.9974,
        "cagr": 0.1343,
        "annualized_volatility": 0.1779,
        "sharpe_ratio": 0.80,
        "maximum_drawdown": -0.3372,
    }
    table = benchmark_comparison_table(summary, benchmark)
    assert table.loc[0, "Strategy"] == "108.85%"
    assert table.loc[0, "Benchmark"] == "299.74%"
    assert table.loc[3, "Strategy"] == "0.65"


def test_configuration_rows_are_human_readable_not_raw_json():
    rows = configuration_rows("Moving Average Trend", {"fast_window": 20, "slow_window": 50})
    assert rows == [("Fast SMA", "20 days"), ("Slow SMA", "50 days")]


def test_interpretation_text_reports_return_and_risk_tradeoff():
    summary = {
        "total_return": 1.0885,
        "annualized_volatility": 0.1131,
        "maximum_drawdown": -0.292,
    }
    benchmark = {
        "total_return": 2.9974,
        "annualized_volatility": 0.1779,
        "maximum_drawdown": -0.3372,
    }
    text = interpretation_text(summary, benchmark)
    assert "trailed the benchmark" in text
    assert "lower annualized volatility" in text
    assert "shallower maximum drawdown" in text
