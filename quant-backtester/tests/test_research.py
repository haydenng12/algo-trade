import numpy as np
import pandas as pd
import pytest

from analytics.performance import performance_summary
from research.benchmark import buy_and_hold_equity, benchmark_summary
from research.cost_sensitivity import cost_sensitivity
from research.oos import moving_average_oos_signals
from research.grid import evaluate_oos, grid_search_moving_average, select_best_parameters
from research.split import chronological_split, date_split
from research.walk_forward import walk_forward_moving_average


def market_data(n=180, start="2020-01-01"):
    idx = pd.bdate_range(start, periods=n)
    # Smooth deterministic trend with cycles gives SMA variants different behavior.
    t = np.arange(n, dtype=float)
    close = 100 + 0.15 * t + 5 * np.sin(t / 8)
    return pd.DataFrame({
        "Open": close * 0.999,
        "High": close * 1.01,
        "Low": close * 0.99,
        "Close": close,
        "Adj Close": close,
        "Volume": np.full(n, 1_000_000),
    }, index=pd.DatetimeIndex(idx, name="Date"))


def test_buy_and_hold_benchmark_scales_with_price():
    data = market_data(5)
    equity = buy_and_hold_equity(data, initial_capital=1_000)
    assert equity.iloc[0] == pytest.approx(1_000)
    assert equity.iloc[-1] == pytest.approx(1_000 * data["Adj Close"].iloc[-1] / data["Adj Close"].iloc[0])


def test_chronological_split_never_shuffles():
    data = market_data(10)
    train, test = chronological_split(data, 0.6)
    assert len(train) == 6
    assert len(test) == 4
    assert train.index.max() < test.index.min()


def test_date_split_separates_past_and_future():
    data = market_data(10)
    split = data.index[6]
    train, test = date_split(data, split)
    assert (train.index < split).all()
    assert (test.index >= split).all()


def test_grid_search_only_contains_valid_parameter_pairs():
    train = market_data(120)
    grid = grid_search_moving_average("TEST", train, [5, 10, 30], [20, 40])
    assert not grid.empty
    assert (grid["fast_window"] < grid["slow_window"]).all()
    assert "selection_score" in grid.columns


def test_oos_uses_fixed_parameters_and_returns_only_test_dates():
    data = market_data(150)
    train, test = chronological_split(data, 0.7)
    result = evaluate_oos("TEST", train, test, 5, 20)
    assert result.equity_curve.index.equals(test.index)
    assert result.signals.index.equals(test.index)


def test_test_future_cannot_change_training_parameter_selection():
    data = market_data(150)
    train, test = chronological_split(data, 0.7)
    grid1 = grid_search_moving_average("TEST", train, [5, 10], [20, 30])
    best1 = select_best_parameters(grid1)

    # Deliberately mutate the untouched test future. Training selection must remain identical.
    mutated_test = test.copy()
    mutated_test[["Open", "High", "Low", "Close", "Adj Close"]] *= 10
    grid2 = grid_search_moving_average("TEST", train, [5, 10], [20, 30])
    best2 = select_best_parameters(grid2)
    assert best1 == best2


def test_cost_sensitivity_reduces_ending_equity_as_friction_rises():
    data = market_data(180)
    train, test = chronological_split(data, 0.7)
    table = cost_sensitivity("TEST", train, test, 5, 20, [0, 5, 25], initial_capital=10_000)
    assert list(table["cost_bps_each"]) == [0.0, 5.0, 25.0]
    assert table.loc[0, "ending_equity"] >= table.loc[1, "ending_equity"] >= table.loc[2, "ending_equity"]
    assert table.loc[2, "total_fees"] > table.loc[1, "total_fees"] > 0


def test_walk_forward_train_window_always_precedes_test_window():
    data = market_data(180)
    wf = walk_forward_moving_average(
        "TEST", data, [5, 10], [20, 30], train_size=80, test_size=25,
        engine_kwargs={"initial_capital": 10_000},
    )
    assert not wf.folds.empty
    assert (wf.folds["train_end"] < wf.folds["test_start"]).all()
    assert wf.equity_curve.index.is_monotonic_increasing
    assert wf.ending_equity > 0


def test_oos_indicators_use_past_training_context():
    data = market_data(140)
    train, test = chronological_split(data, 0.75)

    with_context = moving_average_oos_signals(train, test, 5, 20)
    test_only = __import__("strategies.moving_average", fromlist=["MovingAverageStrategy"]).MovingAverageStrategy(5, 20).generate_signals(test)

    # Enough pre-test history exists, so the first OOS signal can be valid immediately.
    assert with_context.iloc[0] in (0, 1)
    assert test_only.iloc[:19].eq(0).all()
    # Our deterministic trending data should make the context-aware first signal long.
    assert with_context.iloc[0] == 1


def test_oos_lookback_does_not_carry_training_portfolio_state():
    data = market_data(140)
    train, test = chronological_split(data, 0.75)
    initial_capital = 12_345.0

    result = evaluate_oos(
        "TEST", train, test, 5, 20,
        engine_kwargs={"initial_capital": initial_capital, "execution_delay": 1},
    )

    first = result.equity_curve.iloc[0]
    assert first["cash"] == pytest.approx(initial_capital)
    assert first["quantity"] == pytest.approx(0.0)
    assert first["equity"] == pytest.approx(initial_capital)
    assert pd.isna(first["executed_signal"])


def test_cost_sensitivity_matches_oos_when_configuration_is_identical():
    data = market_data(180)
    train, test = chronological_split(data, 0.7)
    config = {
        "initial_capital": 10_000.0,
        "transaction_cost_bps": 5.0,
        "slippage_bps": 5.0,
        "position_fraction": 1.0,
        "execution_delay": 1,
    }

    oos = evaluate_oos("TEST", train, test, 5, 20, engine_kwargs=config)
    table = cost_sensitivity(
        "TEST", train, test, 5, 20, [5],
        initial_capital=config["initial_capital"],
        position_fraction=config["position_fraction"],
        execution_delay=config["execution_delay"],
    )

    row = table.iloc[0]
    assert row["ending_equity"] == pytest.approx(oos.ending_equity)
    assert row["total_return"] == pytest.approx(performance_summary(oos)["total_return"])
    assert row["sharpe_ratio"] == pytest.approx(performance_summary(oos)["sharpe_ratio"])


def test_oos_context_must_end_before_evaluation_period():
    data = market_data(80)
    train, test = chronological_split(data, 0.7)
    overlapping = pd.concat([train, test.iloc[:1]])
    with pytest.raises(ValueError, match="must end before"):
        moving_average_oos_signals(overlapping, test, 5, 20)
