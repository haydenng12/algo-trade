import pandas as pd
import pytest

from backtest.engine import BacktestEngine
from strategies.base import Strategy


class FixedSignalStrategy(Strategy):
    def __init__(self, signals):
        self._signals = signals

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(self._signals, index=data.index, name="signal", dtype=int)


def flat_open_data():
    dates = pd.date_range("2026-01-01", periods=5, freq="D")
    return pd.DataFrame(
        {
            "Open": [100.0] * 5,
            "High": [101.0] * 5,
            "Low": [99.0] * 5,
            "Close": [100.0] * 5,
            "Volume": [1000] * 5,
        },
        index=pd.DatetimeIndex(dates, name="Date"),
    )


def test_costs_and_slippage_reduce_equity_and_reconcile_trade_pnl():
    data = flat_open_data()
    result = BacktestEngine(
        initial_capital=1_000,
        transaction_cost_bps=10,
        slippage_bps=10,
    ).run("TEST", data, FixedSignalStrategy([1, 0, 0, 0, 0]))

    trade = result.trades.iloc[0]
    assert trade["entry_price"] > trade["entry_market_price"]
    assert trade["exit_price"] < trade["exit_market_price"]
    assert trade["gross_pnl"] == pytest.approx(0.0)
    assert trade["transaction_costs"] > 0
    assert trade["slippage_costs"] > 0
    assert trade["costs"] == pytest.approx(trade["transaction_costs"] + trade["slippage_costs"])
    assert trade["net_pnl"] == pytest.approx(trade["gross_pnl"] - trade["costs"])
    assert result.ending_equity < 1_000


def test_cash_constraint_never_allows_negative_cash():
    data = flat_open_data()
    result = BacktestEngine(
        initial_capital=1_000,
        transaction_cost_bps=25,
        slippage_bps=25,
        position_fraction=1.0,
    ).run("TEST", data, FixedSignalStrategy([1, 1, 1, 1, 1]))

    assert (result.equity_curve["cash"] >= -1e-9).all()
    assert result.equity_curve.iloc[1]["quantity"] > 0


def test_half_sizing_keeps_approximately_half_capital_in_cash_before_marking():
    data = flat_open_data()
    result = BacktestEngine(
        initial_capital=1_000,
        position_fraction=0.5,
    ).run("TEST", data, FixedSignalStrategy([1, 1, 1, 1, 1]))

    row = result.equity_curve.iloc[1]
    assert row["quantity"] == pytest.approx(5.0)
    assert row["cash"] == pytest.approx(500.0)
    assert row["market_value"] == pytest.approx(500.0)
    assert row["equity"] == pytest.approx(1_000.0)


def test_configurable_execution_delay_is_respected():
    data = flat_open_data()
    result = BacktestEngine(initial_capital=1_000, execution_delay=2).run(
        "TEST", data, FixedSignalStrategy([1, 1, 1, 0, 0])
    )

    assert result.equity_curve.iloc[0]["quantity"] == 0
    assert result.equity_curve.iloc[1]["quantity"] == 0
    assert result.equity_curve.iloc[2]["quantity"] > 0


def test_zero_delay_is_rejected_to_protect_against_lookahead():
    with pytest.raises(ValueError):
        BacktestEngine(execution_delay=0)
