import pandas as pd
import pytest

from backtest.engine import BacktestEngine
from strategies.base import Strategy


class FixedSignalStrategy(Strategy):
    def __init__(self, signals):
        self._signals = signals

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(self._signals, index=data.index, name="signal", dtype=int)


def toy_data():
    dates = pd.date_range("2026-01-01", periods=5, freq="D")
    return pd.DataFrame(
        {
            "Open": [100.0, 110.0, 120.0, 130.0, 140.0],
            "High": [101.0, 112.0, 125.0, 135.0, 145.0],
            "Low": [99.0, 109.0, 119.0, 129.0, 139.0],
            "Close": [100.0, 111.0, 124.0, 134.0, 144.0],
            "Volume": [1000, 1000, 1000, 1000, 1000],
        },
        index=pd.DatetimeIndex(dates, name="Date"),
    )


def test_signal_executes_only_on_next_day_open():
    data = toy_data()
    result = BacktestEngine(initial_capital=1_000).run(
        "TEST", data, FixedSignalStrategy([1, 1, 0, 0, 0])
    )

    # Day-1 signal is known at day-1 close; it cannot own shares on day 1.
    assert result.equity_curve.iloc[0]["quantity"] == 0
    assert result.equity_curve.iloc[0]["equity"] == pytest.approx(1_000)

    # It enters at day-2 Open = 110, not day-1 Close/Open.
    assert result.equity_curve.iloc[1]["quantity"] == pytest.approx(1_000 / 110)
    assert result.trades.iloc[0]["entry_price"] == pytest.approx(110)

    # Day-3 signal is 0 only after day-3 close, so exit occurs day-4 Open = 130.
    assert result.trades.iloc[0]["exit_price"] == pytest.approx(130)


def test_round_trip_accounting_reconciles_exactly():
    data = toy_data()
    result = BacktestEngine(initial_capital=1_000).run(
        "TEST", data, FixedSignalStrategy([1, 1, 0, 0, 0])
    )

    qty = 1_000 / 110
    expected_final = qty * 130
    expected_pnl = qty * (130 - 110)

    assert result.ending_equity == pytest.approx(expected_final)
    assert result.trades.iloc[0]["gross_pnl"] == pytest.approx(expected_pnl)
    assert result.trades.iloc[0]["net_pnl"] == pytest.approx(expected_pnl)
    assert result.trades.iloc[0]["costs"] == 0
    assert result.equity_curve.iloc[-1]["cash"] == pytest.approx(expected_final)
    assert result.equity_curve.iloc[-1]["quantity"] == 0


def test_equity_equals_cash_plus_marked_position_each_day():
    data = toy_data()
    result = BacktestEngine(initial_capital=1_000).run(
        "TEST", data, FixedSignalStrategy([1, 1, 0, 0, 0])
    )

    expected = result.equity_curve["cash"] + result.equity_curve["market_value"]
    pd.testing.assert_series_equal(
        result.equity_curve["equity"], expected, check_names=False
    )


def test_open_position_is_marked_to_market_without_fake_forced_exit():
    data = toy_data()
    result = BacktestEngine(initial_capital=1_000).run(
        "TEST", data, FixedSignalStrategy([1, 1, 1, 1, 1])
    )

    assert result.trades.empty
    assert result.equity_curve.iloc[-1]["quantity"] > 0
    assert result.ending_equity == pytest.approx((1_000 / 110) * 144)
