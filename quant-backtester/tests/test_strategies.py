import pandas as pd
import pytest

from strategies.moving_average import MovingAverageStrategy


def test_moving_average_strategy_known_signals():
    dates = pd.date_range("2026-01-01", periods=6, freq="D")
    data = pd.DataFrame({"Adj Close": [1, 1, 1, 2, 3, 1]}, index=dates)
    strategy = MovingAverageStrategy(fast_window=2, slow_window=3)

    signals = strategy.generate_signals(data)

    assert signals.tolist() == [0, 0, 0, 1, 1, 0]


def test_moving_average_rejects_invalid_windows():
    with pytest.raises(ValueError):
        MovingAverageStrategy(fast_window=5, slow_window=5)
