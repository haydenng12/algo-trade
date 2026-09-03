import math

import pandas as pd

from data.cleaner import clean_ohlcv
from features.returns import log_returns, simple_returns


def test_simple_returns_known_sequence():
    prices = pd.Series([100.0, 105.0, 102.9])
    result = simple_returns(prices)
    assert math.isnan(result.iloc[0])
    assert abs(result.iloc[1] - 0.05) < 1e-12
    assert abs(result.iloc[2] - (-0.02)) < 1e-12


def test_adjusted_close_is_preferred(clean_frame):
    data, _ = clean_ohlcv(clean_frame)
    result = simple_returns(data)
    expected = 50.5 / 51.0 - 1.0
    assert abs(result.iloc[1] - expected) < 1e-12


def test_can_explicitly_use_raw_close(clean_frame):
    data, _ = clean_ohlcv(clean_frame)
    result = simple_returns(data, prefer_adjusted=False)
    expected = 101.0 / 102.0 - 1.0
    assert abs(result.iloc[1] - expected) < 1e-12


def test_log_returns_add_over_time():
    prices = pd.Series([100.0, 110.0, 121.0])
    result = log_returns(prices)
    assert abs(result.iloc[1] + result.iloc[2] - math.log(121.0 / 100.0)) < 1e-12
