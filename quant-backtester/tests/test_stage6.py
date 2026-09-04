import numpy as np
import pandas as pd
import pytest

from backtest.multi_asset import MultiAssetTargetWeightEngine
from portfolio.volatility_targeting import VolatilityTargetSizer
from strategies.cross_sectional_momentum import CrossSectionalMomentumStrategy
from strategies.rsi import RSIMeanReversionStrategy
from strategies.zscore_mean_reversion import ZScoreMeanReversionStrategy


def frame(prices, start="2020-01-01"):
    prices = np.asarray(prices, dtype=float)
    idx = pd.bdate_range(start, periods=len(prices), name="Date")
    return pd.DataFrame({
        "Open": prices,
        "High": prices * 1.01,
        "Low": prices * 0.99,
        "Close": prices,
        "Adj Close": prices,
        "Volume": np.full(len(prices), 1_000_000),
    }, index=idx)


def test_zscore_mean_reversion_enters_on_deviation_and_exits_on_reversion():
    data = frame([100, 100, 100, 100, 100, 90, 94, 98, 100])
    strategy = ZScoreMeanReversionStrategy(lookback=5, entry_z=-1.5, exit_z=0.0)
    signals = strategy.generate_signals(data)
    assert signals.iloc[5] == 1
    assert signals.iloc[-1] == 0
    assert set(signals.unique()).issubset({0, 1})


def test_zscore_rejects_invalid_threshold_order():
    with pytest.raises(ValueError):
        ZScoreMeanReversionStrategy(20, entry_z=0.5, exit_z=0.0)


def test_rsi_rising_market_is_high_and_falling_market_is_low():
    up = frame(np.arange(100, 130))
    down = frame(np.arange(130, 100, -1))
    strategy = RSIMeanReversionStrategy(lookback=5)
    assert strategy.rsi(up).dropna().iloc[-1] == pytest.approx(100.0)
    assert strategy.rsi(down).dropna().iloc[-1] == pytest.approx(0.0)


def test_rsi_strategy_enters_oversold():
    data = frame([110, 109, 108, 107, 106, 105, 104, 103, 102, 101, 100, 99])
    strategy = RSIMeanReversionStrategy(lookback=5, oversold=30, exit_level=50)
    signals = strategy.generate_signals(data)
    assert signals.iloc[-1] == 1


def test_cross_sectional_momentum_selects_strongest_asset():
    a = frame(np.linspace(100, 150, 20))
    b = frame(np.linspace(100, 110, 20))
    c = frame(np.linspace(100, 90, 20))
    strategy = CrossSectionalMomentumStrategy(lookback=5, top_n=1, rebalance_every=1)
    weights = strategy.generate_target_weights({"A": a, "B": b, "C": c})
    assert weights.iloc[-1]["A"] == pytest.approx(1.0)
    assert weights.iloc[-1][["B", "C"]].sum() == pytest.approx(0.0)
    assert (weights.sum(axis=1) <= 1.0 + 1e-12).all()


def test_cross_sectional_momentum_requires_multiple_assets():
    with pytest.raises(ValueError, match="at least two"):
        CrossSectionalMomentumStrategy(5, 1, 1).generate_target_weights({"A": frame(range(100, 120))})


def test_volatility_targeting_reduces_exposure_when_volatility_rises():
    quiet = pd.Series(100 + np.sin(np.arange(80) / 20) * 0.5)
    volatile = pd.Series(100 + np.sin(np.arange(80)) * 10)
    sizer = VolatilityTargetSizer(target_volatility=0.10, lookback=20, max_fraction=1.0)
    quiet_fraction = sizer.target_fractions(quiet).iloc[-1]
    volatile_fraction = sizer.target_fractions(volatile).iloc[-1]
    assert quiet_fraction > volatile_fraction
    assert 0 <= volatile_fraction <= 1
    assert 0 <= quiet_fraction <= 1


def test_volatility_target_applies_long_flat_signal():
    prices = pd.Series(np.linspace(100, 120, 60))
    signal = pd.Series([0] * 30 + [1] * 30)
    sizer = VolatilityTargetSizer(target_volatility=0.10, lookback=10)
    weights = sizer.apply_to_signal(prices, signal)
    assert weights.iloc[:30].eq(0).all()
    assert weights.iloc[-1] >= 0
    assert weights.max() <= 1


def test_multi_asset_engine_executes_weights_with_one_day_delay():
    a = frame(np.linspace(100, 110, 8))
    b = frame(np.linspace(100, 100, 8))
    weights = pd.DataFrame(0.0, index=a.index, columns=["A", "B"])
    weights.loc[:, "A"] = 1.0
    engine = MultiAssetTargetWeightEngine(initial_capital=10_000, execution_delay=1)
    result = engine.run({"A": a, "B": b}, weights)
    assert result.equity_curve.iloc[0]["cash"] == pytest.approx(10_000)
    assert result.holdings.iloc[0]["A"] == pytest.approx(0.0)
    assert result.holdings.iloc[1]["A"] > 0
    assert result.ending_equity > 10_000


def test_multi_asset_engine_costs_reduce_equity():
    a = frame(np.linspace(100, 120, 20))
    b = frame(np.linspace(100, 95, 20))
    strategy = CrossSectionalMomentumStrategy(lookback=3, top_n=1, rebalance_every=2)
    weights = strategy.generate_target_weights({"A": a, "B": b})
    zero = MultiAssetTargetWeightEngine(initial_capital=10_000).run({"A": a, "B": b}, weights)
    costly = MultiAssetTargetWeightEngine(initial_capital=10_000, transaction_cost_bps=10, slippage_bps=10).run({"A": a, "B": b}, weights)
    assert costly.ending_equity < zero.ending_equity
    assert costly.equity_curve["cumulative_transaction_costs"].iloc[-1] > 0
