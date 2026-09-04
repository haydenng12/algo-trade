import pytest

from portfolio.sizing import EqualWeightSizer, FixedFractionSizer


def test_fixed_fraction_target_notional():
    assert FixedFractionSizer(0.25).target_notional(100_000) == pytest.approx(25_000)


def test_fixed_fraction_rejects_invalid_fraction():
    with pytest.raises(ValueError):
        FixedFractionSizer(0)
    with pytest.raises(ValueError):
        FixedFractionSizer(1.1)


def test_equal_weight_sizer():
    weights = EqualWeightSizer().weights(["SPY", "QQQ", "IWM", "DIA"])
    assert weights == {"SPY": 0.25, "QQQ": 0.25, "IWM": 0.25, "DIA": 0.25}
    assert sum(weights.values()) == pytest.approx(1.0)
