import pytest

from execution.costs import TransactionCostModel
from execution.slippage import SlippageModel


def test_transaction_cost_exact_basis_point_calculation():
    model = TransactionCostModel(bps=10)
    assert model.calculate(10_000) == pytest.approx(10.0)


def test_slippage_worsens_buys_and_sells():
    model = SlippageModel(bps=10)
    assert model.execution_price(100, "BUY") == pytest.approx(100.10)
    assert model.execution_price(100, "SELL") == pytest.approx(99.90)


def test_execution_models_reject_invalid_inputs():
    with pytest.raises(ValueError):
        TransactionCostModel(-1)
    with pytest.raises(ValueError):
        SlippageModel(-1)
    with pytest.raises(ValueError):
        SlippageModel(5).execution_price(100, "HOLD")
