from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TransactionCostModel:
    """Linear trading-fee model expressed in basis points of traded notional."""

    bps: float = 0.0

    def __post_init__(self) -> None:
        if self.bps < 0:
            raise ValueError("Transaction-cost bps cannot be negative.")

    @property
    def rate(self) -> float:
        return self.bps / 10_000.0

    def calculate(self, notional: float) -> float:
        if notional < 0:
            raise ValueError("Trade notional cannot be negative.")
        return float(notional * self.rate)
