from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SlippageModel:
    """Symmetric adverse slippage around the observed market price."""

    bps: float = 0.0

    def __post_init__(self) -> None:
        if self.bps < 0:
            raise ValueError("Slippage bps cannot be negative.")

    @property
    def rate(self) -> float:
        return self.bps / 10_000.0

    def execution_price(self, market_price: float, side: str) -> float:
        if market_price <= 0:
            raise ValueError("market_price must be positive.")
        side = side.upper()
        if side == "BUY":
            return float(market_price * (1.0 + self.rate))
        if side == "SELL":
            return float(market_price * (1.0 - self.rate))
        raise ValueError("side must be 'BUY' or 'SELL'.")
