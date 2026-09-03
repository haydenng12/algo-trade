from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Position:
    symbol: str
    quantity: float = 0.0
    entry_price: float | None = None
    entry_date: object | None = None

    @property
    def is_open(self) -> bool:
        return self.quantity > 0


@dataclass
class Portfolio:
    initial_capital: float
    cash: float | None = None
    position: Position | None = None

    def __post_init__(self) -> None:
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive.")
        if self.cash is None:
            self.cash = float(self.initial_capital)

    def equity(self, mark_price: float) -> float:
        quantity = 0.0 if self.position is None else self.position.quantity
        return float(self.cash + quantity * mark_price)
