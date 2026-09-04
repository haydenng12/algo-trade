from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence


@dataclass(frozen=True)
class FixedFractionSizer:
    """Allocate a fixed fraction of available portfolio capital to a position."""

    fraction: float = 1.0

    def __post_init__(self) -> None:
        if not 0 < self.fraction <= 1:
            raise ValueError("fraction must be in (0, 1].")

    def target_notional(self, portfolio_value: float) -> float:
        if portfolio_value < 0:
            raise ValueError("portfolio_value cannot be negative.")
        return float(portfolio_value * self.fraction)


class EqualWeightSizer:
    """Return equal target weights for an active multi-symbol universe.

    Stage 4's single-symbol engine does not consume this class yet, but the
    interface is ready for the later multi-asset research stages.
    """

    def weights(self, symbols: Sequence[str]) -> dict[str, float]:
        unique = [s.upper().strip() for s in symbols if s and s.strip()]
        if not unique:
            return {}
        if len(set(unique)) != len(unique):
            raise ValueError("symbols must be unique.")
        weight = 1.0 / len(unique)
        return {symbol: weight for symbol in unique}
