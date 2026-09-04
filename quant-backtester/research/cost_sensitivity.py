from __future__ import annotations

import pandas as pd

from analytics.performance import performance_summary
from research.oos import run_oos_moving_average


def cost_sensitivity(
    symbol: str,
    context_data: pd.DataFrame,
    evaluation_data: pd.DataFrame,
    fast_window: int,
    slow_window: int,
    cost_levels_bps: list[float] | tuple[float, ...] = (0, 5, 10, 25),
    *,
    initial_capital: float = 100_000.0,
    position_fraction: float = 1.0,
    execution_delay: int = 1,
) -> pd.DataFrame:
    """Re-run one OOS strategy under multiple equal fee/slippage assumptions.

    The indicator is warmed up with ``context_data`` for every run, while
    portfolio accounting starts fresh on ``evaluation_data``. This makes each
    row directly comparable with ``evaluate_oos`` when all engine assumptions
    are identical.
    """
    rows = []
    for bps in cost_levels_bps:
        result = run_oos_moving_average(
            symbol,
            context_data,
            evaluation_data,
            fast_window,
            slow_window,
            engine_kwargs={
                "initial_capital": initial_capital,
                "transaction_cost_bps": bps,
                "slippage_bps": bps,
                "position_fraction": position_fraction,
                "execution_delay": execution_delay,
            },
        )
        metrics = performance_summary(result)
        rows.append({
            "cost_bps_each": float(bps),
            "ending_equity": result.ending_equity,
            "total_fees": float(result.equity_curve["transaction_costs"].sum()),
            "total_slippage": float(result.equity_curve["slippage_costs"].sum()),
            **metrics,
        })
    return pd.DataFrame(rows)
