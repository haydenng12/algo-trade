"""Stage 5 research framework: benchmarking, splits, grids, walk-forward, costs."""

from research.benchmark import buy_and_hold_equity, benchmark_summary
from research.split import chronological_split
from research.grid import grid_search_moving_average, evaluate_oos
from research.oos import moving_average_oos_signals, run_oos_moving_average
from research.walk_forward import walk_forward_moving_average
from research.cost_sensitivity import cost_sensitivity

__all__ = [
    "buy_and_hold_equity", "benchmark_summary", "chronological_split",
    "grid_search_moving_average", "evaluate_oos", "moving_average_oos_signals",
    "run_oos_moving_average", "walk_forward_moving_average", "cost_sensitivity",
]
