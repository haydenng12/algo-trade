from __future__ import annotations

import math
from typing import Any

import pandas as pd


PERCENT_METRICS = {"Total Return", "CAGR", "Volatility", "Max Drawdown"}
RATIO_METRICS = {"Sharpe", "Sortino"}


def _finite(value: float) -> bool:
    return not pd.isna(value) and math.isfinite(float(value))


def format_percent(value: float) -> str:
    return "—" if not _finite(value) else f"{float(value):.2%}"


def format_ratio(value: float) -> str:
    return "—" if not _finite(value) else f"{float(value):.2f}"


def benchmark_comparison_table(strategy_summary: dict[str, Any], benchmark: dict[str, float]) -> pd.DataFrame:
    rows = [
        ("Total Return", float(strategy_summary["total_return"]), benchmark["total_return"]),
        ("CAGR", float(strategy_summary["cagr"]), benchmark["cagr"]),
        ("Volatility", float(strategy_summary["annualized_volatility"]), benchmark["annualized_volatility"]),
        ("Sharpe", float(strategy_summary["sharpe_ratio"]), benchmark["sharpe_ratio"]),
        ("Max Drawdown", float(strategy_summary["maximum_drawdown"]), benchmark["maximum_drawdown"]),
    ]
    formatted = []
    for metric, strategy, bench in rows:
        formatter = format_percent if metric in PERCENT_METRICS else format_ratio
        formatted.append({"Metric": metric, "Strategy": formatter(strategy), "Benchmark": formatter(bench)})
    return pd.DataFrame(formatted)


def configuration_rows(strategy_name: str, params: dict[str, Any]) -> list[tuple[str, str]]:
    if strategy_name == "Moving Average Trend":
        return [("Fast SMA", f"{int(params['fast_window'])} days"), ("Slow SMA", f"{int(params['slow_window'])} days")]
    if strategy_name == "Z-Score Mean Reversion":
        return [
            ("Lookback", f"{int(params['lookback'])} days"),
            ("Entry Z", f"{float(params['entry_z']):.2f}"),
            ("Exit Z", f"{float(params['exit_z']):.2f}"),
        ]
    if strategy_name == "RSI Mean Reversion":
        return [
            ("RSI Lookback", f"{int(params['lookback'])} days"),
            ("Oversold", f"{float(params['oversold']):.0f}"),
            ("Exit RSI", f"{float(params['exit_level']):.0f}"),
        ]
    if strategy_name == "Cross-Sectional Momentum":
        return [
            ("Lookback", f"{int(params['lookback'])} days"),
            ("Rebalance", f"Every {int(params['rebalance_every'])} days"),
            ("Assets Held", str(int(params['top_n']))),
        ]
    if strategy_name == "Volatility-Targeted Trend":
        return [
            ("Fast SMA", f"{int(params['fast_window'])} days"),
            ("Slow SMA", f"{int(params['slow_window'])} days"),
            ("Vol Target", f"{float(params['target_volatility']):.0%}"),
            ("Vol Lookback", f"{int(params['vol_lookback'])} days"),
        ]
    return [(str(k).replace("_", " ").title(), str(v)) for k, v in params.items()]


def interpretation_text(strategy_summary: dict[str, Any], benchmark: dict[str, float]) -> str:
    strategy_return = float(strategy_summary["total_return"])
    benchmark_return = float(benchmark["total_return"])
    strategy_vol = float(strategy_summary["annualized_volatility"])
    benchmark_vol = float(benchmark["annualized_volatility"])
    strategy_dd = float(strategy_summary["maximum_drawdown"])
    benchmark_dd = float(benchmark["maximum_drawdown"])

    return_gap_pp = (strategy_return - benchmark_return) * 100.0
    return_phrase = (
        f"outperformed the benchmark by {abs(return_gap_pp):.1f} percentage points"
        if return_gap_pp >= 0
        else f"trailed the benchmark by {abs(return_gap_pp):.1f} percentage points"
    )

    if benchmark_vol > 0:
        vol_change = (strategy_vol / benchmark_vol - 1.0) * 100.0
        vol_phrase = (
            f"{abs(vol_change):.1f}% lower annualized volatility"
            if vol_change <= 0
            else f"{abs(vol_change):.1f}% higher annualized volatility"
        )
    else:
        vol_phrase = "an unavailable volatility comparison"

    dd_improvement_pp = (abs(benchmark_dd) - abs(strategy_dd)) * 100.0
    dd_phrase = (
        f"a {abs(dd_improvement_pp):.1f}-point shallower maximum drawdown"
        if dd_improvement_pp >= 0
        else f"a {abs(dd_improvement_pp):.1f}-point deeper maximum drawdown"
    )
    return f"The strategy {return_phrase}, with {vol_phrase} and {dd_phrase}."
