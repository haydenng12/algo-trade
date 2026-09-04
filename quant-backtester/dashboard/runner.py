from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from analytics.drawdown import drawdown_series
from analytics.performance import performance_summary
from backtest.engine import BacktestEngine
from backtest.multi_asset import MultiAssetBacktestResult, MultiAssetTargetWeightEngine
from portfolio.volatility_targeting import VolatilityTargetSizer
from research.benchmark import buy_and_hold_equity
from research.strategy_comparison import multi_asset_performance_summary
from strategies.cross_sectional_momentum import CrossSectionalMomentumStrategy
from strategies.moving_average import MovingAverageStrategy
from strategies.rsi import RSIMeanReversionStrategy
from strategies.zscore_mean_reversion import ZScoreMeanReversionStrategy


SINGLE_ASSET_STRATEGIES = {
    "Moving Average Trend",
    "Z-Score Mean Reversion",
    "RSI Mean Reversion",
}
MULTI_ASSET_STRATEGIES = {
    "Cross-Sectional Momentum",
    "Volatility-Targeted Trend",
}


@dataclass(frozen=True)
class StrategyConfig:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class DashboardRun:
    name: str
    summary: dict[str, float | int]
    equity: pd.Series
    benchmark_equity: pd.Series
    benchmark_name: str
    drawdown: pd.Series
    costs: pd.DataFrame
    trades: pd.DataFrame | None = None
    target_weights: pd.DataFrame | None = None
    holdings: pd.DataFrame | None = None
    signals: pd.Series | None = None


def _engine_kwargs(initial_capital: float, fees_bps: float, slippage_bps: float) -> dict[str, float | int]:
    return {
        "initial_capital": float(initial_capital),
        "transaction_cost_bps": float(fees_bps),
        "slippage_bps": float(slippage_bps),
        "execution_delay": 1,
    }


def equal_weight_buy_and_hold_equity(
    frames: dict[str, pd.DataFrame],
    initial_capital: float = 100_000.0,
) -> pd.Series:
    """Frictionless equal-weight buy-and-hold benchmark for a multi-asset universe.

    Each asset receives the same initial dollar allocation on the first common
    date and is then held without rebalancing. This is intentionally simple and
    transparent: cross-sectional momentum should be judged against access to
    the same investment universe, not only against the first ticker.
    """
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive.")
    if not frames:
        raise ValueError("frames cannot be empty.")

    common: pd.DatetimeIndex | None = None
    prices: dict[str, pd.Series] = {}
    for raw_symbol, frame in frames.items():
        symbol = raw_symbol.upper().strip()
        if not symbol:
            raise ValueError("symbols cannot be blank.")
        col = "Adj Close" if "Adj Close" in frame.columns else "Close"
        if col not in frame.columns:
            raise KeyError(f"{symbol} requires Adj Close or Close.")
        series = frame[col].astype(float)
        common = series.index if common is None else common.intersection(series.index)
        prices[symbol] = series

    if common is None or len(common) < 2:
        raise ValueError("Assets need at least two common dates for a benchmark.")
    common = pd.DatetimeIndex(common).sort_values()
    price_frame = pd.DataFrame({s: p.reindex(common) for s, p in prices.items()})
    if price_frame.isna().any().any() or (price_frame <= 0).any().any():
        raise ValueError("Benchmark prices must be positive and non-missing.")

    normalized = price_frame / price_frame.iloc[0]
    equity = float(initial_capital) * normalized.mean(axis=1)
    equity.name = "equal_weight_benchmark_equity"
    return equity


def _cost_frame(curve: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "transaction_costs",
        "slippage_costs",
        "cumulative_transaction_costs",
        "cumulative_slippage_costs",
        "traded_notional",
    ]
    return curve[[c for c in columns if c in curve.columns]].copy()


def _single_asset_run(
    symbol: str,
    frame: pd.DataFrame,
    config: StrategyConfig,
    initial_capital: float,
    fees_bps: float,
    slippage_bps: float,
) -> DashboardRun:
    p = config.params
    if config.name == "Moving Average Trend":
        strategy = MovingAverageStrategy(int(p.get("fast_window", 20)), int(p.get("slow_window", 50)))
    elif config.name == "Z-Score Mean Reversion":
        strategy = ZScoreMeanReversionStrategy(
            lookback=int(p.get("lookback", 20)),
            entry_z=float(p.get("entry_z", -1.5)),
            exit_z=float(p.get("exit_z", 0.0)),
        )
    elif config.name == "RSI Mean Reversion":
        strategy = RSIMeanReversionStrategy(
            lookback=int(p.get("lookback", 14)),
            oversold=float(p.get("oversold", 30.0)),
            exit_level=float(p.get("exit_level", 50.0)),
        )
    else:
        raise ValueError(f"Unsupported single-asset strategy: {config.name}")

    result = BacktestEngine(**_engine_kwargs(initial_capital, fees_bps, slippage_bps)).run(symbol, frame, strategy)
    equity = result.equity_curve["equity"].astype(float)
    return DashboardRun(
        name=config.name,
        summary=performance_summary(result),
        equity=equity,
        benchmark_equity=buy_and_hold_equity(frame, initial_capital),
        benchmark_name=f"{symbol} Buy & Hold",
        drawdown=drawdown_series(equity),
        costs=_cost_frame(result.equity_curve),
        trades=result.trades.copy(),
        signals=result.signals.copy(),
    )


def _multi_result_run(
    name: str,
    result: MultiAssetBacktestResult,
    benchmark: pd.Series,
    benchmark_name: str,
) -> DashboardRun:
    equity = result.equity_curve["equity"].astype(float)
    return DashboardRun(
        name=name,
        summary=multi_asset_performance_summary(result),
        equity=equity,
        benchmark_equity=benchmark.reindex(equity.index),
        benchmark_name=benchmark_name,
        drawdown=drawdown_series(equity),
        costs=_cost_frame(result.equity_curve),
        target_weights=result.target_weights.copy(),
        holdings=result.holdings.copy(),
    )


def run_strategy(
    frames: dict[str, pd.DataFrame],
    primary_symbol: str,
    config: StrategyConfig,
    *,
    initial_capital: float = 100_000.0,
    fees_bps: float = 5.0,
    slippage_bps: float = 5.0,
) -> DashboardRun:
    if not frames:
        raise ValueError("frames cannot be empty.")
    normalized = {s.upper().strip(): f for s, f in frames.items()}
    primary = primary_symbol.upper().strip()
    if primary not in normalized:
        raise KeyError(f"Primary symbol {primary!r} is not loaded.")

    if config.name in SINGLE_ASSET_STRATEGIES:
        return _single_asset_run(
            primary,
            normalized[primary],
            config,
            initial_capital,
            fees_bps,
            slippage_bps,
        )

    engine = MultiAssetTargetWeightEngine(**_engine_kwargs(initial_capital, fees_bps, slippage_bps))
    p = config.params

    if config.name == "Cross-Sectional Momentum":
        strategy = CrossSectionalMomentumStrategy(
            lookback=int(p.get("lookback", 126)),
            top_n=int(p.get("top_n", min(2, len(normalized)))),
            rebalance_every=int(p.get("rebalance_every", 21)),
        )
        weights = strategy.generate_target_weights(normalized)
        result = engine.run(normalized, weights)
        benchmark = equal_weight_buy_and_hold_equity(normalized, initial_capital)
        return _multi_result_run(
            config.name,
            result,
            benchmark,
            "Equal-Weight Universe Buy & Hold",
        )

    if config.name == "Volatility-Targeted Trend":
        frame = normalized[primary]
        fast = int(p.get("fast_window", 20))
        slow = int(p.get("slow_window", 50))
        target_vol = float(p.get("target_volatility", 0.10))
        vol_lookback = int(p.get("vol_lookback", 20))
        signal = MovingAverageStrategy(fast, slow).generate_signals(frame)
        price_col = "Adj Close" if "Adj Close" in frame.columns else "Close"
        weight = VolatilityTargetSizer(
            target_volatility=target_vol,
            lookback=vol_lookback,
            max_fraction=1.0,
        ).apply_to_signal(frame[price_col], signal)
        weights = pd.DataFrame({primary: weight})
        result = engine.run({primary: frame}, weights)
        benchmark = buy_and_hold_equity(frame, initial_capital)
        run = _multi_result_run(config.name, result, benchmark, f"{primary} Buy & Hold")
        run.signals = signal
        return run

    raise ValueError(f"Unsupported strategy: {config.name}")
