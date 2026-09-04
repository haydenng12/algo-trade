from __future__ import annotations

import argparse

import pandas as pd

from analytics.performance import performance_summary
from backtest.engine import BacktestEngine
from backtest.multi_asset import MultiAssetTargetWeightEngine
from data.loader import MarketDataLoader
from portfolio.volatility_targeting import VolatilityTargetSizer
from research.strategy_comparison import multi_asset_performance_summary
from strategies.cross_sectional_momentum import CrossSectionalMomentumStrategy
from strategies.moving_average import MovingAverageStrategy
from strategies.rsi import RSIMeanReversionStrategy
from strategies.zscore_mean_reversion import ZScoreMeanReversionStrategy


def print_metrics(name: str, summary: dict):
    print(f"\n{name}")
    print(f"  Total return:   {summary['total_return']:.2%}")
    print(f"  CAGR:           {summary['cagr']:.2%}")
    print(f"  Volatility:     {summary['annualized_volatility']:.2%}")
    print(f"  Sharpe:         {summary['sharpe_ratio']:.2f}")
    print(f"  Sortino:        {summary['sortino_ratio']:.2f}")
    print(f"  Max drawdown:   {summary['maximum_drawdown']:.2%}")


def main():
    parser = argparse.ArgumentParser(description="Stage 6 multi-strategy research demo")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2026-01-01")
    parser.add_argument("--capital", type=float, default=100_000)
    parser.add_argument("--fees-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--universe", nargs="+", default=["SPY", "QQQ", "IWM", "EFA"])
    args = parser.parse_args()

    frames = {}
    for symbol in args.universe:
        frame, _ = MarketDataLoader.from_yahoo(symbol, args.start, args.end)
        frames[symbol.upper()] = frame

    primary = args.universe[0].upper()
    data = frames[primary]
    engine_kwargs = dict(
        initial_capital=args.capital,
        transaction_cost_bps=args.fees_bps,
        slippage_bps=args.slippage_bps,
        execution_delay=1,
    )

    print("STAGE 6 — MULTI-STRATEGY RESEARCH")
    print(f"Single-asset symbol: {primary}")
    print(f"Universe:            {', '.join(frames)}")
    print(f"Costs:               {args.fees_bps:.2f} bps fees + {args.slippage_bps:.2f} bps slippage")

    strategies = {
        "Moving Average 20/50": MovingAverageStrategy(20, 50),
        "Z-Score Mean Reversion": ZScoreMeanReversionStrategy(lookback=20, entry_z=-1.5, exit_z=0.0),
        "RSI Mean Reversion": RSIMeanReversionStrategy(lookback=14, oversold=30, exit_level=50),
    }
    for name, strategy in strategies.items():
        result = BacktestEngine(**engine_kwargs).run(primary, data, strategy)
        print_metrics(name, performance_summary(result))

    momentum = CrossSectionalMomentumStrategy(lookback=126, top_n=min(2, len(frames)), rebalance_every=21)
    momentum_weights = momentum.generate_target_weights(frames)
    momentum_result = MultiAssetTargetWeightEngine(**{k: v for k, v in engine_kwargs.items() if k != "position_fraction"}).run(frames, momentum_weights)
    print_metrics("Cross-Sectional Momentum (6-month lookback, monthly rebalance)", multi_asset_performance_summary(momentum_result))

    # Volatility-targeted exposure applied to a 20/50 trend signal. Indicator
    # information is observed at Close and the engine executes weights next Open.
    signal = MovingAverageStrategy(20, 50).generate_signals(data)
    price_col = "Adj Close" if "Adj Close" in data.columns else "Close"
    vol_sizer = VolatilityTargetSizer(target_volatility=0.10, lookback=20, max_fraction=1.0)
    spy_weight = vol_sizer.apply_to_signal(data[price_col], signal)
    vol_weights = pd.DataFrame({primary: spy_weight})
    vol_result = MultiAssetTargetWeightEngine(**{k: v for k, v in engine_kwargs.items() if k != "position_fraction"}).run({primary: data}, vol_weights)
    print_metrics("Volatility-Targeted 20/50 Trend (10% target)", multi_asset_performance_summary(vol_result))


if __name__ == "__main__":
    main()
