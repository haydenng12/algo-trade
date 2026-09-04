from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def equity_figure(strategy_equity: pd.Series, benchmark_equity: pd.Series, strategy_name: str, benchmark_name: str):
    combined = pd.concat(
        [strategy_equity.rename(strategy_name), benchmark_equity.rename(benchmark_name)],
        axis=1,
    ).dropna()
    fig, ax = plt.subplots(figsize=(10, 4.8))
    combined.plot(ax=ax, linewidth=1.8)
    ax.set_title("Growth of $1")
    ax.set_ylabel("Growth multiple")
    ax.set_xlabel("")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def normalized_equity_figure(strategy_equity: pd.Series, benchmark_equity: pd.Series, strategy_name: str, benchmark_name: str):
    combined = pd.concat(
        [strategy_equity.rename(strategy_name), benchmark_equity.rename(benchmark_name)],
        axis=1,
    ).dropna()
    normalized = combined / combined.iloc[0]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    normalized.plot(ax=ax, linewidth=1.8)
    ax.set_title("Strategy vs Benchmark — Growth of $1")
    ax.set_ylabel("Growth multiple")
    ax.set_xlabel("")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def drawdown_figure(drawdown: pd.Series):
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.fill_between(drawdown.index, drawdown.values * 100.0, 0, alpha=0.35)
    ax.plot(drawdown.index, drawdown.values * 100.0, linewidth=1.0)
    ax.set_title("Portfolio Drawdown")
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def weights_figure(weights: pd.DataFrame):
    weights = weights.loc[:, (weights.abs().sum(axis=0) > 0)]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    if weights.empty:
        ax.text(0.5, 0.5, "No invested weights during this period", ha="center", va="center")
        ax.axis("off")
    else:
        weights.plot.area(ax=ax, stacked=True, linewidth=0.2)
        ax.set_ylim(0, 1.05)
        ax.set_title("Target Portfolio Weights")
        ax.set_ylabel("Portfolio weight")
        ax.set_xlabel("")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def cost_sensitivity_figure(table: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4.0))
    ax.plot(table["cost_bps_each"], table["ending_equity"], marker="o")
    ax.set_title("Cost Sensitivity")
    ax.set_xlabel("Fees + slippage assumption (bps each)")
    ax.set_ylabel("Ending equity ($)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig
