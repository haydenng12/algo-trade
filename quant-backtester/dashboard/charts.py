from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


BG = "#0e1117"
PANEL = "#161b22"
TEXT = "#e6edf3"
MUTED = "#9aa4b2"
GRID = "#30363d"
STRATEGY = "#58a6ff"
BENCHMARK = "#f0883e"
DRAWDOWN = "#f85149"
ACCENT = "#3fb950"
PALETTE = ["#58a6ff", "#f0883e", "#3fb950", "#d2a8ff", "#f85149", "#79c0ff"]


def _style(fig, ax) -> None:
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(color=GRID, alpha=0.55, linewidth=0.7)


def equity_figure(strategy_equity: pd.Series, benchmark_equity: pd.Series, strategy_name: str, benchmark_name: str):
    combined = pd.concat(
        [strategy_equity.rename(strategy_name), benchmark_equity.rename(benchmark_name)], axis=1
    ).dropna()
    normalized = combined / combined.iloc[0]
    fig, ax = plt.subplots(figsize=(10, 4.0))
    ax.plot(normalized.index, normalized.iloc[:, 0], label=strategy_name, linewidth=2.0, color=STRATEGY)
    ax.plot(normalized.index, normalized.iloc[:, 1], label=benchmark_name, linewidth=2.0, color=BENCHMARK)
    _style(fig, ax)
    ax.set_title("Growth of $1", loc="left", fontsize=13, fontweight="bold")
    ax.set_ylabel("Growth multiple")
    ax.set_xlabel("")
    legend = ax.legend(frameon=False, loc="upper left")
    for text in legend.get_texts():
        text.set_color(TEXT)
    fig.tight_layout()
    return fig


def normalized_equity_figure(strategy_equity: pd.Series, benchmark_equity: pd.Series, strategy_name: str, benchmark_name: str):
    return equity_figure(strategy_equity, benchmark_equity, strategy_name, benchmark_name)


def drawdown_figure(drawdown: pd.Series):
    fig, ax = plt.subplots(figsize=(10, 3.2))
    values = drawdown.values * 100.0
    ax.fill_between(drawdown.index, values, 0, alpha=0.25, color=DRAWDOWN)
    ax.plot(drawdown.index, values, linewidth=1.2, color=DRAWDOWN)
    _style(fig, ax)
    ax.set_title("Portfolio Drawdown", loc="left", fontsize=13, fontweight="bold")
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("")
    fig.tight_layout()
    return fig


def weights_figure(weights: pd.DataFrame):
    weights = weights.loc[:, (weights.abs().sum(axis=0) > 0)]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    _style(fig, ax)
    if weights.empty:
        ax.text(0.5, 0.5, "No invested weights during this period", ha="center", va="center", color=TEXT)
        ax.axis("off")
    else:
        weights.plot.area(ax=ax, stacked=True, linewidth=0.2, color=PALETTE[: len(weights.columns)])
        ax.set_ylim(0, 1.05)
        ax.set_title("Target Portfolio Weights", loc="left", fontsize=13, fontweight="bold")
        ax.set_ylabel("Portfolio weight")
        ax.set_xlabel("")
        legend = ax.legend(frameon=False, loc="upper left", ncol=min(4, len(weights.columns)))
        for text in legend.get_texts():
            text.set_color(TEXT)
    fig.tight_layout()
    return fig


def cost_sensitivity_figure(table: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(table["cost_bps_each"], table["ending_equity"], marker="o", linewidth=2.0, color=ACCENT)
    _style(fig, ax)
    ax.set_title("Cost Sensitivity", loc="left", fontsize=13, fontweight="bold")
    ax.set_xlabel("Fees + slippage assumption (bps each)")
    ax.set_ylabel("Ending equity ($)")
    fig.tight_layout()
    return fig
