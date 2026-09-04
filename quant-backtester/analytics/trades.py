from __future__ import annotations

import math

import numpy as np
import pandas as pd


def trade_summary(trades: pd.DataFrame) -> dict[str, float | int]:
    """Summarize completed trades produced by the backtest engine."""
    required = {
        "net_pnl",
        "return_pct",
        "holding_period_days",
    }
    missing = required.difference(trades.columns)
    if missing:
        raise ValueError(f"Trade table missing required columns: {sorted(missing)}")

    count = int(len(trades))
    if count == 0:
        return {
            "number_of_trades": 0,
            "win_rate": float("nan"),
            "profit_factor": float("nan"),
            "average_trade_return": float("nan"),
            "average_winner": float("nan"),
            "average_loser": float("nan"),
            "best_trade": float("nan"),
            "worst_trade": float("nan"),
            "average_holding_period_days": float("nan"),
        }

    pnl = trades["net_pnl"].astype(float)
    returns = trades["return_pct"].astype(float)
    holding = trades["holding_period_days"].astype(float)

    winners = pnl[pnl > 0]
    losers = pnl[pnl < 0]
    gross_profit = float(winners.sum())
    gross_loss = float(-losers.sum())

    if gross_loss == 0.0:
        profit_factor = math.inf if gross_profit > 0 else float("nan")
    else:
        profit_factor = gross_profit / gross_loss

    winner_returns = returns[pnl > 0]
    loser_returns = returns[pnl < 0]

    return {
        "number_of_trades": count,
        "win_rate": float((pnl > 0).mean()),
        "profit_factor": float(profit_factor),
        "average_trade_return": float(returns.mean()),
        "average_winner": float(winner_returns.mean()) if not winner_returns.empty else float("nan"),
        "average_loser": float(loser_returns.mean()) if not loser_returns.empty else float("nan"),
        "best_trade": float(returns.max()),
        "worst_trade": float(returns.min()),
        "average_holding_period_days": float(holding.mean()),
    }
