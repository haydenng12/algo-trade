from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from features.returns import select_return_price


def plot_price_history(
    data: pd.DataFrame,
    *,
    symbol: str | None = None,
    prefer_adjusted: bool = True,
):
    """Create a simple Stage 1 inspection chart and return (figure, axes)."""
    price = select_return_price(data, prefer_adjusted=prefer_adjusted)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(price.index, price.values)
    label = "Adjusted Close" if prefer_adjusted and "Adj Close" in data.columns else "Close"
    ax.set_title(f"{symbol + ' - ' if symbol else ''}{label} Price History")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig, ax
