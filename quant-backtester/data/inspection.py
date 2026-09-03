from __future__ import annotations

import pandas as pd

from features.returns import simple_returns


def market_data_summary(data: pd.DataFrame) -> dict[str, object]:
    """Return compact diagnostics useful before downstream research begins."""
    returns = simple_returns(data).dropna()
    return {
        "rows": len(data),
        "start": data.index.min(),
        "end": data.index.max(),
        "columns": list(data.columns),
        "missing_values": int(data.isna().sum().sum()),
        "duplicate_dates": int(data.index.duplicated().sum()),
        "mean_daily_return": float(returns.mean()) if not returns.empty else None,
        "daily_return_std": float(returns.std(ddof=1)) if len(returns) > 1 else None,
    }
