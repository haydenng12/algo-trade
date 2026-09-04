from __future__ import annotations

import pandas as pd


def chronological_split(data: pd.DataFrame, train_fraction: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological train/test split; never shuffles time-series observations."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")
    if len(data) < 2:
        raise ValueError("Need at least two observations to split data.")
    cut = int(len(data) * train_fraction)
    cut = min(max(cut, 1), len(data) - 1)
    return data.iloc[:cut].copy(), data.iloc[cut:].copy()


def date_split(data: pd.DataFrame, test_start: str | pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    ts = pd.Timestamp(test_start)
    train = data.loc[data.index < ts].copy()
    test = data.loc[data.index >= ts].copy()
    if train.empty or test.empty:
        raise ValueError("Date split must leave non-empty train and test periods.")
    return train, test
