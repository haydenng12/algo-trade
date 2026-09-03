from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


REQUIRED_OHLCV_COLUMNS = ("Open", "High", "Low", "Close", "Volume")
OPTIONAL_PRICE_COLUMNS = ("Adj Close",)


class DataValidationError(ValueError):
    """Raised when market data violates the Stage 1 schema or sanity rules."""


@dataclass(frozen=True)
class CleaningReport:
    rows_before: int
    rows_after: int
    duplicate_dates_removed: int
    rows_dropped_for_missing_values: int


def _normalize_column_name(name: str) -> str:
    key = str(name).strip().lower().replace("_", " ")
    mapping = {
        "date": "Date",
        "datetime": "Date",
        "timestamp": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "adj close": "Adj Close",
        "adjusted close": "Adj Close",
        "volume": "Volume",
    }
    return mapping.get(key, str(name).strip())


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with common market-data column names normalized."""
    out = df.copy()
    out.columns = [_normalize_column_name(c) for c in out.columns]
    return out


def validate_ohlcv(
    df: pd.DataFrame,
    required_columns: Iterable[str] = REQUIRED_OHLCV_COLUMNS,
) -> None:
    """Validate schema and basic OHLCV sanity constraints.

    Expected index: unique, increasing ``DatetimeIndex``.
    Required columns: Open, High, Low, Close, Volume.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise DataValidationError("Market data index must be a pandas DatetimeIndex.")
    if df.index.has_duplicates:
        raise DataValidationError("Market data index contains duplicate timestamps.")
    if not df.index.is_monotonic_increasing:
        raise DataValidationError("Market data index must be sorted ascending.")

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")

    if df.empty:
        raise DataValidationError("Market data is empty.")

    required = list(required_columns)
    if df[required].isna().any().any():
        raise DataValidationError("Required OHLCV columns contain missing values.")

    price_cols = [c for c in ("Open", "High", "Low", "Close", "Adj Close") if c in df.columns]
    if price_cols and (df[price_cols] <= 0).any().any():
        raise DataValidationError("Prices must be strictly positive.")
    if (df["Volume"] < 0).any():
        raise DataValidationError("Volume cannot be negative.")

    if (df["High"] < df[["Open", "Close", "Low"]].max(axis=1)).any():
        raise DataValidationError("High must be at least as large as Open, Close, and Low.")
    if (df["Low"] > df[["Open", "Close", "High"]].min(axis=1)).any():
        raise DataValidationError("Low must be at most as small as Open, Close, and High.")


def clean_ohlcv(
    df: pd.DataFrame,
    *,
    drop_missing: bool = True,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Normalize, clean, sort, and validate one-symbol OHLCV data.

    Duplicate dates keep the last occurrence. Required-row missing values are
    dropped by default because Stage 1 favors explicit, deterministic behavior
    over silent interpolation of market prices.
    """
    out = normalize_columns(df)
    rows_before = len(out)

    if "Date" in out.columns:
        parsed_dates = pd.to_datetime(out["Date"], errors="coerce")
        out = out.drop(columns=["Date"])
        out.index = parsed_dates
    elif not isinstance(out.index, pd.DatetimeIndex):
        parsed_dates = pd.to_datetime(out.index, errors="coerce")
        out.index = parsed_dates

    invalid_dates = int(out.index.isna().sum())
    if invalid_dates:
        out = out.loc[~out.index.isna()].copy()

    out.index.name = "Date"
    out = out.sort_index()

    duplicate_dates_removed = int(out.index.duplicated(keep="last").sum())
    out = out.loc[~out.index.duplicated(keep="last")].copy()

    numeric_cols = [c for c in (*REQUIRED_OHLCV_COLUMNS, *OPTIONAL_PRICE_COLUMNS) if c in out.columns]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    missing_before = len(out)
    if drop_missing:
        present_required = [c for c in REQUIRED_OHLCV_COLUMNS if c in out.columns]
        if present_required:
            out = out.dropna(subset=present_required)
    rows_dropped_for_missing_values = missing_before - len(out)

    validate_ohlcv(out)

    report = CleaningReport(
        rows_before=rows_before,
        rows_after=len(out),
        duplicate_dates_removed=duplicate_dates_removed,
        rows_dropped_for_missing_values=rows_dropped_for_missing_values + invalid_dates,
    )
    return out, report
