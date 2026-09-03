from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd

from data.cleaner import CleaningReport, clean_ohlcv


class MarketDataLoader:
    """Stage 1 market-data entry point.

    The core loader is deliberately provider-agnostic. CSV loading is always
    available and deterministic; Yahoo Finance support is optional via
    ``yfinance`` for convenient historical-data retrieval.
    """

    @staticmethod
    def from_csv(path: str | Path) -> tuple[pd.DataFrame, CleaningReport]:
        raw = pd.read_csv(Path(path))
        return clean_ohlcv(raw)

    @staticmethod
    def from_frames(
        frames: Mapping[str, pd.DataFrame],
    ) -> tuple[dict[str, pd.DataFrame], dict[str, CleaningReport]]:
        cleaned: dict[str, pd.DataFrame] = {}
        reports: dict[str, CleaningReport] = {}
        for symbol, frame in frames.items():
            symbol_key = symbol.upper().strip()
            if not symbol_key:
                raise ValueError("Symbol names cannot be blank.")
            cleaned[symbol_key], reports[symbol_key] = clean_ohlcv(frame)
        return cleaned, reports

    @staticmethod
    def from_yahoo(
        symbol: str,
        start: str,
        end: str,
        *,
        auto_adjust: bool = False,
    ) -> tuple[pd.DataFrame, CleaningReport]:
        """Download daily historical data using optional ``yfinance``.

        ``auto_adjust=False`` preserves both raw OHLC prices and an adjusted
        close when Yahoo provides it. Stage 1 return code can then explicitly
        prefer adjusted close for corporate-action-safe return calculations.
        """
        try:
            import yfinance as yf
        except ImportError as exc:
            raise ImportError(
                "Yahoo loading requires the optional 'yfinance' package. "
                "Install project requirements first."
            ) from exc

        raw = yf.download(
            symbol,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=auto_adjust,
            actions=False,
            progress=False,
            group_by="column",
        )
        if raw.empty:
            raise ValueError(f"No market data returned for {symbol!r}.")

        # yfinance may return a one-symbol MultiIndex depending on version.
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.reset_index()
        return clean_ohlcv(raw)
