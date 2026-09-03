import pandas as pd
import pytest

from data.cleaner import DataValidationError, clean_ohlcv, validate_ohlcv
from data.loader import MarketDataLoader


def test_cleaner_normalizes_sorts_and_deduplicates():
    raw = pd.DataFrame(
        {
            "date": ["2026-01-05", "2026-01-02", "2026-01-05"],
            "open": [102, 100, 103],
            "high": [104, 103, 105],
            "low": [100, 99, 101],
            "close": [101, 102, 104],
            "volume": [1100, 1000, 1200],
        }
    )
    cleaned, report = clean_ohlcv(raw)

    assert list(cleaned.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert isinstance(cleaned.index, pd.DatetimeIndex)
    assert cleaned.index.is_monotonic_increasing
    assert cleaned.loc[pd.Timestamp("2026-01-05"), "Close"] == 104
    assert report.duplicate_dates_removed == 1


def test_invalid_ohlc_relationship_is_rejected(clean_frame):
    cleaned, _ = clean_ohlcv(clean_frame)
    cleaned.loc[cleaned.index[0], "High"] = 98.0
    with pytest.raises(DataValidationError):
        validate_ohlcv(cleaned)


def test_negative_volume_is_rejected(clean_frame):
    clean_frame.loc[0, "Volume"] = -1
    with pytest.raises(DataValidationError):
        clean_ohlcv(clean_frame)


def test_csv_loader_round_trip(tmp_path, clean_frame):
    path = tmp_path / "prices.csv"
    clean_frame.to_csv(path, index=False)
    loaded, report = MarketDataLoader.from_csv(path)
    assert len(loaded) == 3
    assert report.rows_after == 3
    assert loaded.index.name == "Date"
