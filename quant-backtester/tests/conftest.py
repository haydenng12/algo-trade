import pandas as pd
import pytest


@pytest.fixture
def clean_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": ["2026-01-02", "2026-01-05", "2026-01-06"],
            "Open": [100.0, 102.0, 101.0],
            "High": [103.0, 104.0, 103.0],
            "Low": [99.0, 100.0, 100.0],
            "Close": [102.0, 101.0, 103.0],
            "Adj Close": [51.0, 50.5, 51.5],
            "Volume": [1_000, 1_100, 1_050],
        }
    )
