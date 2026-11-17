import pandas as pd
import pytest


def make_candles(opens: list[float], closes: list[float] | None = None) -> pd.DataFrame:
    """Hourly candles with the given opens; closes default to the next candle's open."""
    if closes is None:
        closes = opens[1:] + opens[-1:]
    index = pd.date_range("2024-01-01", periods=len(opens), freq="1h", tz="UTC", name="timestamp")
    return pd.DataFrame(
        {
            "open": opens,
            "high": [max(o, c) for o, c in zip(opens, closes, strict=True)],
            "low": [min(o, c) for o, c in zip(opens, closes, strict=True)],
            "close": closes,
            "volume": 1.0,
        },
        index=index,
    )


@pytest.fixture
def candles_factory():
    return make_candles
