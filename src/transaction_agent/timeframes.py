"""Supported candle timeframes. Crypto trades 24/7, so a year is 365 full days."""

import pandas as pd

_TIMEFRAMES: dict[str, pd.Timedelta] = {
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
    "1d": pd.Timedelta(days=1),
}

SUPPORTED = tuple(_TIMEFRAMES)


def to_timedelta(timeframe: str) -> pd.Timedelta:
    try:
        return _TIMEFRAMES[timeframe]
    except KeyError:
        raise ValueError(
            f"Unsupported timeframe {timeframe!r}; choose one of {', '.join(SUPPORTED)}"
        ) from None


def periods_per_year(timeframe: str) -> float:
    return pd.Timedelta(days=365) / to_timedelta(timeframe)
