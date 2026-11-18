"""Supported candle timeframes, and how many candles each market has in a year."""

import pandas as pd

_TIMEFRAMES: dict[str, pd.Timedelta] = {
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
    "1d": pd.Timedelta(days=1),
}

SUPPORTED = tuple(_TIMEFRAMES)
STOCK_TRADING_DAYS = 252  # US market days in a typical year


def is_crypto(symbol: str) -> bool:
    """Alpaca writes crypto pairs with a slash (BTC/USD) and stocks without (SPY)."""
    return "/" in symbol


def to_timedelta(timeframe: str) -> pd.Timedelta:
    try:
        return _TIMEFRAMES[timeframe]
    except KeyError:
        raise ValueError(
            f"Unsupported timeframe {timeframe!r}; choose one of {', '.join(SUPPORTED)}"
        ) from None


def periods_per_year(timeframe: str, crypto: bool = True) -> float:
    """Candles per year. Crypto trades every day of the year; stocks on market days only."""
    if crypto:
        return pd.Timedelta(days=365) / to_timedelta(timeframe)
    if timeframe != "1d":
        raise ValueError("Stocks are supported on daily candles only")
    return float(STOCK_TRADING_DAYS)
