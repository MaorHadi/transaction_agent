"""Per-coin tables for the breakout basket, each row built from past data only."""

import numpy as np
import pandas as pd

from transaction_agent.strategies.breakout import ChannelBreakout
from transaction_agent.volatility import annualized_volatility, realized_variance, walk_forward_har


def breakout_signal(candles: pd.DataFrame, window: int) -> pd.Series:
    """The ChannelBreakout signal for every candle at once.

    Gives the same values as calling `ChannelBreakout(window).signal` on the
    history up to each candle, which the tests verify, but in one pass.
    """
    top = candles["high"].rolling(window).max().shift(1)
    bottom = candles["low"].rolling(window).min().shift(1)
    close = candles["close"]
    signal = np.where(
        close >= top, 1.0, np.where(close <= (top + bottom) / 2, -1.0, ChannelBreakout.HOLD)
    )
    undefined = top.isna() | (top == bottom)
    return pd.Series(signal, index=candles.index).mask(undefined, 0.0)


def breakout_signals(candles_by_symbol: dict[str, pd.DataFrame], window: int) -> pd.DataFrame:
    return pd.DataFrame(
        {symbol: breakout_signal(candles, window) for symbol, candles in candles_by_symbol.items()}
    )


def volatility_forecasts(
    hourly_by_symbol: dict[str, pd.DataFrame], train_days: int = 365, test_days: int = 90
) -> pd.DataFrame:
    """Forecast yearly volatility for the following day, per coin, from hourly candles.

    A row dated D holds forecasts made once day D is complete, for day D + 1, by
    HAR models fitted only on days before D. Rows before a coin has a full
    training window are NaN for that coin. Where a day's hourly data is too
    patchy to forecast from, the previous day's forecast is carried forward.
    """
    return pd.DataFrame(
        {
            symbol: annualized_volatility(
                walk_forward_har(realized_variance(hourly), train_days, test_days)["har"]
            )
            for symbol, hourly in hourly_by_symbol.items()
        }
    ).ffill()
