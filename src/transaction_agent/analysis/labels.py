import pandas as pd


def forward_returns(candles: pd.DataFrame, horizon: int) -> pd.Series:
    """The gain from the trade the bot could actually make after each candle.

    Buy at the next candle's open, sell `horizon` candles later at that candle's
    open. The value for candle i therefore depends on prices up to candle
    i + 1 + horizon, so it is only for scoring, never an input.
    The last `horizon + 1` candles get NaN.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    entry = candles["open"].shift(-1)
    exit_ = candles["open"].shift(-1 - horizon)
    return (exit_ / entry - 1).rename("forward_return")
