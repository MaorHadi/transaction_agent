import pandas as pd


def forward_labels(candles: pd.DataFrame, horizon: int, min_gain: float = 0.0) -> pd.Series:
    """Label each candle 1.0 if buying after it would have gained more than `min_gain`.

    The trade is the one the bot could actually make: buy at the next candle's
    open, sell `horizon` candles later at that candle's open. The label for candle
    i therefore depends on prices up to candle i + 1 + horizon, so it is only for
    training and scoring, never an input. The last `horizon + 1` candles get NaN.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    entry = candles["open"].shift(-1)
    exit_ = candles["open"].shift(-1 - horizon)
    gain = exit_ / entry - 1
    return (gain > min_gain).astype(float).where(gain.notna()).rename("label")
