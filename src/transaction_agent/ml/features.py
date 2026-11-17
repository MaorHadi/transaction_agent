"""Model inputs computed from candles.

Every feature in a row uses only that candle and earlier ones, so a row is exactly
what the bot would know when that candle closes.
"""

import numpy as np
import pandas as pd

RETURN_WINDOWS = (1, 3, 6, 12, 42)  # in candles; 42 four-hour candles is one week
VOLATILITY_WINDOWS = (12, 42)
AVERAGE_WINDOWS = (20, 100)
RSI_WINDOW = 14
VOLUME_WINDOW = 42


def build_features(candles: pd.DataFrame) -> pd.DataFrame:
    """Return one row of features per candle. Early rows contain NaN until windows fill."""
    close = candles["close"]
    log_close = np.log(close)
    step_return = log_close.diff()
    features = {}

    for window in RETURN_WINDOWS:
        features[f"return_{window}"] = log_close.diff(window)
    for window in VOLATILITY_WINDOWS:
        features[f"volatility_{window}"] = step_return.rolling(window).std()
    for window in AVERAGE_WINDOWS:
        features[f"average_gap_{window}"] = close / close.rolling(window).mean() - 1

    # How many standard deviations the close is from its 20-candle average.
    deviation = close.rolling(20).std()
    features["zscore_20"] = ((close - close.rolling(20).mean()) / deviation).where(deviation > 0, 0)

    # Relative strength index: share of recent movement that was upward, 0 to 100.
    change = close.diff()
    gains = change.clip(lower=0).rolling(RSI_WINDOW).mean()
    losses = (-change.clip(upper=0)).rolling(RSI_WINDOW).mean()
    total = gains + losses
    features["rsi_14"] = (100 * gains / total).where(total > 0, 50).where(total.notna())

    log_volume = np.log1p(candles["volume"])
    features["volume_vs_average"] = log_volume - log_volume.rolling(VOLUME_WINDOW).mean()
    features["candle_range"] = (candles["high"] - candles["low"]) / close
    features["candle_body"] = close / candles["open"] - 1

    return pd.DataFrame(features, index=candles.index)
