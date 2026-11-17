"""The most that could have been made with perfect knowledge of the future.

These are ceilings, not strategies: they use prices nobody could have known.
They show how much movement was there to be captured, for scale.
"""

import numpy as np
import pandas as pd


def best_single_coin(opens: pd.DataFrame, cost_rate: float) -> tuple[str, float]:
    """Growth from buying the one coin that ended up rising most, and holding it."""
    growth = opens.iloc[-1] / opens.iloc[0] * (1 - cost_rate) / (1 + cost_rate)
    return str(growth.idxmax()), float(growth.max())


def perfect_foresight_log_growth(opens: pd.DataFrame, cost_rate: float) -> float:
    """Natural log of the best possible growth, deciding at every open with hindsight.

    The account is always entirely in one coin or in cash and may switch at each
    candle's open, paying `cost_rate` on every buy and every sell, and ends in
    cash. Found exactly by dynamic programming over "where is the money now".
    The log is returned because the growth itself can be too large for a float.
    """
    buy, sell = -np.log1p(cost_rate), np.log1p(-cost_rate)
    gains = np.log(opens.to_numpy()[1:] / opens.to_numpy()[:-1])
    in_cash = 0.0
    in_coin = np.full(opens.shape[1], -np.inf)
    for gain in gains:
        best_after_selling = (in_coin + sell).max()
        enter = max(in_cash, best_after_selling) + buy
        in_coin = np.maximum(in_coin, enter) + gain
        in_cash = max(in_cash, best_after_selling)
    return float(max(in_cash, (in_coin + sell).max()))
