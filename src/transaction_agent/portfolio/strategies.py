"""Strategies that decide what share of the account each coin should hold."""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class WeightStrategy(ABC):
    name: str

    @abstractmethod
    def target_weights(self, closes: pd.DataFrame, current: pd.Series) -> pd.Series | None:
        """Return the share of the account each coin should hold, or None for no change.

        `closes` has one column per coin and holds closed candles only, oldest
        first. `current` is the share each coin holds now. Weights are between
        0 and 1 and sum to at most 1; whatever is left stays in cash.
        """


class HoldEqual(WeightStrategy):
    """Buy every coin in equal parts once and never trade again. The benchmark."""

    name = "hold"

    def target_weights(self, closes: pd.DataFrame, current: pd.Series) -> pd.Series | None:
        if current.sum() > 0:
            return None
        return pd.Series(1 / len(closes.columns), index=closes.columns)


class VolatilityTarget(WeightStrategy):
    """Hold less when prices are swinging wildly and more when they are calm.

    Each coin gets an equal slice of the account. Within its slice, the share
    actually invested is target volatility divided by the coin's recent
    volatility, capped at 100%. Both are annualised.
    """

    name = "vol_target"

    def __init__(self, target: float, window: int, periods_per_year: float):
        if target <= 0 or window < 2:
            raise ValueError("target must be positive and window at least 2")
        self.target = target
        self.window = window
        self.periods_per_year = periods_per_year

    def target_weights(self, closes: pd.DataFrame, current: pd.Series) -> pd.Series | None:
        if len(closes) <= self.window:
            return None
        returns = np.log(closes.iloc[-self.window - 1 :]).diff().dropna()
        volatility = returns.std() * np.sqrt(self.periods_per_year)
        scale = (self.target / volatility.where(volatility > 0)).clip(upper=1.0).fillna(1.0)
        return scale / len(closes.columns)
