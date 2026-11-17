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


class BreakoutBasket(WeightStrategy):
    """Hold each coin while its breakout rule says so, sized by forecast volatility.

    Every coin gets an equal slice of the account. A coin's slice is invested
    when its breakout signal says buy, stays invested while the signal says hold,
    and returns to cash when it says sell. With a volatility forecast, the
    invested part of a slice is target volatility divided by the coin's forecast,
    capped at 100%; without one, the whole slice is invested.

    `signals` and `volatility` are tables with one row per candle and one column
    per coin. Each row must have been computed from that candle and earlier ones
    only (see `portfolio.signals`); this class trusts that and just looks rows up.
    """

    name = "breakout_basket"

    def __init__(
        self,
        signals: pd.DataFrame,
        volatility: pd.DataFrame | None = None,
        target: float | None = None,
    ):
        if (volatility is None) != (target is None):
            raise ValueError("Give both a volatility forecast and a target, or neither")
        if target is not None and target <= 0:
            raise ValueError("target must be positive")
        self.signals = signals
        self.volatility = volatility
        self.target = target

    def target_weights(self, closes: pd.DataFrame, current: pd.Series) -> pd.Series | None:
        day = closes.index[-1]
        if day not in self.signals.index:
            return None
        signal = self.signals.loc[day].reindex(closes.columns).fillna(0.0)
        holding = current > 0
        wanted = (signal >= 1.0) | ((signal > 0) & holding)
        scale = pd.Series(1.0, index=closes.columns)
        if self.volatility is not None:
            if day not in self.volatility.index:
                return None
            forecast = self.volatility.loc[day].reindex(closes.columns)
            # A coin with no forecast is not sized, so it is not held.
            scale = (self.target / forecast.where(forecast > 0)).clip(upper=1.0).fillna(0.0)
        return wanted * scale / len(closes.columns)


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
