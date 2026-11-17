from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    """Turns candle history into a signal strength.

    The same strategy object is used in backtest, paper and live trading; it never
    places orders itself. The caller compares the strength with its thresholds.
    """

    name: str

    @abstractmethod
    def signal(self, history: pd.DataFrame) -> float:
        """Return a strength in [-1, 1]: positive is bullish, negative bearish.

        `history` holds closed candles only, oldest first; the last row is the most
        recent closed candle. Return 0.0 when there is not enough history.
        """
