import pandas as pd

from transaction_agent.strategies.base import Strategy


class MovingAverageCrossover(Strategy):
    """Bullish when the fast moving average of closes is above the slow one.

    Strength is the gap between the averages as a fraction of the slow average,
    scaled so that a gap of `full_strength_gap` (or more) gives strength 1.0.
    """

    name = "ma_crossover"

    def __init__(self, fast: int = 20, slow: int = 100, full_strength_gap: float = 0.02):
        if not 0 < fast < slow:
            raise ValueError("Need 0 < fast < slow")
        if full_strength_gap <= 0:
            raise ValueError("full_strength_gap must be positive")
        self.fast = fast
        self.slow = slow
        self.full_strength_gap = full_strength_gap

    def signal(self, history: pd.DataFrame) -> float:
        if len(history) < self.slow:
            return 0.0
        closes = history["close"].to_numpy()
        fast_avg = closes[-self.fast :].mean()
        slow_avg = closes[-self.slow :].mean()
        gap = (fast_avg - slow_avg) / slow_avg
        return float(max(-1.0, min(1.0, gap / self.full_strength_gap)))
