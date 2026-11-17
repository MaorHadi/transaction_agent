import pandas as pd

from transaction_agent.strategies.base import Strategy


class MeanReversion(Strategy):
    """Bullish when the price is unusually far below its recent average.

    Measures how many standard deviations the latest close sits from the average
    of the last `window` closes (a Bollinger-band z-score). A close
    `full_strength_z` deviations below the average gives strength 1.0; a close at
    or above the average gives strength 0 or less, which is the exit signal.
    """

    name = "mean_reversion"

    def __init__(self, window: int = 20, full_strength_z: float = 2.0):
        if window < 2:
            raise ValueError("window must be at least 2")
        if full_strength_z <= 0:
            raise ValueError("full_strength_z must be positive")
        self.window = window
        self.full_strength_z = full_strength_z

    def signal(self, history: pd.DataFrame) -> float:
        if len(history) < self.window:
            return 0.0
        closes = history["close"].to_numpy()[-self.window :]
        deviation = closes.std()
        if deviation == 0:
            return 0.0
        z_score = (closes[-1] - closes.mean()) / deviation
        return float(max(-1.0, min(1.0, -z_score / self.full_strength_z)))
