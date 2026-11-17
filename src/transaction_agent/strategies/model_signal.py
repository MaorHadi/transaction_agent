import pandas as pd

from transaction_agent.strategies.base import Strategy


class ModelSignal(Strategy):
    """Replays a model's probabilities as signal strengths.

    A probability p that the price rises becomes strength 2p - 1, so 50% maps to 0
    (the exit level) and an entry threshold of 0.2 means "at least 60% confident".
    The probabilities must already be out-of-sample: each one produced from that
    candle's features by a model trained only on earlier data.
    """

    name = "model"

    def __init__(self, probabilities: pd.Series):
        self._strengths = (2 * probabilities - 1).to_dict()

    def signal(self, history: pd.DataFrame) -> float:
        return float(self._strengths.get(history.index[-1], 0.0))

    @staticmethod
    def threshold_for(confidence: float) -> float:
        """The entry threshold that corresponds to a minimum probability."""
        return 2 * confidence - 1
