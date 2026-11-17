import pandas as pd

from transaction_agent.strategies.base import Strategy


class ChannelBreakout(Strategy):
    """Trend following: buy at a new high, sell when the price falls back to mid-range.

    The channel is the highest high and lowest low of the previous `window`
    candles, not counting the current one. The signal has three states only:

    - 1.0 when the latest close is at or above the top of the channel (buy);
    - -1.0 when it is at or below the middle of the channel (sell);
    - `HOLD` otherwise: keep an existing position, but do not open one.

    There is deliberately no strength scale. Measured over 890 trades on 14 coins,
    how close the price was to the top at entry had no relation to the trade's
    result (correlation 0.01), so an entry threshold would only be a dial to
    overfit with. The window length is the rule's single setting.
    """

    name = "breakout"

    # Above the engine's exit level (0) and below any sensible entry threshold.
    HOLD = 0.01

    def __init__(self, window: int = 50):
        if window < 2:
            raise ValueError("window must be at least 2")
        self.window = window

    def signal(self, history: pd.DataFrame) -> float:
        if len(history) < self.window + 1:
            return 0.0
        top = history["high"].to_numpy()[-self.window - 1 : -1].max()
        bottom = history["low"].to_numpy()[-self.window - 1 : -1].min()
        close = history["close"].iloc[-1]
        if top == bottom:
            return 0.0
        if close >= top:
            return 1.0
        if close <= (top + bottom) / 2:
            return -1.0
        return self.HOLD
