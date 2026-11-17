import pandas as pd

from transaction_agent.execution.broker import Fill, Side


class SimBroker:
    """Simulated account for backtests.

    Orders never fill at the price that triggered them: they wait for the next
    candle's open, pay slippage against the trader, and pay a proportional fee.
    """

    def __init__(self, cash: float, fee_rate: float, slippage: float):
        if cash <= 0:
            raise ValueError("Starting cash must be positive")
        if fee_rate < 0 or slippage < 0:
            raise ValueError("fee_rate and slippage cannot be negative")
        self._cash = cash
        self._quantity = 0.0
        self._fee_rate = fee_rate
        self._slippage = slippage
        self._pending_buy: float | None = None
        self._pending_sell = False
        self.fills: list[Fill] = []

    def cash(self) -> float:
        return self._cash

    def position(self) -> float:
        return self._quantity

    def equity(self, price: float) -> float:
        return self._cash + self._quantity * price

    def buy_notional(self, notional: float) -> None:
        if notional <= 0:
            raise ValueError("notional must be positive")
        self._pending_buy = notional

    def sell_all(self) -> None:
        self._pending_sell = True

    def on_open(self, timestamp: pd.Timestamp, open_price: float) -> None:
        """Fill any pending order at this candle's open."""
        if self._pending_buy is not None:
            notional = min(self._pending_buy, self._cash)
            self._pending_buy = None
            if notional > 0:
                price = open_price * (1 + self._slippage)
                quantity = notional / (price * (1 + self._fee_rate))
                fee = quantity * price * self._fee_rate
                self._cash -= quantity * price + fee
                self._quantity += quantity
                self.fills.append(Fill(timestamp, Side.BUY, quantity, price, fee))
        if self._pending_sell:
            self._pending_sell = False
            if self._quantity > 0:
                price = open_price * (1 - self._slippage)
                proceeds = self._quantity * price
                fee = proceeds * self._fee_rate
                self._cash += proceeds - fee
                self.fills.append(Fill(timestamp, Side.SELL, self._quantity, price, fee))
                self._quantity = 0.0
