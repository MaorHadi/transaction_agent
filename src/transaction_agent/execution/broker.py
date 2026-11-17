from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import pandas as pd


class Side(StrEnum):
    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True)
class Fill:
    timestamp: pd.Timestamp
    side: Side
    quantity: float
    price: float  # after slippage
    fee: float  # in quote currency


class Broker(Protocol):
    """One trading account for one symbol, long-only.

    SimBroker implements this for backtests; an Alpaca implementation will do the
    same for paper and live trading, so trading logic never depends on which is used.
    """

    def cash(self) -> float: ...

    def position(self) -> float:
        """Quantity of the base asset held."""
        ...

    def buy_notional(self, notional: float) -> None:
        """Submit a market buy spending `notional` of quote currency, fees included."""
        ...

    def sell_all(self) -> None:
        """Submit a market sell of the whole position."""
        ...
