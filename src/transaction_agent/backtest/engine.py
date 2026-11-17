from dataclasses import dataclass

import pandas as pd

from transaction_agent.execution.broker import Fill
from transaction_agent.execution.sim import SimBroker
from transaction_agent.strategies.base import Strategy


@dataclass(frozen=True)
class BacktestConfig:
    initial_cash: float = 10_000.0
    # Alpaca crypto taker fee at tier 1 (under $100k 30-day volume), as of November 2026.
    # Market orders are taker orders. The maker fee at this tier is 0.15%.
    fee_rate: float = 0.0025
    slippage: float = 0.0005
    # Enter a long position when signal strength reaches this level...
    entry_threshold: float = 0.5
    # ...and leave it when strength falls to this level or below.
    exit_threshold: float = 0.0

    def __post_init__(self) -> None:
        if self.exit_threshold >= self.entry_threshold:
            raise ValueError("exit_threshold must be below entry_threshold")


@dataclass(frozen=True)
class BacktestResult:
    config: BacktestConfig
    equity: pd.Series  # account value at each candle's close
    fills: list[Fill]


def run_backtest(
    candles: pd.DataFrame, strategy: Strategy, config: BacktestConfig | None = None
) -> BacktestResult:
    """Replay candles one at a time through a strategy.

    Lookahead is prevented by construction: at candle i the strategy sees only
    candles[0..i], and any order it triggers fills at the open of candle i+1.
    A position still open at the end is valued at the last close, not sold.
    """
    config = config or BacktestConfig()
    if len(candles) < 2:
        raise ValueError("Need at least two candles to backtest")
    broker = SimBroker(config.initial_cash, config.fee_rate, config.slippage)
    opens = candles["open"].to_numpy()
    closes = candles["close"].to_numpy()
    last = len(candles) - 1
    equity = []
    for i, timestamp in enumerate(candles.index):
        broker.on_open(timestamp, opens[i])
        equity.append(broker.equity(closes[i]))
        if i == last:
            break  # an order placed now could never fill
        strength = strategy.signal(candles.iloc[: i + 1])
        if broker.position() == 0:
            if strength >= config.entry_threshold:
                broker.buy_notional(broker.cash())
        elif strength <= config.exit_threshold:
            broker.sell_all()
    return BacktestResult(
        config=config,
        equity=pd.Series(equity, index=candles.index, name="equity"),
        fills=broker.fills,
    )
