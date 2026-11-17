from dataclasses import dataclass

import pandas as pd

from transaction_agent.execution.broker import Fill
from transaction_agent.execution.sim import SimBroker
from transaction_agent.risk import DrawdownGuard, RiskLimits
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
    halted_at: pd.Timestamp | None = None  # when the kill switch fired, if it did


def run_backtest(
    candles: pd.DataFrame,
    strategy: Strategy,
    config: BacktestConfig | None = None,
    start: int = 0,
    risk: RiskLimits | None = None,
) -> BacktestResult:
    """Replay candles one at a time through a strategy.

    Lookahead is prevented by construction: at candle i the strategy sees only
    candles[0..i], and any order it triggers fills at the open of candle i+1.
    A position still open at the end is valued at the last close, not sold.

    Trading begins at candle `start`. Earlier candles are history the strategy can
    read but not trade on, and the returned equity covers `start` onwards.

    Risk limits, when given:
    - Stop-loss: if a candle's low reaches the stop price, the position is sold
      inside that candle at the stop price, or at the open if the candle opened
      below it. After a stop-out the bot stays out until the signal has fallen
      to the exit threshold once, so it does not buy straight back in.
    - Kill switch: once the account's drawdown from its peak reaches the limit
      at a candle's close, the position is sold at the next open and trading stops.
    """
    config = config or BacktestConfig()
    risk = risk or RiskLimits()
    if not 0 <= start <= len(candles) - 2:
        raise ValueError("Need at least two candles to trade on")
    broker = SimBroker(config.initial_cash, config.fee_rate, config.slippage)
    guard = DrawdownGuard(risk.max_drawdown)
    opens = candles["open"].to_numpy()
    lows = candles["low"].to_numpy()
    closes = candles["close"].to_numpy()
    last = len(candles) - 1
    equity = []
    halted_at = None
    stopped_out = False
    for i in range(start, len(candles)):
        timestamp = candles.index[i]
        broker.on_open(timestamp, opens[i])
        if broker.position() > 0:
            stop_price = risk.stop_price(broker.entry_price)
            if stop_price is not None and lows[i] <= stop_price:
                broker.sell_now(timestamp, min(opens[i], stop_price), "stop_loss")
                stopped_out = True
        equity.append(broker.equity(closes[i]))
        if i == last:
            break  # an order placed now could never fill
        if halted_at is None and guard.tripped(equity[-1]):
            halted_at = timestamp
            if broker.position() > 0:
                broker.sell_all("kill_switch")
        if halted_at is not None:
            continue
        strength = strategy.signal(candles.iloc[: i + 1])
        if strength <= config.exit_threshold:
            stopped_out = False
        if broker.position() == 0:
            if strength >= config.entry_threshold and not stopped_out:
                broker.buy_notional(broker.cash() * risk.position_fraction)
        elif strength <= config.exit_threshold:
            broker.sell_all()
    return BacktestResult(
        config=config,
        equity=pd.Series(equity, index=candles.index[start:], name="equity"),
        fills=broker.fills,
        halted_at=halted_at,
    )
