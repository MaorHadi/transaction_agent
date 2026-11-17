from dataclasses import dataclass

import numpy as np
import pandas as pd

from transaction_agent.backtest.engine import BacktestResult
from transaction_agent.execution.broker import Side


@dataclass(frozen=True)
class Metrics:
    total_return: float
    buy_and_hold_return: float
    sharpe: float  # annualised, risk-free rate taken as zero
    max_drawdown: float  # negative fraction, e.g. -0.25
    trades: int  # completed round trips
    win_rate: float  # share of round trips that made money after fees
    fees_paid: float
    exposure: float  # share of candles spent holding a position


def buy_and_hold_return(candles: pd.DataFrame, fee_rate: float, slippage: float) -> float:
    """Return from buying at the first open (paying costs) and holding to the last close."""
    entry = candles["open"].iloc[0] * (1 + slippage) * (1 + fee_rate)
    return float(candles["close"].iloc[-1] / entry - 1)


def max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1).min())


def round_trip_profits(result: BacktestResult) -> list[float]:
    """Profit of each buy followed by its sell, net of both fees."""
    profits = []
    cost = None
    for fill in result.fills:
        value = fill.quantity * fill.price
        if fill.side is Side.BUY:
            cost = value + fill.fee
        elif cost is not None:
            profits.append(value - fill.fee - cost)
            cost = None
    return profits


def compute_metrics(
    result: BacktestResult, candles: pd.DataFrame, periods_per_year: float
) -> Metrics:
    equity = result.equity
    returns = equity.pct_change().dropna()
    volatility = returns.std()
    sharpe = returns.mean() / volatility * np.sqrt(periods_per_year) if volatility > 0 else 0.0
    profits = round_trip_profits(result)

    holding = pd.Series(0.0, index=equity.index)
    for fill in result.fills:
        holding.loc[fill.timestamp :] = 1.0 if fill.side is Side.BUY else 0.0

    return Metrics(
        total_return=float(equity.iloc[-1] / result.config.initial_cash - 1),
        buy_and_hold_return=buy_and_hold_return(
            candles, result.config.fee_rate, result.config.slippage
        ),
        sharpe=float(sharpe),
        max_drawdown=max_drawdown(equity),
        trades=len(profits),
        win_rate=sum(p > 0 for p in profits) / len(profits) if profits else 0.0,
        fees_paid=sum(fill.fee for fill in result.fills),
        exposure=float(holding.mean()),
    )
