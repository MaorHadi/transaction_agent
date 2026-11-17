"""Backtester for strategies that hold several coins in chosen proportions."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from transaction_agent.portfolio.strategies import WeightStrategy


@dataclass(frozen=True)
class PortfolioConfig:
    initial_cash: float = 10_000.0
    fee_rate: float = 0.0025  # Alpaca crypto taker fee at tier 1
    slippage: float = 0.0005
    # Ask the strategy for new weights every this many candles.
    rebalance_every: int = 1
    # Skip adjustments smaller than this share of the account, to avoid paying
    # fees for tiny corrections. Closing a position completely is never skipped.
    min_trade_fraction: float = 0.05

    def __post_init__(self) -> None:
        if self.rebalance_every < 1:
            raise ValueError("rebalance_every must be at least 1")


@dataclass(frozen=True)
class PortfolioResult:
    config: PortfolioConfig
    equity: pd.Series  # account value at each candle's close
    trades: int  # individual coin buys and sells
    costs_paid: float  # fees plus slippage
    invested: float  # average share of the account held in coins


@dataclass(frozen=True)
class Summary:
    total_return: float
    sharpe: float  # annualised, risk-free rate taken as zero
    max_drawdown: float


def summarize(equity: pd.Series, periods_per_year: float) -> Summary:
    returns = equity.pct_change().dropna()
    volatility = returns.std()
    sharpe = returns.mean() / volatility * np.sqrt(periods_per_year) if volatility > 0 else 0.0
    return Summary(
        total_return=float(equity.iloc[-1] / equity.iloc[0] - 1),
        sharpe=float(sharpe),
        max_drawdown=float((equity / equity.cummax() - 1).min()),
    )


def align(candles_by_symbol: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (opens, closes) tables with one column per coin, on shared timestamps."""
    opens = pd.DataFrame({s: c["open"] for s, c in candles_by_symbol.items()}).dropna()
    closes = pd.DataFrame({s: c["close"] for s, c in candles_by_symbol.items()}).dropna()
    shared = opens.index.intersection(closes.index)
    return opens.loc[shared], closes.loc[shared]


def run_portfolio_backtest(
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    strategy: WeightStrategy,
    config: PortfolioConfig | None = None,
    start: int = 0,
) -> PortfolioResult:
    """Replay candles, letting the strategy set what share of the account each coin gets.

    As in the single-coin engine, the strategy sees only closed candles up to the
    current one, and the trades needed to reach its target weights happen at the
    next candle's open. Every trade pays the fee and slippage on its value.
    The account starts in cash at candle `start`; earlier candles are history only.
    """
    config = config or PortfolioConfig()
    if not 0 <= start <= len(closes) - 2:
        raise ValueError("Need at least two candles to trade on")
    symbols = list(closes.columns)
    open_prices = opens[symbols].to_numpy()
    close_prices = closes.to_numpy()
    cost_rate = config.fee_rate + config.slippage
    quantity = np.zeros(len(symbols))
    cash = config.initial_cash
    pending: np.ndarray | None = None
    equity, invested = [], []
    trades, costs = 0, 0.0
    last = len(closes) - 1

    for i in range(start, len(closes)):
        if pending is not None:
            prices = open_prices[i]
            values = quantity * prices
            account = cash + values.sum()
            change = pending * account - values
            small = np.abs(change) < config.min_trade_fraction * account
            closing = (pending == 0) & (quantity > 0)
            change[small & ~closing] = 0.0

            sells = -np.minimum(change, 0.0)
            cash += sells.sum() * (1 - cost_rate)
            quantity -= sells / prices
            buys = np.maximum(change, 0.0)
            if buys.sum() > 0:
                buys *= min(1.0, cash / (buys.sum() * (1 + cost_rate)))
                cash -= buys.sum() * (1 + cost_rate)
                quantity += buys / prices
            quantity[closing] = 0.0
            costs += (sells.sum() + buys.sum()) * cost_rate
            trades += int(np.count_nonzero(sells + buys))
            pending = None

        holdings = quantity * close_prices[i]
        account = cash + holdings.sum()
        equity.append(account)
        invested.append(holdings.sum() / account)
        if i == last:
            break
        if (i - start) % config.rebalance_every == 0:
            current = pd.Series(holdings / account, index=symbols)
            target = strategy.target_weights(closes.iloc[: i + 1], current)
            if target is not None:
                weights = target.reindex(symbols).fillna(0.0).to_numpy(dtype=float)
                if (weights < 0).any() or weights.sum() > 1 + 1e-9:
                    raise ValueError("Target weights must be non-negative and sum to at most 1")
                pending = weights

    return PortfolioResult(
        config=config,
        equity=pd.Series(equity, index=closes.index[start:], name="equity"),
        trades=trades,
        costs_paid=costs,
        invested=float(np.mean(invested)),
    )
