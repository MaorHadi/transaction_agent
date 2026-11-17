"""Walk-forward evaluation: choose settings on past data, score them on unseen data."""

from dataclasses import dataclass, replace

import pandas as pd

from transaction_agent.backtest.engine import BacktestConfig, BacktestResult, run_backtest
from transaction_agent.backtest.metrics import buy_and_hold_return, round_trip_profits
from transaction_agent.risk import RiskLimits
from transaction_agent.strategies.base import Strategy


@dataclass(frozen=True)
class Candidate:
    """One combination of strategy settings and entry threshold to choose between."""

    label: str
    strategy: Strategy
    entry_threshold: float


@dataclass(frozen=True)
class Fold:
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    chosen: str
    train_return: float
    test_return: float
    test_buy_and_hold: float
    test_trades: int


@dataclass(frozen=True)
class WalkForwardResult:
    folds: list[Fold]
    buy_and_hold_return: float  # holding across the whole tested span

    @property
    def total_return(self) -> float:
        """Test-period returns compounded, as if trading each fold in turn."""
        growth = 1.0
        for fold in self.folds:
            growth *= 1 + fold.test_return
        return growth - 1

    @property
    def trades(self) -> int:
        return sum(fold.test_trades for fold in self.folds)


def _total_return(result: BacktestResult) -> float:
    return float(result.equity.iloc[-1] / result.config.initial_cash - 1)


def walk_forward(
    candles: pd.DataFrame,
    candidates: list[Candidate],
    train_size: int,
    test_size: int,
    config: BacktestConfig | None = None,
    risk: RiskLimits | None = None,
) -> WalkForwardResult:
    """Repeatedly pick the best candidate on a training window, then test it on the next.

    Each fold picks the candidate with the highest return over `train_size` candles
    and trades it over the following `test_size` candles, which it has never been
    scored on. The window then slides forward by `test_size`. Only the test-period
    results count.

    Each test period starts with no position. A position still open when a test
    period ends is valued at the last close without paying an exit fee.
    """
    if not candidates:
        raise ValueError("Need at least one candidate")
    if train_size < 2 or test_size < 2:
        raise ValueError("train_size and test_size must be at least 2")
    config = config or BacktestConfig()
    folds = []
    train_start = 0
    while train_start + train_size + 2 <= len(candles):
        test_start = train_start + train_size
        test_end = min(test_start + test_size, len(candles))

        def run(candidate: Candidate, end: int, start: int) -> BacktestResult:
            settings = replace(config, entry_threshold=candidate.entry_threshold)
            return run_backtest(
                candles.iloc[:end], candidate.strategy, settings, start=start, risk=risk
            )

        train_returns = [
            _total_return(run(candidate, test_start, train_start)) for candidate in candidates
        ]
        best = max(range(len(candidates)), key=train_returns.__getitem__)
        test = run(candidates[best], test_end, test_start)
        folds.append(
            Fold(
                test_start=candles.index[test_start],
                test_end=candles.index[test_end - 1],
                chosen=candidates[best].label,
                train_return=train_returns[best],
                test_return=_total_return(test),
                test_buy_and_hold=buy_and_hold_return(
                    candles.iloc[test_start:test_end], config.fee_rate, config.slippage
                ),
                test_trades=len(round_trip_profits(test)),
            )
        )
        train_start += test_size
    if not folds:
        raise ValueError("Not enough candles for one training and one test window")
    return WalkForwardResult(
        folds=folds,
        buy_and_hold_return=buy_and_hold_return(
            candles.iloc[train_size:], config.fee_rate, config.slippage
        ),
    )
