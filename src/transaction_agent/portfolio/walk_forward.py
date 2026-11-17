from dataclasses import dataclass, replace

import pandas as pd

from transaction_agent.backtest.walk_forward import Fold
from transaction_agent.portfolio.engine import (
    PortfolioConfig,
    PortfolioResult,
    run_portfolio_backtest,
)
from transaction_agent.portfolio.strategies import HoldEqual, WeightStrategy


@dataclass(frozen=True)
class PortfolioCandidate:
    label: str
    strategy: WeightStrategy
    rebalance_every: int = 1


@dataclass(frozen=True)
class PortfolioWalkForward:
    folds: list[Fold]
    equity: pd.Series  # growth of 1.0 across all test windows, in order
    benchmark: pd.Series  # growth of 1.0 from buying equal parts once and holding

    @property
    def trades(self) -> int:
        return sum(fold.test_trades for fold in self.folds)


def _growth(result: PortfolioResult) -> pd.Series:
    return result.equity / result.config.initial_cash


def walk_forward_portfolio(
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    candidates: list[PortfolioCandidate],
    train_size: int,
    test_size: int,
    config: PortfolioConfig | None = None,
) -> PortfolioWalkForward:
    """Pick the best candidate on each training window and trade it on the next window.

    Works like the single-coin walk-forward. Each test window starts in cash, so
    re-entering at every window boundary is paid for in the result.
    """
    if not candidates:
        raise ValueError("Need at least one candidate")
    config = config or PortfolioConfig()

    def run(candidate: PortfolioCandidate, end: int, start: int) -> PortfolioResult:
        settings = replace(config, rebalance_every=candidate.rebalance_every)
        return run_portfolio_backtest(
            opens.iloc[:end], closes.iloc[:end], candidate.strategy, settings, start
        )

    hold = PortfolioCandidate("hold", HoldEqual())
    folds, pieces = [], []
    growth = 1.0
    train_start = 0
    while train_start + train_size + 2 <= len(closes):
        test_start = train_start + train_size
        test_end = min(test_start + test_size, len(closes))
        train_returns = [
            _growth(run(candidate, test_start, train_start)).iloc[-1] - 1
            for candidate in candidates
        ]
        best = max(range(len(candidates)), key=train_returns.__getitem__)
        test = run(candidates[best], test_end, test_start)
        test_growth = _growth(test)
        pieces.append(test_growth * growth)
        growth *= test_growth.iloc[-1]
        folds.append(
            Fold(
                test_start=closes.index[test_start],
                test_end=closes.index[test_end - 1],
                chosen=candidates[best].label,
                train_return=float(train_returns[best]),
                test_return=float(test_growth.iloc[-1] - 1),
                test_buy_and_hold=float(_growth(run(hold, test_end, test_start)).iloc[-1] - 1),
                test_trades=test.trades,
            )
        )
        train_start += test_size
    if not folds:
        raise ValueError("Not enough candles for one training and one test window")
    return PortfolioWalkForward(
        folds=folds,
        equity=pd.concat(pieces),
        benchmark=_growth(run(hold, len(closes), train_size)),
    )
