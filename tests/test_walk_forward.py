import pandas as pd
import pytest

from transaction_agent.backtest import BacktestConfig, Candidate, run_backtest, walk_forward
from transaction_agent.strategies import Strategy


class Always(Strategy):
    """Always emits the same strength."""

    name = "always"

    def __init__(self, strength: float):
        self.strength = strength

    def signal(self, history: pd.DataFrame) -> float:
        return self.strength


NO_COSTS = BacktestConfig(initial_cash=1000.0, fee_rate=0.0, slippage=0.0)
HOLD = Candidate("hold", Always(1.0), entry_threshold=0.5)
STAY_OUT = Candidate("stay out", Always(0.0), entry_threshold=0.5)


def test_start_skips_trading_but_keeps_history(candles_factory):
    candles = candles_factory([100.0, 200.0, 300.0, 400.0, 500.0])
    result = run_backtest(candles, Always(1.0), NO_COSTS, start=2)

    assert list(result.equity.index) == list(candles.index[2:])
    # Signal on candle 2 fills at candle 3's open; nothing trades before `start`.
    assert result.fills[0].timestamp == candles.index[3]


def test_picks_best_training_candidate_and_scores_it_on_the_next_window(candles_factory):
    # Rises for 6 candles (holding wins in training), then falls (holding loses in test).
    candles = candles_factory([100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 140.0, 130.0, 120.0])
    result = walk_forward(candles, [STAY_OUT, HOLD], train_size=5, test_size=4, config=NO_COSTS)

    assert len(result.folds) == 1
    fold = result.folds[0]
    assert fold.chosen == "hold"
    assert fold.train_return > 0
    assert fold.test_start == candles.index[5]
    # Buys at candle 6's open (140) and is valued at the last close (120).
    assert fold.test_return == pytest.approx(120.0 / 140.0 - 1)
    assert result.total_return == pytest.approx(fold.test_return)


def test_folds_slide_by_test_size_and_compound(candles_factory):
    candles = candles_factory([100.0 + i for i in range(12)])
    result = walk_forward(candles, [HOLD], train_size=4, test_size=4, config=NO_COSTS)

    assert [f.test_start for f in result.folds] == [candles.index[4], candles.index[8]]
    assert result.folds[1].test_end == candles.index[11]
    expected = (1 + result.folds[0].test_return) * (1 + result.folds[1].test_return) - 1
    assert result.total_return == pytest.approx(expected)


def test_too_little_data_is_an_error(candles_factory):
    with pytest.raises(ValueError):
        walk_forward(candles_factory([100.0] * 5), [HOLD], train_size=5, test_size=4)
