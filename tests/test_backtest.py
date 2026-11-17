import pandas as pd
import pytest

from transaction_agent.backtest import (
    BacktestConfig,
    buy_and_hold_return,
    compute_metrics,
    run_backtest,
)
from transaction_agent.backtest.metrics import max_drawdown
from transaction_agent.execution import Side
from transaction_agent.strategies import ChannelBreakout, Strategy


class Scripted(Strategy):
    """Emits a fixed list of strengths, one per candle, and records what it was shown."""

    name = "scripted"

    def __init__(self, strengths: list[float]):
        self.strengths = strengths
        self.seen: list[pd.Timestamp] = []

    def signal(self, history: pd.DataFrame) -> float:
        self.seen.append(history.index[-1])
        return self.strengths[len(history) - 1]


NO_COSTS = BacktestConfig(initial_cash=1000.0, fee_rate=0.0, slippage=0.0)


def test_order_fills_at_next_candle_open(candles_factory):
    candles = candles_factory([100.0, 110.0, 120.0, 130.0])
    result = run_backtest(candles, Scripted([1, 1, 1, 1]), NO_COSTS)

    assert len(result.fills) == 1
    fill = result.fills[0]
    # Signal came from candle 0, so the fill is at candle 1's open, not candle 0's.
    assert fill.timestamp == candles.index[1]
    assert fill.price == 110.0


def test_strategy_never_sees_future_candles(candles_factory):
    candles = candles_factory([100.0, 110.0, 120.0, 130.0])
    strategy = Scripted([0, 0, 0, 0])
    run_backtest(candles, strategy, NO_COSTS)

    # Called once per candle except the last, each time ending at the current candle.
    assert strategy.seen == list(candles.index[:-1])


def test_changing_the_future_does_not_change_past_fills(candles_factory):
    # A triangle wave: six candles up, six down, so the breakout rule buys and sells repeatedly.
    opens = [100.0 + 8 * min(i % 12, 12 - i % 12) for i in range(60)]
    altered = opens[:40] + [o * 3 for o in opens[40:]]
    strategy = ChannelBreakout(window=5)
    config = BacktestConfig()

    original = run_backtest(candles_factory(opens), strategy, config)
    changed = run_backtest(candles_factory(altered), strategy, config)

    cutoff = candles_factory(opens).index[39]
    assert len([f for f in original.fills if f.timestamp < cutoff]) >= 4
    assert [f for f in original.fills if f.timestamp < cutoff] == [
        f for f in changed.fills if f.timestamp < cutoff
    ]
    assert original.equity.iloc[:39].equals(changed.equity.iloc[:39])


def test_round_trip_without_costs(candles_factory):
    candles = candles_factory([100.0, 100.0, 150.0, 150.0, 150.0])
    # Buy signal on candle 0 -> buy at 100. Exit signal on candle 1 -> sell at 150.
    result = run_backtest(candles, Scripted([1, 0, 0, 0, 0]), NO_COSTS)

    assert [f.side for f in result.fills] == [Side.BUY, Side.SELL]
    assert result.equity.iloc[-1] == pytest.approx(1500.0)


def test_fees_and_slippage_are_charged_on_both_sides(candles_factory):
    candles = candles_factory([100.0, 100.0, 100.0, 100.0])
    config = BacktestConfig(initial_cash=1000.0, fee_rate=0.01, slippage=0.02)
    result = run_backtest(candles, Scripted([1, 0, 0, 0]), config)

    buy, sell = result.fills
    assert buy.price == pytest.approx(102.0)
    assert sell.price == pytest.approx(98.0)
    quantity = 1000.0 / (102.0 * 1.01)
    assert buy.quantity == pytest.approx(quantity)
    assert buy.fee == pytest.approx(quantity * 102.0 * 0.01)
    # A flat market must lose money once costs are included.
    expected = quantity * 98.0 * 0.99
    assert result.equity.iloc[-1] == pytest.approx(expected)
    assert expected < 1000.0


def test_no_trade_below_entry_threshold(candles_factory):
    candles = candles_factory([100.0, 110.0, 120.0, 130.0])
    config = BacktestConfig(entry_threshold=0.6)
    result = run_backtest(candles, Scripted([0.59, 0.59, 0.59, 0.59]), config)

    assert result.fills == []
    assert (result.equity == config.initial_cash).all()


def test_open_position_is_marked_to_last_close(candles_factory):
    candles = candles_factory([100.0, 100.0, 100.0], closes=[100.0, 100.0, 120.0])
    result = run_backtest(candles, Scripted([1, 1, 1]), NO_COSTS)

    assert len(result.fills) == 1
    assert result.equity.iloc[-1] == pytest.approx(1200.0)


def test_buy_and_hold_matches_manual_calculation(candles_factory):
    candles = candles_factory([100.0, 120.0, 90.0], closes=[120.0, 90.0, 150.0])

    assert buy_and_hold_return(candles, fee_rate=0.0, slippage=0.0) == pytest.approx(0.5)
    with_costs = 150.0 / (100.0 * 1.001 * 1.0025) - 1
    assert buy_and_hold_return(candles, fee_rate=0.0025, slippage=0.001) == pytest.approx(
        with_costs
    )


def test_max_drawdown():
    equity = pd.Series([100.0, 120.0, 90.0, 110.0, 60.0, 130.0])
    assert max_drawdown(equity) == pytest.approx(60.0 / 120.0 - 1)


def test_metrics_for_one_winning_and_one_losing_trade(candles_factory):
    #                         buy    sell   buy    sell
    candles = candles_factory([100.0, 100.0, 200.0, 200.0, 100.0, 100.0])
    result = run_backtest(candles, Scripted([1, 0, 1, 0, 0, 0]), NO_COSTS)
    metrics = compute_metrics(result, candles, periods_per_year=8760)

    assert metrics.trades == 2
    assert metrics.win_rate == pytest.approx(0.5)
    assert metrics.total_return == pytest.approx(0.0)  # doubled, then halved
    assert metrics.max_drawdown == pytest.approx(-0.5)
    assert metrics.fees_paid == 0.0
    # Held during candles 1 and 3 only.
    assert metrics.exposure == pytest.approx(2 / 6)


def test_exit_threshold_must_be_below_entry():
    with pytest.raises(ValueError):
        BacktestConfig(entry_threshold=0.2, exit_threshold=0.2)
