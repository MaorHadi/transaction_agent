import numpy as np
import pandas as pd
import pytest

from transaction_agent.portfolio import (
    HoldEqual,
    PortfolioCandidate,
    PortfolioConfig,
    VolatilityTarget,
    WeightStrategy,
    run_portfolio_backtest,
    summarize,
    walk_forward_portfolio,
)
from transaction_agent.strategies import ChannelBreakout

NO_COSTS = PortfolioConfig(initial_cash=1000.0, fee_rate=0.0, slippage=0.0, min_trade_fraction=0.0)


def table(**columns: list[float]) -> pd.DataFrame:
    count = len(next(iter(columns.values())))
    index = pd.date_range("2024-01-01", periods=count, freq="1D", tz="UTC")
    return pd.DataFrame(columns, index=index)


class Fixed(WeightStrategy):
    """Always asks for the same weights and records how much history it was shown."""

    name = "fixed"

    def __init__(self, **weights: float):
        self.weights = pd.Series(weights)
        self.seen: list[int] = []

    def target_weights(self, closes, current):
        self.seen.append(len(closes))
        return self.weights


def test_trades_happen_at_the_next_open_and_strategy_sees_no_future():
    opens = table(A=[10.0, 20.0, 20.0, 20.0])
    closes = table(A=[10.0, 20.0, 20.0, 40.0])
    strategy = Fixed(A=1.0)
    result = run_portfolio_backtest(opens, closes, strategy, NO_COSTS)

    # Decided at candle 0's close, bought at candle 1's open (20), not at 10.
    assert list(result.equity) == pytest.approx([1000.0, 1000.0, 1000.0, 2000.0])
    assert strategy.seen == [1, 2, 3]


def test_hold_equal_buys_once_and_matches_manual_calculation():
    opens = table(A=[10.0, 10.0, 10.0], B=[100.0, 100.0, 100.0])
    closes = table(A=[10.0, 10.0, 20.0], B=[100.0, 100.0, 50.0])
    result = run_portfolio_backtest(opens, closes, HoldEqual(), NO_COSTS)

    # 500 in each: A doubles to 1000, B halves to 250.
    assert result.equity.iloc[-1] == pytest.approx(1250.0)
    assert result.trades == 2


def test_costs_are_charged_on_every_trade():
    opens = table(A=[10.0, 10.0, 10.0])
    closes = table(A=[10.0, 10.0, 10.0])
    config = PortfolioConfig(initial_cash=1000.0, fee_rate=0.01, slippage=0.01)
    result = run_portfolio_backtest(opens, closes, Fixed(A=1.0), config)

    assert result.equity.iloc[-1] == pytest.approx(1000.0 / 1.02)
    assert result.costs_paid == pytest.approx(1000.0 - 1000.0 / 1.02)


def test_small_adjustments_are_skipped():
    opens = table(A=[10.0, 10.0, 10.0, 10.0])
    closes = table(A=[10.0, 10.0, 10.2, 10.2])
    config = PortfolioConfig(initial_cash=1000.0, fee_rate=0.0, slippage=0.0)
    # Target 50%; a 2% price rise moves the weight only slightly, below the 5% band.
    result = run_portfolio_backtest(opens, closes, Fixed(A=0.5), config)

    assert result.trades == 1


def test_rejects_weights_above_one():
    opens = table(A=[10.0, 10.0, 10.0])
    with pytest.raises(ValueError):
        run_portfolio_backtest(opens, opens, Fixed(A=1.5), NO_COSTS)


def test_volatility_target_holds_less_of_the_wilder_coin():
    rng = np.random.default_rng(0)
    calm = 100 * np.exp(np.cumsum(rng.normal(0, 0.005, 60)))
    wild = 100 * np.exp(np.cumsum(rng.normal(0, 0.05, 60)))
    closes = table(calm=list(calm), wild=list(wild))
    strategy = VolatilityTarget(target=0.3, window=30, periods_per_year=365)
    weights = strategy.target_weights(closes, pd.Series(0.0, index=closes.columns))

    wild_volatility = np.log(closes["wild"].iloc[-31:]).diff().dropna().std() * np.sqrt(365)
    assert weights["calm"] == pytest.approx(0.5)  # calm enough to be fully invested
    assert weights["wild"] == pytest.approx(0.5 * 0.3 / wild_volatility)
    assert weights["wild"] < weights["calm"]


def test_channel_breakout_has_three_states(candles_factory):
    strategy = ChannelBreakout(window=3)
    opens = [10.0, 12.0, 14.0, 16.0]  # previous three candles span 10 to 16

    def signal(last_close: float) -> float:
        return strategy.signal(candles_factory(opens, closes=[12.0, 14.0, 16.0, last_close]))

    assert signal(20.0) == 1.0  # new high: buy
    assert signal(16.0) == 1.0  # exactly at the top counts
    assert signal(15.9) == ChannelBreakout.HOLD  # near the top is not a buy
    assert signal(13.1) == ChannelBreakout.HOLD  # upper half: keep holding
    assert signal(13.0) == -1.0  # back to the middle: sell
    assert signal(9.0) == -1.0
    assert strategy.signal(candles_factory(opens[:3])) == 0.0  # not enough history


def test_channel_breakout_trades_only_on_new_highs_and_mid_range_exits(candles_factory):
    from transaction_agent.backtest import BacktestConfig, run_backtest

    # A 10-to-11 range, new high (buy), drifts in the upper half (hold), falls to mid-range (sell).
    closes = [10.0, 11.0, 10.0, 12.0, 11.8, 11.9, 10.5, 10.5]
    candles = candles_factory([10.0] + closes[:-1], closes=closes)
    config = BacktestConfig(initial_cash=1000.0, fee_rate=0.0, slippage=0.0)
    result = run_backtest(candles, ChannelBreakout(window=3), config)

    assert [fill.side.value for fill in result.fills] == ["buy", "sell"]
    assert result.fills[0].timestamp == candles.index[4]  # the open after the new high
    assert result.fills[1].timestamp == candles.index[7]


def test_walk_forward_chains_test_windows_and_reports_a_benchmark():
    prices = [100.0 + i for i in range(13)]
    opens, closes = table(A=prices), table(A=prices[1:] + prices[-1:])
    result = walk_forward_portfolio(
        opens, closes, [PortfolioCandidate("all in", Fixed(A=1.0))], 4, 4, NO_COSTS
    )

    assert [fold.chosen for fold in result.folds] == ["all in", "all in"]
    expected = (1 + result.folds[0].test_return) * (1 + result.folds[1].test_return)
    assert result.equity.iloc[-1] == pytest.approx(expected)
    assert result.benchmark.index[0] == closes.index[4]
    assert summarize(result.benchmark, 365).total_return > 0
