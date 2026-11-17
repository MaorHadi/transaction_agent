import math

import numpy as np
import pandas as pd
import pytest

from transaction_agent.portfolio import (
    BreakoutBasket,
    best_single_coin,
    perfect_foresight_log_growth,
)
from transaction_agent.portfolio.signals import breakout_signal
from transaction_agent.strategies import ChannelBreakout

HOLD = ChannelBreakout.HOLD


def random_candles(count: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.03, count)))
    open_ = np.concatenate([[100.0], close[:-1]])
    index = pd.date_range("2022-01-01", periods=count, freq="1D", tz="UTC")
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) * 1.01,
            "low": np.minimum(open_, close) * 0.99,
            "close": close,
        },
        index=index,
    )


def table(**columns: list[float]) -> pd.DataFrame:
    count = len(next(iter(columns.values())))
    index = pd.date_range("2024-01-01", periods=count, freq="1D", tz="UTC")
    return pd.DataFrame(columns, index=index)


def test_signal_table_matches_the_rule_applied_one_candle_at_a_time():
    candles = random_candles(200, seed=3)
    rule = ChannelBreakout(window=20)
    one_at_a_time = [rule.signal(candles.iloc[: i + 1]) for i in range(len(candles))]

    assert list(breakout_signal(candles, 20)) == one_at_a_time
    assert {1.0, -1.0, HOLD} <= set(one_at_a_time)  # the data exercises every state


def test_basket_buys_on_breakout_holds_then_sells():
    closes = table(A=[1.0, 1.0, 1.0, 1.0], B=[1.0, 1.0, 1.0, 1.0])
    signals = pd.DataFrame(
        {"A": [1.0, HOLD, HOLD, -1.0], "B": [HOLD, HOLD, 1.0, 1.0]}, index=closes.index
    )
    basket = BreakoutBasket(signals)
    nothing = pd.Series({"A": 0.0, "B": 0.0})
    holding_a = pd.Series({"A": 0.5, "B": 0.0})

    # Day 0: A breaks out; B is only "hold", which does not open a position.
    assert basket.target_weights(closes.iloc[:1], nothing).to_dict() == {"A": 0.5, "B": 0.0}
    # Day 1: "hold" keeps A only because it is already held.
    assert basket.target_weights(closes.iloc[:2], holding_a).to_dict() == {"A": 0.5, "B": 0.0}
    assert basket.target_weights(closes.iloc[:2], nothing).to_dict() == {"A": 0.0, "B": 0.0}
    # Day 3: A's sell signal closes it; B has broken out.
    assert basket.target_weights(closes, holding_a).to_dict() == {"A": 0.0, "B": 0.5}


def test_basket_sizes_each_slice_by_forecast_volatility():
    closes = table(calm=[1.0], wild=[1.0], unknown=[1.0])
    signals = pd.DataFrame({"calm": [1.0], "wild": [1.0], "unknown": [1.0]}, index=closes.index)
    forecast = pd.DataFrame(
        {"calm": [0.3], "wild": [1.2], "unknown": [np.nan]}, index=closes.index
    )
    basket = BreakoutBasket(signals, forecast, target=0.6)
    weights = basket.target_weights(closes, pd.Series(0.0, index=closes.columns))

    assert weights["calm"] == pytest.approx(1 / 3)  # calmer than the target: full slice
    assert weights["wild"] == pytest.approx(1 / 3 * 0.6 / 1.2)  # twice the target: half a slice
    assert weights["unknown"] == 0.0  # no forecast, no position


def test_basket_needs_forecast_and_target_together():
    signals = pd.DataFrame({"A": [1.0]})
    with pytest.raises(ValueError):
        BreakoutBasket(signals, volatility=signals)
    with pytest.raises(ValueError):
        BreakoutBasket(signals, target=0.5)


def test_best_single_coin():
    opens = table(A=[10.0, 12.0, 20.0], B=[10.0, 30.0, 15.0])
    coin, growth = best_single_coin(opens, cost_rate=0.0)
    assert (coin, growth) == ("A", 2.0)


def test_perfect_foresight_rides_every_rise_and_sits_out_every_fall():
    # A doubles then halves; B halves then triples. Best path: A, then B.
    opens = table(A=[10.0, 20.0, 10.0], B=[10.0, 5.0, 15.0])

    assert math.exp(perfect_foresight_log_growth(opens, cost_rate=0.0)) == pytest.approx(6.0)
    # With costs: buy A, sell A, buy B, sell B.
    with_costs = 6.0 * (0.99 / 1.01) ** 2
    assert math.exp(perfect_foresight_log_growth(opens, cost_rate=0.01)) == pytest.approx(
        with_costs
    )


def test_perfect_foresight_stays_in_cash_when_nothing_rises_enough_to_pay_costs():
    opens = table(A=[10.0, 10.05, 10.0], B=[10.0, 9.0, 8.0])
    assert perfect_foresight_log_growth(opens, cost_rate=0.01) == 0.0


def test_perfect_foresight_keeps_a_position_instead_of_paying_to_switch():
    # Both rise 10% a day; switching between them would only cost fees.
    opens = table(A=[10.0, 11.0, 12.1], B=[10.0, 11.0, 12.1])
    expected = 1.21 * 0.99 / 1.01
    assert math.exp(perfect_foresight_log_growth(opens, cost_rate=0.01)) == pytest.approx(expected)
