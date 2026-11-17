import pytest

from transaction_agent.strategies import MeanReversion, MovingAverageCrossover


def test_zero_until_enough_history(candles_factory):
    strategy = MovingAverageCrossover(fast=2, slow=5)
    assert strategy.signal(candles_factory([100.0] * 4)) == 0.0


def test_strength_scales_with_gap(candles_factory):
    strategy = MovingAverageCrossover(fast=1, slow=2, full_strength_gap=0.10)
    # closes are 100 then 110: fast avg 110, slow avg 105, gap = 5/105.
    candles = candles_factory([100.0, 100.0], closes=[100.0, 110.0])
    assert strategy.signal(candles) == pytest.approx((5 / 105) / 0.10)


def test_strength_is_clipped(candles_factory):
    strategy = MovingAverageCrossover(fast=1, slow=2, full_strength_gap=0.01)
    assert strategy.signal(candles_factory([1.0, 1.0], closes=[100.0, 200.0])) == 1.0
    assert strategy.signal(candles_factory([1.0, 1.0], closes=[200.0, 100.0])) == -1.0


def test_rejects_bad_lengths():
    with pytest.raises(ValueError):
        MovingAverageCrossover(fast=10, slow=10)


def test_mean_reversion_is_bullish_below_the_average(candles_factory):
    strategy = MeanReversion(window=4, full_strength_z=2.0)
    closes = [100.0, 100.0, 100.0, 96.0]
    # mean 99, population std sqrt(3), z = -3/sqrt(3) = -sqrt(3)
    expected = (3**0.5) / 2.0
    assert strategy.signal(candles_factory(closes, closes=closes)) == pytest.approx(expected)


def test_mean_reversion_is_bearish_above_the_average(candles_factory):
    strategy = MeanReversion(window=4, full_strength_z=1.0)
    closes = [100.0, 100.0, 100.0, 104.0]
    assert strategy.signal(candles_factory(closes, closes=closes)) == -1.0


def test_mean_reversion_flat_prices_and_short_history(candles_factory):
    strategy = MeanReversion(window=4)
    assert strategy.signal(candles_factory([100.0] * 4, closes=[100.0] * 4)) == 0.0
    assert strategy.signal(candles_factory([100.0, 90.0])) == 0.0
