import pytest

from transaction_agent.strategies import MovingAverageCrossover


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
