import pandas as pd
import pytest

from transaction_agent.backtest import BacktestConfig, compute_metrics, run_backtest
from transaction_agent.execution import Side
from transaction_agent.risk import DrawdownGuard, RiskLimits
from transaction_agent.strategies import Strategy


class Scripted(Strategy):
    name = "scripted"

    def __init__(self, strengths: list[float]):
        self.strengths = strengths

    def signal(self, history: pd.DataFrame) -> float:
        return self.strengths[len(history) - 1]


NO_COSTS = BacktestConfig(initial_cash=1000.0, fee_rate=0.0, slippage=0.0)


def candles_with_lows(opens, lows, closes):
    index = pd.date_range("2024-01-01", periods=len(opens), freq="4h", tz="UTC")
    highs = [max(o, c) for o, c in zip(opens, closes, strict=True)]
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": 1.0}, index=index
    )


def test_stop_loss_sells_inside_the_candle_at_the_stop_price():
    # Buy at candle 1's open (100). Candle 2 dips to 85: the 10% stop sells at 90.
    candles = candles_with_lows(
        opens=[100.0, 100.0, 100.0, 95.0],
        lows=[100.0, 99.0, 85.0, 95.0],
        closes=[100.0, 100.0, 95.0, 95.0],
    )
    result = run_backtest(candles, Scripted([1, 1, 1, 1]), NO_COSTS, risk=RiskLimits(stop_loss=0.1))

    buy, sell = result.fills
    assert buy.side is Side.BUY
    assert sell.reason == "stop_loss"
    assert sell.timestamp == candles.index[2]
    assert sell.price == pytest.approx(90.0)
    assert result.equity.iloc[-1] == pytest.approx(900.0)
    assert compute_metrics(result, candles, 2190).stop_outs == 1


def test_stop_loss_fills_at_the_open_when_the_candle_gaps_below_the_stop():
    candles = candles_with_lows(
        opens=[100.0, 100.0, 80.0, 80.0],
        lows=[100.0, 99.0, 75.0, 80.0],
        closes=[100.0, 100.0, 80.0, 80.0],
    )
    result = run_backtest(candles, Scripted([1, 1, 1, 1]), NO_COSTS, risk=RiskLimits(stop_loss=0.1))

    assert result.fills[1].price == pytest.approx(80.0)


def test_no_reentry_after_a_stop_until_the_signal_resets():
    opens = [100.0] * 8
    lows = [100.0, 100.0, 85.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    candles = candles_with_lows(opens, lows, closes=opens)
    # Signal stays bullish through the stop-out, resets on candle 4, returns on candle 5.
    strengths = [1, 1, 1, 1, 0, 1, 1, 1]
    result = run_backtest(candles, Scripted(strengths), NO_COSTS, risk=RiskLimits(stop_loss=0.1))

    buys = [f.timestamp for f in result.fills if f.side is Side.BUY]
    assert buys == [candles.index[1], candles.index[6]]


def test_position_fraction_limits_the_size_of_each_buy():
    candles = candles_with_lows([100.0] * 3, [100.0] * 3, [100.0] * 3)
    result = run_backtest(
        candles, Scripted([1, 1, 1]), NO_COSTS, risk=RiskLimits(position_fraction=0.25)
    )

    assert result.fills[0].quantity == pytest.approx(2.5)  # 250 of 1000 at price 100


def test_kill_switch_sells_and_stops_trading():
    opens = [100.0, 100.0, 100.0, 70.0, 70.0, 70.0, 70.0]
    closes = [100.0, 100.0, 70.0, 70.0, 70.0, 70.0, 70.0]
    candles = candles_with_lows(opens, lows=closes, closes=closes)
    # Bullish throughout: without the kill switch the bot would stay in or re-enter.
    result = run_backtest(candles, Scripted([1] * 7), NO_COSTS, risk=RiskLimits(max_drawdown=0.2))

    assert result.halted_at == candles.index[2]
    assert [f.reason for f in result.fills] == ["signal", "kill_switch"]
    assert result.fills[1].timestamp == candles.index[3]
    assert result.equity.iloc[-1] == pytest.approx(700.0)


def test_drawdown_guard_measures_from_the_peak():
    guard = DrawdownGuard(0.25)
    assert not guard.tripped(100.0)
    assert not guard.tripped(200.0)
    assert not guard.tripped(151.0)
    assert guard.tripped(150.0)


def test_no_limits_by_default():
    assert not DrawdownGuard(None).tripped(0.0)
    assert RiskLimits().stop_price(100.0) is None


@pytest.mark.parametrize(
    "kwargs", [{"position_fraction": 0}, {"stop_loss": 1.5}, {"max_drawdown": -0.1}]
)
def test_rejects_invalid_limits(kwargs):
    with pytest.raises(ValueError):
        RiskLimits(**kwargs)
