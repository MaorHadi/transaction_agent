import numpy as np
import pandas as pd
import pytest

from transaction_agent.analysis import feature_correlations
from transaction_agent.volatility import (
    annualized_volatility,
    har_inputs,
    input_correlations,
    realized_variance,
    score_forecasts,
    walk_forward_har,
)


def hourly_candles(closes: list[float]) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=len(closes), freq="1h", tz="UTC")
    return pd.DataFrame({"close": closes}, index=index)


def clustered_variance(days: int, seed: int = 0) -> pd.Series:
    """Daily variance whose logarithm drifts slowly, so calm and wild spells persist."""
    rng = np.random.default_rng(seed)
    level = np.zeros(days)
    for day in range(1, days):
        level[day] = 0.97 * level[day - 1] + rng.normal(0, 0.15)
    noisy = level + rng.normal(0, 0.5, days)
    index = pd.date_range("2021-01-01", periods=days, freq="1D", tz="UTC")
    return pd.Series(np.exp(noisy - 7), index=index)


def test_realized_variance_sums_squared_hourly_returns():
    # Day one: flat. Day two: alternates between 100 and 110 every hour.
    closes = [100.0] * 24 + [110.0 if hour % 2 == 0 else 100.0 for hour in range(24)]
    variance = realized_variance(hourly_candles(closes))

    assert len(variance) == 1  # the flat day has zero variance and is dropped
    assert variance.iloc[0] == pytest.approx(24 * np.log(1.1) ** 2)


def test_days_with_too_few_candles_are_dropped():
    closes = list(100 + np.arange(30) % 2)  # one full day and six hours of the next
    variance = realized_variance(hourly_candles(closes))

    assert list(variance.index) == [pd.Timestamp("2024-01-01", tz="UTC")]


def test_inputs_use_only_that_day_and_earlier():
    variance = clustered_variance(120)
    full = har_inputs(variance)

    pd.testing.assert_frame_equal(full.iloc[:80], har_inputs(variance.iloc[:80]))
    assert full["last_week"].iloc[6] == pytest.approx(np.log(variance.iloc[:7].mean()))


def test_forecasts_are_made_by_models_fitted_on_earlier_days():
    variance = clustered_variance(600)
    forecasts = walk_forward_har(variance, train_size=300, test_size=100)

    assert forecasts.index[0] == variance.index[300]
    assert forecasts.index.is_unique
    # A forecast made on day d is scored against day d + 1.
    day = forecasts.index[10]
    following = variance.index[variance.index.get_loc(day) + 1]
    assert forecasts.loc[day, "actual"] == pytest.approx(np.log(variance.loc[following]))
    assert forecasts.loc[day, "naive"] == pytest.approx(np.log(variance.loc[day]))


def test_changing_later_days_does_not_change_earlier_forecasts():
    variance = clustered_variance(600)
    altered = variance.copy()
    altered.iloc[450:] *= 10
    original = walk_forward_har(variance, 300, 100)
    changed = walk_forward_har(altered, 300, 100)

    # Forecasts in the first test window (days 300 to 399) are fitted on days before 300.
    early = original.index[:100]
    assert original.loc[early, "har"].equals(changed.loc[early, "har"])


def test_har_beats_naive_when_volatility_clusters():
    scores = score_forecasts(walk_forward_har(clustered_variance(1500), 365, 90))

    assert scores.loc["naive", "vs_naive"] == pytest.approx(0.0)
    assert scores.loc["har", "vs_naive"] > 0.2
    assert scores.loc["har", "error"] < scores.loc["naive", "error"]


def test_input_correlations_are_strong_for_clustered_and_absent_for_random_variance():
    clustered, band = input_correlations(clustered_variance(1500))
    assert (clustered > 5 * band).all()

    rng = np.random.default_rng(1)
    index = pd.date_range("2021-01-01", periods=1500, freq="1D", tz="UTC")
    random, band = input_correlations(pd.Series(np.exp(rng.normal(-7, 0.5, 1500)), index=index))
    assert (random.abs() < 2 * band).all()


def test_annualized_volatility():
    daily_variance = 0.03**2  # a 3% daily move
    result = annualized_volatility(pd.Series([np.log(daily_variance)]))
    assert result.iloc[0] == pytest.approx(0.03 * np.sqrt(365))


def test_feature_correlations_report_direction_and_size():
    rng = np.random.default_rng(2)
    count = 6000
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, count)))
    open_ = np.concatenate([[100.0], close[:-1]])
    index = pd.date_range("2022-01-01", periods=count, freq="4h", tz="UTC")
    candles = pd.DataFrame(
        {
            "open": open_,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.uniform(1, 10, count),
        },
        index=index,
    )
    table, band = feature_correlations(candles, horizon=6)

    assert list(table.columns) == ["direction", "size"]
    assert len(table) == 14  # one row per feature
    assert band == pytest.approx(2 / np.sqrt((count - 106) / 6), rel=0.01)
    # A random walk has no predictable direction: nearly all inputs sit inside the band.
    assert (table["direction"].abs() < band).mean() > 0.8
