"""Forecasting tomorrow's volatility with the HAR model.

HAR (heterogeneous autoregressive) predicts the next day's realized variance from
three numbers: the variance of the last day, the average of the last week and the
average of the last month. It is a plain linear regression, here on logarithms.
Realized variance for a day is the sum of squared hourly returns within it.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

INPUTS = ("last_day", "last_week", "last_month")
METHODS = ("har", "naive", "average")  # naive: tomorrow = today; average: last month's mean


def realized_variance(hourly: pd.DataFrame, min_candles: int = 20) -> pd.Series:
    """Daily realized variance from hourly candles. Days with too few candles are dropped."""
    squared = (np.log(hourly["close"]).diff() ** 2).dropna()
    by_day = squared.groupby(squared.index.floor("D"))
    variance = by_day.sum()[by_day.count() >= min_candles]
    return variance[variance > 0].rename("realized_variance")


def annualized_volatility(log_variance: pd.Series) -> pd.Series:
    """Convert a daily log variance into volatility per year, e.g. 0.6 for 60%."""
    return np.sqrt(np.exp(log_variance) * 365)


def har_inputs(variance: pd.Series) -> pd.DataFrame:
    """The three model inputs for each day, using that day and earlier days only."""
    return pd.DataFrame(
        {
            "last_day": np.log(variance),
            "last_week": np.log(variance.rolling(7).mean()),
            "last_month": np.log(variance.rolling(30).mean()),
        }
    )


def _target(variance: pd.Series) -> pd.Series:
    return np.log(variance).shift(-1).rename("actual")


def input_correlations(variance: pd.Series) -> tuple[pd.Series, float]:
    """Rank correlation of each input with the next day's variance, and the noise band.

    A correlation inside plus or minus the band is what unrelated data would show
    by chance about 95% of the time.
    """
    table = har_inputs(variance).join(_target(variance)).dropna()
    correlations = table[list(INPUTS)].corrwith(table["actual"], method="spearman")
    return correlations, 2 / np.sqrt(len(table))


def walk_forward_har(
    variance: pd.Series, train_size: int = 365, test_size: int = 90
) -> pd.DataFrame:
    """Forecast each day's next-day log variance with a model fitted on earlier days only.

    Returns one row per forecast with the actual outcome and three forecasts:
    `har`, `naive` (tomorrow equals today) and `average` (the last month's mean).
    A row's index is the day the forecast was made; it predicts the following day.
    """
    inputs = har_inputs(variance)
    actual = _target(variance)
    usable = inputs.notna().all(axis=1) & actual.notna()

    pieces = []
    train_start = 0
    while train_start + train_size + 1 <= len(variance):
        test_start = train_start + train_size
        test_end = min(test_start + test_size, len(variance))
        # The last training row's outcome is the day before the first forecast is made.
        train = usable.iloc[train_start : test_start - 1]
        test = usable.iloc[test_start:test_end]
        train_days, test_days = train[train].index, test[test].index
        if len(train_days) > len(INPUTS) and len(test_days):
            model = LinearRegression().fit(inputs.loc[train_days], actual.loc[train_days])
            pieces.append(
                pd.DataFrame(
                    {
                        "actual": actual.loc[test_days],
                        "har": model.predict(inputs.loc[test_days]),
                        "naive": inputs.loc[test_days, "last_day"],
                        "average": inputs.loc[test_days, "last_month"],
                    }
                )
            )
        train_start += test_size
    if not pieces:
        raise ValueError("Not enough days for one training and one test window")
    return pd.concat(pieces)


def score_forecasts(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Compare each forecasting method with what actually happened.

    Columns: `correlation` with the actual log variance; `error`, the typical miss
    in annualized volatility (0.10 means ten percentage points); and
    `vs_naive`, the share of the naive method's squared error removed
    (positive is better than naive, zero is the same).
    """
    actual_volatility = annualized_volatility(forecasts["actual"])
    naive_error = ((forecasts["naive"] - forecasts["actual"]) ** 2).mean()
    rows = {}
    for method in METHODS:
        squared_error = ((forecasts[method] - forecasts["actual"]) ** 2).mean()
        rows[method] = {
            "correlation": forecasts[method].corr(forecasts["actual"]),
            "error": (annualized_volatility(forecasts[method]) - actual_volatility).abs().mean(),
            "vs_naive": 1 - squared_error / naive_error,
        }
    return pd.DataFrame(rows).T
