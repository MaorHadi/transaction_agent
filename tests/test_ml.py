import numpy as np
import pandas as pd
import pytest

from transaction_agent.backtest import BacktestConfig, run_backtest
from transaction_agent.ml import (
    build_features,
    classification_report,
    forward_labels,
    make_model,
    walk_forward_predict,
)
from transaction_agent.strategies import ModelSignal


def random_candles(count: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, count)))
    open_ = np.concatenate([[100.0], close[:-1]])
    index = pd.date_range("2022-01-01", periods=count, freq="4h", tz="UTC")
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) * 1.002,
            "low": np.minimum(open_, close) * 0.998,
            "close": close,
            "volume": rng.uniform(1, 10, count),
        },
        index=index,
    )


class SpyModel:
    """Records what it was trained on and always predicts 70%."""

    trained_on: list[pd.DatetimeIndex] = []

    def fit(self, x, y):
        SpyModel.trained_on.append(x.index)
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, x):
        return np.tile([0.3, 0.7], (len(x), 1))


def test_features_do_not_look_ahead():
    candles = random_candles(300)
    full = build_features(candles)
    truncated = build_features(candles.iloc[:200])

    # Adding later candles must not change any earlier row.
    pd.testing.assert_frame_equal(full.iloc[:200], truncated)
    assert full.iloc[150:].notna().all().all()


def test_labels_describe_the_trade_the_bot_could_make(candles_factory):
    # Bought at the next open, sold two candles after that.
    candles = candles_factory([100.0, 100.0, 105.0, 110.0, 90.0, 80.0])
    labels = forward_labels(candles, horizon=2)

    assert labels.iloc[0] == 1.0  # buy at 100 (candle 1), sell at 110 (candle 3)
    assert labels.iloc[1] == 0.0  # buy at 105, sell at 90
    assert labels.iloc[2] == 0.0  # buy at 110, sell at 80
    assert labels.iloc[3:].isna().all()  # the future is not known yet


def test_min_gain_raises_the_bar(candles_factory):
    candles = candles_factory([100.0, 100.0, 101.0, 102.0])
    assert forward_labels(candles, horizon=1).iloc[0] == 1.0
    assert forward_labels(candles, horizon=1, min_gain=0.02).iloc[0] == 0.0


def test_training_never_uses_prices_from_the_test_window():
    candles = random_candles(700)
    horizon = 6
    SpyModel.trained_on = []
    predictions = walk_forward_predict(
        candles, SpyModel, train_size=300, test_size=100, horizon=horizon
    )

    assert predictions.folds == len(SpyModel.trained_on) == 4
    for fold, trained_index in enumerate(SpyModel.trained_on):
        test_start = 300 + fold * 100
        last_trained = candles.index.get_loc(trained_index[-1])
        # The label of the last training row ends before the test window opens.
        assert last_trained + 1 + horizon < test_start
    assert predictions.probabilities.index[0] == candles.index[300]
    assert predictions.probabilities.index.is_unique
    assert (predictions.probabilities == 0.7).all()


def test_report_scores_probabilities_against_outcomes():
    candles = random_candles(700)
    predictions = walk_forward_predict(candles, SpyModel, 300, 100, horizon=6)
    report = classification_report(predictions)

    assert report.samples == predictions.labels.notna().sum()
    assert report.roc_auc == pytest.approx(0.5)  # a constant prediction ranks nothing
    assert report.accuracy == pytest.approx(report.base_rate)  # always calls "up"
    assert report.calibration["count"].sum() == report.samples


def test_model_signal_maps_probability_to_strength(candles_factory):
    candles = candles_factory([100.0, 100.0, 100.0])
    signal = ModelSignal(pd.Series([0.8, 0.5], index=candles.index[:2]))

    assert signal.signal(candles.iloc[:1]) == pytest.approx(0.6)
    assert signal.signal(candles.iloc[:2]) == pytest.approx(0.0)
    assert signal.signal(candles) == 0.0  # no prediction for this candle
    assert ModelSignal.threshold_for(0.6) == pytest.approx(0.2)


@pytest.mark.parametrize("name", ["logistic", "random_forest", "gradient_boosting"])
def test_real_models_run_end_to_end(name):
    candles = random_candles(600, seed=1)
    predictions = walk_forward_predict(
        candles, lambda: make_model(name), train_size=350, test_size=125, horizon=6
    )

    assert predictions.probabilities.between(0, 1).all()
    config = BacktestConfig(entry_threshold=ModelSignal.threshold_for(0.55))
    result = run_backtest(
        candles, ModelSignal(predictions.probabilities), config, predictions.first_test_index
    )
    assert len(result.equity) == 250


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError):
        make_model("neural_net")
