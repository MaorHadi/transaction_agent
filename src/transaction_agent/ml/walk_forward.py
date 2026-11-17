"""Walk-forward training: every prediction comes from a model that never saw that period."""

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import brier_score_loss, roc_auc_score

from transaction_agent.ml.features import build_features
from transaction_agent.ml.labels import forward_labels

CALIBRATION_BINS = (0.0, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 1.0)


@dataclass(frozen=True)
class Predictions:
    """Out-of-sample probabilities that the price rises over the horizon."""

    probabilities: pd.Series  # one per candle in the tested span
    labels: pd.Series  # what actually happened; NaN where the future is not yet known
    first_test_index: int  # position in the candles where the tested span begins
    folds: int


@dataclass(frozen=True)
class ClassificationReport:
    samples: int
    base_rate: float  # how often the price actually rose
    accuracy: float  # share of correct calls when calling "up" at probability > 0.5
    roc_auc: float  # 0.5 is coin-flipping, 1.0 is perfect ranking
    brier: float  # mean squared error of the probabilities; lower is better
    brier_baseline: float  # the score from always predicting the base rate
    calibration: pd.DataFrame  # per probability bin: count, mean predicted, actual rate


def walk_forward_predict(
    candles: pd.DataFrame,
    model_factory: Callable[[], ClassifierMixin],
    train_size: int,
    test_size: int,
    horizon: int,
    min_gain: float = 0.0,
) -> Predictions:
    """Train on a rolling window, predict the next window, slide forward, repeat.

    Each fold trains a fresh model on `train_size` candles and predicts the
    following `test_size` candles. The last `horizon + 1` training candles are
    dropped (purged) because their labels depend on prices inside the test window.
    """
    if train_size <= horizon + 1:
        raise ValueError("train_size must be larger than horizon + 1")
    features = build_features(candles)
    labels = forward_labels(candles, horizon, min_gain)
    usable = features.notna().all(axis=1)

    pieces = []
    train_start = 0
    while train_start + train_size + 1 <= len(candles):
        test_start = train_start + train_size
        test_end = min(test_start + test_size, len(candles))

        train_rows = slice(train_start, test_start - horizon - 1)
        train_mask = usable.iloc[train_rows] & labels.iloc[train_rows].notna()
        x_train = features.iloc[train_rows][train_mask]
        y_train = labels.iloc[train_rows][train_mask]

        test_mask = usable.iloc[test_start:test_end]
        x_test = features.iloc[test_start:test_end][test_mask]

        if len(x_test) and y_train.nunique() == 2:
            model = model_factory()
            model.fit(x_train, y_train)
            up_column = list(model.classes_).index(1.0)
            pieces.append(
                pd.Series(model.predict_proba(x_test)[:, up_column], index=x_test.index)
            )
        train_start += test_size
    if not pieces:
        raise ValueError("Not enough candles for one training and one test window")
    probabilities = pd.concat(pieces).rename("probability")
    return Predictions(
        probabilities=probabilities,
        labels=labels.loc[probabilities.index],
        first_test_index=train_size,
        folds=len(pieces),
    )


def classification_report(predictions: Predictions) -> ClassificationReport:
    """Score the probabilities against what actually happened."""
    known = predictions.labels.notna()
    actual = predictions.labels[known]
    predicted = predictions.probabilities[known]
    base_rate = float(actual.mean())

    bins = pd.cut(predicted, CALIBRATION_BINS, include_lowest=True)
    calibration = pd.DataFrame(
        {
            "count": predicted.groupby(bins, observed=True).size(),
            "predicted": predicted.groupby(bins, observed=True).mean(),
            "actual": actual.groupby(bins, observed=True).mean(),
        }
    )
    return ClassificationReport(
        samples=len(actual),
        base_rate=base_rate,
        accuracy=float(((predicted > 0.5) == (actual == 1.0)).mean()),
        roc_auc=float(roc_auc_score(actual, predicted)),
        brier=float(brier_score_loss(actual, predicted)),
        brier_baseline=float(((actual - base_rate) ** 2).mean()),
        calibration=calibration,
    )
