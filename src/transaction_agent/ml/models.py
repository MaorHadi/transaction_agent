"""The scikit-learn classifiers the project compares."""

from sklearn.base import ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODELS = ("logistic", "random_forest", "gradient_boosting")


def make_model(name: str) -> ClassifierMixin:
    """Build an untrained classifier whose probabilities are calibrated.

    Calibration fits a correction on data the base model was not trained on, split
    in time order, so that a predicted 60% is right about 60% of the time on the
    training period. Whether that holds on later data is checked separately.
    """
    if name == "logistic":
        base = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    elif name == "random_forest":
        # Large leaves keep the trees from memorising individual candles.
        base = RandomForestClassifier(
            n_estimators=200, min_samples_leaf=25, n_jobs=-1, random_state=0
        )
    elif name == "gradient_boosting":
        base = HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=150, random_state=0
        )
    else:
        raise ValueError(f"Unknown model {name!r}; choose one of {', '.join(MODELS)}")
    return CalibratedClassifierCV(base, method="sigmoid", cv=TimeSeriesSplit(n_splits=3))
