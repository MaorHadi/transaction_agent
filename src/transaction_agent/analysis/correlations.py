"""Check whether each model input has any relation to what the model must predict."""

import numpy as np
import pandas as pd

from transaction_agent.analysis.features import build_features
from transaction_agent.analysis.labels import forward_returns


def feature_correlations(candles: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, float]:
    """Rank correlation of every feature with the future return, and the noise band.

    Two columns: `direction`, the correlation with the forward return itself, and
    `size`, the correlation with its absolute value (how far the price moves,
    whichever way). A correlation inside plus or minus the band is what an
    unrelated input would show by chance about 95% of the time. The band allows
    for forward returns overlapping: only one in `horizon` rows is independent.
    """
    table = build_features(candles).join(forward_returns(candles, horizon)).dropna()
    features = table.drop(columns="forward_return")
    result = pd.DataFrame(
        {
            "direction": features.corrwith(table["forward_return"], method="spearman"),
            "size": features.corrwith(table["forward_return"].abs(), method="spearman"),
        }
    )
    band = 2 / np.sqrt(len(table) / horizon)
    return result.sort_values("direction", key=np.abs, ascending=False), float(band)
