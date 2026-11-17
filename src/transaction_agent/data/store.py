"""Local Parquet cache for candles, plus integrity checks."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from transaction_agent.timeframes import to_timedelta

CANDLE_COLUMNS = ("open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class CandleReport:
    rows: int
    duplicates: int
    gaps: int  # places where one or more candles are missing
    missing_candles: int

    @property
    def clean(self) -> bool:
        return self.duplicates == 0 and self.gaps == 0


def candle_path(data_dir: Path, symbol: str, timeframe: str) -> Path:
    return data_dir / f"{symbol.replace('/', '-')}_{timeframe}.parquet"


def save_candles(candles: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    candles.to_parquet(path)


def load_candles(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No candles at {path}. Run the download command first.")
    return pd.read_parquet(path)


def validate_candles(candles: pd.DataFrame, timeframe: str) -> CandleReport:
    """Count duplicate timestamps and missing candles. Requires a sorted index."""
    if not candles.index.is_monotonic_increasing:
        raise ValueError("Candles must be sorted by timestamp")
    step = to_timedelta(timeframe)
    duplicates = int(candles.index.duplicated().sum())
    deltas = candles.index.unique().to_series().diff().dropna()
    oversized = deltas[deltas > step]
    missing = int((oversized / step).round().sum() - len(oversized))
    return CandleReport(
        rows=len(candles), duplicates=duplicates, gaps=len(oversized), missing_candles=missing
    )
