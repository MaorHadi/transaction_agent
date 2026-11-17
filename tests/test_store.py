import pandas as pd
import pytest

from transaction_agent.data import store


def test_clean_candles(candles_factory):
    report = store.validate_candles(candles_factory([1.0] * 10), "1h")
    assert report.clean
    assert report.rows == 10


def test_counts_gaps_and_missing_candles(candles_factory):
    candles = candles_factory([1.0] * 10).drop(index=candles_factory([1.0] * 10).index[[3, 4, 7]])
    report = store.validate_candles(candles, "1h")
    assert report.gaps == 2
    assert report.missing_candles == 3
    assert not report.clean


def test_counts_duplicates(candles_factory):
    candles = candles_factory([1.0] * 5)
    report = store.validate_candles(pd.concat([candles, candles.iloc[[2]]]).sort_index(), "1h")
    assert report.duplicates == 1
    assert report.gaps == 0


def test_rejects_unsorted(candles_factory):
    with pytest.raises(ValueError):
        store.validate_candles(candles_factory([1.0] * 5).iloc[::-1], "1h")


def test_save_and_load_round_trip(candles_factory, tmp_path):
    candles = candles_factory([1.0, 2.0, 3.0])
    path = store.candle_path(tmp_path, "BTC/USD", "1h")
    store.save_candles(candles, path)

    assert path.name == "BTC-USD_1h.parquet"
    pd.testing.assert_frame_equal(store.load_candles(path), candles, check_freq=False)
