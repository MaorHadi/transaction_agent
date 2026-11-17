"""Historical crypto candles from Alpaca. Alpaca is the project's only market API."""

import time
from datetime import datetime

import pandas as pd
from alpaca.common.exceptions import APIError
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from transaction_agent.config import Settings
from transaction_agent.data.store import CANDLE_COLUMNS
from transaction_agent.timeframes import to_timedelta

_ALPACA_TIMEFRAMES = {
    "1h": TimeFrame(1, TimeFrameUnit.Hour),
    "4h": TimeFrame(4, TimeFrameUnit.Hour),
    "1d": TimeFrame(1, TimeFrameUnit.Day),
}

# Alpaca returns at most 10,000 candles per request. Each chunk stays under that, so a
# download uses as few requests as possible and a rate limit costs one retry, not the lot.
_CANDLES_PER_CHUNK = 9_000
_RETRY_WAITS = (15, 30, 60, 120, 240)  # seconds; the limit without API keys is strict
_KEYLESS_PAUSE = 5  # seconds between requests when no API key is configured


def fetch_candles(
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
    settings: Settings | None = None,
) -> pd.DataFrame:
    """Download OHLCV candles, indexed by UTC timestamp, oldest first."""
    if timeframe not in _ALPACA_TIMEFRAMES:
        raise ValueError(f"Unsupported timeframe {timeframe!r}")
    settings = settings or Settings()
    # Crypto market data needs no keys; they only raise the rate limit.
    client = CryptoHistoricalDataClient(settings.alpaca_api_key, settings.alpaca_secret_key)
    chunk_span = to_timedelta(timeframe) * _CANDLES_PER_CHUNK
    chunks = []
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + chunk_span, end)
        request = CryptoBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=_ALPACA_TIMEFRAMES[timeframe],
            start=chunk_start,
            end=chunk_end,
        )
        frame = _get_bars(client, request)
        if not frame.empty:
            chunks.append(frame.xs(symbol, level="symbol"))
        chunk_start = chunk_end
        if chunk_start < end and not settings.alpaca_api_key:
            time.sleep(_KEYLESS_PAUSE)
    if not chunks:
        raise RuntimeError(f"Alpaca returned no candles for {symbol} {timeframe}")
    frame = pd.concat(chunks)
    frame.index = pd.to_datetime(frame.index, utc=True)
    frame.index.name = "timestamp"
    # Chunk boundaries are inclusive on both sides, so drop the repeated candle.
    frame = frame[~frame.index.duplicated()]
    return frame[list(CANDLE_COLUMNS)].sort_index()


def _get_bars(client: CryptoHistoricalDataClient, request: CryptoBarsRequest) -> pd.DataFrame:
    """Fetch one chunk, waiting and retrying when Alpaca rate-limits the request."""
    for wait in (*_RETRY_WAITS, None):
        try:
            return client.get_crypto_bars(request).df
        except APIError as error:
            if error.status_code != 429 or wait is None:
                raise
            time.sleep(wait)
    raise AssertionError("unreachable")
