"""Historical candles from Alpaca. Alpaca is the project's only market API."""

import time
from datetime import UTC, datetime, timedelta

import pandas as pd
from alpaca.common.exceptions import APIError
from alpaca.data.enums import Adjustment
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from transaction_agent.config import Settings
from transaction_agent.data.store import CANDLE_COLUMNS
from transaction_agent.timeframes import is_crypto, to_timedelta

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
# Free Alpaca accounts may not query the most recent minutes of stock data.
_STOCK_DATA_DELAY = timedelta(minutes=20)


def fetch_candles(
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
    settings: Settings | None = None,
) -> pd.DataFrame:
    """Download OHLCV candles, indexed by UTC timestamp, oldest first.

    Crypto pairs (BTC/USD) work on every timeframe. Stocks and ETFs (SPY) are
    daily only, need API keys, and are adjusted for splits and dividends so that
    a price series reflects what a holder actually earned.
    """
    if timeframe not in _ALPACA_TIMEFRAMES:
        raise ValueError(f"Unsupported timeframe {timeframe!r}")
    settings = settings or Settings()
    crypto = is_crypto(symbol)
    if crypto:
        # Crypto market data needs no keys; they only raise the rate limit.
        client = CryptoHistoricalDataClient(settings.alpaca_api_key, settings.alpaca_secret_key)
    else:
        if timeframe != "1d":
            raise ValueError("Stocks are supported on daily candles only")
        if not settings.alpaca_api_key:
            raise ValueError("Stock data needs Alpaca API keys in .env")
        client = StockHistoricalDataClient(settings.alpaca_api_key, settings.alpaca_secret_key)
        end = min(end, datetime.now(UTC) - _STOCK_DATA_DELAY)

    chunk_span = to_timedelta(timeframe) * _CANDLES_PER_CHUNK
    chunks = []
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + chunk_span, end)
        window = {
            "symbol_or_symbols": symbol,
            "timeframe": _ALPACA_TIMEFRAMES[timeframe],
            "start": chunk_start,
            "end": chunk_end,
        }
        if crypto:
            frame = _with_retries(lambda w=window: client.get_crypto_bars(CryptoBarsRequest(**w)))
        else:
            request = StockBarsRequest(**window, adjustment=Adjustment.ALL)
            frame = _with_retries(lambda r=request: client.get_stock_bars(r))
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


def _with_retries(fetch) -> pd.DataFrame:
    """Run one request, waiting and retrying when Alpaca rate-limits it."""
    for wait in (*_RETRY_WAITS, None):
        try:
            return fetch().df
        except APIError as error:
            if error.status_code != 429 or wait is None:
                raise
            time.sleep(wait)
    raise AssertionError("unreachable")
