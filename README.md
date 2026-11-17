# transaction_agent

A crypto trading bot built around an honest backtester. It trades BTC/USD and ETH/USD on hourly-to-daily candles through the [Alpaca](https://alpaca.markets) API, and only enters a trade when its signal is strong enough.

This is a learning project. The goal is sound engineering and truthful results, not a claim of profit. The first strategy loses money, and the numbers below say so.

## Status

| Phase | State |
|---|---|
| Data download and local cache (Alpaca → Parquet) | Done |
| Backtester with fees, slippage and no lookahead | Done |
| Moving-average crossover strategy | Done |
| Mean-reversion strategy, walk-forward evaluation | Planned |
| Risk limits, paper trading on Alpaca | Planned |
| ML models (scikit-learn) with calibrated confidence | Planned |

## Quick start

Requires Python 3.11 or newer.

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows; use `source .venv/bin/activate` elsewhere
pip install -e ".[dev]"

transaction-agent download --symbol BTC/USD --timeframe 1h
transaction-agent backtest --strategy ma_crossover --threshold 0.6
pytest
```

Downloading historical crypto candles needs no API key. To raise Alpaca's rate limit, copy `.env.example` to `.env` and add your keys.

## How the backtester stays honest

- **No lookahead.** At each candle the strategy sees only candles that have already closed. Any order it triggers fills at the *next* candle's open.
- **Costs on every trade.** Each fill pays a proportional fee (default 0.25%, Alpaca's crypto taker fee for accounts under $100k monthly volume) and slippage (default 0.05%) against the trader.
- **Compared with doing nothing.** Every result is shown next to buy-and-hold over the same period, with the same costs.
- **Tested against known answers.** The tests use hand-built candle series, including one that changes future prices and checks that past trades do not move.

## First results

Moving-average crossover (20 and 100 candles), BTC/USD hourly, 2024-11-17 to 2026-11-17, starting cash 10,000. Parameters are the defaults and were not tuned.

| Entry threshold | Strategy return | Buy-and-hold | Max drawdown | Trades | Win rate | Fees paid |
|---|---|---|---|---|---|---|
| 0.3 | -62.7% | -29.2% | -66.8% | 90 | 31.1% | 2,946 |
| 0.6 | -45.0% | -29.2% | -48.4% | 65 | 33.8% | 2,393 |
| 0.9 | -48.4% | -29.2% | -49.6% | 52 | 26.9% | 1,864 |

The strategy does worse than holding at every threshold. Fees alone consume 19% to 29% of the starting cash. A higher threshold cuts the number of trades and the fees, but does not make the strategy profitable. This is the baseline that later strategies and models have to beat.

The downloaded data has 2 missing hourly candles out of 17,520; they are left as gaps, not filled.

## Layout

```
src/transaction_agent/
  data/        download and cache candles, integrity checks
  strategies/  Strategy base class and implementations
  backtest/    engine and metrics
  execution/   Broker interface and the simulated broker
  cli.py       download / backtest commands
tests/
```

Strategies return a signal strength between -1 and 1 and never place orders themselves. Trading logic talks to a `Broker` interface, so the same strategy will run unchanged in backtest, paper and live trading.

## Disclaimer

Nothing here is financial advice. Trading crypto can lose all the money you put in.
