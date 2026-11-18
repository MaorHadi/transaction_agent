# transaction_agent

A research tool that tests trading rules on historical market data from
[Alpaca](https://alpaca.markets), with trading fees included, and reports honestly
whether they beat simply buying and holding.

The project started as an attempt to build a trading bot that recognises price
patterns in crypto. The main result is negative: no rule or model tested here
predicted price direction well enough to beat buy-and-hold after fees. What did
hold up is that volatility is forecastable, and that sizing a position by it
reduces losses.

THIS IS AN EXPERIMENT AND SHOULD NOT BE ENTRUSTED WITH REAL MONEY NOTHING IN HERE IS A FINANCIAL ADVICE

## Findings

All figures include Alpaca's crypto fee (0.25% per trade) and 0.05% slippage, and
come from data up to November 2026. "Unseen data" means the settings were chosen
on an earlier period and scored on a later one.

### 1. Price direction was not predictable

| Idea | Result on unseen data |
|---|---|
| Moving-average crossover | Lost to buy-and-hold; removed |
| Mean reversion (buying dips) | Lost to buy-and-hold; removed |
| Momentum rotation and pair rotation between coins | Lost to buy-and-hold; removed |
| Direction classifiers (logistic regression, random forest, gradient boosting) | ROC AUC of 0.48 to 0.51, where 0.50 is a coin flip; removed |
| Confidence threshold on the breakout signal | Signal strength had no relation to the trade's outcome; removed |

A correlation check explains the classifier result. Of 14 candidate inputs, none
related to the *direction* of the next move beyond what chance produces. The
volatility-type inputs did relate to the *size* of the next move, with rank
correlations of +0.22 to +0.29.

A scan of BTC daily candles for commonly claimed patterns (momentum over 3 to 90
days, weekday effects, bounces after crash days, returns above long moving
averages) found all of them inside the noise band. The only clear relation was
that large moves follow large moves.

### 2. Volatility was forecastable

A HAR model (a linear regression on the last day's, week's and month's realized
variance, computed from hourly candles) forecasts the next day's volatility. On
BTC, ETH, LINK and LTC it removed 25% to 31% of the squared error of the naive
forecast "tomorrow equals today".

### 3. Trend following cut losses but did not add return

The channel breakout rule buys at a new N-candle high and sells when the price
falls back to the middle of the channel.

| Test on unseen data | Breakout | Buy-and-hold |
|---|---|---|
| BTC, daily candles | +158.7% | +287.7% |
| BTC, 4-hour candles | -8.0% | +35.7% |
| 13 coins in equal slices, 2024 to 2026 | +5.2% (worst fall -33.5%) | -40.2% (worst fall -77.4%) |
| 30 US-listed ETFs in equal slices, 2017 to 2026 | +19.8% (worst fall -8.6%) | +154.1% (worst fall -30.2%) |

It protected the account in a falling market (the 13 coins) and missed most of
the gain in rising ones. The best channel length also changed between timeframes,
which is a sign of an unstable rule.

### 4. Volatility targeting was the one result that beat holding

Volatility targeting holds less of a coin when its recent volatility is high and
more when it is calm. It makes no prediction about direction.

| BTC, 4-hour candles, unseen data, 2022 to 2026 | Return | Worst fall | Sharpe |
|---|---|---|---|
| Volatility targeting | +89.8% | -50.3% | 0.53 |
| Buy-and-hold | +37.0% | -67.3% | 0.38 |

This is a single asset over a single period, so it is promising rather than
proven.

### Conclusion

With public price data and simple methods, direction is not predictable, which
matches the published research. Risk is. For most people a broad index fund held
for many years is the better use of money than trading these rules: the 30 ETFs
simply held made +154% where the best rule here unforunatly only made +20%.

## How the testing works

- No lookahead. A rule sees a candle only after it has closed, and its order
  fills at the next candle's open.
- Costs on every trade. 0.25% fee plus 0.05% slippage for crypto; slippage
  only for stocks and ETFs, which have no commission on Alpaca.
- Walk-forward evaluation. Settings are chosen on a training window and scored
  on the window that follows, which the rule has never seen. The windows then
  roll forward and the scores are joined.
- Baselines. Every result is shown next to buy-and-hold and staying in cash.
- Noise band. A correlation over *n* observations counts as real only outside
  ±2/√n, the range unrelated data falls in about 95% of the time.
- Theoretical maximum. The `basket` command also reports what perfect daily
  foresight would have earned, to show how far any real rule is from the ceiling.

## Setup

Requires Python 3.11 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; on macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
```

Copy `.env.example` to `.env`. Crypto data downloads without API keys, but slowly
because of rate limits. Stock and ETF data needs the keys of a free Alpaca
account:

```
API_ALPACA_ENDPOINT=https://paper-api.alpaca.markets/v2
API_ALPACA_KEY=        # the key ID
API_ALPACA_CLIENT=     # the secret key
DATA_DIR=data
```

## Usage

Download candles first; they are cached as Parquet files in `data/`. Timeframes
are `1h`, `4h` and `1d` for crypto, and `1d` only for stocks and ETFs.

```bash
transaction-agent download --symbol BTC/USD --timeframe 4h --start 2021-01-01
transaction-agent download --symbol BTC/USD --timeframe 1d --start 2021-01-01
transaction-agent download --symbol BTC/USD --timeframe 1h --start 2021-01-01
transaction-agent download --symbol SPY --timeframe 1d --start 2016-01-01
```

Then run any of the tests:

| Command | What it does |
|---|---|
| `transaction-agent backtest --window 50` | Runs the breakout rule with one channel length over the whole history |
| `transaction-agent walkforward` | Chooses the channel length on past data and scores it on unseen data |
| `transaction-agent portfolio --symbols BTC/USD,ETH/USD` | Walk-forward volatility targeting on one or more coins |
| `transaction-agent basket --universe crypto` | Breakout across 13 coins on daily candles, with and without forecast sizing (needs `1d` and `1h` candles for each coin) |
| `transaction-agent basket --universe etfs` | The same rule across 30 ETFs (needs `1d` candles for each) |
| `transaction-agent volatility --symbol BTC/USD` | Scores the HAR volatility forecast (needs `1h` candles) |
| `transaction-agent correlations --horizon 6` | Shows how each candidate input relates to the direction and size of the next move |

`backtest`, `walkforward` and `correlations` take `--symbol` and `--timeframe`
(default `BTC/USD` on `4h`). `backtest` and `walkforward` also accept `--fee`,
`--slippage`, `--stop-loss`, `--max-drawdown` and `--position-fraction`. Run any
command with `--help` for the full list.

## Project layout

```
src/transaction_agent/
  data/        download from Alpaca, Parquet cache, gap and duplicate checks
  strategies/  the channel breakout rule
  backtest/    single-asset engine, metrics, walk-forward
  execution/   broker interface and the simulated broker used in backtests
  risk/        stop-loss, position size, drawdown kill switch
  portfolio/   multi-asset engine, volatility targeting, breakout basket,
               perfect-foresight ceiling
  volatility/  HAR volatility forecast (scikit-learn)
  analysis/    candidate inputs and their correlation with future moves
  cli.py       the commands above
tests/         56 tests, built on hand-made candle series with known answers
```

## Development

```bash
pytest
ruff check .
```

## Limits

- Alpaca's crypto history starts in 2021, so the tests cover one or two market
  cycles. Slow patterns cannot be tested on this data.
- Many ideas were tried, which raises the chance that a good-looking result is
  luck. Walk-forward scoring reduces this but does not remove it.
- Slippage is a fixed percentage, not a model of the order book.
- Paper and live trading through Alpaca were planned, didnt built this at the end.
