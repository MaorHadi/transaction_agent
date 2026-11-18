"""Download candles from Alpaca and evaluate trading strategies on them."""

import argparse
import math
from datetime import UTC, datetime, timedelta

import pandas as pd

from transaction_agent import timeframes
from transaction_agent.backtest import (
    BacktestConfig,
    Candidate,
    compute_metrics,
    run_backtest,
    walk_forward,
)
from transaction_agent.config import Settings
from transaction_agent.data import store
from transaction_agent.portfolio import (
    BreakoutBasket,
    HoldEqual,
    PortfolioCandidate,
    PortfolioConfig,
    VolatilityTarget,
    best_single_coin,
    breakout_signals,
    perfect_foresight_log_growth,
    run_portfolio_backtest,
    summarize,
    volatility_forecasts,
    walk_forward_portfolio,
)
from transaction_agent.portfolio.engine import Summary, align
from transaction_agent.risk import RiskLimits
from transaction_agent.strategies import ChannelBreakout

# The "do nothing" baseline: never trading keeps the account exactly where it started.
CASH_RETURN = 0.0
BREAKOUT_WINDOWS = (20, 50, 100)
VOLATILITY_TARGETS = (0.4, 0.6, 0.8)  # yearly volatility aimed for per coin
# Coins with complete daily history on Alpaca back to 2021 (MKR left out: delisted in 2025).
BASKET_SYMBOLS = (
    "BTC/USD", "ETH/USD", "LINK/USD", "LTC/USD", "BCH/USD", "AAVE/USD", "UNI/USD",
    "DOGE/USD", "AVAX/USD", "GRT/USD", "SUSHI/USD", "YFI/USD", "BAT/USD",
)  # fmt: skip
# Large, liquid US-listed funds covering stock indexes, sectors, countries, bonds and
# commodities. Funds are used instead of single stocks so the list is not a hand-pick
# of today's winners. All have traded since before 2016, where Alpaca's data begins.
ETF_SYMBOLS = (
    "SPY", "QQQ", "IWM", "DIA", "EFA", "EEM", "EWJ", "EWZ", "FXI", "EWG", "EWU",
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "VNQ",
    "TLT", "IEF", "LQD", "HYG", "GLD", "SLV", "USO", "DBC",
)  # fmt: skip
# Per universe: its symbols and the fee charged on each trade. Alpaca charges 0.25%
# on crypto and no commission on US stocks and funds.
UNIVERSES = {
    "crypto": (BASKET_SYMBOLS, PortfolioConfig.fee_rate),
    "etfs": (ETF_SYMBOLS, 0.0),
}


def _breakout_candidates() -> list[Candidate]:
    """The settings walk-forward chooses between: breakout's window length only."""
    return [
        Candidate(f"window={window}", ChannelBreakout(window), BacktestConfig.entry_threshold)
        for window in BREAKOUT_WINDOWS
    ]


def _volatility_target_candidates(per_day: float, per_year: float) -> list[PortfolioCandidate]:
    """Settings the multi-coin walk-forward chooses between. Windows are in days."""
    return [
        PortfolioCandidate(
            f"target={target:.0%} window={days}d",
            VolatilityTarget(target, round(days * per_day), per_year),
            rebalance_every=round(per_day),
        )
        for target in (0.3, 0.5, 0.7)
        for days in (14, 30)
    ]


def _parse_date(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=UTC)


def _add_market_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--symbol", default="BTC/USD")
    parser.add_argument("--timeframe", default="4h", choices=timeframes.SUPPORTED)


def _add_account_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--fee", type=float, default=BacktestConfig.fee_rate)
    parser.add_argument("--slippage", type=float, default=BacktestConfig.slippage)
    parser.add_argument("--cash", type=float, default=BacktestConfig.initial_cash)
    parser.add_argument("--stop-loss", type=float, help="Sell this far below entry, e.g. 0.05")
    parser.add_argument(
        "--max-drawdown", type=float, help="Stop trading at this loss from the peak, e.g. 0.25"
    )
    parser.add_argument(
        "--position-fraction", type=float, default=1.0, help="Share of cash used per trade"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="transaction-agent", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    download = commands.add_parser("download", help="Download candles from Alpaca")
    _add_market_args(download)
    download.add_argument("--start", type=_parse_date, help="YYYY-MM-DD, default 2 years ago")
    download.add_argument("--end", type=_parse_date, help="YYYY-MM-DD, default now")

    backtest = commands.add_parser("backtest", help="Backtest the breakout rule on cached candles")
    _add_market_args(backtest)
    backtest.add_argument("--window", type=int, default=50, help="Channel length, in candles")
    _add_account_args(backtest)

    forward = commands.add_parser(
        "walkforward", help="Choose breakout's window on past data, score it on unseen data"
    )
    _add_market_args(forward)
    _add_account_args(forward)
    forward.add_argument("--train-days", type=int, default=365)
    forward.add_argument("--test-days", type=int, default=90)

    portfolio = commands.add_parser(
        "portfolio", help="Walk-forward volatility targeting across one or more coins"
    )
    portfolio.add_argument(
        "--symbols", default="BTC/USD,ETH/USD", help="Comma-separated, e.g. BTC/USD,ETH/USD"
    )
    portfolio.add_argument("--timeframe", default="4h", choices=timeframes.SUPPORTED)
    portfolio.add_argument("--train-days", type=int, default=365)
    portfolio.add_argument("--test-days", type=int, default=90)
    portfolio.add_argument("--fee", type=float, default=PortfolioConfig.fee_rate)
    portfolio.add_argument("--slippage", type=float, default=PortfolioConfig.slippage)

    basket = commands.add_parser(
        "basket", help="Breakout across many coins on daily candles, sized by forecast volatility"
    )
    basket.add_argument("--universe", default="crypto", choices=tuple(UNIVERSES))
    basket.add_argument("--symbols", help="Comma-separated; overrides the universe's own list")
    basket.add_argument("--train-days", type=int, default=365)
    basket.add_argument("--test-days", type=int, default=180)
    basket.add_argument("--fee", type=float, help="Default: the universe's usual fee")
    basket.add_argument("--slippage", type=float, default=PortfolioConfig.slippage)

    volatility = commands.add_parser(
        "volatility", help="Forecast next-day volatility from hourly candles and score it"
    )
    volatility.add_argument("--symbol", default="BTC/USD")
    volatility.add_argument("--train-days", type=int, default=365)
    volatility.add_argument("--test-days", type=int, default=90)

    correlations = commands.add_parser(
        "correlations", help="Show how each candidate input relates to the future price move"
    )
    _add_market_args(correlations)
    correlations.add_argument("--horizon", type=int, default=6, help="Candles ahead")
    return parser


def _load(args: argparse.Namespace, settings: Settings) -> pd.DataFrame:
    return store.load_candles(store.candle_path(settings.data_dir, args.symbol, args.timeframe))


def _per_year(args: argparse.Namespace) -> float:
    """Candles per year for the command's symbol and timeframe."""
    return timeframes.periods_per_year(args.timeframe, timeframes.is_crypto(args.symbol))


def _config(args: argparse.Namespace) -> BacktestConfig:
    return BacktestConfig(initial_cash=args.cash, fee_rate=args.fee, slippage=args.slippage)


def _risk(args: argparse.Namespace) -> RiskLimits:
    return RiskLimits(
        position_fraction=args.position_fraction,
        stop_loss=args.stop_loss,
        max_drawdown=args.max_drawdown,
    )


def _span(candles: pd.DataFrame) -> str:
    return f"{candles.index[0]:%Y-%m-%d} to {candles.index[-1]:%Y-%m-%d}, {len(candles)} candles"


def _download(args: argparse.Namespace, settings: Settings) -> None:
    from transaction_agent.data.alpaca import fetch_candles  # imports the Alpaca SDK

    end = args.end or datetime.now(UTC)
    start = args.start or end - timedelta(days=730)
    candles = fetch_candles(args.symbol, args.timeframe, start, end, settings)
    path = store.candle_path(settings.data_dir, args.symbol, args.timeframe)
    store.save_candles(candles, path)
    report = store.validate_candles(candles, args.timeframe)
    print(f"Saved {report.rows} candles to {path}")
    print(f"  from {candles.index[0]} to {candles.index[-1]}")
    if timeframes.is_crypto(args.symbol):
        print(
            f"  duplicates: {report.duplicates}, gaps: {report.gaps} "
            f"({report.missing_candles} missing candles)"
        )
    else:
        # Stock markets close on weekends and holidays, so gaps in time are normal.
        print(f"  duplicates: {report.duplicates}")


def _backtest(args: argparse.Namespace, settings: Settings) -> None:
    candles = _load(args, settings)
    strategy = ChannelBreakout(window=args.window)
    result = run_backtest(candles, strategy, _config(args), risk=_risk(args))
    metrics = compute_metrics(result, candles, _per_year(args))

    print(f"{strategy.name} (window {args.window}) on {args.symbol} {args.timeframe}")
    print(_span(candles))
    rows = [
        ("Strategy return", f"{metrics.total_return:+.2%}"),
        ("Buy-and-hold return", f"{metrics.buy_and_hold_return:+.2%}"),
        ("Stay-in-cash return", f"{CASH_RETURN:+.2%}"),
        ("Sharpe (annualised)", f"{metrics.sharpe:.2f}"),
        ("Max drawdown", f"{metrics.max_drawdown:.2%}"),
        ("Completed trades", f"{metrics.trades}"),
        ("Win rate", f"{metrics.win_rate:.1%}"),
        ("Fees paid", f"{metrics.fees_paid:,.2f}"),
        ("Time in market", f"{metrics.exposure:.1%}"),
        ("Stop-loss exits", f"{metrics.stop_outs}"),
    ]
    for label, value in rows:
        print(f"  {label:<22}{value:>12}")
    if result.halted_at is not None:
        print(f"  Kill switch stopped trading on {result.halted_at:%Y-%m-%d}")


def _walkforward(args: argparse.Namespace, settings: Settings) -> None:
    candles = _load(args, settings)
    per_day = _per_year(args) / 365
    result = walk_forward(
        candles,
        _breakout_candidates(),
        train_size=round(args.train_days * per_day),
        test_size=round(args.test_days * per_day),
        config=_config(args),
        risk=_risk(args),
    )

    print(f"breakout on {args.symbol} {args.timeframe}, walk-forward")
    print(f"train {args.train_days} days, test {args.test_days} days; {_span(candles)}")
    print(
        f"  {'test period':<24}{'chosen on training data':<28}"
        f"{'train':>8}{'test':>8}{'hold':>8}{'trades':>8}"
    )
    for fold in result.folds:
        period = f"{fold.test_start:%Y-%m-%d} to {fold.test_end:%m-%d}"
        print(
            f"  {period:<24}{fold.chosen:<28}{fold.train_return:>+8.1%}"
            f"{fold.test_return:>+8.1%}{fold.test_buy_and_hold:>+8.1%}{fold.test_trades:>8}"
        )
    print(f"  Unseen-data return: {result.total_return:+.1%} over {result.trades} trades")
    print(f"  Buy-and-hold over the same span: {result.buy_and_hold_return:+.1%}")
    print(f"  Staying in cash over the same span: {CASH_RETURN:+.1%}")


def _portfolio(args: argparse.Namespace, settings: Settings) -> None:
    symbols = [symbol.strip() for symbol in args.symbols.split(",")]
    opens, closes = align(
        {
            symbol: store.load_candles(store.candle_path(settings.data_dir, symbol, args.timeframe))
            for symbol in symbols
        }
    )
    per_year = timeframes.periods_per_year(args.timeframe)
    per_day = per_year / 365
    result = walk_forward_portfolio(
        opens,
        closes,
        _volatility_target_candidates(per_day, per_year),
        train_size=round(args.train_days * per_day),
        test_size=round(args.test_days * per_day),
        config=PortfolioConfig(fee_rate=args.fee, slippage=args.slippage),
    )
    strategy, hold = summarize(result.equity, per_year), summarize(result.benchmark, per_year)

    print(f"volatility targeting on {', '.join(symbols)} {args.timeframe}, walk-forward")
    print(f"train {args.train_days} days, test {args.test_days} days; {_span(closes)}")
    header = f"  {'test period':<24}{'chosen on training data':<28}"
    print(f"{header}{'test':>8}{'hold':>8}{'trades':>7}")
    for fold in result.folds:
        period = f"{fold.test_start:%Y-%m-%d} to {fold.test_end:%m-%d}"
        print(
            f"  {period:<24}{fold.chosen:<28}{fold.test_return:>+8.1%}"
            f"{fold.test_buy_and_hold:>+8.1%}{fold.test_trades:>7}"
        )
    print(f"  Unseen data, {result.trades} trades:")
    print(f"  {'':<16}{'return':>10}{'max drawdown':>14}{'Sharpe':>8}")
    cash = Summary(total_return=CASH_RETURN, sharpe=0.0, max_drawdown=0.0)
    rows = (("strategy", strategy), ("hold equal parts", hold), ("stay in cash", cash))
    for label, summary in rows:
        print(
            f"  {label:<16}{summary.total_return:>+10.1%}"
            f"{summary.max_drawdown:>14.1%}{summary.sharpe:>8.2f}"
        )


def _basket(args: argparse.Namespace, settings: Settings) -> None:
    universe_symbols, universe_fee = UNIVERSES[args.universe]
    symbols = (
        [symbol.strip() for symbol in args.symbols.split(",")]
        if args.symbols
        else list(universe_symbols)
    )
    fee = universe_fee if args.fee is None else args.fee
    crypto = all(timeframes.is_crypto(symbol) for symbol in symbols)

    def load(timeframe: str) -> dict[str, pd.DataFrame]:
        return {
            symbol: store.load_candles(store.candle_path(settings.data_dir, symbol, timeframe))
            for symbol in symbols
        }

    daily = load("1d")
    opens, closes = align(daily)
    signals = {window: breakout_signals(daily, window) for window in BREAKOUT_WINDOWS}
    plain = [
        PortfolioCandidate(f"window={window}", BreakoutBasket(signals[window]))
        for window in BREAKOUT_WINDOWS
    ]
    sized = []
    if crypto:
        # Forecast sizing needs hourly candles around the clock, which only crypto has.
        forecasts = volatility_forecasts(load("1h")).dropna()
        # Trade only on days where every coin has a price and a volatility forecast.
        days = closes.index.intersection(forecasts.index)
        opens, closes = opens.loc[days], closes.loc[days]
        sized = [
            PortfolioCandidate(
                f"window={window} target={target:.0%}",
                BreakoutBasket(signals[window], forecasts, target),
            )
            for window in BREAKOUT_WINDOWS
            for target in VOLATILITY_TARGETS
        ]
    days = closes.index

    per_year = timeframes.periods_per_year("1d", crypto)
    train_size = round(args.train_days * per_year / 365)
    test_size = round(args.test_days * per_year / 365)
    cost_rate = fee + args.slippage
    # Slices are a small share of the account, so allow adjustments down to 1% of it.
    config = PortfolioConfig(fee_rate=fee, slippage=args.slippage, min_trade_fraction=0.01)
    results = {}
    if sized:
        results["breakout + forecast sizing"] = walk_forward_portfolio(
            opens, closes, sized, train_size, test_size, config
        )
    results["breakout, full slices"] = walk_forward_portfolio(
        opens, closes, plain, train_size, test_size, config
    )
    main_result = next(iter(results.values()))
    unseen = main_result.equity.index

    print(f"Breakout basket on {len(symbols)} {args.universe}, daily candles, walk-forward")
    print(f"fee {fee:.2%} and slippage {args.slippage:.2%} per trade")
    print(f"train {args.train_days} days, test {args.test_days} days")
    print(f"unseen data: {unseen[0]:%Y-%m-%d} to {unseen[-1]:%Y-%m-%d}, {len(unseen)} days")
    print()
    header = f"  {'test period':<24}{'chosen on training data':<28}"
    print(f"{header}{'test':>8}{'hold':>8}{'trades':>7}")
    for fold in main_result.folds:
        period = f"{fold.test_start:%Y-%m-%d} to {fold.test_end:%m-%d}"
        print(
            f"  {period:<24}{fold.chosen:<28}{fold.test_return:>+8.1%}"
            f"{fold.test_buy_and_hold:>+8.1%}{fold.test_trades:>7}"
        )
    print()
    print(f"  {'on unseen data':<28}{'return':>10}{'worst drop':>12}{'Sharpe':>8}{'trades':>8}")
    rows = [(label, summarize(r.equity, per_year), r.trades) for label, r in results.items()]
    held = summarize(main_result.benchmark, per_year)
    rows.append(("hold all equally", held, len(symbols)))
    rows.append(("stay in cash", Summary(CASH_RETURN, 0.0, 0.0), 0))
    for label, summary, trades in rows:
        print(
            f"  {label:<28}{summary.total_return:>+10.1%}{summary.max_drawdown:>12.1%}"
            f"{summary.sharpe:>8.2f}{trades:>8}"
        )

    unseen_opens = opens.loc[unseen]
    coin, growth = best_single_coin(unseen_opens, cost_rate)
    log_growth = perfect_foresight_log_growth(unseen_opens, cost_rate)
    daily_gain = math.expm1(log_growth / (len(unseen) - 1))
    print()
    print("Theoretical maximum over the same days, knowing the future (not achievable):")
    print(f"  holding only the best one ({coin}): {growth - 1:+.1%}")
    print(
        f"  switching to the best one or cash every day: about 10^{log_growth / math.log(10):.0f} "
        f"times the starting cash ({daily_gain:+.1%} per day on average)"
    )

    print()
    print("Every setting, traded without interruption over all available days")
    print(f"({days[0]:%Y-%m-%d} to {days[-1]:%Y-%m-%d}; no setting is chosen, nothing is fitted):")
    print(f"  {'setting':<28}{'return':>10}{'worst drop':>12}{'Sharpe':>8}{'trades':>8}")
    for candidate in sized + plain:
        result = run_portfolio_backtest(opens, closes, candidate.strategy, config)
        summary = summarize(result.equity, per_year)
        print(
            f"  {candidate.label:<28}{summary.total_return:>+10.1%}"
            f"{summary.max_drawdown:>12.1%}{summary.sharpe:>8.2f}{result.trades:>8}"
        )
    hold = summarize(run_portfolio_backtest(opens, closes, HoldEqual(), config).equity, per_year)
    print(
        f"  {'hold all equally':<28}{hold.total_return:>+10.1%}"
        f"{hold.max_drawdown:>12.1%}{hold.sharpe:>8.2f}{len(symbols):>8}"
    )


def _volatility(args: argparse.Namespace, settings: Settings) -> None:
    from transaction_agent.volatility import (
        input_correlations,
        realized_variance,
        score_forecasts,
        walk_forward_har,
    )

    hourly = store.load_candles(store.candle_path(settings.data_dir, args.symbol, "1h"))
    variance = realized_variance(hourly)
    correlations, band = input_correlations(variance)
    forecasts = walk_forward_har(variance, args.train_days, args.test_days)
    scores = score_forecasts(forecasts)

    print(f"Next-day volatility forecast (HAR) on {args.symbol}, from hourly candles")
    print(f"train {args.train_days} days, test {args.test_days} days")
    print(
        f"unseen data: {forecasts.index[0]:%Y-%m-%d} to {forecasts.index[-1]:%Y-%m-%d}, "
        f"{len(forecasts)} days"
    )
    print()
    print(f"Do the inputs relate to tomorrow's volatility? (noise band: +/-{band:.2f})")
    for name, value in correlations.items():
        verdict = "related" if abs(value) > band else "noise"
        print(f"  {name.replace('_', ' '):<14}{value:>+8.2f}   {verdict}")
    print()
    print("Forecast quality on unseen data:")
    print(f"  {'method':<26}{'correlation':>12}{'typical miss':>14}{'vs naive':>10}")
    labels = {
        "har": "HAR model",
        "naive": "naive: tomorrow = today",
        "average": "last month's average",
    }
    for method, label in labels.items():
        row = scores.loc[method]
        print(
            f"  {label:<26}{row['correlation']:>12.2f}"
            f"{row['error'] * 100:>10.1f} pts{row['vs_naive']:>+10.1%}"
        )
    print("  (typical miss is in points of yearly volatility; vs naive is squared error removed)")


def _correlations(args: argparse.Namespace, settings: Settings) -> None:
    from transaction_agent.analysis import feature_correlations

    candles = _load(args, settings)
    table, band = feature_correlations(candles, args.horizon)

    print(f"Candidate inputs on {args.symbol} {args.timeframe}, against the move over the next")
    print(f"{args.horizon} candles. {_span(candles)}")
    print(f"Noise band: +/-{band:.3f}. Anything inside it is what an unrelated input shows.")
    print()
    print(f"  {'input':<20}{'direction':>10}{'':>10}{'size of move':>14}")
    for name, row in table.iterrows():
        direction = "related" if abs(row["direction"]) > band else "noise"
        size = "related" if abs(row["size"]) > band else "noise"
        print(
            f"  {name:<20}{row['direction']:>+10.3f}  {direction:<8}"
            f"{row['size']:>+14.3f}  {size}"
        )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    settings = Settings()
    handlers = {
        "download": _download,
        "backtest": _backtest,
        "walkforward": _walkforward,
        "portfolio": _portfolio,
        "basket": _basket,
        "volatility": _volatility,
        "correlations": _correlations,
    }
    try:
        handlers[args.command](args, settings)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"error: {error}") from None


if __name__ == "__main__":
    main()
