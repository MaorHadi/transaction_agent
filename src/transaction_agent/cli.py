import argparse
from datetime import UTC, datetime, timedelta

from transaction_agent import timeframes
from transaction_agent.backtest import BacktestConfig, compute_metrics, run_backtest
from transaction_agent.config import Settings
from transaction_agent.data import store
from transaction_agent.strategies import MovingAverageCrossover

STRATEGIES = ("ma_crossover",)


def _parse_date(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=UTC)


def _add_market_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--symbol", default="BTC/USD")
    parser.add_argument("--timeframe", default="1h", choices=timeframes.SUPPORTED)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="transaction-agent", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    download = commands.add_parser("download", help="Download candles from Alpaca")
    _add_market_args(download)
    download.add_argument("--start", type=_parse_date, help="YYYY-MM-DD, default 2 years ago")
    download.add_argument("--end", type=_parse_date, help="YYYY-MM-DD, default now")

    backtest = commands.add_parser("backtest", help="Backtest a strategy on cached candles")
    _add_market_args(backtest)
    backtest.add_argument("--strategy", default="ma_crossover", choices=STRATEGIES)
    backtest.add_argument("--threshold", type=float, default=BacktestConfig.entry_threshold)
    backtest.add_argument("--fee", type=float, default=BacktestConfig.fee_rate)
    backtest.add_argument("--slippage", type=float, default=BacktestConfig.slippage)
    backtest.add_argument("--cash", type=float, default=BacktestConfig.initial_cash)
    backtest.add_argument("--fast", type=int, default=20, help="Fast average length, in candles")
    backtest.add_argument("--slow", type=int, default=100, help="Slow average length, in candles")
    return parser


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
    print(
        f"  duplicates: {report.duplicates}, gaps: {report.gaps} "
        f"({report.missing_candles} missing candles)"
    )


def _backtest(args: argparse.Namespace, settings: Settings) -> None:
    candles = store.load_candles(store.candle_path(settings.data_dir, args.symbol, args.timeframe))
    strategy = MovingAverageCrossover(fast=args.fast, slow=args.slow)
    config = BacktestConfig(
        initial_cash=args.cash,
        fee_rate=args.fee,
        slippage=args.slippage,
        entry_threshold=args.threshold,
    )
    result = run_backtest(candles, strategy, config)
    metrics = compute_metrics(result, candles, timeframes.periods_per_year(args.timeframe))

    print(f"{strategy.name} on {args.symbol} {args.timeframe}, entry threshold {args.threshold}")
    print(f"{candles.index[0]:%Y-%m-%d} to {candles.index[-1]:%Y-%m-%d}, {len(candles)} candles")
    rows = [
        ("Strategy return", f"{metrics.total_return:+.2%}"),
        ("Buy-and-hold return", f"{metrics.buy_and_hold_return:+.2%}"),
        ("Sharpe (annualised)", f"{metrics.sharpe:.2f}"),
        ("Max drawdown", f"{metrics.max_drawdown:.2%}"),
        ("Completed trades", f"{metrics.trades}"),
        ("Win rate", f"{metrics.win_rate:.1%}"),
        ("Fees paid", f"{metrics.fees_paid:,.2f}"),
        ("Time in market", f"{metrics.exposure:.1%}"),
    ]
    for label, value in rows:
        print(f"  {label:<22}{value:>12}")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    settings = Settings()
    try:
        if args.command == "download":
            _download(args, settings)
        else:
            _backtest(args, settings)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"error: {error}") from None


if __name__ == "__main__":
    main()
