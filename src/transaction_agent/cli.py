"""Download candles from Alpaca and evaluate trading strategies on them."""

import argparse
from dataclasses import replace
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
from transaction_agent.risk import RiskLimits
from transaction_agent.strategies import MeanReversion, MovingAverageCrossover, Strategy

STRATEGIES = ("ma_crossover", "mean_reversion")
SWEEP_THRESHOLDS = (0.2, 0.4, 0.6, 0.8, 1.0)


def _candidates(strategy: str) -> list[Candidate]:
    """The fixed, deliberately small set of settings that walk-forward chooses between."""
    if strategy == "ma_crossover":
        return [
            Candidate(
                f"fast={fast} slow={slow} thr={threshold}",
                MovingAverageCrossover(fast, slow),
                threshold,
            )
            for fast, slow in ((10, 50), (20, 100), (50, 200))
            for threshold in (0.3, 0.6, 0.9)
        ]
    return [
        Candidate(f"window={window} thr={threshold}", MeanReversion(window), threshold)
        for window in (20, 50, 100)
        for threshold in (0.5, 0.75, 1.0)
    ]


def _parse_date(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=UTC)


def _add_market_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--symbol", default="BTC/USD")
    parser.add_argument("--timeframe", default="4h", choices=timeframes.SUPPORTED)


def _add_cost_args(parser: argparse.ArgumentParser) -> None:
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


def _add_strategy_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--strategy", default="ma_crossover", choices=STRATEGIES)
    parser.add_argument("--fast", type=int, default=20, help="ma_crossover: fast average length")
    parser.add_argument("--slow", type=int, default=100, help="ma_crossover: slow average length")
    parser.add_argument("--window", type=int, default=20, help="mean_reversion: lookback length")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="transaction-agent", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    download = commands.add_parser("download", help="Download candles from Alpaca")
    _add_market_args(download)
    download.add_argument("--start", type=_parse_date, help="YYYY-MM-DD, default 2 years ago")
    download.add_argument("--end", type=_parse_date, help="YYYY-MM-DD, default now")

    backtest = commands.add_parser("backtest", help="Backtest a strategy on cached candles")
    _add_market_args(backtest)
    _add_strategy_args(backtest)
    _add_cost_args(backtest)
    backtest.add_argument("--threshold", type=float, default=BacktestConfig.entry_threshold)

    sweep = commands.add_parser("sweep", help="Backtest a strategy across entry thresholds")
    _add_market_args(sweep)
    _add_strategy_args(sweep)
    _add_cost_args(sweep)

    forward = commands.add_parser(
        "walkforward", help="Choose settings on past data, score them on unseen data"
    )
    _add_market_args(forward)
    forward.add_argument("--strategy", default="ma_crossover", choices=STRATEGIES)
    _add_cost_args(forward)
    forward.add_argument("--train-days", type=int, default=180)
    forward.add_argument("--test-days", type=int, default=60)
    return parser


def _load(args: argparse.Namespace, settings: Settings) -> pd.DataFrame:
    return store.load_candles(store.candle_path(settings.data_dir, args.symbol, args.timeframe))


def _strategy(args: argparse.Namespace) -> Strategy:
    if args.strategy == "ma_crossover":
        return MovingAverageCrossover(fast=args.fast, slow=args.slow)
    return MeanReversion(window=args.window)


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
    print(
        f"  duplicates: {report.duplicates}, gaps: {report.gaps} "
        f"({report.missing_candles} missing candles)"
    )


def _backtest(args: argparse.Namespace, settings: Settings) -> None:
    candles = _load(args, settings)
    strategy = _strategy(args)
    config = replace(_config(args), entry_threshold=args.threshold)
    result = run_backtest(candles, strategy, config, risk=_risk(args))
    metrics = compute_metrics(result, candles, timeframes.periods_per_year(args.timeframe))

    print(f"{strategy.name} on {args.symbol} {args.timeframe}, entry threshold {args.threshold}")
    print(_span(candles))
    rows = [
        ("Strategy return", f"{metrics.total_return:+.2%}"),
        ("Buy-and-hold return", f"{metrics.buy_and_hold_return:+.2%}"),
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


def _sweep(args: argparse.Namespace, settings: Settings) -> None:
    candles = _load(args, settings)
    strategy = _strategy(args)
    periods = timeframes.periods_per_year(args.timeframe)

    print(f"{strategy.name} on {args.symbol} {args.timeframe}")
    print(_span(candles))
    print(
        f"  {'threshold':>9}{'return':>10}{'max dd':>10}"
        f"{'trades':>8}{'win rate':>10}{'fees':>10}"
    )
    for threshold in SWEEP_THRESHOLDS:
        config = replace(_config(args), entry_threshold=threshold)
        result = run_backtest(candles, strategy, config, risk=_risk(args))
        metrics = compute_metrics(result, candles, periods)
        print(
            f"  {threshold:>9.1f}{metrics.total_return:>+10.1%}{metrics.max_drawdown:>10.1%}"
            f"{metrics.trades:>8}{metrics.win_rate:>10.1%}{metrics.fees_paid:>10,.0f}"
        )
    print(f"  buy-and-hold: {metrics.buy_and_hold_return:+.1%}")


def _walkforward(args: argparse.Namespace, settings: Settings) -> None:
    candles = _load(args, settings)
    per_day = timeframes.periods_per_year(args.timeframe) / 365
    result = walk_forward(
        candles,
        _candidates(args.strategy),
        train_size=round(args.train_days * per_day),
        test_size=round(args.test_days * per_day),
        config=_config(args),
        risk=_risk(args),
    )

    print(f"{args.strategy} on {args.symbol} {args.timeframe}, walk-forward")
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


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    settings = Settings()
    handlers = {
        "download": _download,
        "backtest": _backtest,
        "sweep": _sweep,
        "walkforward": _walkforward,
    }
    try:
        handlers[args.command](args, settings)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"error: {error}") from None


if __name__ == "__main__":
    main()
