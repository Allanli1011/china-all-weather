from __future__ import annotations

import argparse
import csv
import time
from dataclasses import replace
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, List

from .backtest import run_event_backtest
from .features import build_features, build_market_features
from .fetchers import CurlTextClient, FetchError, HKEXFetcher, HTTPTextClient, SFCFetcher, YahooPriceFetcher
from .models import ShortTurnoverReport
from .parsers import (
    ParseError,
    ReportUnavailable,
    parse_daily_quote_short_turnover,
    parse_short_turnover_page,
)
from .price import parse_yahoo_chart
from .reports import build_monitor_report
from .sfc import parse_sfc_positions_csv
from .storage import ShortTurnoverStore
from .strategies import generate_signals


DEFAULT_DB = Path("data/hkex_short_oi.sqlite")
DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_REPORT_DIR = Path("reports")


def main(argv: Iterable[str] = None) -> int:
    parser = argparse.ArgumentParser(description="HKEX short-selling turnover tracker")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_current = subparsers.add_parser("fetch-current", help="Fetch current HKEX short-selling pages")
    fetch_current.add_argument("--market", choices=["main", "gem", "all"], default="all")
    fetch_current.add_argument("--session", choices=["morning", "day", "auto", "all"], default="auto")
    fetch_current.add_argument("--db", type=Path, default=DEFAULT_DB)
    fetch_current.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    fetch_current.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    fetch_current.add_argument("--no-raw", action="store_true")

    fetch_quote = subparsers.add_parser("fetch-daily-quote", help="Fetch HKEX daily quotation short section")
    fetch_quote.add_argument("--date", required=True, help="Trade date, YYYY-MM-DD")
    fetch_quote.add_argument("--market", choices=["main", "gem", "all"], default="all")
    fetch_quote.add_argument("--db", type=Path, default=DEFAULT_DB)
    fetch_quote.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    fetch_quote.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    fetch_quote.add_argument("--no-raw", action="store_true")

    report_cmd = subparsers.add_parser("report", help="Generate a report from local SQLite data")
    report_cmd.add_argument("--db", type=Path, default=DEFAULT_DB)
    report_cmd.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    report_cmd.add_argument("--date", help="Report date, YYYY-MM-DD. Defaults to latest.")

    fetch_sfc = subparsers.add_parser("fetch-sfc-latest", help="Fetch latest SFC weekly short positions CSV")
    fetch_sfc.add_argument("--db", type=Path, default=DEFAULT_DB)
    fetch_sfc.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    fetch_sfc.add_argument("--no-raw", action="store_true")

    backfill = subparsers.add_parser("backfill-hkex", help="Backfill HKEX daily quotation short-selling history")
    backfill.add_argument("--start", required=True, help="Start date, YYYY-MM-DD")
    backfill.add_argument("--end", required=True, help="End date, YYYY-MM-DD")
    backfill.add_argument("--market", choices=["main", "gem", "all"], default="all")
    backfill.add_argument("--db", type=Path, default=DEFAULT_DB)
    backfill.add_argument("--pause", type=float, default=0.25)
    backfill.add_argument("--timeout", type=int, default=10)
    backfill.add_argument("--retries", type=int, default=1)
    backfill.add_argument("--include-weekends", action="store_true")
    backfill.add_argument("--save-raw", action="store_true")
    backfill.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)

    fetch_prices = subparsers.add_parser("fetch-prices", help="Fetch Yahoo OHLCV for HK stocks")
    fetch_prices.add_argument("--start", required=True, help="Start date, YYYY-MM-DD")
    fetch_prices.add_argument("--end", required=True, help="End date, YYYY-MM-DD")
    fetch_prices.add_argument("--codes", help="Comma-separated HKEX 5-digit codes. Defaults to latest top short turnover.")
    fetch_prices.add_argument("--top-n", type=int, default=50)
    fetch_prices.add_argument("--db", type=Path, default=DEFAULT_DB)
    fetch_prices.add_argument("--pause", type=float, default=0.25)
    fetch_prices.add_argument("--timeout", type=int, default=20)
    fetch_prices.add_argument("--retries", type=int, default=0)

    backtest = subparsers.add_parser("backtest", help="Run event-study backtests from local data")
    backtest.add_argument("--start", help="Signal start date, YYYY-MM-DD")
    backtest.add_argument("--end", help="Signal end date, YYYY-MM-DD")
    backtest.add_argument("--horizons", default="1,3,5,10")
    backtest.add_argument("--min-short-value", type=float, default=10_000_000)
    backtest.add_argument("--db", type=Path, default=DEFAULT_DB)
    backtest.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)

    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "fetch-current":
        reports = _fetch_current(args)
        if not reports:
            print("No HKEX reports were loaded.")
            return 1
        path = _save_and_report(reports, args.db, args.report_dir)
        print(f"Saved {sum(len(report.records) for report in reports)} rows to {args.db}")
        print(f"Report: {path}")
        return 0

    if args.command == "fetch-daily-quote":
        reports = _fetch_daily_quote(args)
        if not reports:
            print("No HKEX daily quotation reports were loaded.")
            return 1
        path = _save_and_report(reports, args.db, args.report_dir)
        print(f"Saved {sum(len(report.records) for report in reports)} rows to {args.db}")
        print(f"Report: {path}")
        return 0

    if args.command == "report":
        store = ShortTurnoverStore(args.db)
        target_date = date.fromisoformat(args.date) if args.date else store.latest_trade_date()
        if target_date is None:
            print(f"No local data found in {args.db}")
            return 1
        records = store.load_records(end_date=target_date)
        current_records = [record for record in records if record.trade_date == target_date]
        if not current_records:
            print(f"No local records found for {target_date}")
            return 1
        reports = _reports_from_records(current_records)
        path = _write_report_from_store(store, reports, args.report_dir, target_date)
        print(f"Report: {path}")
        return 0

    if args.command == "fetch-sfc-latest":
        fetched_at = datetime.now()
        client = HTTPTextClient()
        fetcher = SFCFetcher(client)
        try:
            result = fetcher.fetch_latest_positions_csv()
            if not args.no_raw:
                _write_raw(args.raw_dir, "sfc", "positions", result.text, fetched_at)
            records = parse_sfc_positions_csv(result.text, source_url=result.url, fetched_at=fetched_at)
        except (FetchError, ParseError) as exc:
            print(f"Could not fetch SFC positions: {exc}")
            return 1
        store = ShortTurnoverStore(args.db)
        store.save_sfc_positions(records)
        report_date = records[0].report_date if records else "n/a"
        print(f"Saved {len(records)} SFC short-position rows for {report_date} to {args.db}")
        print(f"Source: {result.url}")
        return 0

    if args.command == "backfill-hkex":
        return _run_backfill_hkex(args)

    if args.command == "fetch-prices":
        return _run_fetch_prices(args)

    if args.command == "backtest":
        return _run_backtest(args)

    parser.error("Unknown command")
    return 2


def _fetch_current(args) -> List[ShortTurnoverReport]:
    client = HTTPTextClient()
    fetcher = HKEXFetcher(client)
    markets = ["main", "gem"] if args.market == "all" else [args.market]
    reports: List[ShortTurnoverReport] = []
    fetched_at = datetime.now()

    for market in markets:
        sessions = _sessions_for_current(args.session)
        for session in sessions:
            try:
                result = fetcher.fetch_current_short_turnover(market, session)
                if not args.no_raw:
                    _write_raw(args.raw_dir, market, session, result.text, fetched_at)
                report = parse_short_turnover_page(
                    result.text,
                    market=market,
                    session=session,
                    source_url=result.url,
                    fetched_at=fetched_at,
                )
            except ReportUnavailable as exc:
                print(f"Skip {market}/{session}: {exc}")
                continue
            except (FetchError, ParseError) as exc:
                print(f"Skip {market}/{session}: {exc}")
                continue
            reports.append(report)
            if args.session == "auto":
                break
    return reports


def _fetch_daily_quote(args) -> List[ShortTurnoverReport]:
    client = HTTPTextClient()
    fetcher = HKEXFetcher(client)
    trade_date = date.fromisoformat(args.date)
    markets = ["main", "gem"] if args.market == "all" else [args.market]
    reports: List[ShortTurnoverReport] = []
    fetched_at = datetime.now()

    for market in markets:
        try:
            result = fetcher.fetch_daily_quote(market, trade_date)
            if not args.no_raw:
                _write_raw(args.raw_dir, market, "daily_quote", result.text, fetched_at)
            reports.append(
                parse_daily_quote_short_turnover(
                    result.text,
                    market=market,
                    source_url=result.url,
                    fetched_at=fetched_at,
                )
            )
        except (FetchError, ParseError) as exc:
            print(f"Skip {market}/daily_quote/{trade_date}: {exc}")
    return reports


def _run_backfill_hkex(args) -> int:
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    if end < start:
        print("--end must be >= --start")
        return 2

    client = HTTPTextClient(timeout=args.timeout, retries=args.retries)
    fetcher = HKEXFetcher(client)
    store = ShortTurnoverStore(args.db)
    markets = ["main", "gem"] if args.market == "all" else [args.market]
    fetched_at = datetime.now()
    loaded_reports = 0
    loaded_rows = 0
    skipped = 0

    for trade_date in _date_range(start, end):
        if trade_date.weekday() >= 5 and not args.include_weekends:
            continue
        for market in markets:
            try:
                result = fetcher.fetch_daily_quote(market, trade_date)
                if args.save_raw:
                    _write_raw(args.raw_dir, market, f"daily_quote_{trade_date:%Y%m%d}", result.text, fetched_at)
                report = parse_daily_quote_short_turnover(
                    result.text,
                    market=market,
                    source_url=result.url,
                    fetched_at=fetched_at,
                )
            except (FetchError, ParseError) as exc:
                skipped += 1
                print(f"Skip {market}/{trade_date}: {exc}", flush=True)
                continue
            store.save_report(report)
            loaded_reports += 1
            loaded_rows += len(report.records)
            print(f"Loaded {market}/{trade_date}: {len(report.records)} rows", flush=True)
            if args.pause:
                time.sleep(args.pause)

    print(f"Backfill complete: {loaded_reports} reports, {loaded_rows} rows, {skipped} skipped", flush=True)
    print(f"DB: {args.db}", flush=True)
    return 0 if loaded_reports else 1


def _run_fetch_prices(args) -> int:
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    if end < start:
        print("--end must be >= --start")
        return 2

    store = ShortTurnoverStore(args.db)
    codes = _resolve_price_codes(store, args.codes, args.top_n)
    if not codes:
        print("No codes found for price fetch.")
        return 1

    client = CurlTextClient(timeout=args.timeout, retries=args.retries)
    fetcher = YahooPriceFetcher(client)
    fetched_at = datetime.now()
    total_bars = 0
    failed = 0
    for code in codes:
        try:
            result = fetcher.fetch_chart(code, start, end)
            bars = [
                replace(bar, fetched_at=fetched_at)
                for bar in parse_yahoo_chart(result.text, code=code, source_url=result.url)
            ]
        except (FetchError, ParseError, ValueError) as exc:
            failed += 1
            print(f"Skip price {code}: {exc}", flush=True)
            continue
        store.save_price_bars(bars)
        total_bars += len(bars)
        print(f"Loaded price {code}: {len(bars)} bars", flush=True)
        if args.pause:
            time.sleep(args.pause)

    print(f"Price fetch complete: {len(codes) - failed}/{len(codes)} codes, {total_bars} bars", flush=True)
    print(f"DB: {args.db}", flush=True)
    return 0 if total_bars else 1


def _run_backtest(args) -> int:
    store = ShortTurnoverStore(args.db)
    signal_start = date.fromisoformat(args.start) if args.start else None
    signal_end = date.fromisoformat(args.end) if args.end else store.latest_trade_date()
    if signal_end is None:
        print("No local HKEX short-turnover data found.")
        return 1

    records = store.load_records(end_date=signal_end, session="day")
    if not records:
        print("No short-turnover records found for backtest.")
        return 1
    features = build_features(records)
    market_features = build_market_features(records)
    
    if signal_start:
        features = [row for row in features if row.record.trade_date >= signal_start]
        market_features = [row for row in market_features if row.trade_date >= signal_start]
    
    features = [row for row in features if row.record.trade_date <= signal_end]
    market_features = [row for row in market_features if row.trade_date <= signal_end]
    
    price_bars = store.load_price_bars(end_date=signal_end + timedelta(days=45), source="yahoo")
    horizons = [int(item.strip()) for item in args.horizons.split(",") if item.strip()]

    result = run_event_backtest(
        features=features,
        price_bars=price_bars,
        horizons=horizons,
        min_short_value=args.min_short_value,
        market_features=market_features,
        market_index_code="03033",  # Using CSOP Hang Seng Tech ETF as default proxy
    )
    report_path = _write_backtest_report(result, args.report_dir, signal_end)
    print(f"Backtest events: {len(result.events)}")
    print(f"Report: {report_path}")
    return 0 if result.events else 1


def _sessions_for_current(session: str) -> List[str]:
    if session == "auto":
        return ["day", "morning"]
    if session == "all":
        return ["day", "morning"]
    return [session]


def _date_range(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def _resolve_price_codes(store: ShortTurnoverStore, codes_arg: str, top_n: int) -> List[str]:
    if codes_arg:
        return [_normalize_cli_code(code) for code in codes_arg.split(",") if code.strip()]
    if top_n <= 0:
        return store.get_unique_codes()
    latest_date = store.latest_trade_date()
    if latest_date is None:
        return []
    records = store.load_records(end_date=latest_date, session="day")
    latest_records = [record for record in records if record.trade_date == latest_date and not record.is_non_hkd]
    latest_records.sort(key=lambda record: record.short_value, reverse=True)
    codes = []
    for record in latest_records:
        if record.code not in codes:
            codes.append(record.code)
        if len(codes) >= top_n:
            break
    return codes


def _normalize_cli_code(value: str) -> str:
    if value.startswith("^") or value.endswith(".HK"):
        return value.upper()
    digits = "".join(ch for ch in value if ch.isdigit())
    return digits.zfill(5)


def _save_and_report(reports: List[ShortTurnoverReport], db_path: Path, report_dir: Path) -> Path:
    store = ShortTurnoverStore(db_path)
    for report in reports:
        store.save_report(report)
    latest_date = max(report.trade_date for report in reports)
    return _write_report_from_store(store, reports, report_dir, latest_date)


def _write_report_from_store(
    store: ShortTurnoverStore,
    reports: List[ShortTurnoverReport],
    report_dir: Path,
    target_date: date,
) -> Path:
    records = store.load_records(end_date=target_date)
    features = build_features(records)
    current_features = [row for row in features if row.record.trade_date == target_date]
    
    price_start_date = target_date - timedelta(days=60)
    price_bars = store.load_price_bars(end_date=target_date, source="yahoo")
    filtered_price_bars = [bar for bar in price_bars if bar.trade_date >= price_start_date]
    prices_by_code = {}
    for bar in filtered_price_bars:
        prices_by_code.setdefault(bar.code, []).append(bar)
        
    signals = generate_signals(current_features, prices_by_code=prices_by_code)
    summaries = store.load_summaries(trade_date=target_date)
    sfc_positions = store.load_sfc_positions()
    markdown = build_monitor_report(
        reports=reports,
        features=current_features,
        signals=signals,
        summaries=summaries,
        sfc_positions=sfc_positions,
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"hkex_short_oi_{target_date.strftime('%Y%m%d')}_{datetime.now().strftime('%H%M%S')}.md"
    report_path.write_text(markdown, encoding="utf-8")
    return report_path


def _reports_from_records(records) -> List[ShortTurnoverReport]:
    grouped = {}
    for record in records:
        key = (record.trade_date, record.market, record.session, record.source_url)
        grouped.setdefault(key, []).append(record)
    reports = []
    for (trade_date, market, session, source_url), group_records in grouped.items():
        reports.append(
            ShortTurnoverReport(
                trade_date=trade_date,
                market=market,
                session=session,
                source_url=source_url,
                records=group_records,
                summaries=[],
            )
        )
    return reports


def _write_raw(raw_dir: Path, market: str, session: str, text: str, fetched_at: datetime) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{fetched_at.strftime('%Y%m%d_%H%M%S')}_{market}_{session}.html"
    path.write_text(text, encoding="utf-8")
    return path


def _write_backtest_report(result, report_dir: Path, signal_end: date) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"hkex_short_oi_backtest_{signal_end.strftime('%Y%m%d')}_{datetime.now().strftime('%H%M%S')}.md"
    csv_path = report_path.with_suffix(".events.csv")

    lines = [
        f"# HKEX Short OI Event Backtest - through {signal_end}",
        "",
        "Signals use HKEX daily short-selling turnover features and Yahoo OHLCV.",
        "Returns enter at the next available open and exit at the horizon close.",
        "",
        "## Summary",
        "",
        "| Strategy | Horizon | Count | Avg Return | Net Return | Win Rate | Max DD | Min | Max | IC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for strategy in sorted(result.by_strategy):
        for horizon in sorted(result.by_strategy[strategy]):
            stats = result.by_strategy[strategy][horizon]
            lines.append(
                "| {strategy} | {horizon} | {count} | {avg} | {net} | {win} | {dd} | {min_ret} | {max_ret} | {ic} |".format(
                    strategy=strategy,
                    horizon=horizon,
                    count=stats.count,
                    avg=_fmt_return(stats.avg_return),
                    net=_fmt_return(stats.avg_net_return),
                    win=_fmt_return(stats.win_rate),
                    dd=_fmt_return(stats.max_drawdown),
                    min_ret=_fmt_return(stats.min_return),
                    max_ret=_fmt_return(stats.max_return),
                    ic=f"{stats.ic:.3f}" if stats.ic is not None else "n/a",
                )
            )
    if not result.by_strategy:
        lines.append("| n/a | n/a | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")

    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- This is an event study, not a production execution simulator.",
            "- Yahoo prices are public OHLCV proxy data; verify with broker/HKEX data before trading.",
            "- Short-selling turnover is not short interest; use SFC weekly positions for context.",
            "",
            f"Event rows: `{csv_path.name}`",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    _write_events_csv(csv_path, result.events)
    return report_path


def _write_events_csv(path: Path, events: List[dict]) -> None:
    fieldnames = [
        "signal_date",
        "entry_date",
        "exit_date",
        "code",
        "name",
        "strategy",
        "direction",
        "horizon",
        "entry_price",
        "exit_price",
        "raw_return",
        "strategy_return",
        "net_return",
        "drawdown",
        "score",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for event in events:
            writer.writerow(event)


def _fmt_return(value) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2%}"


if __name__ == "__main__":
    raise SystemExit(main())
