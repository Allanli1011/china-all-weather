# HKEX Short OI

HKEX short-selling turnover tracker and strategy signal lab.

Important vocabulary:

- HKEX daily pages provide short-selling turnover, not short interest or open short positions.
- SFC weekly aggregated reportable short positions are the better public proxy for net short-position pressure.
- Current MVP stores both, but strategy signals remain watchlist items until price/volume confirmation is added.

## What Works

- Fetch current HKEX Main Board and GEM short-selling turnover pages.
- Parse fixed-width HKEX text reports into normalized records.
- Store data idempotently in SQLite.
- Fetch latest SFC weekly aggregated short-position CSV.
- Generate a Markdown daily report with top short-selling turnover and strategy watchlist sections.
- Compute rolling z-scores and percentiles once enough local history exists.
- Backfill HKEX daily quotation short-selling history.
- Fetch Yahoo OHLCV for Hong Kong stocks.
- Run first-pass event-study backtests.

## Commands

Run from this project directory:

```bash
PYTHONPATH=src python3 -m hkex_short_oi fetch-current \
  --market all \
  --session auto \
  --db data/hkex_short_oi.sqlite \
  --raw-dir data/raw \
  --report-dir reports
```

Fetch latest SFC weekly short-position proxy:

```bash
PYTHONPATH=src python3 -m hkex_short_oi fetch-sfc-latest \
  --db data/hkex_short_oi.sqlite \
  --raw-dir data/raw
```

Backfill HKEX daily quotation short-selling history:

```bash
PYTHONPATH=src python3 -m hkex_short_oi backfill-hkex \
  --start 2026-03-25 \
  --end 2026-04-28 \
  --market main \
  --db data/hkex_short_oi.sqlite
```

Fetch Yahoo OHLCV for top short-turnover names:

```bash
PYTHONPATH=src python3 -m hkex_short_oi fetch-prices \
  --start 2026-03-25 \
  --end 2026-04-28 \
  --top-n 40 \
  --db data/hkex_short_oi.sqlite
```

Run event-study backtest:

```bash
PYTHONPATH=src python3 -m hkex_short_oi backtest \
  --start 2026-04-22 \
  --end 2026-04-27 \
  --horizons 1,3 \
  --db data/hkex_short_oi.sqlite
```

Regenerate a report from local data:

```bash
PYTHONPATH=src python3 -m hkex_short_oi report \
  --db data/hkex_short_oi.sqlite \
  --report-dir reports
```

Try a historical daily quotation page when HKEX exposes it:

```bash
PYTHONPATH=src python3 -m hkex_short_oi fetch-daily-quote \
  --date 2026-04-28 \
  --market all \
  --db data/hkex_short_oi.sqlite
```

## Project Layout

```text
src/hkex_short_oi/
  fetchers.py    HTTP clients and HKEX/SFC fetchers
  parsers.py     HKEX short-turnover parsers
  sfc.py         SFC weekly short-position CSV parser
  price.py       Yahoo chart parser and HK code mapping
  storage.py     SQLite persistence
  features.py    Rolling z-score and percentile features
  strategies.py  Watchlist signal rules
  backtest.py    Event-study backtester
  reports.py     Markdown report builder
  __main__.py    CLI entrypoint
tests/
  test_parsers.py
  test_storage_features_strategies.py
```

## Strategy Ideas Implemented As Watchlists

- `crowded_reversal_watch`: short pressure is unusually high. Look for price failing to make new lows before considering a rebound trade.
- `pressure_breakdown_watch`: short pressure is unusually high and can confirm bearish continuation if price breaks support.
- `short_pressure_fade`: short pressure drops materially versus recent history. Look for stabilization.

## Next Build Steps

1. Extend HKEX history to 6-12 months; the current local backfill is only a short validation window.
2. Add price-confirmation filters directly to strategy definitions.
3. Join SFC weekly position deltas onto daily HKEX turnover features.
4. Expand event-study backtests to 1-day, 3-day, 5-day, and 10-day forward returns over a larger sample.
5. Add an OpenClaw cron job only after backtest validation.
