# HKEX Short OI Status

Updated: 2026-04-28 22:35 CST

## Completed

- Created the project scaffold under `/Users/kumamon/.openclaw/workspace/agents/main/projects/hkex-short-oi`.
- Implemented a Python package in `src/hkex_short_oi/`.
- Added CLI commands:
  - `fetch-current`: fetch HKEX current Main Board/GEM short-selling turnover pages.
  - `fetch-daily-quote`: fetch historical HKEX daily quotation short-selling sections when available.
  - `fetch-sfc-latest`: fetch latest SFC weekly aggregated reportable short positions CSV.
  - `report`: regenerate a Markdown report from local SQLite data.
- Implemented parsers for:
  - HKEX current fixed-width short-selling turnover reports.
  - HKEX daily quotation short-selling sections.
  - SFC weekly aggregated reportable short-position CSV files.
- Implemented SQLite storage with idempotent upserts for:
  - HKEX daily short-selling turnover records.
  - HKEX market summary sections.
  - SFC weekly short-position proxy records.
- Implemented initial feature builders:
  - Rolling short-value z-scores.
  - Rolling short-ratio z-scores when total turnover is available.
  - Rolling percentiles and moving averages.
- Implemented initial strategy watchlist rules:
  - `crowded_reversal_watch`
  - `pressure_breakdown_watch`
  - `short_pressure_fade`
- Implemented Markdown report generation.
- Verified real HKEX fetch for 2026-04-28:
  - Main Board: 833 rows.
  - GEM: 3 rows.
  - Total: 836 rows.
- Verified real SFC fetch:
  - Latest report date: 2026-04-17.
  - 1200 weekly short-position proxy rows saved.
- Latest generated report:
  - `/Users/kumamon/.openclaw/workspace/agents/main/projects/hkex-short-oi/reports/hkex_short_oi_20260428_214411.md`
- Tests pass:
  - `PYTHONPATH=src python3 -m unittest discover -s tests`
  - 10 tests OK.
- Added `backfill-hkex`, `fetch-prices`, and `backtest` CLI commands.
- Backfilled Main Board HKEX daily quotation short-selling history:
  - Range: 2026-03-25 to 2026-04-28.
  - Loaded: 22 reports, 18,076 rows.
  - Skipped: 3 no-trade/no-date pages.
- Added Yahoo OHLCV ingestion using system `curl` because Yahoo returned HTTP 429 to Python `urllib`.
- Fetched prices for the latest top 40 short-turnover names:
  - Range: 2026-03-25 to 2026-04-28.
  - Loaded: 880 daily bars.
- Ran first-pass event-study backtest:
  - Signal window: 2026-04-22 to 2026-04-27.
  - Horizons: 1 and 3 trading days.
  - Events: 72.
  - Latest backtest report: `/Users/kumamon/.openclaw/workspace/agents/main/projects/hkex-short-oi/reports/hkex_short_oi_backtest_20260427_223343.md`

## Important Clarification

- A first-pass event-study backtest has been run, but it is only a small-sample plumbing validation.
- Current strategy rules remain watchlist hypotheses, not validated trading signals.
- Current local HKEX history is still short, so `Z20` is barely usable and `Z60` is not fully mature.
- Price confirmation filters are not yet wired into strategy selection; reversal and breakdown hypotheses can currently fire from the same high-short-pressure event.
- HKEX daily data is short-selling turnover, not true short interest or open short positions.
- SFC weekly aggregated reportable short positions should be treated as the public short-position proxy.

## Next Tasks

1. Extend HKEX history.
   - Target at least 6-12 months.
   - HKEX daily quotation pages are large and slow, so backfill may need batching.

2. Expand Hong Kong stock OHLCV coverage.
   - Current run fetched only the latest top 40 short-turnover names.
   - Need a broader universe before drawing conclusions.

3. Improve event-study backtests.
   - Add 5-day and 10-day horizons once enough future prices are available.
   - Add IC, quantile spread, drawdown, and transaction-cost-adjusted returns.

4. Add price confirmation filters.
   - Separate reversal from continuation instead of firing both directions from the same stress event.
   - Suggested filters: break below 20-day low, reclaim prior-day high, bullish candle, hold above support, volume confirmation.

5. Test core hypotheses.
   - High short ratio / high short z-score alone.
   - High short pressure plus price breakdown.
   - High short pressure but price fails to make new lows.
   - Falling short pressure after prior crowding.
   - SFC short-position increase plus HKEX daily short-turnover acceleration.

6. Add sector/index aggregation.
   - Build short-pressure indexes for Hang Seng / HSTECH heavyweights.
   - Compare index-level pressure against HSI/HSTECH ETF returns.

7. Only after backtest validation, add cron automation.
   - Daily HKEX fetch after market close.
   - Weekly SFC fetch after SFC updates.
   - Daily report delivery if signals are meaningful.
