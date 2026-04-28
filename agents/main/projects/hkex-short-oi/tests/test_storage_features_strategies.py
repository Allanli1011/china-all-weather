import tempfile
import unittest
from datetime import date
from pathlib import Path

from hkex_short_oi.features import build_features
from hkex_short_oi.backtest import run_event_backtest
from hkex_short_oi.models import PriceBar, ShortTurnoverRecord, ShortTurnoverReport
from hkex_short_oi.price import parse_yahoo_chart
from hkex_short_oi.storage import ShortTurnoverStore
from hkex_short_oi.strategies import generate_signals


def make_record(trade_date, value, total_value=100_000_000):
    return ShortTurnoverRecord(
        trade_date=trade_date,
        market="main",
        session="day",
        code="00700",
        raw_code="700",
        name="TENCENT",
        short_shares=1_000_000,
        short_value=value,
        total_shares=2_000_000,
        total_value=total_value,
        currency="HKD",
        is_non_hkd=False,
        source_url="https://example.test",
    )


class StorageFeatureStrategyTests(unittest.TestCase):
    def test_store_upserts_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test.sqlite"
            store = ShortTurnoverStore(db_path)
            report = ShortTurnoverReport(
                trade_date=date(2026, 4, 28),
                market="main",
                session="day",
                source_url="https://example.test",
                records=[make_record(date(2026, 4, 28), 10_000_000)],
                summaries=[],
            )

            store.save_report(report)
            store.save_report(report)

            records = store.load_records()
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].code, "00700")

    def test_store_upserts_price_bars(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test.sqlite"
            store = ShortTurnoverStore(db_path)
            bars = [
                PriceBar(
                    trade_date=date(2026, 4, 20),
                    code="00700",
                    source="yahoo",
                    open=100.0,
                    high=102.0,
                    low=99.0,
                    close=101.0,
                    adj_close=101.0,
                    volume=1000,
                    turnover=None,
                    source_url="https://example.test/chart",
                )
            ]

            store.save_price_bars(bars)
            store.save_price_bars(bars)

            loaded = store.load_price_bars(code="00700")
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].close, 101.0)

    def test_features_and_signals_detect_crowded_shorting(self):
        records = [
            make_record(date(2026, 4, day), 10_000_000)
            for day in range(1, 21)
        ]
        records.append(make_record(date(2026, 4, 21), 40_000_000))

        features = build_features(records, short_windows=(20,))
        latest = features[-1]

        self.assertGreater(latest.short_value_z[20], 2.0)
        signals = generate_signals([latest], min_short_value=5_000_000)
        strategies = {signal.strategy for signal in signals}
        self.assertIn("crowded_reversal", strategies)
        self.assertIn("pressure_breakdown", strategies)

    def test_yahoo_chart_parser(self):
        payload = """
        {
          "chart": {
            "result": [{
              "timestamp": [1776633600, 1776720000],
              "indicators": {
                "quote": [{
                  "open": [100.0, 102.0],
                  "high": [103.0, 104.0],
                  "low": [99.0, 101.0],
                  "close": [102.0, 103.0],
                  "volume": [1000, 1200]
                }],
                "adjclose": [{"adjclose": [102.0, 103.0]}]
              }
            }]
          }
        }
        """
        bars = parse_yahoo_chart(payload, code="00700", source_url="https://example.test/chart")

        self.assertEqual(len(bars), 2)
        self.assertEqual(bars[0].code, "00700")
        self.assertEqual(bars[0].close, 102.0)

    def test_event_backtest_uses_next_open_and_future_close(self):
        records = [
            make_record(date(2026, 4, day), 10_000_000)
            for day in range(1, 21)
        ]
        records.append(make_record(date(2026, 4, 21), 40_000_000))
        features = build_features(records, short_windows=(20,))
        bars = [
            PriceBar(date(2026, 3, day), "00700", "test", 105, 106, 104, 105, 105, 1000, None, "")
            for day in range(1, 25)
        ] + [
            PriceBar(date(2026, 4, 21), "00700", "test", 95, 101, 94, 100, 100, 1000, None, ""),
            PriceBar(date(2026, 4, 22), "00700", "test", 101, 103, 100, 102, 102, 1000, None, ""),
            PriceBar(date(2026, 4, 23), "00700", "test", 102, 106, 101, 105, 105, 1000, None, ""),
        ]

        result = run_event_backtest(features, bars, horizons=(1, 2), min_short_value=1)

        long_one_day = result.by_strategy["crowded_reversal"][1]
        short_one_day = result.by_strategy["pressure_breakdown"][1]
        self.assertEqual(long_one_day.count, 1)
        self.assertAlmostEqual(long_one_day.avg_return, (102 / 101) - 1)
        self.assertAlmostEqual(short_one_day.avg_return, -((102 / 101) - 1))


if __name__ == "__main__":
    unittest.main()
