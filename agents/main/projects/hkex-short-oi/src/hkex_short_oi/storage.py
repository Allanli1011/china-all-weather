from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, List, Optional

from .models import MarketSummary, PriceBar, ShortPositionRecord, ShortTurnoverRecord, ShortTurnoverReport


class ShortTurnoverStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def save_report(self, report: ShortTurnoverReport) -> None:
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO short_turnover (
                    trade_date, market, session, code, raw_code, name,
                    short_shares, short_value, total_shares, total_value,
                    currency, is_non_hkd, source_url, fetched_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(trade_date, market, session, code, currency)
                DO UPDATE SET
                    raw_code=excluded.raw_code,
                    name=excluded.name,
                    short_shares=excluded.short_shares,
                    short_value=excluded.short_value,
                    total_shares=excluded.total_shares,
                    total_value=excluded.total_value,
                    is_non_hkd=excluded.is_non_hkd,
                    source_url=excluded.source_url,
                    fetched_at=excluded.fetched_at
                """,
                [self._record_tuple(record) for record in report.records],
            )
            conn.executemany(
                """
                INSERT INTO market_summary (
                    trade_date, market, session, section, short_shares,
                    short_values_json, market_turnover_hkd, short_pct_market
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(trade_date, market, session, section)
                DO UPDATE SET
                    short_shares=excluded.short_shares,
                    short_values_json=excluded.short_values_json,
                    market_turnover_hkd=excluded.market_turnover_hkd,
                    short_pct_market=excluded.short_pct_market
                """,
                [self._summary_tuple(summary) for summary in report.summaries],
            )

    def save_sfc_positions(self, records: Iterable[ShortPositionRecord]) -> None:
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO sfc_short_positions (
                    report_date, code, raw_code, name, short_position_shares,
                    short_position_hkd, source_url, fetched_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(report_date, code)
                DO UPDATE SET
                    raw_code=excluded.raw_code,
                    name=excluded.name,
                    short_position_shares=excluded.short_position_shares,
                    short_position_hkd=excluded.short_position_hkd,
                    source_url=excluded.source_url,
                    fetched_at=excluded.fetched_at
                """,
                [self._sfc_tuple(record) for record in records],
            )

    def load_sfc_positions(self, report_date: Optional[date] = None) -> List[ShortPositionRecord]:
        clauses = []
        params: List[object] = []
        if report_date:
            clauses.append("report_date = ?")
            params.append(report_date.isoformat())
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        query = f"""
            SELECT report_date, code, raw_code, name, short_position_shares,
                   short_position_hkd, source_url, fetched_at
            FROM sfc_short_positions
            {where}
            ORDER BY report_date, code
        """
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_sfc_position(row) for row in rows]

    def save_price_bars(self, bars: Iterable[PriceBar]) -> None:
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO price_bars (
                    trade_date, code, source, open, high, low, close, adj_close,
                    volume, turnover, source_url, fetched_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(trade_date, code, source)
                DO UPDATE SET
                    open=excluded.open,
                    high=excluded.high,
                    low=excluded.low,
                    close=excluded.close,
                    adj_close=excluded.adj_close,
                    volume=excluded.volume,
                    turnover=excluded.turnover,
                    source_url=excluded.source_url,
                    fetched_at=excluded.fetched_at
                """,
                [self._price_bar_tuple(bar) for bar in bars],
            )

    def load_price_bars(
        self,
        code: Optional[str] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        source: Optional[str] = None,
    ) -> List[PriceBar]:
        clauses = []
        params: List[object] = []
        if code:
            clauses.append("code = ?")
            params.append(code)
        if start_date:
            clauses.append("trade_date >= ?")
            params.append(start_date.isoformat())
        if end_date:
            clauses.append("trade_date <= ?")
            params.append(end_date.isoformat())
        if source:
            clauses.append("source = ?")
            params.append(source)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        query = f"""
            SELECT trade_date, code, source, open, high, low, close, adj_close,
                   volume, turnover, source_url, fetched_at
            FROM price_bars
            {where}
            ORDER BY code, trade_date
        """
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_price_bar(row) for row in rows]

    def load_records(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        market: Optional[str] = None,
        session: Optional[str] = None,
    ) -> List[ShortTurnoverRecord]:
        clauses = []
        params: List[object] = []
        if start_date:
            clauses.append("trade_date >= ?")
            params.append(start_date.isoformat())
        if end_date:
            clauses.append("trade_date <= ?")
            params.append(end_date.isoformat())
        if market:
            clauses.append("market = ?")
            params.append(market)
        if session:
            clauses.append("session = ?")
            params.append(session)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        query = f"""
            SELECT trade_date, market, session, code, raw_code, name,
                   short_shares, short_value, total_shares, total_value,
                   currency, is_non_hkd, source_url, fetched_at
            FROM short_turnover
            {where}
            ORDER BY trade_date, market, session, code
        """
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_record(row) for row in rows]

    def latest_trade_date(self) -> Optional[date]:
        with self._connect() as conn:
            row = conn.execute("SELECT MAX(trade_date) FROM short_turnover").fetchone()
        if not row or not row[0]:
            return None
        return date.fromisoformat(row[0])

    def get_unique_codes(self) -> List[str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT DISTINCT code FROM short_turnover ORDER BY code").fetchall()
        return [row[0] for row in rows]

    def load_summaries(
        self,
        trade_date: Optional[date] = None,
        market: Optional[str] = None,
        session: Optional[str] = None,
    ) -> List[MarketSummary]:
        clauses = []
        params: List[object] = []
        if trade_date:
            clauses.append("trade_date = ?")
            params.append(trade_date.isoformat())
        if market:
            clauses.append("market = ?")
            params.append(market)
        if session:
            clauses.append("session = ?")
            params.append(session)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        query = f"""
            SELECT trade_date, market, session, section, short_shares,
                   short_values_json, market_turnover_hkd, short_pct_market
            FROM market_summary
            {where}
            ORDER BY trade_date, market, session, section
        """
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_summary(row) for row in rows]

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS short_turnover (
                    trade_date TEXT NOT NULL,
                    market TEXT NOT NULL,
                    session TEXT NOT NULL,
                    code TEXT NOT NULL,
                    raw_code TEXT NOT NULL,
                    name TEXT NOT NULL,
                    short_shares INTEGER NOT NULL,
                    short_value REAL NOT NULL,
                    total_shares INTEGER,
                    total_value REAL,
                    currency TEXT NOT NULL,
                    is_non_hkd INTEGER NOT NULL,
                    source_url TEXT NOT NULL,
                    fetched_at TEXT,
                    PRIMARY KEY (trade_date, market, session, code, currency)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS market_summary (
                    trade_date TEXT NOT NULL,
                    market TEXT NOT NULL,
                    session TEXT NOT NULL,
                    section TEXT NOT NULL,
                    short_shares INTEGER,
                    short_values_json TEXT NOT NULL,
                    market_turnover_hkd REAL,
                    short_pct_market REAL,
                    PRIMARY KEY (trade_date, market, session, section)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_short_turnover_code_date "
                "ON short_turnover(code, trade_date)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sfc_short_positions (
                    report_date TEXT NOT NULL,
                    code TEXT NOT NULL,
                    raw_code TEXT NOT NULL,
                    name TEXT NOT NULL,
                    short_position_shares INTEGER NOT NULL,
                    short_position_hkd REAL NOT NULL,
                    source_url TEXT NOT NULL,
                    fetched_at TEXT,
                    PRIMARY KEY (report_date, code)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sfc_short_positions_code_date "
                "ON sfc_short_positions(code, report_date)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS price_bars (
                    trade_date TEXT NOT NULL,
                    code TEXT NOT NULL,
                    source TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    adj_close REAL,
                    volume INTEGER,
                    turnover REAL,
                    source_url TEXT NOT NULL,
                    fetched_at TEXT,
                    PRIMARY KEY (trade_date, code, source)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_price_bars_code_date "
                "ON price_bars(code, trade_date)"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.path))

    @staticmethod
    def _record_tuple(record: ShortTurnoverRecord) -> tuple:
        fetched_at = record.fetched_at.isoformat() if record.fetched_at else None
        return (
            record.trade_date.isoformat(),
            record.market,
            record.session,
            record.code,
            record.raw_code,
            record.name,
            record.short_shares,
            record.short_value,
            record.total_shares,
            record.total_value,
            record.currency,
            1 if record.is_non_hkd else 0,
            record.source_url,
            fetched_at,
        )

    @staticmethod
    def _summary_tuple(summary: MarketSummary) -> tuple:
        return (
            summary.trade_date.isoformat(),
            summary.market,
            summary.session,
            summary.section,
            summary.short_shares,
            json.dumps(summary.short_values, sort_keys=True),
            summary.market_turnover_hkd,
            summary.short_pct_market,
        )

    @staticmethod
    def _sfc_tuple(record: ShortPositionRecord) -> tuple:
        fetched_at = record.fetched_at.isoformat() if record.fetched_at else None
        return (
            record.report_date.isoformat(),
            record.code,
            record.raw_code,
            record.name,
            record.short_position_shares,
            record.short_position_hkd,
            record.source_url,
            fetched_at,
        )

    @staticmethod
    def _price_bar_tuple(bar: PriceBar) -> tuple:
        fetched_at = bar.fetched_at.isoformat() if bar.fetched_at else None
        return (
            bar.trade_date.isoformat(),
            bar.code,
            bar.source,
            bar.open,
            bar.high,
            bar.low,
            bar.close,
            bar.adj_close,
            bar.volume,
            bar.turnover,
            bar.source_url,
            fetched_at,
        )

    @staticmethod
    def _row_to_record(row: Iterable[object]) -> ShortTurnoverRecord:
        values = list(row)
        fetched_at = datetime.fromisoformat(values[13]) if values[13] else None
        return ShortTurnoverRecord(
            trade_date=date.fromisoformat(values[0]),
            market=values[1],
            session=values[2],
            code=values[3],
            raw_code=values[4],
            name=values[5],
            short_shares=int(values[6]),
            short_value=float(values[7]),
            total_shares=int(values[8]) if values[8] is not None else None,
            total_value=float(values[9]) if values[9] is not None else None,
            currency=values[10],
            is_non_hkd=bool(values[11]),
            source_url=values[12],
            fetched_at=fetched_at,
        )

    @staticmethod
    def _row_to_summary(row: Iterable[object]) -> MarketSummary:
        values = list(row)
        return MarketSummary(
            trade_date=date.fromisoformat(values[0]),
            market=values[1],
            session=values[2],
            section=values[3],
            short_shares=int(values[4]) if values[4] is not None else None,
            short_values=json.loads(values[5]),
            market_turnover_hkd=float(values[6]) if values[6] is not None else None,
            short_pct_market=float(values[7]) if values[7] is not None else None,
        )

    @staticmethod
    def _row_to_sfc_position(row: Iterable[object]) -> ShortPositionRecord:
        values = list(row)
        fetched_at = datetime.fromisoformat(values[7]) if values[7] else None
        return ShortPositionRecord(
            report_date=date.fromisoformat(values[0]),
            code=values[1],
            raw_code=values[2],
            name=values[3],
            short_position_shares=int(values[4]),
            short_position_hkd=float(values[5]),
            source_url=values[6],
            fetched_at=fetched_at,
        )

    @staticmethod
    def _row_to_price_bar(row: Iterable[object]) -> PriceBar:
        values = list(row)
        fetched_at = datetime.fromisoformat(values[11]) if values[11] else None
        return PriceBar(
            trade_date=date.fromisoformat(values[0]),
            code=values[1],
            source=values[2],
            open=float(values[3]),
            high=float(values[4]),
            low=float(values[5]),
            close=float(values[6]),
            adj_close=float(values[7]) if values[7] is not None else None,
            volume=int(values[8]) if values[8] is not None else None,
            turnover=float(values[9]) if values[9] is not None else None,
            source_url=values[10],
            fetched_at=fetched_at,
        )
