"""数据持久化管理 — Parquet 增量存储"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .settings import Settings


class DataStore:
    """管理所有本地 Parquet 缓存的读写。

    目录结构：
        data/raw/futures/{SYMBOL}_daily.parquet
        data/raw/contracts/{SYMBOL}_contract_calendar.parquet
        data/processed/returns_daily.parquet
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._compression = settings.data.get("cache_compression", "snappy")
        # 确保目录存在
        settings.raw_futures_dir.mkdir(parents=True, exist_ok=True)
        settings.raw_contracts_dir.mkdir(parents=True, exist_ok=True)
        settings.processed_dir.mkdir(parents=True, exist_ok=True)
        settings.reports_dir.mkdir(parents=True, exist_ok=True)

    # ── 期货日线数据 ───────────────────────────────────────────────────────

    def futures_daily_path(self, symbol_sina: str) -> Path:
        return self._settings.raw_futures_dir / f"{symbol_sina}_daily.parquet"

    def save_futures_daily(self, symbol_sina: str, df: pd.DataFrame) -> None:
        """保存（覆盖写入）期货日线数据"""
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        df.to_parquet(
            self.futures_daily_path(symbol_sina),
            compression=self._compression,
        )

    def upsert_futures_daily(self, symbol_sina: str, new_df: pd.DataFrame) -> pd.DataFrame:
        """增量更新期货日线数据（新数据覆盖同日期旧数据）。

        Returns:
            合并后的完整历史数据
        """
        existing = self.load_futures_daily(symbol_sina)
        if existing is None:
            merged = new_df.copy()
        else:
            merged = pd.concat([existing, new_df])
            merged = merged[~merged.index.duplicated(keep="last")]
        merged.index = pd.to_datetime(merged.index)
        merged.sort_index(inplace=True)
        self.save_futures_daily(symbol_sina, merged)
        return merged

    def load_futures_daily(self, symbol_sina: str) -> pd.DataFrame | None:
        """加载期货日线数据，不存在时返回 None"""
        path = self.futures_daily_path(symbol_sina)
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        return df

    # ── 主力合约代码日历 ──────────────────────────────────────────────────

    def contract_calendar_path(self, symbol_sina: str) -> Path:
        return self._settings.raw_contracts_dir / f"{symbol_sina}_contract_calendar.parquet"

    def save_contract_calendar(self, symbol_sina: str, df: pd.DataFrame) -> None:
        """保存主力合约代码日历（date → contract_code）"""
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        df.to_parquet(
            self.contract_calendar_path(symbol_sina),
            compression=self._compression,
        )

    def load_contract_calendar(self, symbol_sina: str) -> pd.DataFrame | None:
        path = self.contract_calendar_path(symbol_sina)
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        return df

    # ── 加工后数据 ────────────────────────────────────────────────────────

    def returns_path(self) -> Path:
        return self._settings.processed_dir / "returns_daily.parquet"

    def save_returns(self, df: pd.DataFrame) -> None:
        """保存日收益率宽表（index=date, columns=symbol）"""
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        df.to_parquet(self.returns_path(), compression=self._compression)

    def load_returns(self) -> pd.DataFrame | None:
        path = self.returns_path()
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        return df

    def prices_path(self) -> Path:
        return self._settings.processed_dir / "prices_daily.parquet"

    def save_prices(self, df: pd.DataFrame) -> None:
        """保存日收盘价宽表（index=date, columns=symbol）"""
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        df.to_parquet(self.prices_path(), compression=self._compression)

    def load_prices(self) -> pd.DataFrame | None:
        path = self.prices_path()
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        return df

    # ── 回测报告 ──────────────────────────────────────────────────────────

    def save_report_df(self, name: str, df: pd.DataFrame) -> Path:
        """保存任意 DataFrame 到 reports/ 目录"""
        path = self._settings.reports_dir / f"{name}.parquet"
        df.to_parquet(path, compression=self._compression)
        return path

    def save_report_csv(self, name: str, df: pd.DataFrame) -> Path:
        path = self._settings.reports_dir / f"{name}.csv"
        df.to_csv(path, index=True, encoding="utf-8-sig")
        return path
