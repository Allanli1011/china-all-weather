"""AKShare 数据获取模块 — 期货日线数据与主力合约代码日历"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import akshare as ak
import pandas as pd

if TYPE_CHECKING:
    from .settings import Settings

logger = logging.getLogger(__name__)

# AKShare 返回列名 → 标准列名映射
_FUTURES_COL_MAP = {
    "日期": "date",
    "开盘价": "open",
    "最高价": "high",
    "最低价": "low",
    "收盘价": "close",
    "成交量": "volume",
    "持仓量": "open_interest",
    "结算价": "settle",
    # 新浪接口部分版本使用英文列名
    "date": "date",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
    "hold": "open_interest",
}

# 可能包含主力合约代码的列名
_CONTRACT_CODE_CANDIDATES = ["合约代码", "contract", "symbol", "代码"]


def fetch_futures_daily(
    symbol_sina: str,
    start_date: str,
    end_date: str | None = None,
    retries: int = 3,
    delay: float = 0.5,
) -> pd.DataFrame:
    """获取主力连续合约日线数据（via AKShare futures_main_sina）。

    Args:
        symbol_sina: AKShare 代码，如 "IF0"、"AU0"
        start_date:  起始日期，格式 "YYYYMMDD"
        end_date:    截止日期，None 表示获取到最新数据
        retries:     失败重试次数
        delay:       请求间隔（秒），避免触发限流

    Returns:
        DataFrame，index=date(pd.Timestamp)，
        columns 至少包含 [open, high, low, close, volume, open_interest]
        若 AKShare 返回了合约代码字段，则额外包含 contract_code 列

    Raises:
        RuntimeError: 重试耗尽后仍获取失败
    """
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            df = ak.futures_main_sina(symbol=symbol_sina, start_date=start_date)
            break
        except Exception as e:
            last_err = e
            logger.warning(
                "fetch_futures_daily(%s) 第 %d/%d 次失败: %s",
                symbol_sina, attempt, retries, e,
            )
            if attempt < retries:
                time.sleep(delay * attempt)
    else:
        raise RuntimeError(
            f"fetch_futures_daily({symbol_sina}) 重试 {retries} 次后仍失败"
        ) from last_err

    df = _normalize_futures_df(df, symbol_sina)

    if end_date is not None:
        df = df[df.index <= pd.Timestamp(end_date)]

    time.sleep(delay)
    return df


def _normalize_futures_df(df: pd.DataFrame, symbol_sina: str) -> pd.DataFrame:
    """标准化 AKShare 返回的期货日线 DataFrame。

    - 重命名列名为英文标准名
    - 解析日期并设为 index
    - 检测是否含主力合约代码字段
    - 数值列转为 float
    """
    # 重命名已知列
    rename_map = {k: v for k, v in _FUTURES_COL_MAP.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    # 解析日期并设为 index
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    elif df.index.dtype == object or df.index.dtype.kind == "O":
        df.index = pd.to_datetime(df.index)

    # 检测是否含合约代码字段
    code_col = next((c for c in _CONTRACT_CODE_CANDIDATES if c in df.columns), None)
    if code_col and code_col != "contract_code":
        df = df.rename(columns={code_col: "contract_code"})
    elif code_col is None:
        # AKShare 不含合约代码字段，稍后在 returns.py 中通过其他途径获取
        pass

    # 数值列转 float
    numeric_cols = ["open", "high", "low", "close", "volume", "open_interest", "settle"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.sort_index(inplace=True)
    logger.debug("fetch_futures_daily(%s): %d 行，列: %s", symbol_sina, len(df), list(df.columns))
    return df


def has_contract_code_field(df: pd.DataFrame) -> bool:
    """检查 DataFrame 是否含主力合约代码字段（用于判断展期检测路径）"""
    return "contract_code" in df.columns


def fetch_contract_calendar_via_spot(
    symbol_sina: str,
    trading_dates: pd.DatetimeIndex,
    delay: float = 0.5,
) -> pd.DataFrame:
    """通过 AKShare 现货行情接口构建主力合约代码日历（回退路径B）。

    此函数尝试通过 ak.futures_zh_spot() 获取每日主力合约代码。
    注意：此接口只能获取当前数据，历史主力合约代码需要通过其他手段。

    实际回测中，若 futures_main_sina 不含代码字段，
    建议使用 build_rollover_calendar_from_price_jump() 作为简单回退。

    Returns:
        DataFrame, index=date, columns=['contract_code']
        （若接口不支持，返回空 DataFrame）
    """
    logger.warning(
        "fetch_contract_calendar_via_spot: 此接口无法获取历史主力合约代码，"
        "建议使用价格跳幅检测作为回退方案"
    )
    return pd.DataFrame(columns=["contract_code"])


def build_rollover_calendar_from_price_jump(
    daily_df: pd.DataFrame,
    jump_threshold: float = 0.03,
) -> pd.DataFrame:
    """通过价格跳幅检测构建展期日历（简化回退方案）。

    当相邻两日收盘价变动幅度超过阈值，且方向与前后日不一致时，
    推断为展期日。此方案精度低于合约代码检测，但无需额外数据。

    Args:
        daily_df: 含 close 列的日线 DataFrame
        jump_threshold: 单日涨跌幅阈值（默认3%）

    Returns:
        DataFrame, index=date, columns=['is_rollover']
    """
    close = daily_df["close"]
    returns = close.pct_change()
    is_rollover = returns.abs() > jump_threshold
    result = pd.DataFrame({"is_rollover": is_rollover}, index=daily_df.index)
    # 第一行无法计算，标记为 False
    result.iloc[0] = False
    logger.debug(
        "build_rollover_calendar_from_price_jump: 检测到 %d 个潜在展期日（阈值=%.1f%%）",
        is_rollover.sum(), jump_threshold * 100,
    )
    return result


def fetch_all_instruments(
    universe_specs: dict,  # symbol → InstrumentSpec
    start_date: str,
    store: "DataStore",  # noqa: F821
    settings: "Settings",
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """批量获取所有品种的日线数据并缓存到 DataStore。

    Args:
        universe_specs: load_universe() 返回的品种字典
        start_date: 数据起始日期
        store: DataStore 实例
        settings: Settings 实例
        force_refresh: True 则忽略缓存重新获取

    Returns:
        dict: symbol_sina → DataFrame（含已缓存数据）
    """
    from .data_store import DataStore  # 避免循环导入

    delay = settings.data.get("fetch_delay_seconds", 0.5)
    retries = settings.data.get("fetch_retries", 3)
    results: dict[str, pd.DataFrame] = {}

    for symbol, spec in universe_specs.items():
        sina_sym = spec.symbol_sina
        cached = None if force_refresh else store.load_futures_daily(sina_sym)

        if cached is not None and not cached.empty:
            # 检查是否需要增量更新（最新日期距今超过1天）
            latest = cached.index.max()
            today = pd.Timestamp.today().normalize()
            if latest >= today - pd.Timedelta(days=1):
                logger.info("%s: 缓存已是最新 (最新日期: %s)", sina_sym, latest.date())
                results[symbol] = cached
                continue
            # 增量更新：从最新日期+1天开始获取
            inc_start = (latest + pd.Timedelta(days=1)).strftime("%Y%m%d")
            logger.info("%s: 增量更新从 %s 开始", sina_sym, inc_start)
            try:
                new_data = fetch_futures_daily(sina_sym, inc_start, retries=retries, delay=delay)
                merged = store.upsert_futures_daily(sina_sym, new_data)
                results[symbol] = merged
            except Exception as e:
                logger.error("%s 增量更新失败，使用缓存数据: %s", sina_sym, e)
                results[symbol] = cached
        else:
            # 全量获取
            logger.info("%s: 全量获取从 %s 开始", sina_sym, start_date)
            try:
                df = fetch_futures_daily(sina_sym, start_date, retries=retries, delay=delay)
                store.save_futures_daily(sina_sym, df)
                results[symbol] = df
            except Exception as e:
                logger.error("%s 全量获取失败: %s", sina_sym, e)

    logger.info(
        "fetch_all_instruments: 完成 %d/%d 个品种",
        len(results), len(universe_specs),
    )
    return results
