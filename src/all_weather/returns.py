"""收益率计算模块 — 主力连续合约展期检测与日收益率矩阵构建"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .data_store import DataStore
    from .settings import Settings
    from .universe import InstrumentSpec

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 展期检测
# ─────────────────────────────────────────────────────────────────────────────

def detect_rollover_days(
    daily_df: pd.DataFrame,
    jump_threshold: float = 0.03,
) -> pd.Series:
    """通过价格跳幅检测展期日（回退方案，适用于无合约代码字段时）。

    判断逻辑：
        1. 相邻两日收盘价涨跌幅绝对值超过 jump_threshold
        2. 仅保留单向异常（防止真实大行情被误判），通过比较前后日收益率方向筛选

    由于 AKShare futures_main_sina 不返回合约代码，此方案作为主要检测手段。

    Args:
        daily_df: 含 'close' 列的日线 DataFrame
        jump_threshold: 涨跌幅阈值（默认3%，商品期货建议3%，金融期货可设1%）

    Returns:
        bool Series，index=date，True 表示该日为展期日
    """
    close = daily_df["close"].copy()
    raw_ret = close.pct_change()

    # 基础条件：绝对收益超阈值
    large_move = raw_ret.abs() > jump_threshold

    # 辅助筛选：如果前后日收益方向相反，倾向于是展期（基差跳跃），而非趋势行情
    # 例：昨天 +4%（展期向上跳），但前一天和后一天都是小幅波动
    prev_ret = raw_ret.shift(1)
    next_ret = raw_ret.shift(-1)

    # 展期特征：当日绝对收益大，且（前日或后日）方向相反或绝对值小得多
    likely_rollover = large_move & (
        (prev_ret.abs() < jump_threshold * 0.5) |  # 前日波动正常
        (next_ret.abs() < jump_threshold * 0.5) |  # 后日波动正常
        (np.sign(raw_ret) != np.sign(prev_ret))     # 方向与前日相反
    )

    # 第一行无收益率，强制为 False
    likely_rollover.iloc[0] = False

    n = likely_rollover.sum()
    if n > 0:
        logger.debug(
            "detect_rollover_days: 检测到 %d 个展期日（阈值=%.1f%%）",
            n, jump_threshold * 100,
        )
    return likely_rollover.rename("is_rollover")


# ─────────────────────────────────────────────────────────────────────────────
# 价格矩阵与收益率矩阵构建
# ─────────────────────────────────────────────────────────────────────────────

def build_aligned_price_matrix(
    store: "DataStore",
    universe: dict[str, "InstrumentSpec"],
    start_date: str,
    end_date: str | None = None,
) -> pd.DataFrame:
    """构建对齐到共同交易日历的收盘价宽表。

    对于晚上市的品种（如 SC 2018年上市），早期日期填充 NaN，
    而非前向填充价格——前向填充会导致波动率被低估。

    Args:
        store: DataStore 实例
        universe: symbol → InstrumentSpec
        start_date: 起始日期
        end_date: 截止日期（None = 最新）

    Returns:
        DataFrame, index=date(pd.Timestamp), columns=symbol, values=close price
    """
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date) if end_date else None

    price_series: dict[str, pd.Series] = {}
    for symbol, spec in universe.items():
        df = store.load_futures_daily(spec.symbol_sina)
        if df is None or df.empty:
            logger.warning("build_aligned_price_matrix: %s 无缓存数据，跳过", symbol)
            continue
        close = df["close"].dropna()
        price_series[symbol] = close

    if not price_series:
        raise ValueError("无有效价格数据，请先运行数据获取")

    # 构建共同交易日历（取所有品种交集的超集，用最长历史品种的日历）
    all_dates_union = pd.DatetimeIndex(
        sorted(set().union(*[s.index for s in price_series.values()]))
    )

    # 过滤日期范围
    mask = all_dates_union >= start_ts
    if end_ts is not None:
        mask &= all_dates_union <= end_ts
    calendar = all_dates_union[mask]

    # 将各品种收益率对齐到日历，晚上市品种用 NaN 填充早期
    price_df = pd.DataFrame(index=calendar, dtype=float)
    for symbol, series in price_series.items():
        price_df[symbol] = series.reindex(calendar)

    logger.info(
        "build_aligned_price_matrix: %d 个品种, %d 个交易日 (%s ~ %s)",
        len(price_df.columns),
        len(price_df),
        price_df.index.min().date(),
        price_df.index.max().date(),
    )
    return price_df


def compute_daily_returns(
    price_df: pd.DataFrame,
    rollover_masks: dict[str, pd.Series] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """计算日收益率宽表，展期日收益率置0。

    Args:
        price_df: 收盘价宽表（index=date, columns=symbol）
        rollover_masks: dict[symbol → bool Series]，True 表示展期日
                        若为 None，不进行展期处理

    Returns:
        (returns_df, is_rollover_df):
            - returns_df: 日收益率宽表，展期日收益率为 0
            - is_rollover_df: 展期日标记宽表（bool）
    """
    # 逐品种计算 pct_change
    returns_dict: dict[str, pd.Series] = {}
    rollover_dict: dict[str, pd.Series] = {}

    for symbol in price_df.columns:
        prices = price_df[symbol].dropna()
        if len(prices) < 2:
            continue

        ret = prices.pct_change()

        # 展期日处理
        if rollover_masks and symbol in rollover_masks:
            mask = rollover_masks[symbol].reindex(ret.index, fill_value=False)
            ret = ret.where(~mask, other=0.0)
            rollover_dict[symbol] = mask
        else:
            rollover_dict[symbol] = pd.Series(False, index=ret.index)

        returns_dict[symbol] = ret

    returns_df = pd.DataFrame(returns_dict)
    is_rollover_df = pd.DataFrame(rollover_dict)

    # 对齐到完整日历（晚上市品种早期为 NaN）
    returns_df = returns_df.reindex(price_df.index)
    is_rollover_df = is_rollover_df.reindex(price_df.index, fill_value=False)

    logger.info(
        "compute_daily_returns: %d 个品种，%d 天",
        len(returns_df.columns), len(returns_df),
    )
    return returns_df, is_rollover_df


def build_returns_pipeline(
    store: "DataStore",
    universe: dict[str, "InstrumentSpec"],
    settings: "Settings",
    force_rebuild: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """完整的收益率构建流水线（带缓存）。

    流程：
        1. 加载缓存的收盘价宽表（如不存在则从原始数据构建）
        2. 对每个品种运行 detect_rollover_days()
        3. 调用 compute_daily_returns() 生成收益率宽表
        4. 将收盘价宽表和收益率宽表保存到 DataStore

    Args:
        store: DataStore 实例
        universe: symbol → InstrumentSpec
        settings: Settings 实例
        force_rebuild: 是否强制重建（忽略缓存）

    Returns:
        (price_df, returns_df, is_rollover_df)
    """
    if not force_rebuild:
        cached_returns = store.load_returns()
        cached_prices = store.load_prices()
        if cached_returns is not None and cached_prices is not None:
            logger.info("使用缓存的收益率数据")
            # 重建展期标记（轻量操作，无需缓存）
            rollover_masks = _build_rollover_masks(cached_prices, settings)
            _, is_rollover_df = compute_daily_returns(cached_prices, rollover_masks)
            return cached_prices, cached_returns, is_rollover_df

    # 构建价格矩阵
    price_df = build_aligned_price_matrix(
        store=store,
        universe=universe,
        start_date=settings.data_start_date,
    )

    # 展期检测
    rollover_masks = _build_rollover_masks(price_df, settings)

    # 计算收益率
    returns_df, is_rollover_df = compute_daily_returns(price_df, rollover_masks)

    # 缓存
    store.save_prices(price_df)
    store.save_returns(returns_df)
    logger.info("收益率矩阵已保存到 DataStore")

    return price_df, returns_df, is_rollover_df


def _build_rollover_masks(
    price_df: pd.DataFrame,
    settings: "Settings",
) -> dict[str, pd.Series]:
    """为每个品种构建展期日 bool mask"""
    threshold = settings.rollover.get("price_jump_threshold", 0.03)
    masks: dict[str, pd.Series] = {}
    for symbol in price_df.columns:
        prices = price_df[symbol].dropna()
        if prices.empty:
            continue
        # detect_rollover_days 期望含 "close" 列的 DataFrame
        df_for_detect = prices.rename("close").to_frame()
        masks[symbol] = detect_rollover_days(df_for_detect, jump_threshold=threshold)
    return masks


def compute_rollover_cost_series(
    is_rollover_df: pd.DataFrame,
    price_df: pd.DataFrame,
    universe: dict[str, "InstrumentSpec"],
) -> pd.Series:
    """计算每日总展期成本（元，用于回测时扣除）。

    展期成本 = 品种单手展期成本 × 当日持有手数
    此函数仅计算"单位持仓（1手）"时的成本率，
    回测引擎会乘以实际持仓手数。

    Returns:
        Series(index=date, values=每日单手展期成本率之和)
    """
    cost_series = pd.Series(0.0, index=price_df.index)
    for symbol in is_rollover_df.columns:
        if symbol not in universe:
            continue
        spec = universe[symbol]
        rollover_days = is_rollover_df[symbol].reindex(price_df.index, fill_value=False)
        prices = price_df[symbol].reindex(price_df.index)
        # 展期成本率 = rollover_cost / (price * multiplier)
        cost_rate = rollover_days * prices.apply(
            lambda p: spec.rollover_cost(p) / (p * spec.multiplier) if p > 0 else 0.0
        )
        cost_series += cost_rate.fillna(0.0)
    return cost_series
