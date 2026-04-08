"""品种宇宙管理 — 期货合约元数据与活跃品种筛选"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .settings import Settings

# 资产类别到标准名的映射（与 YAML 配置键对应）
ASSET_CLASS_ORDER = [
    "equity",
    "bond",
    "gold",
    "industrial_metal",
    "ferrous",
    "energy",
    "agriculture",
]

# 商品大类（在两层ERC中合并为一个虚拟类别）
COMMODITY_CLASSES = {"industrial_metal", "ferrous", "energy", "agriculture"}


@dataclass(frozen=True)
class InstrumentSpec:
    """单个期货品种的完整元数据"""

    symbol: str             # 内部代码（如 "IF"）
    name: str               # 中文名
    exchange: str           # 交易所（CFFEX/SHFE/DCE/CZCE/INE）
    asset_class: str        # 资产类别（equity/bond/gold/industrial_metal/ferrous/energy/agriculture）
    symbol_sina: str        # AKShare 新浪接口代码（如 "IF0"）
    multiplier: int         # 合约乘数（元/点 或 元/吨）
    margin_rate: float      # 保证金率
    tick_size: float        # 最小变动价位
    fee_rate: float         # 手续费率（双边合计，按成交额计）
    slippage_ticks: int     # 模拟滑点（tick数）
    listed_date: str        # 上市日期（YYYYMMDD）

    @property
    def listed_dt(self) -> pd.Timestamp:
        return pd.Timestamp(self.listed_date)

    @property
    def is_commodity(self) -> bool:
        return self.asset_class in COMMODITY_CLASSES

    @property
    def is_financial(self) -> bool:
        return self.asset_class in {"equity", "bond", "gold"}

    def rollover_cost(self, price: float) -> float:
        """展期单次交易成本（元，按单手计算）

        包含：双边手续费 + 双边滑点
        展期 = 平仓近月 + 开仓远月，各一次手续费和滑点
        """
        contract_value = price * self.multiplier
        fee = contract_value * self.fee_rate * 2        # 双边手续费
        slip = self.tick_size * self.slippage_ticks * 2  # 双边滑点
        return fee + slip

    def trade_cost_rate(self, price: float) -> float:
        """单次普通交易（非展期）成本率（相对于合约价值）"""
        slip_cost = self.tick_size * self.slippage_ticks
        return self.fee_rate + slip_cost / (price * self.multiplier)


def load_universe(settings: Settings) -> dict[str, InstrumentSpec]:
    """从 Settings 中加载全部品种元数据。

    Returns:
        dict: symbol → InstrumentSpec
    """
    specs: dict[str, InstrumentSpec] = {}
    for asset_class, instruments in settings.universe.items():
        if not isinstance(instruments, dict):
            continue
        for symbol, cfg in instruments.items():
            if not isinstance(cfg, dict):
                continue
            specs[symbol] = InstrumentSpec(
                symbol=symbol,
                name=cfg["name"],
                exchange=cfg["exchange"],
                asset_class=asset_class,
                symbol_sina=cfg["symbol_sina"],
                multiplier=int(cfg["multiplier"]),
                margin_rate=float(cfg["margin_rate"]),
                tick_size=float(cfg["tick_size"]),
                fee_rate=float(cfg["fee_rate"]),
                slippage_ticks=int(cfg["slippage_ticks"]),
                listed_date=str(cfg["listed_date"]),
            )
    return specs


def filter_active_instruments(
    universe: dict[str, InstrumentSpec],
    as_of_date: pd.Timestamp,
    min_history_days: int = 90,
) -> list[str]:
    """筛选在指定日期可参与策略的品种。

    品种进入组合的条件：
        上市日期 + min_history_days <= as_of_date

    这保证了在计算60日滚动协方差时，品种有足够的历史数据（
    min_history_days 通常设为 vol_window + 安全缓冲）。

    Args:
        universe: 全品种元数据字典
        as_of_date: 判断日期
        min_history_days: 品种上市后需要积累的最少历史天数

    Returns:
        可参与策略的品种代码列表（按资产类别顺序）
    """
    cutoff = as_of_date - pd.Timedelta(days=min_history_days)
    active = [
        symbol
        for symbol, spec in universe.items()
        if spec.listed_dt <= cutoff
    ]
    # 按资产类别排序，保持权重矩阵列顺序稳定
    return sorted(active, key=lambda s: (ASSET_CLASS_ORDER.index(universe[s].asset_class), s))


def group_by_class(
    symbols: list[str],
    universe: dict[str, InstrumentSpec],
) -> dict[str, list[str]]:
    """将品种列表按资产类别分组。

    Returns:
        dict: asset_class → [symbol, ...]
    """
    groups: dict[str, list[str]] = {}
    for sym in symbols:
        cls = universe[sym].asset_class
        groups.setdefault(cls, []).append(sym)
    return groups


def get_class_label(asset_class: str) -> str:
    """将底层资产类别映射到两层ERC的顶层大类标签。

    商品相关类别（industrial_metal / ferrous / energy / agriculture）
    在顶层统一归为 "commodity"，与 equity/bond/gold 并列。
    """
    if asset_class in COMMODITY_CLASSES:
        return "commodity"
    return asset_class


def top_level_classes(
    symbols: list[str],
    universe: dict[str, InstrumentSpec],
) -> dict[str, list[str]]:
    """将品种按顶层大类分组（equity/bond/gold/commodity）。

    Returns:
        dict: top_class → [symbol, ...]
    """
    groups: dict[str, list[str]] = {}
    for sym in symbols:
        label = get_class_label(universe[sym].asset_class)
        groups.setdefault(label, []).append(sym)
    return groups
