"""回测引擎 — 日度循环，阈值触发再平衡（支持目标波动率杠杆 + 分类阈值）"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .settings import Settings
    from .universe import InstrumentSpec

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 结果数据类
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    """回测结果容器"""

    equity_curve: pd.Series               # 组合净值（从1开始）
    portfolio_returns: pd.Series          # 组合日收益率
    weights_history: pd.DataFrame         # 持仓权重历史（再平衡后快照）
    rebalance_log: pd.DataFrame           # 再平衡日志
    leverage_history: pd.Series           # 每日实际杠杆倍数
    performance: dict[str, float] = field(default_factory=dict)
    strategy_name: str = "all_weather_erc"

    @property
    def n_rebalances(self) -> int:
        return len(self.rebalance_log)

    @property
    def avg_annual_rebalances(self) -> float:
        n_days = len(self.equity_curve)
        years = n_days / 252
        return self.n_rebalances / max(years, 0.01)

    @property
    def avg_leverage(self) -> float:
        if self.leverage_history.empty:
            return 1.0
        return float(self.leverage_history.mean())


# ─────────────────────────────────────────────────────────────────────────────
# 杠杆计算
# ─────────────────────────────────────────────────────────────────────────────

def _compute_portfolio_vol(
    weights: pd.Series,
    hist_returns: pd.DataFrame,
    min_periods: int = 20,
) -> float:
    """用历史收益率估计当前组合的年化波动率。

    Args:
        weights: 目标权重（归一化，sum=1）
        hist_returns: 最近 N 天日收益率宽表
        min_periods: 最少有效观测数

    Returns:
        年化组合波动率（若数据不足返回 0）
    """
    syms = [s for s in weights.index if s in hist_returns.columns]
    if not syms or len(hist_returns) < min_periods:
        return 0.0

    w = weights.reindex(syms).fillna(0.0).values
    sub = hist_returns[syms].dropna(how="all")
    if len(sub) < min_periods:
        return 0.0

    cov = sub.cov().values
    port_var = float(w @ cov @ w)
    return float(np.sqrt(max(port_var, 0) * 252))


def _compute_leverage(
    base_weights: pd.Series,
    hist_returns: pd.DataFrame,
    target_vol: float,
    max_leverage: float,
    min_periods: int = 20,
) -> tuple[pd.Series, float]:
    """计算杠杆倍数并缩放权重以命中目标波动率。

    原理：若当前组合波动率 σ < target_vol，则整体加杠杆 k = target_vol/σ，
    使得杠杆后组合波动率 ≈ target_vol。
    缩放后各权重之和 = k（>1时表示借款融资）。

    Args:
        base_weights: ERC 原始归一化权重（sum=1）
        hist_returns: 最近 N 天日收益率
        target_vol: 目标年化波动率
        max_leverage: 杠杆上限
        min_periods: 最少有效观测数

    Returns:
        (scaled_weights, leverage_ratio)
        scaled_weights：各品种目标权重，sum = leverage_ratio
        leverage_ratio：实际施加的杠杆倍数
    """
    current_vol = _compute_portfolio_vol(base_weights, hist_returns, min_periods)
    if current_vol < 1e-4:
        return base_weights.copy(), 1.0

    leverage = min(target_vol / current_vol, max_leverage)
    leverage = max(leverage, 1.0)   # 不降杠杆（仅在波动率偏低时加杠杆）
    scaled = base_weights * leverage
    return scaled, leverage


# ─────────────────────────────────────────────────────────────────────────────
# 再平衡触发逻辑
# ─────────────────────────────────────────────────────────────────────────────

def _compute_l1_deviation(
    current_weights: pd.Series,
    target_weights: pd.Series,
) -> float:
    """计算整体 L1 偏差"""
    all_syms = current_weights.index.union(target_weights.index)
    cur = current_weights.reindex(all_syms, fill_value=0.0)
    tgt = target_weights.reindex(all_syms, fill_value=0.0)
    return float((cur - tgt).abs().sum())


def _compute_class_deviations(
    current_weights: pd.Series,
    target_weights: pd.Series,
    universe: dict[str, "InstrumentSpec"],
) -> dict[str, float]:
    """计算各资产大类的 L1 偏差（用于分类阈值触发）"""
    from .universe import get_class_label

    all_syms = current_weights.index.union(target_weights.index)
    cur = current_weights.reindex(all_syms, fill_value=0.0)
    tgt = target_weights.reindex(all_syms, fill_value=0.0)

    class_devs: dict[str, float] = {}
    for sym in all_syms:
        cls = get_class_label(universe[sym].asset_class) if sym in universe else "other"
        class_devs[cls] = class_devs.get(cls, 0.0) + abs(float(cur[sym]) - float(tgt[sym]))
    return class_devs


def _check_rebalance_trigger(
    current_weights: pd.Series,
    target_weights: pd.Series,
    universe: dict[str, "InstrumentSpec"],
    settings: "Settings",
) -> bool:
    """判断是否触发再平衡。

    模式1（全局阈值）：L1(current, target) > rebalance_threshold
    模式2（分类阈值）：任意大类的类内 L1 偏差 > 该类阈值
    """
    bt_cfg = settings.backtest
    per_class_cfg = bt_cfg.get("per_class_thresholds", {})
    use_per_class = isinstance(per_class_cfg, dict) and per_class_cfg.get("enabled", False)

    if use_per_class:
        class_devs = _compute_class_deviations(current_weights, target_weights, universe)
        fallback = float(bt_cfg.get("rebalance_threshold", 0.10))
        for cls, dev in class_devs.items():
            threshold = float(per_class_cfg.get(cls, fallback))
            if dev > threshold:
                return True
        return False
    else:
        threshold = float(bt_cfg.get("rebalance_threshold", 0.10))
        return _compute_l1_deviation(current_weights, target_weights) > threshold


# ─────────────────────────────────────────────────────────────────────────────
# 核心回测引擎
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(
    returns: pd.DataFrame,
    prices: pd.DataFrame,
    universe: dict[str, "InstrumentSpec"],
    settings: "Settings",
    use_hrp: bool = False,
    strategy_name: str | None = None,
) -> BacktestResult:
    """执行全天候策略回测（支持杠杆 + 分类阈值）。

    日度循环逻辑：
    1. 计算 ERC/HRP 目标权重
    2. [可选] 按目标波动率缩放权重（杠杆）
    3. 检测是否触发再平衡（全局 L1 或分类阈值）
    4. 再平衡日：扣手续费+滑点，更新持仓
    5. 计算当日组合收益率（含融资成本）
    6. 价格漂移更新实际权重
    """
    from .risk_parity import compute_final_weights, compute_hrp_weights
    from .universe import filter_active_instruments

    rp_cfg = settings.risk_parity
    bt_cfg = settings.backtest

    vol_window: int = rp_cfg["vol_window"]
    min_periods: int = rp_cfg["vol_min_periods"]
    min_history: int = rp_cfg["min_history_days"]

    # 杠杆配置
    leverage_enabled: bool = bt_cfg.get("leverage_enabled", False)
    target_vol: float = float(bt_cfg.get("target_vol", 0.10))
    max_leverage: float = float(bt_cfg.get("max_leverage", 3.0))
    financing_rate_daily: float = float(bt_cfg.get("financing_rate", 0.02)) / 252

    # 回测期间
    start_date = pd.Timestamp(bt_cfg.get("start_date", returns.index.min()))
    end_date = (
        pd.Timestamp(bt_cfg["end_date"])
        if bt_cfg.get("end_date")
        else returns.index.max()
    )
    dates = returns.index[(returns.index >= start_date) & (returns.index <= end_date)]
    if len(dates) < vol_window + min_history:
        raise ValueError(
            f"回测数据不足：需要 {vol_window + min_history} 天，实际 {len(dates)} 天"
        )

    # 初始化
    portfolio_value = float(bt_cfg.get("initial_capital", 10_000_000))
    current_weights = pd.Series(dtype=float)   # 当前实际持仓权重（可 > 1 表示杠杆）
    target_weights = pd.Series(dtype=float)    # 最新目标权重（含杠杆缩放）
    current_leverage = 1.0
    rebalance_next_day = False

    daily_returns: list[float] = []
    daily_dates: list[pd.Timestamp] = []
    daily_leverages: list[float] = []
    weights_snapshots: list[dict] = []
    rebalance_records: list[dict] = []

    lev_tag = f"_lev{int(target_vol*100)}pct" if leverage_enabled else ""
    name = strategy_name or ("all_weather_hrp" if use_hrp else f"all_weather_erc{lev_tag}")

    per_class_enabled = bt_cfg.get("per_class_thresholds", {}).get("enabled", False)
    threshold_desc = (
        "per-class" if per_class_enabled
        else f"L1={bt_cfg.get('rebalance_threshold', 0.10):.2f}"
    )
    logger.info(
        "开始回测: %s | %s~%s | threshold=%s | leverage=%s (target_vol=%.0f%%)",
        name, start_date.date(), end_date.date(),
        threshold_desc,
        leverage_enabled, target_vol * 100,
    )

    returns_index = returns.index

    for date in dates:
        date_iloc = returns_index.get_loc(date)

        # ── 1. 计算 ERC/HRP 目标权重 ───────────────────────────────────────
        active = filter_active_instruments(universe, date, min_history_days=min_history)
        hist_start = max(0, date_iloc - vol_window)
        hist_returns = returns.iloc[hist_start:date_iloc]

        if len(hist_returns) >= min_periods and active:
            base_weights = (
                compute_hrp_weights(hist_returns, active, settings)
                if use_hrp
                else compute_final_weights(hist_returns, universe, active, settings)
            )
        else:
            base_weights = pd.Series(dtype=float)

        # ── 2. 目标波动率杠杆缩放 ──────────────────────────────────────────
        if not base_weights.empty:
            if leverage_enabled:
                scaled_weights, current_leverage = _compute_leverage(
                    base_weights, hist_returns, target_vol, max_leverage, min_periods
                )
                target_weights = scaled_weights
            else:
                target_weights = base_weights
                current_leverage = 1.0

        # ── 3. 执行昨日触发的再平衡 ────────────────────────────────────────
        if rebalance_next_day and not target_weights.empty:
            cost = _compute_rebalance_cost(
                current_weights, target_weights,
                prices.loc[date], universe, portfolio_value,
            )
            portfolio_value -= cost
            current_weights = target_weights.copy()
            rebalance_next_day = False

            rebalance_records.append({
                "date": date,
                "cost": cost,
                "cost_rate": cost / portfolio_value if portfolio_value > 0 else 0.0,
                "leverage": current_leverage,
                "n_active": len(active),
            })
            weights_snapshots.append({"date": date, **current_weights.to_dict()})

        # ── 4. 首次建仓 ────────────────────────────────────────────────────
        elif current_weights.empty and not target_weights.empty:
            current_weights = target_weights.copy()
            weights_snapshots.append({"date": date, **current_weights.to_dict()})

        # ── 5. 计算当日组合收益率 ──────────────────────────────────────────
        if not current_weights.empty:
            held = [s for s in current_weights.index if s in returns.columns]
            day_ret = returns.loc[date, held].fillna(0.0)
            w = current_weights.reindex(held).fillna(0.0)
            gross_ret = float(w.dot(day_ret))

            # 融资成本：仅对杠杆部分（权重之和 - 1）计收
            lever = float(current_weights.sum())
            financing_cost = max(lever - 1.0, 0.0) * financing_rate_daily
            port_ret = gross_ret - financing_cost
        else:
            port_ret = 0.0
            lever = 1.0

        portfolio_value *= (1 + port_ret)
        daily_returns.append(port_ret)
        daily_dates.append(date)
        daily_leverages.append(lever)

        # ── 6. 价格漂移更新实际权重 ────────────────────────────────────────
        if not current_weights.empty:
            current_weights = _update_weights_after_drift(
                current_weights, returns.loc[date]
            )

        # ── 7. 检测再平衡触发 ──────────────────────────────────────────────
        if not current_weights.empty and not target_weights.empty:
            if _check_rebalance_trigger(current_weights, target_weights, universe, settings):
                rebalance_next_day = True

    # ── 构建结果 ──────────────────────────────────────────────────────────────
    idx = pd.DatetimeIndex(daily_dates)
    returns_series = pd.Series(daily_returns, index=idx, name=name)
    equity_curve = (1 + returns_series).cumprod()
    equity_curve /= equity_curve.iloc[0]

    weights_df = (
        pd.DataFrame(weights_snapshots).set_index("date")
        if weights_snapshots else pd.DataFrame()
    )
    rebalance_df = (
        pd.DataFrame(rebalance_records)
        if rebalance_records
        else pd.DataFrame(columns=["date", "cost", "cost_rate", "leverage", "n_active"])
    )
    leverage_series = pd.Series(daily_leverages, index=idx, name="leverage")

    logger.info(
        "回测完成: %s | 总收益=%.1f%% | 再平衡=%d次(年均%.1f) | 均杠杆=%.2fx",
        name,
        (equity_curve.iloc[-1] - 1) * 100,
        len(rebalance_df),
        len(rebalance_df) / max(len(dates) / 252, 0.01),
        leverage_series.mean(),
    )

    return BacktestResult(
        equity_curve=equity_curve,
        portfolio_returns=returns_series,
        weights_history=weights_df,
        rebalance_log=rebalance_df,
        leverage_history=leverage_series,
        strategy_name=name,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────────────────────────────────────

def _update_weights_after_drift(
    current_weights: pd.Series,
    day_returns: pd.Series,
) -> pd.Series:
    """价格漂移后更新持仓权重（保持杠杆比例不变）"""
    ret = day_returns.reindex(current_weights.index).fillna(0.0)
    new_values = current_weights * (1 + ret)
    # 归一化到与原始权重相同的"总仓位"（保持杠杆倍数）
    old_total = current_weights.sum()
    new_total = new_values.sum()
    if new_total > 1e-10 and old_total > 1e-10:
        return new_values * (old_total / new_total)
    return current_weights.copy()


def _compute_rebalance_cost(
    current_weights: pd.Series,
    target_weights: pd.Series,
    current_prices: pd.Series,
    universe: dict[str, "InstrumentSpec"],
    portfolio_value: float,
) -> float:
    """计算再平衡成本（手续费 + 滑点）"""
    all_syms = current_weights.index.union(target_weights.index)
    cur = current_weights.reindex(all_syms, fill_value=0.0)
    tgt = target_weights.reindex(all_syms, fill_value=0.0)
    delta_w = (tgt - cur).abs()

    total_cost = 0.0
    for symbol, dw in delta_w.items():
        if dw < 1e-8 or symbol not in universe:
            continue
        spec = universe[symbol]
        price = float(current_prices.get(symbol, 0.0))
        if price <= 0:
            continue
        slip_rate = spec.tick_size * spec.slippage_ticks / price
        cost_rate = spec.fee_rate + slip_rate
        total_cost += dw * portfolio_value * cost_rate

    return total_cost


# ─────────────────────────────────────────────────────────────────────────────
# 对比实验函数
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment_grid(
    returns: pd.DataFrame,
    prices: pd.DataFrame,
    universe: dict[str, "InstrumentSpec"],
    settings: "Settings",
    experiments: list[dict],
) -> dict[str, BacktestResult]:
    """批量运行多组参数实验。

    Args:
        experiments: 每项为 {name, leverage_enabled, target_vol,
                              per_class_enabled, threshold_override}
    """
    results: dict[str, BacktestResult] = {}
    for exp in experiments:
        # 临时 patch 配置
        bt = settings.raw["backtest"]
        orig = {k: bt.get(k) for k in ["leverage_enabled", "target_vol",
                                         "rebalance_threshold"]}
        orig_per = bt.get("per_class_thresholds", {}).copy()

        bt["leverage_enabled"] = exp.get("leverage_enabled", False)
        bt["target_vol"] = exp.get("target_vol", 0.10)
        if "threshold_override" in exp:
            bt["rebalance_threshold"] = exp["threshold_override"]
        pct = bt.setdefault("per_class_thresholds", {})
        pct["enabled"] = exp.get("per_class_enabled", False)

        name = exp["name"]
        logger.info("实验: %s", name)
        result = run_backtest(returns, prices, universe, settings, strategy_name=name)
        results[name] = result

        # 恢复配置
        for k, v in orig.items():
            if v is None:
                bt.pop(k, None)
            else:
                bt[k] = v
        bt["per_class_thresholds"] = orig_per

    return results


def run_threshold_comparison(
    returns: pd.DataFrame,
    prices: pd.DataFrame,
    universe: dict[str, "InstrumentSpec"],
    settings: "Settings",
    thresholds: list[float] | None = None,
) -> dict[str, BacktestResult]:
    """不同全局阈值对比实验（向后兼容）"""
    if thresholds is None:
        thresholds = [0.05, 0.10, 0.15, 0.20]
    experiments = [
        {"name": f"ERC_threshold_{int(t*100)}pct",
         "leverage_enabled": False,
         "threshold_override": t}
        for t in thresholds
    ]
    return run_experiment_grid(returns, prices, universe, settings, experiments)
