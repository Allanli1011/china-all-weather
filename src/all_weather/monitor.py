"""
每日仓位监控 — 全天候期货策略

用法：
    python -m all_weather.monitor --config configs/all_weather.yaml

功能：
    1. 增量拉取最新期货行情（AKShare）
    2. 基于最新 60 日数据重新计算 ERC 目标权重（含杠杆缩放）
    3. 从状态文件加载昨日持仓，按今日价格漂移更新实际权重
    4. 检测是否触发再平衡（全局 L1 或分类阈值）
    5. 输出明日操作指令 + 目标仓位明细
    6. 保存今日状态供明日使用
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .settings import Settings
    from .universe import InstrumentSpec

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 状态文件
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MonitorState:
    """持久化到磁盘的日度监控状态"""

    as_of_date: str                           # 上次运行日期 YYYY-MM-DD
    current_weights: dict[str, float]         # 昨日收盘后持仓权重（含杠杆）
    target_weights: dict[str, float]          # 昨日计算的目标权重
    rebalance_pending: bool                   # 昨日是否触发了再平衡
    last_rebalance_date: str | None           # 上次实际调仓日期
    portfolio_value_est: float                # 估算净值（相对基准1.0）
    leverage: float                           # 昨日实际杠杆倍数

    # --- 工厂方法 ---

    @classmethod
    def load(cls, path: Path) -> "MonitorState | None":
        if not path.exists():
            return None
        try:
            return cls(**json.loads(path.read_text(encoding="utf-8")))
        except Exception as e:
            logger.warning("状态文件加载失败: %s", e)
            return None

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# 核心监控逻辑
# ─────────────────────────────────────────────────────────────────────────────

class PositionMonitor:
    """每日仓位监控器"""

    def __init__(self, settings: "Settings", state_path: Path | None = None):
        self.settings = settings
        self.state_path = state_path or (settings.data_dir / "monitor_state.json")

    def run(self, force_refresh: bool = False) -> dict:
        """执行一次日度监控，返回监控结果字典。"""
        from .backtest import (
            _check_rebalance_trigger,
            _compute_leverage,
            _update_weights_after_drift,
        )
        from .data_fetcher import fetch_all_instruments
        from .data_store import DataStore
        from .returns import build_returns_pipeline
        from .risk_parity import compute_final_weights
        from .universe import filter_active_instruments, load_universe

        settings = self.settings
        universe = load_universe(settings)
        store = DataStore(settings)

        # ── 1. 拉取最新数据 ─────────────────────────────────────────────────
        logger.info("增量获取行情数据...")
        fetch_all_instruments(
            universe_specs=universe,
            start_date=settings.data_start_date,
            store=store,
            settings=settings,
            force_refresh=force_refresh,
        )

        # ── 2. 构建收益率矩阵 ────────────────────────────────────────────────
        price_df, returns_df, is_rollover_df = build_returns_pipeline(
            store=store, universe=universe, settings=settings
        )
        today = returns_df.index[-1]
        logger.info("最新数据日期: %s", today.date())

        # ── 3. 加载状态文件 ──────────────────────────────────────────────────
        state = MonitorState.load(self.state_path)
        is_first_run = state is None
        if is_first_run:
            logger.info("首次运行，将以当前 ERC 权重初始化状态")

        # ── 4. 计算 ERC 目标权重（基于最新 60 日）──────────────────────────
        rp_cfg = settings.risk_parity
        vol_window: int = rp_cfg["vol_window"]
        min_periods: int = rp_cfg["vol_min_periods"]
        min_history: int = rp_cfg["min_history_days"]

        today_iloc = returns_df.index.get_loc(today)
        hist_start = max(0, today_iloc - vol_window)
        hist_returns = returns_df.iloc[hist_start : today_iloc]   # 不含今日

        active = filter_active_instruments(universe, today, min_history_days=min_history)
        if len(hist_returns) < min_periods or not active:
            logger.error("历史数据不足，无法计算目标权重")
            return {"error": "insufficient_data"}

        base_weights = compute_final_weights(hist_returns, universe, active, settings)

        # ── 5. 目标波动率杠杆缩放 ───────────────────────────────────────────
        bt_cfg = settings.backtest
        leverage_enabled: bool = bt_cfg.get("leverage_enabled", False)
        target_vol: float = float(bt_cfg.get("target_vol", 0.10))
        max_leverage: float = float(bt_cfg.get("max_leverage", 3.0))

        if leverage_enabled and not base_weights.empty:
            target_weights, leverage_ratio = _compute_leverage(
                base_weights, hist_returns, target_vol, max_leverage, min_periods
            )
        else:
            target_weights = base_weights
            leverage_ratio = 1.0

        # ── 6. 推算当前实际持仓权重（含今日漂移）──────────────────────────
        today_returns = returns_df.loc[today]
        if is_first_run:
            # 首次：假设当前 = 今日目标（净建仓状态）
            current_weights = target_weights.copy()
            portfolio_value = 1.0
            last_rebalance_date = str(today.date())
        else:
            prev_weights = pd.Series(state.current_weights)
            # 应用今日价格漂移
            current_weights = _update_weights_after_drift(prev_weights, today_returns)
            portfolio_value = state.portfolio_value_est

            # 若昨日触发了再平衡，则今日已按昨日目标执行调仓
            if state.rebalance_pending:
                prev_target = pd.Series(state.target_weights)
                current_weights = prev_target.copy()
                last_rebalance_date = str(today.date())
                logger.info("今日已执行昨日触发的再平衡，当前权重 = 昨日目标权重")
            else:
                last_rebalance_date = state.last_rebalance_date

        # ── 7. 更新净值估算 ──────────────────────────────────────────────────
        held_syms = [s for s in current_weights.index if s in returns_df.columns]
        w_held = current_weights.reindex(held_syms).fillna(0.0)
        gross_ret = float(w_held.dot(today_returns.reindex(held_syms).fillna(0.0)))
        financing_rate_daily = float(bt_cfg.get("financing_rate", 0.02)) / 252
        lever = float(current_weights.sum())
        financing_cost = max(lever - 1.0, 0.0) * financing_rate_daily
        today_net_ret = gross_ret - financing_cost
        portfolio_value *= (1 + today_net_ret)

        # ── 8. 检测再平衡触发 ────────────────────────────────────────────────
        needs_rebalance = _check_rebalance_trigger(
            current_weights, target_weights, universe, settings
        )

        # ── 9. 计算各类偏差 ──────────────────────────────────────────────────
        class_deviations = _compute_class_deviations_detail(
            current_weights, target_weights, universe, settings
        )
        global_deviation = float(
            (current_weights.reindex(target_weights.index.union(current_weights.index), fill_value=0.0)
             - target_weights.reindex(target_weights.index.union(current_weights.index), fill_value=0.0))
            .abs().sum()
        )

        # ── 10. 保存今日状态 ──────────────────────────────────────────────────
        new_state = MonitorState(
            as_of_date=str(today.date()),
            current_weights=current_weights.to_dict(),
            target_weights=target_weights.to_dict(),
            rebalance_pending=needs_rebalance,
            last_rebalance_date=last_rebalance_date,
            portfolio_value_est=round(portfolio_value, 6),
            leverage=round(lever, 4),
        )
        new_state.save(self.state_path)
        logger.info("状态已保存: %s", self.state_path)

        return {
            "as_of_date": str(today.date()),
            "today_return": round(today_net_ret * 100, 4),
            "portfolio_value": round(portfolio_value, 4),
            "leverage": round(lever, 4),
            "needs_rebalance": needs_rebalance,
            "executed_rebalance_today": (not is_first_run and state.rebalance_pending),
            "last_rebalance_date": last_rebalance_date,
            "global_deviation": round(global_deviation * 100, 2),
            "class_deviations": {k: round(v * 100, 2) for k, v in class_deviations["deviations"].items()},
            "class_thresholds": {k: round(v * 100, 2) for k, v in class_deviations["thresholds"].items()},
            "triggered_classes": class_deviations["triggered"],
            "target_weights": target_weights.round(6).to_dict(),
            "current_weights": current_weights.round(6).to_dict(),
            "universe": universe,
            "is_rollover_today": _get_today_rollovers(is_rollover_df, today),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────────────────────────────────────

def _compute_class_deviations_detail(
    current_weights: pd.Series,
    target_weights: pd.Series,
    universe: dict,
    settings: "Settings",
) -> dict:
    """返回各大类偏差、阈值及触发列表"""
    from .universe import get_class_label

    bt_cfg = settings.backtest
    per_class_cfg = bt_cfg.get("per_class_thresholds", {})
    use_per_class = isinstance(per_class_cfg, dict) and per_class_cfg.get("enabled", False)
    global_threshold = float(bt_cfg.get("rebalance_threshold", 0.10))

    all_syms = current_weights.index.union(target_weights.index)
    cur = current_weights.reindex(all_syms, fill_value=0.0)
    tgt = target_weights.reindex(all_syms, fill_value=0.0)

    class_devs: dict[str, float] = {}
    for sym in all_syms:
        cls = get_class_label(universe[sym].asset_class) if sym in universe else "other"
        class_devs[cls] = class_devs.get(cls, 0.0) + abs(float(cur[sym]) - float(tgt[sym]))

    class_thresholds: dict[str, float] = {}
    triggered: list[str] = []
    for cls, dev in class_devs.items():
        if use_per_class:
            thr = float(per_class_cfg.get(cls, global_threshold))
        else:
            thr = global_threshold
        class_thresholds[cls] = thr
        if dev > thr:
            triggered.append(cls)

    return {"deviations": class_devs, "thresholds": class_thresholds, "triggered": triggered}


def _get_today_rollovers(is_rollover_df: pd.DataFrame, today: pd.Timestamp) -> list[str]:
    """返回今日发生展期的品种列表"""
    if is_rollover_df is None or is_rollover_df.empty:
        return []
    if today not in is_rollover_df.index:
        return []
    row = is_rollover_df.loc[today]
    return [str(c) for c in row[row == True].index]


# ─────────────────────────────────────────────────────────────────────────────
# 报告打印
# ─────────────────────────────────────────────────────────────────────────────

def print_monitor_report(result: dict) -> None:
    """打印人类可读的监控报告"""
    from .universe import get_class_label

    w = 68
    sep = "=" * w
    thin = "-" * w

    date_str = result["as_of_date"]
    print(f"\n{sep}")
    print(f"  全天候期货策略 — 日度监控报告  {date_str} (收盘后)")
    print(sep)

    # ── 今日总结 ──────────────────────────────────────────────────────────
    pv = result["portfolio_value"]
    tr = result["today_return"]
    lev = result["leverage"]
    sign = "+" if tr >= 0 else ""
    print(f"\n  今日净值: {pv:.4f}  今日收益: {sign}{tr:.2f}%  杠杆: {lev:.2f}x")

    rollovers = result.get("is_rollover_today", [])
    if rollovers:
        print(f"  ⚡ 今日发生展期: {', '.join(rollovers)}")

    if result.get("executed_rebalance_today"):
        print(f"  ✅ 今日已执行调仓（昨日触发）→ 持仓已更新")

    # ── 明日操作信号 ──────────────────────────────────────────────────────
    print(f"\n{thin}")
    needs = result["needs_rebalance"]
    if needs:
        triggered = result.get("triggered_classes", [])
        cls_zh = {"equity": "股票", "bond": "债券", "gold": "黄金", "commodity": "商品"}
        triggered_str = " / ".join(cls_zh.get(c, c) for c in triggered)
        print(f"  ⚠️  明日操作: 【需要调仓】  触发大类: {triggered_str}")
    else:
        print(f"  ✅ 明日操作: 【无需调仓】  持仓偏差在阈值内")

    lrd = result.get("last_rebalance_date") or "未知"
    print(f"  上次调仓: {lrd}  全局 L1 偏差: {result['global_deviation']:.2f}%")

    # ── 分类偏差 ──────────────────────────────────────────────────────────
    print(f"\n  偏差分析:")
    cls_zh = {"equity": "股票", "bond": "债券", "gold": "黄金",
              "commodity": "商品", "other": "其他"}
    devs = result["class_deviations"]
    thrs = result["class_thresholds"]
    triggered_set = set(result.get("triggered_classes", []))
    for cls, dev in sorted(devs.items()):
        thr = thrs.get(cls, 10.0)
        flag = " ⚠️ 触发" if cls in triggered_set else ""
        bar_filled = int(dev / thr * 20)
        bar = "█" * min(bar_filled, 20) + "░" * (20 - min(bar_filled, 20))
        print(f"  {cls_zh.get(cls, cls):6s}  [{bar}] {dev:5.1f}% / {thr:.0f}%{flag}")

    # ── 目标权重 vs 当前权重 ──────────────────────────────────────────────
    print(f"\n{thin}")
    print(f"  {'品种':6s} {'大类':8s} {'目标权重':>9s} {'当前权重':>9s} {'差异':>8s} {'方向':>6s}")
    print(f"  {'-'*6} {'-'*8} {'-'*9} {'-'*9} {'-'*8} {'-'*6}")

    target_w = result["target_weights"]
    current_w = result["current_weights"]
    universe = result.get("universe", {})

    all_syms = sorted(
        set(target_w.keys()) | set(current_w.keys()),
        key=lambda s: (get_class_label(universe[s].asset_class) if s in universe else "z", s)
    )
    prev_cls = None
    for sym in all_syms:
        tgt = target_w.get(sym, 0.0) * 100
        cur = current_w.get(sym, 0.0) * 100
        diff = tgt - cur
        direction = "买入↑" if diff > 0.05 else ("卖出↓" if diff < -0.05 else "持平 ")
        cls = get_class_label(universe[sym].asset_class) if sym in universe else "other"
        cls_zh_map = {"equity": "股票", "bond": "债券", "gold": "黄金",
                      "commodity": "商品", "other": "其他"}
        cls_label = cls_zh_map.get(cls, cls)
        if cls != prev_cls and prev_cls is not None:
            print(f"  {'':-<6} {'':-<8} {'':-<9} {'':-<9} {'':-<8} {'':-<6}")
        prev_cls = cls
        diff_str = f"{diff:+.2f}%"
        print(f"  {sym:6s} {cls_label:8s} {tgt:8.2f}% {cur:8.2f}% {diff_str:>8s} {direction:>6s}")

    print(f"{sep}\n")


def save_daily_report(result: dict, output_dir: Path) -> Path:
    """保存当日监控报告为 JSON 文件"""
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"monitor_{result['as_of_date']}.json"
    path = output_dir / filename
    # Remove non-serializable items
    serializable = {k: v for k, v in result.items() if k != "universe"}
    path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
