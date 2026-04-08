#!/usr/bin/env python3
# 建议每个交易日 16:05 后运行（上海期货所全面收盘后）
# crontab 示例: 5 16 * * 1-5 cd /path/to/all-weather-strategy && python3 run_monitor.py --leverage --target-vol 0.10 --per-class-thresholds >> logs/monitor.log 2>&1
"""
全天候期货策略 — 每日仓位监控入口

用法：
    # 默认配置（推荐配置：vol=10% + 分类阈值）
    python run_monitor.py

    # 指定配置文件
    python run_monitor.py --config configs/all_weather.yaml

    # 强制重新下载最新数据
    python run_monitor.py --force-refresh

    # 指定状态文件路径（多策略并行监控时使用）
    python run_monitor.py --state data/monitor_state_vol10.json

    # 保存日度报告到指定目录
    python run_monitor.py --report-dir reports/daily_monitor

建议每个交易日 16:00（中国市场收盘后）运行一次。
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent / "src"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="全天候策略日度仓位监控")
    p.add_argument(
        "--config",
        default="configs/all_weather.yaml",
        help="策略配置文件路径（默认: configs/all_weather.yaml）",
    )
    p.add_argument(
        "--state",
        default=None,
        help="状态文件路径（默认: data/monitor_state.json）",
    )
    p.add_argument(
        "--report-dir",
        default=None,
        help="日度报告输出目录（默认不保存）",
    )
    p.add_argument(
        "--force-refresh",
        action="store_true",
        help="强制重新下载所有行情数据",
    )
    p.add_argument(
        "--leverage",
        action="store_true",
        help="启用目标波动率杠杆（覆盖配置文件）",
    )
    p.add_argument(
        "--target-vol",
        type=float,
        default=None,
        help="目标年化波动率（如 0.10）",
    )
    p.add_argument(
        "--per-class-thresholds",
        action="store_true",
        help="使用分类再平衡阈值（覆盖配置文件）",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="只输出操作信号，不打印完整报告",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from all_weather.settings import load_settings
    from all_weather.monitor import PositionMonitor, print_monitor_report, save_daily_report

    # ── 加载并覆盖配置 ───────────────────────────────────────────────────────
    cfg_path = Path(args.config).resolve()
    settings = load_settings(cfg_path)

    if args.leverage:
        settings.raw["backtest"]["leverage_enabled"] = True
    if args.target_vol is not None:
        settings.raw["backtest"]["target_vol"] = args.target_vol
    if args.per_class_thresholds:
        settings.raw["backtest"].setdefault("per_class_thresholds", {})["enabled"] = True

    # ── 运行监控 ─────────────────────────────────────────────────────────────
    state_path = Path(args.state) if args.state else None
    monitor = PositionMonitor(settings=settings, state_path=state_path)

    result = monitor.run(force_refresh=args.force_refresh)

    if "error" in result:
        logger.error("监控失败: %s", result["error"])
        sys.exit(1)

    # ── 打印报告 ─────────────────────────────────────────────────────────────
    if not args.quiet:
        print_monitor_report(result)
    else:
        # 精简输出：只打印操作信号
        date = result["as_of_date"]
        pv = result["portfolio_value"]
        tr = result["today_return"]
        needs = result["needs_rebalance"]
        signal = "【调仓】" if needs else "【持仓】"
        triggered = ", ".join(result.get("triggered_classes", []))
        print(f"{date} | 净值={pv:.4f} | 今日{tr:+.2f}% | {signal} {triggered}")

    # ── 保存日度报告 ─────────────────────────────────────────────────────────
    if args.report_dir:
        report_path = save_daily_report(result, Path(args.report_dir))
        logger.info("日度报告已保存: %s", report_path)

    # ── 退出码：0=无需调仓，1=需要调仓（方便 cron 判断）────────────────────
    sys.exit(0 if not result["needs_rebalance"] else 1)


if __name__ == "__main__":
    main()
