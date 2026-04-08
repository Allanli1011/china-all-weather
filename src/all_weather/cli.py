"""命令行入口 — 全天候期货策略"""

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="全天候期货策略 — 基于风险平价的中国期货市场多资产组合"
    )
    p.add_argument(
        "--config",
        default="configs/all_weather.yaml",
        help="策略配置文件路径（默认: configs/all_weather.yaml）",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="全局再平衡阈值（覆盖配置文件，如 0.10）",
    )
    p.add_argument(
        "--leverage",
        action="store_true",
        help="启用目标波动率杠杆（target_vol 来自配置文件，默认10%%）",
    )
    p.add_argument(
        "--target-vol",
        type=float,
        default=None,
        help="目标年化波动率（配合 --leverage 使用，如 0.10）",
    )
    p.add_argument(
        "--per-class-thresholds",
        action="store_true",
        help="启用分类再平衡阈值（债券20%%/黄金15%%/商品8%%）",
    )
    p.add_argument(
        "--experiment",
        choices=["leverage", "per_class", "combined", "all", "summary"],
        default=None,
        help=(
            "运行预设实验组合: "
            "leverage=杠杆对比, per_class=分类阈值对比, "
            "combined=杠杆+分类阈值, all=全部, "
            "summary=精选4组关键配置对比报告"
        ),
    )
    p.add_argument(
        "--compare-thresholds",
        action="store_true",
        help="运行全局阈值对比实验（0.05/0.10/0.15/0.20）",
    )
    p.add_argument(
        "--compare-hrp",
        action="store_true",
        help="同时运行 HRP 对照实验",
    )
    p.add_argument(
        "--force-refresh",
        action="store_true",
        help="强制重新下载所有数据（忽略本地缓存）",
    )
    p.add_argument(
        "--force-rebuild",
        action="store_true",
        help="强制重建收益率矩阵（忽略处理后的缓存）",
    )
    p.add_argument(
        "--no-report",
        action="store_true",
        help="跳过 HTML 报告生成",
    )
    p.add_argument(
        "--attribution",
        action="store_true",
        help="生成年度收益归因分析报告（按资产大类拆解每年贡献）",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── 1. 加载配置 ───────────────────────────────────────────────────────
    from .settings import load_settings
    cfg_path = Path(args.config).resolve()
    logger.info("加载配置: %s", cfg_path)
    settings = load_settings(cfg_path)

    if args.threshold is not None:
        settings.raw["backtest"]["rebalance_threshold"] = args.threshold
        logger.info("覆盖再平衡阈值: %.2f", args.threshold)

    # ── 2. 加载品种宇宙 ───────────────────────────────────────────────────
    from .universe import load_universe
    universe = load_universe(settings)
    logger.info("品种宇宙: %d 个品种", len(universe))

    # ── 3. 数据获取 ───────────────────────────────────────────────────────
    from .data_fetcher import fetch_all_instruments
    from .data_store import DataStore

    store = DataStore(settings)
    logger.info("开始获取数据（force_refresh=%s）...", args.force_refresh)
    fetch_all_instruments(
        universe_specs=universe,
        start_date=settings.data_start_date,
        store=store,
        settings=settings,
        force_refresh=args.force_refresh,
    )

    # ── 4. 构建收益率矩阵 ─────────────────────────────────────────────────
    from .returns import build_returns_pipeline
    logger.info("构建收益率矩阵（force_rebuild=%s）...", args.force_rebuild)
    price_df, returns_df, is_rollover_df = build_returns_pipeline(
        store=store,
        universe=universe,
        settings=settings,
        force_rebuild=args.force_rebuild,
    )
    logger.info(
        "收益率矩阵: %d 个品种, %d 天 (%s ~ %s)",
        len(returns_df.columns),
        len(returns_df),
        returns_df.index.min().date(),
        returns_df.index.max().date(),
    )

    # ── 5. 应用 CLI 覆盖参数 ──────────────────────────────────────────────
    if args.leverage:
        settings.raw["backtest"]["leverage_enabled"] = True
        logger.info("启用杠杆模式")
    if args.target_vol is not None:
        settings.raw["backtest"]["target_vol"] = args.target_vol
        logger.info("目标波动率: %.0f%%", args.target_vol * 100)
    if args.per_class_thresholds:
        settings.raw["backtest"].setdefault("per_class_thresholds", {})["enabled"] = True
        logger.info("启用分类再平衡阈值")

    # ── 6. 运行回测 ───────────────────────────────────────────────────────
    from .backtest import run_backtest, run_experiment_grid, run_threshold_comparison

    results: dict = {}

    if args.experiment:
        results.update(_run_preset_experiments(
            args.experiment, returns_df, price_df, universe, settings
        ))
    elif args.compare_thresholds:
        logger.info("运行全局阈值对比实验...")
        results.update(run_threshold_comparison(
            returns=returns_df, prices=price_df,
            universe=universe, settings=settings,
            thresholds=[0.05, 0.10, 0.15, 0.20],
        ))
    else:
        # 单次运行
        threshold = settings.rebalance_threshold
        lev = settings.backtest.get("leverage_enabled", False)
        logger.info("运行 ERC 策略（threshold=%.2f, leverage=%s）...", threshold, lev)
        erc_result = run_backtest(
            returns=returns_df, prices=price_df,
            universe=universe, settings=settings, use_hrp=False,
        )
        results[erc_result.strategy_name] = erc_result

    if args.compare_hrp:
        logger.info("运行 HRP 对照策略...")
        hrp_result = run_backtest(
            returns=returns_df, prices=price_df,
            universe=universe, settings=settings,
            use_hrp=True, strategy_name="HRP",
        )
        results["HRP"] = hrp_result

    # ── 7. 基准（IF期货）收益率 ───────────────────────────────────────────
    bm_symbol = settings.benchmark_symbol.replace("0", "")  # "IF0" → "IF"
    benchmark_returns = returns_df.get(bm_symbol)

    # ── 8. 计算绩效指标并保存 ──────────────────────────────────────────────
    from .metrics import compute_performance_metrics, format_metrics_table

    metrics_dict = {}
    for name, result in results.items():
        bm = benchmark_returns.reindex(result.portfolio_returns.index) if benchmark_returns is not None else None
        m = compute_performance_metrics(result.portfolio_returns, bm)
        m["avg_annual_rebalances"] = round(result.avg_annual_rebalances, 1)
        m["n_rebalances"] = result.n_rebalances
        metrics_dict[name] = m
        result.performance = m

    metrics_table = format_metrics_table(metrics_dict)

    # 打印关键结果
    print("\n" + "=" * 70)
    print("全天候期货策略 — 回测结果")
    print("=" * 70)
    for name, result in results.items():
        m = metrics_dict[name]
        print(f"\n[{name}]")
        print(f"  年化收益率:   {m.get('annualized_return', 0):.2%}")
        print(f"  年化波动率:   {m.get('annualized_vol', 0):.2%}")
        print(f"  夏普比率:     {m.get('sharpe_ratio', 0):.4f}")
        print(f"  最大回撤:     {m.get('max_drawdown', 0):.2%}")
        print(f"  Calmar比率:   {m.get('calmar_ratio', 0):.4f}")
        if "information_ratio" in m:
            print(f"  信息比(IR):   {m.get('information_ratio', 0):.4f}")
        print(f"  年均再平衡:   {m.get('avg_annual_rebalances', 0):.1f} 次")
        print(f"  平均杠杆:     {result.avg_leverage:.2f}x")
    print("=" * 70)

    # 保存指标 CSV
    csv_path = store.save_report_csv("backtest_summary", metrics_table)
    logger.info("绩效指标已保存: %s", csv_path)

    # 保存权重历史和再平衡日志
    for name, result in results.items():
        if not result.weights_history.empty:
            store.save_report_df(f"weights_{name}", result.weights_history)
        if not result.rebalance_log.empty:
            store.save_report_csv(f"rebalance_log_{name}", result.rebalance_log)

    # ── 9. 生成 HTML 报告 ──────────────────────────────────────────────────
    if not args.no_report:
        from .report import generate_html_report
        generate_html_report(
            results=results,
            benchmark_returns=benchmark_returns,
            settings=settings,
        )

    # ── 10. 年度归因分析 ────────────────────────────────────────────────────
    if args.attribution:
        from .attribution import (
            compute_yearly_class_attribution,
            compute_instrument_yearly_attribution,
            format_yearly_table,
        )
        from .report_attribution import generate_attribution_report

        logger.info("生成年度归因分析...")
        attribution_data = {}
        for name, result in results.items():
            yearly_cls = compute_yearly_class_attribution(result, returns_df, universe)
            yearly_inst = compute_instrument_yearly_attribution(result, returns_df, universe)
            attribution_data[name] = {
                "yearly_class": yearly_cls,
                "yearly_instrument": yearly_inst,
                "result": result,
            }

        attr_path = generate_attribution_report(
            attribution_data=attribution_data,
            returns=returns_df,
            universe=universe,
            settings=settings,
        )
        logger.info("归因报告已生成: %s", attr_path)

    logger.info("完成")


def _run_preset_experiments(
    experiment: str,
    returns_df,
    price_df,
    universe,
    settings,
) -> dict:
    """运行预设实验组合"""
    from .backtest import run_experiment_grid

    # 实验1：杠杆对比（无杠杆 vs target_vol=8%/10%/12%）
    leverage_experiments = [
        {"name": "ERC_无杠杆",       "leverage_enabled": False, "threshold_override": 0.10},
        {"name": "ERC_杠杆_vol8%",   "leverage_enabled": True,  "target_vol": 0.08,  "threshold_override": 0.10},
        {"name": "ERC_杠杆_vol10%",  "leverage_enabled": True,  "target_vol": 0.10,  "threshold_override": 0.10},
        {"name": "ERC_杠杆_vol12%",  "leverage_enabled": True,  "target_vol": 0.12,  "threshold_override": 0.10},
    ]

    # 实验2：分类阈值对比（全局10% vs 分类阈值）
    per_class_experiments = [
        {"name": "ERC_全局阈值10%",  "leverage_enabled": False, "per_class_enabled": False, "threshold_override": 0.10},
        {"name": "ERC_分类阈值",     "leverage_enabled": False, "per_class_enabled": True},
    ]

    # 实验3：最优组合（杠杆10% + 分类阈值）
    combined_experiments = [
        {"name": "ERC_无杠杆_全局",      "leverage_enabled": False, "per_class_enabled": False, "threshold_override": 0.10},
        {"name": "ERC_杠杆_全局",        "leverage_enabled": True,  "target_vol": 0.10, "per_class_enabled": False, "threshold_override": 0.10},
        {"name": "ERC_无杠杆_分类阈值",  "leverage_enabled": False, "per_class_enabled": True},
        {"name": "ERC_杠杆_分类阈值",    "leverage_enabled": True,  "target_vol": 0.10, "per_class_enabled": True},
    ]

    # 精选4组：无杠杆基线 / 保守杠杆vol8% / 高效vol10%分类阈值 / 最优Sharpe vol12%全局
    summary_experiments = [
        {"name": "ERC_无杠杆",          "leverage_enabled": False, "per_class_enabled": False, "threshold_override": 0.10},
        {"name": "ERC_vol8%_全局",      "leverage_enabled": True,  "target_vol": 0.08,  "per_class_enabled": False, "threshold_override": 0.10},
        {"name": "ERC_vol10%_分类阈值", "leverage_enabled": True,  "target_vol": 0.10,  "per_class_enabled": True},
        {"name": "ERC_vol12%_全局",     "leverage_enabled": True,  "target_vol": 0.12,  "per_class_enabled": False, "threshold_override": 0.10},
    ]

    exp_map = {
        "leverage":   leverage_experiments,
        "per_class":  per_class_experiments,
        "combined":   combined_experiments,
        "all":        leverage_experiments + per_class_experiments[1:] + combined_experiments[2:],
        "summary":    summary_experiments,
    }
    experiments = exp_map.get(experiment, combined_experiments)
    logger.info("运行预设实验 [%s]，共 %d 组...", experiment, len(experiments))
    return run_experiment_grid(returns_df, price_df, universe, settings, experiments)


if __name__ == "__main__":
    main()
