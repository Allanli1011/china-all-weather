"""绩效指标计算模块"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_performance_metrics(
    returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """计算全套绩效指标。

    Args:
        returns: 日收益率序列（不含NaN）
        benchmark_returns: 基准日收益率序列（如 IF 期货），可选
        risk_free_rate: 年化无风险利率（默认2%）
        periods_per_year: 每年交易日数（默认252）

    Returns:
        指标字典，包含：
            annualized_return, annualized_vol, sharpe_ratio,
            max_drawdown, max_drawdown_duration_days,
            calmar_ratio, sortino_ratio,
            win_rate_monthly（月胜率）,
            alpha, beta, information_ratio（若有基准）
    """
    ret = returns.dropna()
    if len(ret) < 2:
        return {}

    rf_daily = risk_free_rate / periods_per_year

    # 基础指标
    ann_ret = (1 + ret).prod() ** (periods_per_year / len(ret)) - 1
    ann_vol = ret.std() * np.sqrt(periods_per_year)
    sharpe = (ann_ret - risk_free_rate) / ann_vol if ann_vol > 0 else 0.0

    # 最大回撤
    cum = (1 + ret).cumprod()
    rolling_max = cum.cummax()
    drawdown = cum / rolling_max - 1
    max_dd = drawdown.min()

    # 最大回撤持续天数
    dd_duration = _max_drawdown_duration(drawdown)

    # Calmar 比率
    calmar = ann_ret / abs(max_dd) if max_dd < 0 else np.inf

    # Sortino 比率（下行波动率）
    downside = ret[ret < rf_daily]
    downside_vol = downside.std() * np.sqrt(periods_per_year) if len(downside) > 1 else 0.0
    sortino = (ann_ret - risk_free_rate) / downside_vol if downside_vol > 0 else 0.0

    # 月胜率
    monthly_ret = (1 + ret).resample("ME").prod() - 1
    win_rate = (monthly_ret > 0).sum() / max(len(monthly_ret), 1)

    result = {
        "annualized_return": round(ann_ret, 6),
        "annualized_vol": round(ann_vol, 6),
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown": round(max_dd, 6),
        "max_drawdown_duration_days": dd_duration,
        "calmar_ratio": round(calmar, 4),
        "sortino_ratio": round(sortino, 4),
        "win_rate_monthly": round(win_rate, 4),
        "total_return": round((1 + ret).prod() - 1, 6),
        "n_days": len(ret),
    }

    # 基准相关指标
    if benchmark_returns is not None:
        bm = benchmark_returns.reindex(ret.index).dropna()
        common = ret.reindex(bm.index).dropna()
        bm = bm.reindex(common.index)

        if len(common) >= 20:
            bm_ann = (1 + bm).prod() ** (periods_per_year / len(bm)) - 1

            # Alpha & Beta（OLS）
            cov_matrix = np.cov(common.values, bm.values)
            beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] > 0 else 0.0
            alpha_daily = common.mean() - beta * bm.mean()
            alpha_ann = alpha_daily * periods_per_year

            # 信息比（超额收益 / 跟踪误差）
            excess = common - bm
            te = excess.std() * np.sqrt(periods_per_year)
            ir = (excess.mean() * periods_per_year) / te if te > 0 else 0.0

            result.update({
                "benchmark_annualized_return": round(bm_ann, 6),
                "alpha": round(alpha_ann, 6),
                "beta": round(beta, 4),
                "information_ratio": round(ir, 4),
                "tracking_error": round(te, 6),
            })

    return result


def _max_drawdown_duration(drawdown: pd.Series) -> int:
    """计算最大回撤持续天数（从回撤开始到完全恢复）"""
    in_dd = drawdown < 0
    if not in_dd.any():
        return 0

    max_dur = 0
    cur_dur = 0
    for v in in_dd:
        if v:
            cur_dur += 1
            max_dur = max(max_dur, cur_dur)
        else:
            cur_dur = 0
    return max_dur


def compute_rolling_metrics(
    returns: pd.Series,
    window: int = 252,
    risk_free_rate: float = 0.02,
) -> pd.DataFrame:
    """计算滚动1年的关键指标。

    Returns:
        DataFrame，columns=[rolling_return, rolling_vol, rolling_sharpe, rolling_drawdown]
    """
    ret = returns.dropna()
    rf_daily = risk_free_rate / window

    rolling_ret = (1 + ret).rolling(window).apply(lambda x: x.prod() - 1, raw=True)
    rolling_vol = ret.rolling(window).std() * np.sqrt(window)
    rolling_sharpe = (rolling_ret - risk_free_rate) / rolling_vol.replace(0, np.nan)

    cum = (1 + ret).cumprod()
    rolling_max = cum.rolling(window, min_periods=1).max()
    rolling_dd = (cum / rolling_max - 1).rolling(window).min()

    return pd.DataFrame({
        "rolling_return": rolling_ret,
        "rolling_vol": rolling_vol,
        "rolling_sharpe": rolling_sharpe,
        "rolling_drawdown": rolling_dd,
    })


def format_metrics_table(
    metrics_dict: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """将多个策略的指标字典格式化为对比表格。

    Args:
        metrics_dict: {strategy_name: metrics_dict}

    Returns:
        DataFrame，行=指标，列=策略名
    """
    rows = {}
    key_order = [
        "annualized_return", "annualized_vol", "sharpe_ratio",
        "max_drawdown", "max_drawdown_duration_days", "calmar_ratio",
        "sortino_ratio", "win_rate_monthly", "total_return",
        "benchmark_annualized_return", "alpha", "beta",
        "information_ratio", "tracking_error",
    ]
    labels = {
        "annualized_return": "年化收益率",
        "annualized_vol": "年化波动率",
        "sharpe_ratio": "夏普比率",
        "max_drawdown": "最大回撤",
        "max_drawdown_duration_days": "最大回撤持续天数",
        "calmar_ratio": "Calmar比率",
        "sortino_ratio": "Sortino比率",
        "win_rate_monthly": "月胜率",
        "total_return": "总收益率",
        "benchmark_annualized_return": "基准年化收益率",
        "alpha": "Alpha（年化）",
        "beta": "Beta",
        "information_ratio": "信息比（IR）",
        "tracking_error": "跟踪误差",
    }
    for key in key_order:
        label = labels.get(key, key)
        row = {name: m.get(key) for name, m in metrics_dict.items()}
        if any(v is not None for v in row.values()):
            rows[label] = row

    return pd.DataFrame(rows).T
