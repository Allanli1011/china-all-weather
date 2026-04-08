"""风险平价权重计算 — 两层 ERC（等风险贡献）+ HRP 对照"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.optimize import minimize

if TYPE_CHECKING:
    from .settings import Settings
    from .universe import InstrumentSpec

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 协方差矩阵估计
# ─────────────────────────────────────────────────────────────────────────────

def compute_rolling_cov(
    returns: pd.DataFrame,
    window: int = 60,
    min_periods: int = 20,
    use_ledoit_wolf: bool = True,
) -> pd.DataFrame | None:
    """计算当前截面的协方差矩阵（使用最近 window 天数据）。

    Args:
        returns: 日收益率宽表（index=date, columns=symbol），可含 NaN
        window: 滚动窗口天数
        min_periods: 品种进入协方差矩阵的最少有效观测数
        use_ledoit_wolf: 是否使用 Ledoit-Wolf 收缩估计

    Returns:
        协方差矩阵 DataFrame（对称，正定），或 None（数据不足时）
    """
    if len(returns) < min_periods:
        return None

    # 取最近 window 天
    recent = returns.iloc[-window:] if len(returns) > window else returns

    # 过滤：仅保留有效观测数 >= min_periods 的品种
    valid_cols = [
        col for col in recent.columns
        if recent[col].notna().sum() >= min_periods
    ]
    if len(valid_cols) < 2:
        logger.debug("compute_rolling_cov: 有效品种数 < 2，无法计算协方差")
        return None

    sub = recent[valid_cols].copy()

    if use_ledoit_wolf:
        try:
            from sklearn.covariance import LedoitWolf
            # 删除含 NaN 的行（LedoitWolf 不支持缺失值）
            sub_clean = sub.dropna()
            if len(sub_clean) < min_periods:
                # 回退到 pandas 标准协方差
                cov_matrix = sub.cov()
            else:
                lw = LedoitWolf(assume_centered=False)
                lw.fit(sub_clean.values)
                cov_matrix = pd.DataFrame(
                    lw.covariance_,
                    index=valid_cols,
                    columns=valid_cols,
                )
        except Exception as e:
            logger.warning("LedoitWolf 失败，回退到标准协方差: %s", e)
            cov_matrix = sub.cov()
    else:
        cov_matrix = sub.cov()

    # 确保正定（添加微小对角线扰动）
    cov_matrix = _ensure_positive_definite(cov_matrix)
    return cov_matrix


def _ensure_positive_definite(cov: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
    """确保协方差矩阵正定（通过添加对角线扰动）"""
    vals = cov.values.copy()
    min_eig = np.linalg.eigvalsh(vals).min()
    if min_eig < eps:
        vals += (abs(min_eig) + eps) * np.eye(len(vals))
    return pd.DataFrame(vals, index=cov.index, columns=cov.columns)


# ─────────────────────────────────────────────────────────────────────────────
# ERC（等风险贡献）优化
# ─────────────────────────────────────────────────────────────────────────────

def solve_erc(
    cov: np.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-10,
    min_weight: float = 0.005,
    max_weight: float = 1.0,
) -> np.ndarray:
    """求解等风险贡献（ERC）权重。

    目标：最小化各品种风险贡献之间的差异
        minimize Σ_i Σ_j (RC_i - RC_j)²
        subject to: Σ w_i = 1, min_weight ≤ w_i ≤ max_weight

    其中 RC_i = w_i * (Σ w)_i（品种 i 的风险贡献）

    Args:
        cov: n×n 协方差矩阵（numpy array）
        max_iter: 最大迭代次数
        tol: 收敛容忍度
        min_weight: 单品种最低权重
        max_weight: 单品种最高权重

    Returns:
        权重数组（归一化，sum=1）
    """
    n = len(cov)
    if n == 1:
        return np.array([1.0])

    def portfolio_rc(weights: np.ndarray) -> np.ndarray:
        """计算各品种的风险贡献向量"""
        sigma = np.sqrt(weights @ cov @ weights)
        if sigma < 1e-12:
            return np.ones(n) / n
        marginal = cov @ weights
        rc = weights * marginal / sigma
        return rc

    def objective(weights: np.ndarray) -> float:
        """目标函数：各品种RC方差之和"""
        rc = portfolio_rc(weights)
        return float(np.sum((rc[:, None] - rc[None, :]) ** 2))

    def gradient(weights: np.ndarray) -> np.ndarray:
        """数值梯度（scipy 需要梯度信息时使用）"""
        eps = 1e-6
        grad = np.zeros(n)
        f0 = objective(weights)
        for i in range(n):
            w_plus = weights.copy()
            w_plus[i] += eps
            w_plus = w_plus / w_plus.sum()
            grad[i] = (objective(w_plus) - f0) / eps
        return grad

    # 初始权重：1/vol（波动率反向，作为热启动点）
    vols = np.sqrt(np.diag(cov))
    vols = np.where(vols < 1e-10, 1e-10, vols)
    w0 = (1.0 / vols)
    w0 = np.clip(w0, min_weight, max_weight)
    w0 /= w0.sum()

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(min_weight, max_weight)] * n

    result = minimize(
        fun=objective,
        x0=w0,
        method="SLSQP",
        jac=gradient,
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": max_iter, "ftol": tol},
    )

    if not result.success and result.fun > 1e-6:
        logger.debug("ERC 优化收敛警告: %s (fun=%.2e)", result.message, result.fun)

    weights = np.clip(result.x, min_weight, max_weight)
    weights /= weights.sum()
    return weights


# ─────────────────────────────────────────────────────────────────────────────
# 两层风险平价权重计算
# ─────────────────────────────────────────────────────────────────────────────

def compute_intra_basket_weights(
    returns: pd.DataFrame,
    basket_symbols: list[str],
    settings: "Settings",
) -> pd.Series:
    """计算单个子篮内的 ERC 权重（第一层）。

    Args:
        returns: 日收益率宽表（最近 window 天的切片）
        basket_symbols: 子篮内的品种列表
        settings: 策略配置

    Returns:
        归一化权重 Series（index=symbol），若数据不足则等权
    """
    rp_cfg = settings.risk_parity
    available = [s for s in basket_symbols if s in returns.columns]
    if not available:
        return pd.Series(dtype=float)

    if len(available) == 1:
        return pd.Series({available[0]: 1.0})

    cov_df = compute_rolling_cov(
        returns[available],
        window=rp_cfg["vol_window"],
        min_periods=rp_cfg["vol_min_periods"],
        use_ledoit_wolf=rp_cfg.get("use_ledoit_wolf", True),
    )

    if cov_df is None:
        # 数据不足，等权
        return pd.Series({s: 1.0 / len(available) for s in available})

    weights = solve_erc(
        cov=cov_df.values,
        max_iter=rp_cfg.get("erc_max_iter", 1000),
        tol=rp_cfg.get("erc_tol", 1e-10),
        min_weight=rp_cfg.get("min_weight", 0.005),
        max_weight=rp_cfg.get("max_weight", 0.40),
    )
    return pd.Series(dict(zip(cov_df.columns, weights)))


def compute_inter_class_weights(
    class_returns: pd.DataFrame,
    settings: "Settings",
) -> pd.Series:
    """计算顶层大类间的 ERC 权重（第二层）。

    Args:
        class_returns: 各大类合成收益率宽表（index=date, columns=class_label）
        settings: 策略配置

    Returns:
        归一化权重 Series（index=class_label）
    """
    rp_cfg = settings.risk_parity
    available_classes = [
        c for c in class_returns.columns
        if class_returns[c].notna().sum() >= rp_cfg["vol_min_periods"]
    ]

    if len(available_classes) == 1:
        return pd.Series({available_classes[0]: 1.0})

    cov_df = compute_rolling_cov(
        class_returns[available_classes],
        window=rp_cfg["vol_window"],
        min_periods=rp_cfg["vol_min_periods"],
        use_ledoit_wolf=rp_cfg.get("use_ledoit_wolf", True),
    )

    if cov_df is None:
        return pd.Series({c: 1.0 / len(available_classes) for c in available_classes})

    weights = solve_erc(
        cov=cov_df.values,
        max_iter=rp_cfg.get("erc_max_iter", 1000),
        tol=rp_cfg.get("erc_tol", 1e-10),
        min_weight=rp_cfg.get("min_weight", 0.005),
        max_weight=rp_cfg.get("max_weight", 0.40),
    )
    return pd.Series(dict(zip(cov_df.columns, weights)))


def compute_final_weights(
    returns: pd.DataFrame,
    universe: dict[str, "InstrumentSpec"],
    active_symbols: list[str],
    settings: "Settings",
) -> pd.Series:
    """两层 ERC 风险平价：计算各品种的最终权重。

    第一层：在每个底层资产类别内求解 ERC 权重
    第二层：将商品相关类别（industrial_metal/ferrous/energy/agriculture）
            合并为 "commodity" 大类，与 equity/bond/gold 并列，
            在顶层四大类间求解 ERC

    Args:
        returns: 日收益率宽表（含最近 vol_window 天历史）
        universe: 品种元数据字典
        active_symbols: 当前可参与组合的品种列表
        settings: 策略配置

    Returns:
        归一化权重 Series（index=symbol），sum=1
    """
    from .universe import COMMODITY_CLASSES, group_by_class

    rp_cfg = settings.risk_parity

    # 过滤：仅保留在 returns 中有足够历史的活跃品种
    min_obs = rp_cfg["vol_min_periods"]
    eligible = [
        s for s in active_symbols
        if s in returns.columns and returns[s].notna().sum() >= min_obs
    ]
    if not eligible:
        logger.warning("compute_final_weights: 无合格品种，返回空权重")
        return pd.Series(dtype=float)

    # 按底层资产类别分组
    class_groups = group_by_class(eligible, universe)

    # ── 第一层：各底层类别内的 ERC 权重 ──────────────────────────────────
    intra_weights: dict[str, pd.Series] = {}
    for cls, symbols in class_groups.items():
        intra_weights[cls] = compute_intra_basket_weights(returns, symbols, settings)

    # ── 计算每个底层类别的合成收益率（用于第二层协方差计算）─────────────
    # 商品相关底层类别合并到一个顶层 "commodity" 大类
    top_class_returns: dict[str, pd.Series] = {}
    commodity_parts: list[pd.Series] = []

    for cls, w_series in intra_weights.items():
        if w_series.empty:
            continue
        syms = w_series.index.tolist()
        cls_ret = returns[syms].mul(w_series, axis=1).sum(axis=1)

        if cls in COMMODITY_CLASSES:
            commodity_parts.append(cls_ret)
        else:
            top_class_returns[cls] = cls_ret

    # 商品大类合成收益率（各商品子篮的简单平均，权重在第二层重新分配）
    if commodity_parts:
        commodity_ret = pd.concat(commodity_parts, axis=1).mean(axis=1)
        top_class_returns["commodity"] = commodity_ret

    if not top_class_returns:
        logger.warning("compute_final_weights: 无有效顶层大类")
        return pd.Series(dtype=float)

    # ── 第二层：顶层大类间 ERC ────────────────────────────────────────────
    class_returns_df = pd.DataFrame(top_class_returns).reindex(returns.index)
    inter_weights = compute_inter_class_weights(class_returns_df, settings)

    # ── 合并两层权重 ──────────────────────────────────────────────────────
    # commodity 大类权重需要再拆分到各商品子篮
    commodity_classes = [c for c in class_groups if c in COMMODITY_CLASSES]
    n_commodity_classes = len(commodity_classes)
    commodity_class_weight = inter_weights.get("commodity", 0.0)

    # 商品子篮间按其各自合成收益率做等权（or 可再做一次 ERC，此处简化为等权分配）
    # 更精确方案：商品子篮间做 ERC，权重之和归一到 commodity_class_weight
    if n_commodity_classes > 0:
        sub_commodity_returns = pd.DataFrame(
            {cls: returns[intra_weights[cls].index.tolist()].mul(intra_weights[cls], axis=1).sum(axis=1)
             for cls in commodity_classes if not intra_weights[cls].empty}
        ).reindex(returns.index)

        if len(sub_commodity_returns.columns) > 1:
            sub_cov = compute_rolling_cov(
                sub_commodity_returns,
                window=rp_cfg["vol_window"],
                min_periods=rp_cfg["vol_min_periods"],
                use_ledoit_wolf=rp_cfg.get("use_ledoit_wolf", True),
            )
            if sub_cov is not None:
                sub_w_arr = solve_erc(
                    sub_cov.values,
                    max_iter=rp_cfg.get("erc_max_iter", 1000),
                    tol=rp_cfg.get("erc_tol", 1e-10),
                    min_weight=rp_cfg.get("min_weight", 0.005),
                    max_weight=rp_cfg.get("max_weight", 0.40),
                )
                sub_commodity_inter = pd.Series(
                    dict(zip(sub_cov.columns, sub_w_arr))
                )
            else:
                sub_commodity_inter = pd.Series(
                    {c: 1.0 / len(commodity_classes) for c in commodity_classes}
                )
        elif len(sub_commodity_returns.columns) == 1:
            sub_commodity_inter = pd.Series({sub_commodity_returns.columns[0]: 1.0})
        else:
            sub_commodity_inter = pd.Series(dtype=float)
    else:
        sub_commodity_inter = pd.Series(dtype=float)

    # 最终权重 = 顶层大类权重 × 类内权重
    final_weights: dict[str, float] = {}

    for cls, w_series in intra_weights.items():
        if w_series.empty:
            continue
        if cls in COMMODITY_CLASSES:
            # 商品子篮权重 = commodity大类权重 × 子篮间权重 × 品种内权重
            sub_cls_weight = sub_commodity_inter.get(cls, 0.0) * commodity_class_weight
        else:
            sub_cls_weight = inter_weights.get(cls, 0.0)

        for symbol, intra_w in w_series.items():
            final_weights[symbol] = sub_cls_weight * intra_w

    result = pd.Series(final_weights)
    total = result.sum()
    if total > 0:
        result /= total

    return result


# ─────────────────────────────────────────────────────────────────────────────
# HRP（层级风险平价）对照实现
# ─────────────────────────────────────────────────────────────────────────────

def compute_hrp_weights(
    returns: pd.DataFrame,
    active_symbols: list[str],
    settings: "Settings",
) -> pd.Series:
    """层级风险平价（Hierarchical Risk Parity）权重。

    基于 Lopez de Prado (2016) 的 HRP 方法：
    1. 计算相关系数矩阵
    2. 层次聚类（基于相关系数距离）
    3. 在聚类树上做二分风险平价

    无需协方差矩阵求逆，对样本量不足时更鲁棒。
    此实现作为 ERC 的对照组。

    Returns:
        归一化权重 Series（index=symbol），sum=1
    """
    rp_cfg = settings.risk_parity
    min_obs = rp_cfg["vol_min_periods"]

    eligible = [
        s for s in active_symbols
        if s in returns.columns and returns[s].notna().sum() >= min_obs
    ]
    if not eligible:
        return pd.Series(dtype=float)

    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform

    recent = returns[eligible].dropna(how="all")
    if len(recent) < min_obs:
        return pd.Series({s: 1.0 / len(eligible) for s in eligible})

    # 相关系数矩阵 → 距离矩阵
    corr = recent.corr().fillna(0.0)
    np.fill_diagonal(corr.values, 1.0)
    dist = np.sqrt(0.5 * (1 - corr.values))
    np.fill_diagonal(dist, 0.0)

    # 层次聚类
    dist_condensed = squareform(dist, checks=False)
    link = linkage(dist_condensed, method="ward")

    # 递归二分分配
    sorted_idx = _quasi_diag(link, len(eligible))
    sorted_symbols = [eligible[i] for i in sorted_idx]

    # 每个品种的方差（用于风险平价分配）
    vols = recent.std() ** 2  # 方差
    vols = vols.reindex(eligible).fillna(vols.mean())

    weights = _recursive_bisection(sorted_symbols, vols)
    result = pd.Series(weights)
    result = result / result.sum()
    return result


def _quasi_diag(link: np.ndarray, n_items: int) -> list[int]:
    """从 scipy linkage 矩阵提取准对角化排序"""
    link = link.astype(int)
    sort_idx = pd.Series([link[-1, 0], link[-1, 1]])

    num_items = link[-1, 3]
    while sort_idx.max() >= n_items:
        sort_idx.index = range(0, sort_idx.shape[0] * 2, 2)
        df0 = sort_idx[sort_idx >= n_items]
        i = df0.index
        j = df0.values - n_items
        sort_idx[i] = link[j, 0]
        df0 = pd.Series(link[j, 1], index=i + 1)
        sort_idx = pd.concat([sort_idx, df0]).sort_index()
        sort_idx = sort_idx.reset_index(drop=True)

    return sort_idx.tolist()


def _recursive_bisection(
    symbols: list[str],
    variances: pd.Series,
) -> dict[str, float]:
    """HRP 递归二分分配权重"""
    weights = {s: 1.0 for s in symbols}
    cluster_items = [symbols]

    while cluster_items:
        cluster = cluster_items.pop(0)
        if len(cluster) <= 1:
            continue

        mid = len(cluster) // 2
        left = cluster[:mid]
        right = cluster[mid:]

        # 每侧的聚类方差（倒数加权）
        left_var = _cluster_var(left, variances)
        right_var = _cluster_var(right, variances)

        total_var = left_var + right_var
        if total_var < 1e-12:
            left_alpha = 0.5
        else:
            left_alpha = right_var / total_var  # 风险平价：方差大的分配权重小

        for s in left:
            weights[s] *= left_alpha
        for s in right:
            weights[s] *= (1 - left_alpha)

        cluster_items.append(left)
        cluster_items.append(right)

    return weights


def _cluster_var(symbols: list[str], variances: pd.Series) -> float:
    """单个聚类的等权组合方差（近似）"""
    n = len(symbols)
    if n == 0:
        return 0.0
    return float(variances.reindex(symbols).fillna(0.0).mean())


# ─────────────────────────────────────────────────────────────────────────────
# 风险贡献验证工具
# ─────────────────────────────────────────────────────────────────────────────

def compute_risk_contributions(
    weights: pd.Series,
    cov: pd.DataFrame,
) -> pd.Series:
    """计算各品种的实际风险贡献（用于验证 ERC 结果）。

    Returns:
        RC Series（index=symbol），sum = 组合标准差
    """
    w = weights.reindex(cov.index).fillna(0.0).values
    sigma = np.sqrt(w @ cov.values @ w)
    if sigma < 1e-12:
        return pd.Series(0.0, index=cov.index)
    marginal = cov.values @ w
    rc = w * marginal / sigma
    return pd.Series(rc, index=cov.index)
