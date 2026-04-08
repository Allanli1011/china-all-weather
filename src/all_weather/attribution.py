"""年度收益归因分析 — 按资产大类拆解每年收益贡献"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .backtest import BacktestResult
    from .universe import InstrumentSpec

# 大类顺序（用于固定图例顺序）
CLASS_ORDER = ["equity", "bond", "gold", "commodity"]
CLASS_LABELS_ZH = {
    "equity":    "股票（IF/IC）",
    "bond":      "债券（T/TF）",
    "gold":      "黄金（AU）",
    "commodity": "商品（工业/黑色/能源/农产品）",
    "other":     "其他",
}


def compute_daily_attribution(
    result: "BacktestResult",
    returns: pd.DataFrame,
    universe: dict[str, "InstrumentSpec"],
) -> pd.DataFrame:
    """计算每日各资产的收益贡献（权重 × 日收益率）。

    权重取法：用再平衡日快照做前向填充，得到每日实际持仓权重。
    注意：不归一化权重（保留杠杆敞口的绝对贡献）。

    Returns:
        DataFrame，index = 交易日，columns = 品种代码
    """
    wh = result.weights_history
    if wh.empty:
        return pd.DataFrame()

    port_idx = result.portfolio_returns.index
    # 前向填充：再平衡日之间权重不变
    daily_w = wh.reindex(port_idx, method="ffill").fillna(0.0)

    # 与日收益率对齐
    common_syms = [s for s in daily_w.columns if s in returns.columns]
    daily_ret = returns[common_syms].reindex(port_idx).fillna(0.0)
    daily_attr = daily_w[common_syms] * daily_ret

    return daily_attr


def compute_class_attribution(
    daily_attr: pd.DataFrame,
    universe: dict[str, "InstrumentSpec"],
) -> pd.DataFrame:
    """将品种级日归因聚合为资产大类级别。

    Returns:
        DataFrame，index = 交易日，columns = 大类名称（英文 key）
    """
    from .universe import get_class_label

    class_attr: dict[str, pd.Series] = {}
    for sym in daily_attr.columns:
        cls = get_class_label(universe[sym].asset_class) if sym in universe else "other"
        if cls not in class_attr:
            class_attr[cls] = daily_attr[sym].copy()
        else:
            class_attr[cls] = class_attr[cls] + daily_attr[sym]

    df = pd.DataFrame(class_attr)
    # 固定列顺序
    ordered = [c for c in CLASS_ORDER if c in df.columns]
    rest = [c for c in df.columns if c not in ordered]
    return df[ordered + rest]


def compute_yearly_class_attribution(
    result: "BacktestResult",
    returns: pd.DataFrame,
    universe: dict[str, "InstrumentSpec"],
) -> pd.DataFrame:
    """按年汇总各大类收益贡献（%）。

    Returns:
        DataFrame，index = 年份(int)，columns = 大类名称，values = 年度贡献(%)
    """
    daily_attr = compute_daily_attribution(result, returns, universe)
    if daily_attr.empty:
        return pd.DataFrame()

    class_attr = compute_class_attribution(daily_attr, universe)

    # 年度加总（连续复利近似用简单加总，日频下误差极小）
    yearly = class_attr.groupby(class_attr.index.year).sum() * 100
    yearly.index.name = "year"

    # 加一列：组合总收益（含成本，来自 portfolio_returns）
    port_ret = result.portfolio_returns
    yearly["total"] = (
        port_ret.groupby(port_ret.index.year)
        .apply(lambda x: float((1 + x).prod() - 1)) * 100
    )
    # 估算交易成本 = 总贡献合计 - 实际组合收益（含融资成本）
    class_cols = [c for c in yearly.columns if c != "total"]
    yearly["cost_drag"] = yearly["total"] - yearly[class_cols].sum(axis=1)

    return yearly


def compute_instrument_yearly_attribution(
    result: "BacktestResult",
    returns: pd.DataFrame,
    universe: dict[str, "InstrumentSpec"],
) -> pd.DataFrame:
    """品种级年度归因（按绝对贡献排序）。

    Returns:
        DataFrame，index = 品种代码，columns = 年份，values = 贡献(%)
    """
    daily_attr = compute_daily_attribution(result, returns, universe)
    if daily_attr.empty:
        return pd.DataFrame()

    yearly = daily_attr.groupby(daily_attr.index.year).sum() * 100
    return yearly.T  # index=symbol, columns=year


def compute_rolling_class_sharpe(
    result: "BacktestResult",
    returns: pd.DataFrame,
    universe: dict[str, "InstrumentSpec"],
    window: int = 252,
) -> pd.DataFrame:
    """各大类的滚动Sharpe（252日），月度采样。"""
    daily_attr = compute_daily_attribution(result, returns, universe)
    if daily_attr.empty:
        return pd.DataFrame()

    class_attr = compute_class_attribution(daily_attr, universe)
    roll_sr: dict[str, pd.Series] = {}
    for cls in class_attr.columns:
        s = class_attr[cls]
        rm = s.rolling(window).mean() * 252
        rv = s.rolling(window).std() * np.sqrt(252)
        sr = (rm / rv.replace(0, np.nan)).dropna()
        roll_sr[cls] = sr.resample("ME").last()

    return pd.DataFrame(roll_sr).dropna(how="all")


def format_yearly_table(yearly: pd.DataFrame) -> str:
    """生成年度归因的 HTML 表格（带颜色）。"""
    class_cols = [c for c in yearly.columns if c not in ("total", "cost_drag")]
    display_cols = class_cols + ["cost_drag", "total"]

    rows = []
    header_cells = "<th>年份</th>" + "".join(
        f"<th>{CLASS_LABELS_ZH.get(c, c)}</th>" for c in display_cols
    )
    rows.append(f"<tr>{header_cells}</tr>")

    for yr, row in yearly.iterrows():
        cells = [f"<td><b>{yr}</b></td>"]
        for col in display_cols:
            val = row.get(col, 0.0)
            if pd.isna(val):
                cells.append("<td>-</td>")
                continue
            # 颜色
            intensity = min(abs(val) / 10.0, 1.0)
            if col == "cost_drag":
                # cost is always negative → red shades
                r = int(220 + 35 * (1 - intensity))
                g = int(255 * (1 - intensity * 0.6))
                b = int(255 * (1 - intensity * 0.6))
            elif val >= 0:
                r = int(255 * (1 - intensity * 0.6))
                g = int(200 + 55 * (1 - intensity))
                b = int(255 * (1 - intensity * 0.6))
            else:
                r = int(200 + 55 * (1 - intensity))
                g = int(255 * (1 - intensity * 0.6))
                b = int(255 * (1 - intensity * 0.6))
            style = f"background:rgb({r},{g},{b});color:#1e293b"
            bold = " font-weight:600" if col == "total" else ""
            cells.append(
                f'<td style="{style}{bold}">{val:+.2f}%</td>'
            )
        rows.append("<tr>" + "".join(cells) + "</tr>")

    return "<table class='attr-table'><thead>" + rows[0] + "</thead><tbody>" + "".join(rows[1:]) + "</tbody></table>"
