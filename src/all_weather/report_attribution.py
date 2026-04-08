"""年度收益归因 HTML 报告"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .backtest import BacktestResult
    from .settings import Settings
    from .universe import InstrumentSpec

_PALETTE = ["#2563eb", "#dc2626", "#16a34a", "#d97706",
            "#7c3aed", "#0891b2", "#db2777", "#65a30d"]

# 大类颜色固定映射
_CLASS_COLORS = {
    "equity":    "#ef4444",   # 红：股票
    "bond":      "#3b82f6",   # 蓝：债券
    "gold":      "#f59e0b",   # 金：黄金
    "commodity": "#22c55e",   # 绿：商品
    "cost_drag": "#94a3b8",   # 灰：成本
    "other":     "#a855f7",
}

_CLASS_ZH = {
    "equity":    "股票",
    "bond":      "债券",
    "gold":      "黄金",
    "commodity": "商品",
    "cost_drag": "交易成本",
    "other":     "其他",
}

# 品种颜色（17个品种）
_SYM_PALETTE = [
    "#3b82f6","#2563eb","#1d4ed8",   # 蓝系：债券
    "#ef4444","#dc2626",              # 红系：股票
    "#f59e0b",                         # 金：黄金
    "#22c55e","#16a34a","#15803d",   # 绿系：工业金属
    "#f97316","#ea580c","#c2410c",   # 橙系：黑色金属
    "#0891b2",                         # 青：能源
    "#a855f7","#9333ea","#7c3aed","#6d28d9",  # 紫系：农产品
]


def generate_attribution_report(
    attribution_data: dict,
    returns: "pd.DataFrame",
    universe: dict[str, "InstrumentSpec"],
    settings: "Settings",
    output_path: "Path | None" = None,
) -> Path:
    """生成年度归因 HTML 报告。"""
    from .attribution import compute_rolling_class_sharpe, CLASS_LABELS_ZH

    if output_path is None:
        output_path = settings.reports_dir / "attribution_report.html"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── 为每个策略构建图表数据 ────────────────────────────────────────────
    strategy_sections = []

    for strategy_name, data in attribution_data.items():
        yearly_cls: pd.DataFrame = data["yearly_class"]
        yearly_inst: pd.DataFrame = data["yearly_instrument"]
        result = data["result"]

        if yearly_cls.empty:
            continue

        years = [str(y) for y in yearly_cls.index]
        class_cols = [c for c in yearly_cls.columns if c not in ("total", "cost_drag")]

        # ① 大类堆叠柱状图数据
        stacked_datasets = []
        for cls in class_cols:
            c = _CLASS_COLORS.get(cls, "#94a3b8")
            stacked_datasets.append({
                "label": _CLASS_ZH.get(cls, cls),
                "data": [round(float(v), 2) for v in yearly_cls[cls].values],
                "backgroundColor": c + "dd",
                "borderColor": c,
                "borderWidth": 0.5,
                "stack": "contribution",
            })
        if "cost_drag" in yearly_cls.columns:
            stacked_datasets.append({
                "label": "交易成本",
                "data": [round(float(v), 2) for v in yearly_cls["cost_drag"].values],
                "backgroundColor": "#94a3b8aa",
                "borderColor": "#94a3b8",
                "borderWidth": 0.5,
                "stack": "contribution",
            })
        # 折线叠加：组合总收益
        stacked_datasets.append({
            "label": "组合总收益",
            "data": [round(float(v), 2) for v in yearly_cls["total"].values],
            "type": "line",
            "borderColor": "#1e293b",
            "backgroundColor": "transparent",
            "borderWidth": 2,
            "pointRadius": 4,
            "pointBackgroundColor": "#1e293b",
            "order": 0,
        })
        stacked_json = json.dumps({"labels": years, "datasets": stacked_datasets})

        # ② 品种级年度贡献热力图（HTML 表格）
        inst_html = _render_instrument_heatmap(yearly_inst, universe)

        # ③ 滚动大类Sharpe
        roll_cls_sr = compute_rolling_class_sharpe(result, returns, universe)
        roll_sr_datasets = []
        roll_sr_dates = []
        if not roll_cls_sr.empty:
            roll_sr_dates = [str(d.date()) for d in roll_cls_sr.index]
            for cls in roll_cls_sr.columns:
                c = _CLASS_COLORS.get(cls, "#94a3b8")
                roll_sr_datasets.append({
                    "label": _CLASS_ZH.get(cls, cls),
                    "data": [round(float(v), 3) if not np.isnan(v) else None
                             for v in roll_cls_sr[cls].values],
                    "borderColor": c,
                    "backgroundColor": "transparent",
                    "borderWidth": 1.5,
                    "pointRadius": 0,
                    "spanGaps": True,
                })
        roll_sr_json = json.dumps({"labels": roll_sr_dates, "datasets": roll_sr_datasets})

        # ④ 年度归因数值表（HTML）
        yearly_table_html = _render_yearly_table(yearly_cls)

        # ⑤ 最大贡献 & 最大拖累（年份级洞察）
        insights = _compute_insights(yearly_cls, yearly_inst, universe)

        strategy_sections.append({
            "name": strategy_name,
            "stacked_json": stacked_json,
            "roll_sr_json": roll_sr_json,
            "yearly_table": yearly_table_html,
            "inst_heatmap": inst_html,
            "insights": insights,
        })

    html = _render_full_html(strategy_sections)
    output_path.write_text(html, encoding="utf-8")
    print(f"归因报告已生成: {output_path}")
    return output_path


def _render_yearly_table(yearly: pd.DataFrame) -> str:
    class_cols = [c for c in yearly.columns if c not in ("total", "cost_drag")]
    display_cols = class_cols + (["cost_drag"] if "cost_drag" in yearly.columns else []) + ["total"]

    header = "<tr><th>年份</th>" + "".join(
        f"<th>{_CLASS_ZH.get(c, c)}</th>" for c in display_cols
    ) + "</tr>"

    rows = []
    for yr, row in yearly.iterrows():
        cells = [f"<td><b>{yr}</b></td>"]
        for col in display_cols:
            val = row.get(col, float("nan"))
            if pd.isna(val):
                cells.append("<td>-</td>")
                continue
            intensity = min(abs(val) / 10.0, 1.0)
            if col == "cost_drag":
                r, g, b = int(220 + 35*(1-intensity)), int(255*(1-intensity*0.5)), int(255*(1-intensity*0.5))
            elif val >= 0:
                r, g, b = int(255*(1-intensity*0.5)), int(200+55*(1-intensity)), int(255*(1-intensity*0.5))
            else:
                r, g, b = int(200+55*(1-intensity)), int(255*(1-intensity*0.5)), int(255*(1-intensity*0.5))
            bold = "font-weight:600;" if col == "total" else ""
            cells.append(
                f'<td style="background:rgb({r},{g},{b});color:#1e293b;{bold}">{val:+.2f}%</td>'
            )
        rows.append("<tr>" + "".join(cells) + "</tr>")

    return (
        "<table class='attr-table'><thead>" + header +
        "</thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def _render_instrument_heatmap(yearly_inst: pd.DataFrame, universe: dict) -> str:
    """品种 × 年度贡献热力图（HTML table）。"""
    from .universe import get_class_label

    if yearly_inst.empty:
        return "<p>暂无品种级数据</p>"

    years = [str(c) for c in yearly_inst.columns]
    # 按大类分组排序品种
    syms = list(yearly_inst.index)
    sym_class = {s: (get_class_label(universe[s].asset_class) if s in universe else "other")
                 for s in syms}
    class_order = ["equity", "bond", "gold", "commodity", "other"]
    syms_sorted = sorted(syms, key=lambda s: (class_order.index(sym_class.get(s, "other"))
                                               if sym_class.get(s, "other") in class_order else 99, s))

    header = "<tr><th>品种</th><th>大类</th>" + "".join(f"<th>{y}</th>" for y in years) + "</tr>"
    rows = []
    prev_cls = None
    for sym in syms_sorted:
        cls = sym_class.get(sym, "other")
        cls_label = _CLASS_ZH.get(cls, cls)
        cls_cell = f'<td style="color:{_CLASS_COLORS.get(cls,"#64748b")};font-weight:600">{cls_label}</td>' \
                   if cls != prev_cls else "<td></td>"
        prev_cls = cls

        cells = [f"<td><b>{sym}</b></td>", cls_cell]
        for yr_col in yearly_inst.columns:
            val = yearly_inst.loc[sym, yr_col] if sym in yearly_inst.index else float("nan")
            if pd.isna(val):
                cells.append("<td>-</td>")
                continue
            intensity = min(abs(val) / 5.0, 1.0)
            if val >= 0:
                r, g, b = int(255*(1-intensity*0.5)), int(200+55*(1-intensity)), int(255*(1-intensity*0.5))
            else:
                r, g, b = int(200+55*(1-intensity)), int(255*(1-intensity*0.5)), int(255*(1-intensity*0.5))
            cells.append(
                f'<td style="background:rgb({r},{g},{b});color:#1e293b">{val:+.2f}%</td>'
            )
        rows.append("<tr>" + "".join(cells) + "</tr>")

    return (
        "<div style='overflow-x:auto'><table class='attr-table'><thead>" + header +
        "</thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def _compute_insights(yearly_cls: pd.DataFrame, yearly_inst: pd.DataFrame, universe: dict) -> list[str]:
    """提取关键洞察：每年最大贡献/最大拖累大类 & 品种。"""
    from .universe import get_class_label
    class_cols = [c for c in yearly_cls.columns if c not in ("total", "cost_drag")]
    insights = []

    for yr, row in yearly_cls.iterrows():
        total = row.get("total", 0.0)
        best_cls = max(class_cols, key=lambda c: row.get(c, -999)) if class_cols else None
        worst_cls = min(class_cols, key=lambda c: row.get(c, 999)) if class_cols else None
        best_val = row.get(best_cls, 0.0) if best_cls else 0.0
        worst_val = row.get(worst_cls, 0.0) if worst_cls else 0.0

        # 品种级最大贡献和最大拖累
        best_sym = worst_sym = best_sym_val = worst_sym_val = None
        if not yearly_inst.empty and yr in yearly_inst.columns:
            col = yearly_inst[yr].dropna()
            if not col.empty:
                best_sym = str(col.idxmax())
                best_sym_val = float(col.max())
                worst_sym = str(col.idxmin())
                worst_sym_val = float(col.min())

        parts = [
            f"<b>{yr}</b>: 总收益 <b>{total:+.1f}%</b>",
            f"最大贡献大类 <span class='tag-pos'>{_CLASS_ZH.get(best_cls, best_cls)} {best_val:+.1f}%</span>",
            f"最大拖累大类 <span class='tag-neg'>{_CLASS_ZH.get(worst_cls, worst_cls)} {worst_val:+.1f}%</span>",
        ]
        if best_sym:
            parts.append(f"最强品种 <b>{best_sym}</b> {best_sym_val:+.1f}%")
        if worst_sym:
            parts.append(f"最弱品种 <b>{worst_sym}</b> {worst_sym_val:+.1f}%")
        insights.append(" · ".join(parts))

    return insights


def _render_full_html(sections: list[dict]) -> str:
    section_htmls = []
    for i, sec in enumerate(sections):
        sid = f"s{i}"
        charts = f"""
<div class="card">
  <h2>{sec['name']} — 年度大类收益贡献（%）</h2>
  <div class="chart-wrap">
    <canvas id="stacked_{sid}"></canvas>
  </div>
</div>

<div class="grid2">
  <div class="card">
    <h2>{sec['name']} — 各大类滚动Sharpe（252日）</h2>
    <div class="chart-wrap-sm">
      <canvas id="rollsr_{sid}"></canvas>
    </div>
  </div>
  <div class="card">
    <h2>年度归因数值表</h2>
    <div style="overflow-x:auto">{sec['yearly_table']}</div>
  </div>
</div>

<div class="card">
  <h2>{sec['name']} — 品种级年度贡献热力图</h2>
  {sec['inst_heatmap']}
</div>

<div class="card">
  <h2>{sec['name']} — 逐年洞察</h2>
  <ul class="insights">
    {''.join(f"<li>{ins}</li>" for ins in sec['insights'])}
  </ul>
</div>
"""
        section_htmls.append(charts)

    # 所有图表的 JS 初始化
    js_inits = []
    for i, sec in enumerate(sections):
        sid = f"s{i}"
        js_inits.append(f"""
new Chart(document.getElementById('stacked_{sid}').getContext('2d'), {{
  type: 'bar',
  data: {sec['stacked_json']},
  options: {{
    responsive: true, maintainAspectRatio: false,
    interaction: {{mode: 'index', intersect: false}},
    plugins: {{
      legend: {{position: 'top', labels: {{boxWidth: 12, font: {{size: 11}}}}}},
      tooltip: {{callbacks: {{label: (c) => `${{c.dataset.label}}: ${{c.parsed.y?.toFixed(2)||'-'}}%`}}}}
    }},
    scales: {{
      x: {{ticks: {{font: {{size: 10}}}}}},
      y: {{
        stacked: true,
        title: {{display: true, text: '年度收益贡献 %'}},
        ticks: {{font: {{size: 10}}, callback: (v) => v + '%'}},
        grid: {{color: '#e2e8f0'}}
      }}
    }}
  }}
}});

new Chart(document.getElementById('rollsr_{sid}').getContext('2d'), {{
  type: 'line',
  data: {sec['roll_sr_json']},
  options: {{
    responsive: true, maintainAspectRatio: false,
    interaction: {{mode: 'index', intersect: false}},
    plugins: {{
      legend: {{position: 'top', labels: {{boxWidth: 12, font: {{size: 11}}}}}},
      annotation: {{annotations: {{zero: {{type: 'line', yMin: 0, yMax: 0, borderColor: '#94a3b8', borderWidth: 1}}}}}}
    }},
    scales: {{
      x: {{type: 'category', ticks: {{maxTicksLimit: 10, maxRotation: 0, font: {{size: 10}}}}}},
      y: {{
        title: {{display: true, text: 'Sharpe'}},
        ticks: {{font: {{size: 10}}, callback: (v) => v.toFixed(1)}},
        grid: {{color: '#e2e8f0'}}
      }}
    }}
  }}
}});
""")

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>全天候策略 — 年度归因分析</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  *{{box-sizing:border-box}}
  body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
       margin:0;padding:20px 24px;background:#f1f5f9;color:#1e293b}}
  h1{{font-size:1.5rem;font-weight:700;margin:0 0 4px}}
  h2{{font-size:0.95rem;font-weight:600;color:#475569;margin:0 0 14px}}
  .subtitle{{color:#64748b;font-size:0.85rem;margin-bottom:28px}}
  .grid2{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:20px}}
  @media(max-width:900px){{.grid2{{grid-template-columns:1fr}}}}
  .card{{background:white;border-radius:12px;padding:22px 24px;
         box-shadow:0 1px 3px rgba(0,0,0,.08);margin-bottom:20px}}
  .chart-wrap{{position:relative;height:320px}}
  .chart-wrap-sm{{position:relative;height:260px}}
  .attr-table{{border-collapse:collapse;width:100%;font-size:0.80rem;white-space:nowrap}}
  .attr-table th,.attr-table td{{padding:5px 10px;text-align:center;
    border:1px solid #e2e8f0}}
  .attr-table th{{background:#f8fafc;font-weight:600;position:sticky;top:0}}
  .attr-table td:first-child,.attr-table th:first-child{{text-align:left;min-width:60px}}
  .insights{{list-style:none;padding:0;margin:0;font-size:0.85rem;line-height:2.0}}
  .insights li{{padding:6px 0;border-bottom:1px solid #f1f5f9}}
  .tag-pos{{background:#dcfce7;color:#15803d;padding:1px 6px;border-radius:4px;font-weight:600}}
  .tag-neg{{background:#fee2e2;color:#b91c1c;padding:1px 6px;border-radius:4px;font-weight:600}}
  hr{{border:none;border-top:2px solid #e2e8f0;margin:32px 0}}
</style>
</head>
<body>
<h1>全天候期货策略 — 年度收益归因分析</h1>
<p class="subtitle">按资产大类（股票 / 债券 / 黄金 / 商品）拆解每年收益贡献 | 2015–2026</p>

{'<hr>'.join(section_htmls)}

<script>
{chr(10).join(js_inits)}
</script>
</body>
</html>"""
