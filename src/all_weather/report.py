"""HTML 可视化报告生成（增强版）"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .backtest import BacktestResult
    from .settings import Settings


def generate_html_report(
    results: dict[str, "BacktestResult"],
    benchmark_returns: pd.Series | None,
    settings: "Settings",
    output_path: Path | None = None,
) -> Path:
    """生成增强版 HTML 回测报告。

    包含：净值曲线、回撤曲线、年度收益、滚动Sharpe、
    权重构成、杠杆历史、月度收益热力图、绩效指标表。
    """
    from .metrics import compute_performance_metrics, format_metrics_table

    if output_path is None:
        output_path = settings.reports_dir / "backtest_report.html"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── 计算绩效指标 ───────────────────────────────────────────────────────
    metrics_dict: dict[str, dict] = {}
    for name, result in results.items():
        bm = benchmark_returns.reindex(result.portfolio_returns.index) if benchmark_returns is not None else None
        m = compute_performance_metrics(result.portfolio_returns, bm)
        m["avg_annual_rebalances"] = round(result.avg_annual_rebalances, 1)
        m["n_rebalances"] = result.n_rebalances
        metrics_dict[name] = m
        result.performance = m

    if benchmark_returns is not None:
        bm_name = settings.benchmark_symbol
        metrics_dict[f"基准({bm_name})"] = compute_performance_metrics(
            benchmark_returns.dropna()
        )

    metrics_table = format_metrics_table(metrics_dict)

    # ── 净值曲线 ──────────────────────────────────────────────────────────
    equity_data: dict[str, dict] = {}
    for name, result in results.items():
        equity_data[name] = {
            "dates": [str(d.date()) for d in result.equity_curve.index],
            "values": [round(float(v), 6) for v in result.equity_curve.values],
        }
    if benchmark_returns is not None:
        first_result = list(results.values())[0]
        bm_ret = benchmark_returns.reindex(first_result.portfolio_returns.index).fillna(0)
        bm_eq = (1 + bm_ret).cumprod()
        bm_eq = bm_eq / bm_eq.iloc[0]
        equity_data[f"基准({settings.benchmark_symbol})"] = {
            "dates": [str(d.date()) for d in bm_eq.index],
            "values": [round(float(v), 6) for v in bm_eq.values],
        }

    # ── 回撤曲线 ──────────────────────────────────────────────────────────
    drawdown_data: dict[str, dict] = {}
    for name, result in results.items():
        eq = result.equity_curve
        rolling_max = eq.cummax()
        dd = (eq / rolling_max - 1) * 100
        drawdown_data[name] = {
            "dates": [str(d.date()) for d in dd.index],
            "values": [round(float(v), 4) for v in dd.values],
        }
    if benchmark_returns is not None:
        bm_eq_series = pd.Series(
            equity_data[f"基准({settings.benchmark_symbol})"]["values"],
            index=pd.DatetimeIndex(equity_data[f"基准({settings.benchmark_symbol})"]["dates"])
        )
        bm_dd = (bm_eq_series / bm_eq_series.cummax() - 1) * 100
        drawdown_data[f"基准({settings.benchmark_symbol})"] = {
            "dates": [str(d.date()) for d in bm_dd.index],
            "values": [round(float(v), 4) for v in bm_dd.values],
        }

    # ── 年度收益率 ────────────────────────────────────────────────────────
    annual_data: dict[str, dict] = {}
    for name, result in results.items():
        rets = result.portfolio_returns
        annual = rets.groupby(rets.index.year).apply(
            lambda x: float((1 + x).prod() - 1) * 100
        )
        annual_data[name] = {
            "years": [str(y) for y in annual.index],
            "values": [round(v, 2) for v in annual.values],
        }
    if benchmark_returns is not None:
        first_result = list(results.values())[0]
        bm_ret_aligned = benchmark_returns.reindex(first_result.portfolio_returns.index).fillna(0)
        bm_annual = bm_ret_aligned.groupby(bm_ret_aligned.index.year).apply(
            lambda x: float((1 + x).prod() - 1) * 100
        )
        annual_data[f"基准({settings.benchmark_symbol})"] = {
            "years": [str(y) for y in bm_annual.index],
            "values": [round(v, 2) for v in bm_annual.values],
        }
    all_years = sorted({y for d in annual_data.values() for y in d["years"]})

    # ── 滚动Sharpe (252日) ────────────────────────────────────────────────
    rolling_sharpe_data: dict[str, dict] = {}
    for name, result in results.items():
        rets = result.portfolio_returns
        roll_mean = rets.rolling(252).mean() * 252
        roll_std = rets.rolling(252).std() * np.sqrt(252)
        roll_sr = (roll_mean / roll_std.replace(0, np.nan)).dropna()
        # 每月采样一次降低数据量
        roll_sr_m = roll_sr.resample("ME").last().dropna()
        rolling_sharpe_data[name] = {
            "dates": [str(d.date()) for d in roll_sr_m.index],
            "values": [round(float(v), 4) for v in roll_sr_m.values],
        }

    # ── 权重构成（前3个策略的最新权重饼图 + 历史面积图）────────────────────
    weight_pie_data: dict[str, dict] = {}
    weight_area_data: dict[str, dict] = {}
    for name, result in list(results.items())[:4]:
        if result.weights_history.empty:
            continue
        # 按资产大类聚合最新权重
        last_w = result.weights_history.iloc[-1].dropna()
        total = last_w.sum()
        if total > 0:
            last_w = last_w / total
        weight_pie_data[name] = {
            "labels": list(last_w.index),
            "values": [round(float(v) * 100, 2) for v in last_w.values],
        }
        # 权重面积图（月度采样）
        wh = result.weights_history
        if hasattr(wh.index, 'freq') or len(wh) > 0:
            wh_m = wh.resample("ME").last().ffill() if len(wh) > 10 else wh
            syms = [c for c in wh_m.columns if not wh_m[c].isna().all()]
            row_sums = wh_m[syms].sum(axis=1).replace(0, np.nan)
            wh_norm = wh_m[syms].divide(row_sums, axis=0).fillna(0)
            weight_area_data[name] = {
                "dates": [str(d.date()) for d in wh_norm.index],
                "symbols": syms,
                "series": {s: [round(float(v), 4) for v in wh_norm[s].values] for s in syms},
            }

    # ── 杠杆历史 ─────────────────────────────────────────────────────────
    leverage_data: dict[str, dict] = {}
    for name, result in results.items():
        if result.leverage_history.empty:
            continue
        lev_m = result.leverage_history.resample("ME").mean()
        leverage_data[name] = {
            "dates": [str(d.date()) for d in lev_m.index],
            "values": [round(float(v), 4) for v in lev_m.values],
        }

    # ── 月度收益热力图（取第一个策略）────────────────────────────────────
    monthly_heatmap_data: dict | None = None
    if results:
        first_name, first_result = next(iter(results.items()))
        rets = first_result.portfolio_returns
        monthly = rets.groupby([rets.index.year, rets.index.month]).apply(
            lambda x: float((1 + x).prod() - 1) * 100
        )
        monthly_dict: dict[int, dict[int, float]] = {}
        for (yr, mo), val in monthly.items():
            monthly_dict.setdefault(yr, {})[mo] = round(val, 2)
        monthly_heatmap_data = {
            "strategy": first_name,
            "data": {str(yr): {str(mo): v for mo, v in months.items()}
                     for yr, months in sorted(monthly_dict.items())},
        }

    # ── 再平衡统计 ────────────────────────────────────────────────────────
    rebal_info = {
        name: {
            "n_rebalances": result.n_rebalances,
            "avg_annual": round(result.avg_annual_rebalances, 1),
            "avg_leverage": round(result.avg_leverage, 2),
        }
        for name, result in results.items()
    }

    html = _render_html(
        metrics_table=metrics_table,
        equity_data=equity_data,
        drawdown_data=drawdown_data,
        annual_data=annual_data,
        all_years=all_years,
        rolling_sharpe_data=rolling_sharpe_data,
        weight_pie_data=weight_pie_data,
        weight_area_data=weight_area_data,
        leverage_data=leverage_data,
        monthly_heatmap_data=monthly_heatmap_data,
        rebal_info=rebal_info,
    )
    output_path.write_text(html, encoding="utf-8")
    print(f"HTML 报告已生成: {output_path}")
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# HTML 渲染
# ─────────────────────────────────────────────────────────────────────────────

_PALETTE = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed",
            "#0891b2", "#db2777", "#65a30d", "#ea580c", "#0d9488"]

_WEIGHT_PALETTE = [
    "#3b82f6","#ef4444","#22c55e","#f59e0b","#a855f7",
    "#06b6d4","#ec4899","#84cc16","#f97316","#14b8a6",
    "#6366f1","#8b5cf6","#10b981","#f43f5e","#0ea5e9",
    "#fbbf24","#34d399",
]


def _mk_datasets(data_dict: dict, palette=None, fill=False, tension=0.0):
    palette = palette or _PALETTE
    datasets = []
    for i, (name, d) in enumerate(data_dict.items()):
        c = palette[i % len(palette)]
        datasets.append({
            "label": name,
            "data": d["values"],
            "borderColor": c,
            "backgroundColor": c + ("40" if fill else "20"),
            "borderWidth": 1.5,
            "pointRadius": 0,
            "fill": fill,
            "tension": tension,
        })
    return datasets


def _render_html(
    metrics_table: pd.DataFrame,
    equity_data: dict,
    drawdown_data: dict,
    annual_data: dict,
    all_years: list[str],
    rolling_sharpe_data: dict,
    weight_pie_data: dict,
    weight_area_data: dict,
    leverage_data: dict,
    monthly_heatmap_data: dict | None,
    rebal_info: dict,
) -> str:

    metrics_html = metrics_table.to_html(
        classes="metrics-table", border=0,
        float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x),
        na_rep="-",
    )

    first_dates = list(equity_data.values())[0]["dates"] if equity_data else []
    first_dd_dates = list(drawdown_data.values())[0]["dates"] if drawdown_data else []

    eq_datasets = _mk_datasets(equity_data)
    dd_datasets = _mk_datasets(drawdown_data, fill=True)
    roll_sr_datasets = _mk_datasets(rolling_sharpe_data)

    # Annual bar datasets (grouped by year)
    annual_datasets = []
    for i, (name, d) in enumerate(annual_data.items()):
        val_map = dict(zip(d["years"], d["values"]))
        annual_datasets.append({
            "label": name,
            "data": [val_map.get(y, None) for y in all_years],
            "backgroundColor": _PALETTE[i % len(_PALETTE)] + "cc",
        })

    # Leverage line datasets
    lev_datasets = []
    lev_dates = []
    for i, (name, d) in enumerate(leverage_data.items()):
        if not lev_dates:
            lev_dates = d["dates"]
        lev_datasets.append({
            "label": name,
            "data": d["values"],
            "borderColor": _PALETTE[i % len(_PALETTE)],
            "backgroundColor": "transparent",
            "borderWidth": 1.5,
            "pointRadius": 0,
        })

    # Weight area chart (first strategy with history)
    weight_area_js = "null"
    weight_area_name = ""
    if weight_area_data:
        wname, wdata = next(iter(weight_area_data.items()))
        weight_area_name = wname
        syms = wdata["symbols"]
        area_datasets = []
        for j, sym in enumerate(syms):
            c = _WEIGHT_PALETTE[j % len(_WEIGHT_PALETTE)]
            area_datasets.append({
                "label": sym,
                "data": wdata["series"][sym],
                "borderColor": c,
                "backgroundColor": c + "cc",
                "borderWidth": 0.5,
                "pointRadius": 0,
                "fill": True,
                "tension": 0.2,
            })
        weight_area_js = json.dumps({
            "labels": wdata["dates"],
            "datasets": area_datasets,
        })

    # Monthly heatmap HTML
    heatmap_html = ""
    if monthly_heatmap_data:
        mo_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        rows = [f"<tr><th>年份</th>" + "".join(f"<th>{m}</th>" for m in mo_names) + "</tr>"]
        for yr_str, months in sorted(monthly_heatmap_data["data"].items()):
            cells = []
            for mo in range(1, 13):
                val = months.get(str(mo))
                if val is None:
                    cells.append("<td>-</td>")
                else:
                    # color: green positive, red negative
                    intensity = min(abs(val) / 8.0, 1.0)
                    if val >= 0:
                        r, g, b = int(255*(1-intensity*0.7)), int(200+55*(1-intensity)), int(255*(1-intensity*0.7))
                    else:
                        r, g, b = int(200+55*(1-intensity)), int(255*(1-intensity*0.7)), int(255*(1-intensity*0.7))
                    cells.append(f'<td style="background:rgb({r},{g},{b});color:#1e293b">{val:.1f}%</td>')
            rows.append(f"<tr><td><b>{yr_str}</b></td>" + "".join(cells) + "</tr>")
        heatmap_html = f"""
<div class="card">
  <h2>月度收益热力图 — {monthly_heatmap_data['strategy']}</h2>
  <div style="overflow-x:auto">
  <table class="heatmap-table">
    {"".join(rows)}
  </table>
  </div>
</div>"""

    # Rebalance table
    rebal_rows = "".join(
        f"<tr><td>{name}</td><td>{info['n_rebalances']}</td>"
        f"<td>{info['avg_annual']}</td><td>{info['avg_leverage']}x</td></tr>"
        for name, info in rebal_info.items()
    )

    eq_json = json.dumps({"labels": first_dates, "datasets": eq_datasets})
    dd_json = json.dumps({"labels": first_dd_dates, "datasets": dd_datasets})
    annual_json = json.dumps({"labels": all_years, "datasets": annual_datasets})
    roll_sr_dates = list(rolling_sharpe_data.values())[0]["dates"] if rolling_sharpe_data else []
    roll_sr_json = json.dumps({"labels": roll_sr_dates, "datasets": roll_sr_datasets})
    lev_json = json.dumps({"labels": lev_dates, "datasets": lev_datasets})

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>全天候期货策略 — 回测报告</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  *{{box-sizing:border-box}}
  body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
       margin:0;padding:20px 24px;background:#f1f5f9;color:#1e293b}}
  h1{{font-size:1.6rem;font-weight:700;margin:0 0 4px}}
  h2{{font-size:1.0rem;font-weight:600;color:#475569;margin:0 0 14px}}
  .subtitle{{color:#64748b;font-size:0.88rem;margin-bottom:28px}}
  .grid2{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:20px}}
  @media(max-width:900px){{.grid2{{grid-template-columns:1fr}}}}
  .card{{background:white;border-radius:12px;padding:22px 24px;
         box-shadow:0 1px 3px rgba(0,0,0,.08);margin-bottom:20px}}
  .chart-wrap{{position:relative;height:320px}}
  .chart-wrap-sm{{position:relative;height:260px}}
  .metrics-table{{border-collapse:collapse;width:100%;font-size:0.83rem}}
  .metrics-table th,.metrics-table td{{padding:7px 12px;text-align:right;
    border-bottom:1px solid #e2e8f0}}
  .metrics-table th:first-child,.metrics-table td:first-child{{text-align:left}}
  .metrics-table thead th{{background:#f8fafc;font-weight:600;color:#334155;
    position:sticky;top:0}}
  .rebal-table{{border-collapse:collapse;width:100%;font-size:0.85rem}}
  .rebal-table th,.rebal-table td{{padding:7px 14px;text-align:center;
    border-bottom:1px solid #e2e8f0}}
  .rebal-table thead th{{background:#f8fafc;font-weight:600}}
  .heatmap-table{{border-collapse:collapse;font-size:0.80rem;white-space:nowrap}}
  .heatmap-table th,.heatmap-table td{{padding:5px 8px;text-align:center;
    border:1px solid #e2e8f0;min-width:52px}}
  .heatmap-table th{{background:#f8fafc;font-weight:600}}
  .overflow-x{{overflow-x:auto}}
</style>
</head>
<body>
<h1>全天候期货策略 — 回测报告</h1>
<p class="subtitle">基于风险平价（ERC）的中国期货市场多资产组合 | 阈值触发再平衡 | 2015–2026</p>

<div class="card">
  <h2>净值曲线对比</h2>
  <div class="chart-wrap">
    <canvas id="equityChart"></canvas>
  </div>
</div>

<div class="grid2">
  <div class="card">
    <h2>回撤曲线（%）</h2>
    <div class="chart-wrap-sm">
      <canvas id="ddChart"></canvas>
    </div>
  </div>
  <div class="card">
    <h2>滚动Sharpe（252日，月度采样）</h2>
    <div class="chart-wrap-sm">
      <canvas id="rollSrChart"></canvas>
    </div>
  </div>
</div>

<div class="card">
  <h2>年度收益率（%）</h2>
  <div class="chart-wrap">
    <canvas id="annualChart"></canvas>
  </div>
</div>

<div class="grid2">
  <div class="card">
    <h2>权重构成历史 — {weight_area_name}</h2>
    <div class="chart-wrap-sm">
      <canvas id="weightAreaChart"></canvas>
    </div>
  </div>
  <div class="card">
    <h2>平均杠杆倍数（月度均值）</h2>
    <div class="chart-wrap-sm">
      <canvas id="levChart"></canvas>
    </div>
  </div>
</div>

{heatmap_html}

<div class="card overflow-x">
  <h2>绩效指标汇总</h2>
  {metrics_html}
</div>

<div class="card">
  <h2>再平衡 & 杠杆统计</h2>
  <table class="rebal-table">
    <thead><tr><th>策略</th><th>总调仓次数</th><th>年均调仓</th><th>平均杠杆</th></tr></thead>
    <tbody>{rebal_rows}</tbody>
  </table>
</div>

<script>
const OPT_LINE = (title, yFmt) => ({{
  responsive:true, maintainAspectRatio:false,
  interaction:{{mode:'index',intersect:false}},
  plugins:{{legend:{{position:'top',labels:{{boxWidth:12,font:{{size:11}}}}}},
            tooltip:{{callbacks:{{label:(c)=>`${{c.dataset.label}}: ${{(yFmt||((v)=>v.toFixed(4)))(c.parsed.y)}}`}}}}}},
  scales:{{
    x:{{type:'category',ticks:{{maxTicksLimit:12,maxRotation:0,font:{{size:10}}}}}},
    y:{{title:{{display:!!title,text:title}},ticks:{{font:{{size:10}},
       callback:(v)=>(yFmt||((x)=>x.toFixed(2)))(v)}}}}
  }}
}});

new Chart(document.getElementById('equityChart').getContext('2d'),{{
  type:'line', data:{eq_json},
  options:OPT_LINE('净值',(v)=>v.toFixed(3))
}});

new Chart(document.getElementById('ddChart').getContext('2d'),{{
  type:'line', data:{dd_json},
  options:OPT_LINE('回撤 %',(v)=>v.toFixed(1)+'%')
}});

new Chart(document.getElementById('rollSrChart').getContext('2d'),{{
  type:'line', data:{roll_sr_json},
  options:OPT_LINE('Sharpe',(v)=>v.toFixed(2))
}});

new Chart(document.getElementById('annualChart').getContext('2d'),{{
  type:'bar', data:{annual_json},
  options:{{
    responsive:true, maintainAspectRatio:false,
    plugins:{{legend:{{position:'top',labels:{{boxWidth:12,font:{{size:11}}}}}},
              tooltip:{{callbacks:{{label:(c)=>`${{c.dataset.label}}: ${{c.parsed.y?.toFixed(1)||'-'}}%`}}}}}},
    scales:{{
      x:{{ticks:{{font:{{size:10}}}}}},
      y:{{title:{{display:true,text:'年度收益 %'}},ticks:{{font:{{size:10}},callback:(v)=>v+'%'}},
          grid:{{color:'#e2e8f0'}}}}
    }}
  }}
}});

const weightAreaData = {weight_area_js};
if (weightAreaData) {{
  new Chart(document.getElementById('weightAreaChart').getContext('2d'),{{
    type:'line', data:weightAreaData,
    options:{{
      responsive:true, maintainAspectRatio:false,
      interaction:{{mode:'index',intersect:false}},
      plugins:{{legend:{{position:'top',labels:{{boxWidth:10,font:{{size:10}}}}}}}},
      scales:{{
        x:{{type:'category',stacked:true,ticks:{{maxTicksLimit:10,maxRotation:0,font:{{size:10}}}}}},
        y:{{stacked:true,min:0,max:1,title:{{display:true,text:'权重'}},
            ticks:{{callback:(v)=>(v*100).toFixed(0)+'%',font:{{size:10}}}}}}
      }}
    }}
  }});
}} else {{
  document.getElementById('weightAreaChart').parentElement.innerHTML='<p style="color:#94a3b8;padding-top:80px;text-align:center">暂无权重历史数据</p>';
}}

new Chart(document.getElementById('levChart').getContext('2d'),{{
  type:'line', data:{lev_json},
  options:OPT_LINE('杠杆倍数',(v)=>v.toFixed(2)+'x')
}});
</script>
</body>
</html>"""
