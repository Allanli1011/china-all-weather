# China All Weather Strategy

基于风险平价（ERC）的中国期货市场全天候多资产策略，覆盖17个高流动性品种，支持目标波动率杠杆与分类再平衡阈值。

## 策略简介

桥水全天候策略的中国期货市场适配版本：
- **核心算法**：等风险贡献（ERC）两层架构 + Ledoit-Wolf 收缩协方差矩阵
- **品种宇宙**：17个品种，覆盖股票（IF/IC）、债券（T/TF）、黄金（AU）、工业金属、黑色金属、能源、农产品
- **再平衡机制**：分类阈值触发（商品8% / 债券20% / 黄金15% / 股票15%），下一交易日执行
- **目标波动率杠杆**：自动缩放权重以命中目标年化波动率（推荐10%，均杠杆约2x）
- **展期处理**：价格跳跃识别展期日，当日收益置零，独立计展期交易成本

## 回测结果（2015-01-05 ~ 2026-04-07）

| 配置 | 年化收益 | Sharpe | 最大回撤 | 年均调仓 |
|------|---------|--------|---------|---------|
| 无杠杆（基线） | 5.81% | 0.649 | -18.1% | 5.9次 |
| vol=8% 全局阈值 | 8.24% | 0.713 | -18.5% | 23.3次 |
| **vol=10% 分类阈值（推荐）** | **9.77%** | **0.721** | **-21.9%** | **9.8次** |
| vol=12% 全局阈值 | 11.00% | 0.732 | -26.3% | 29.9次 |
| HRP 对照 | 5.03% | 0.543 | -19.0% | 93.1次 |

## 快速开始

```bash
# 安装依赖
pip install akshare pandas numpy scipy scikit-learn pyyaml pyarrow

# 运行回测（推荐配置）
python3 -m all_weather.cli \
    --config configs/all_weather.yaml \
    --leverage --target-vol 0.10 \
    --per-class-thresholds

# 多组对比实验（无杠杆/vol8%/vol10%分类/vol12%）
python3 -m all_weather.cli \
    --config configs/all_weather.yaml \
    --experiment summary

# 生成年度归因报告
python3 -m all_weather.cli \
    --config configs/all_weather.yaml \
    --experiment summary --attribution --no-report

# 每日仓位监控（收盘后运行）
python3 run_monitor.py \
    --leverage --target-vol 0.10 --per-class-thresholds
```

## 项目结构

```
all-weather-strategy/
├── configs/
│   └── all_weather.yaml          # 品种元数据、策略参数
├── data/
│   ├── raw/futures/              # AKShare 原始日线数据（Parquet）
│   ├── processed/                # 处理后收益率矩阵
│   └── monitor_state.json        # 每日监控状态（自动更新）
├── docs/
│   └── research_report.md        # 完整研究报告（含展期处理专章）
├── reports/
│   ├── backtest_report.html      # 回测可视化报告
│   ├── attribution_report.html   # 年度归因报告
│   └── backtest_summary.csv      # 绩效指标汇总
├── run_monitor.py                # 每日监控入口
├── src/all_weather/
│   ├── settings.py               # 配置加载
│   ├── universe.py               # 品种宇宙定义
│   ├── data_fetcher.py           # AKShare 行情获取（增量）
│   ├── data_store.py             # Parquet 存储管理
│   ├── returns.py                # 收益率计算 + 展期检测
│   ├── risk_parity.py            # ERC / HRP 权重计算
│   ├── backtest.py               # 回测引擎（阈值触发再平衡）
│   ├── metrics.py                # 绩效指标
│   ├── attribution.py            # 年度收益归因
│   ├── monitor.py                # 每日监控逻辑
│   ├── report.py                 # HTML 回测报告
│   └── report_attribution.py    # HTML 归因报告
└── tests/
    ├── test_risk_parity.py
    ├── test_returns.py
    └── test_backtest.py
```

## 品种宇宙（17个）

| 大类 | 品种 | 符号 |
|------|------|------|
| 股票 | 沪深300、中证500 | IF0、IC0 |
| 债券 | 10年国债、5年国债 | T0、TF0 |
| 黄金 | 黄金 | AU0 |
| 工业金属 | 铜、铝、锌 | CU0、AL0、ZN0 |
| 黑色金属 | 铁矿石、焦炭、螺纹钢 | I0、J0、RB0 |
| 能源 | 原油 | SC0 |
| 农产品 | 豆粕、豆油、棉花、白糖、玉米 | M0、Y0、CF0、SR0、C0 |

## 每日监控

监控程序在每个交易日收盘后（推荐16:05）运行，自动：
1. 增量拉取最新行情（AKShare）
2. 用最新60日数据重算ERC目标权重
3. 从状态文件加载昨日持仓，按今日价格漂移更新实际权重
4. 检测是否触发再平衡（分类阈值）
5. 输出明日操作指令 + 目标仓位明细

退出码：`0` = 无需调仓，`1` = 需要调仓（可配合 shell 脚本发送通知）

## 研究文档

详见 [`docs/research_report.md`](docs/research_report.md)，重点章节：
- **第4章：期货展期处理** — AKShare 数据局限、价格跳跃识别方案、展期日三步处理、误识别影响分析
- **第8章：年度归因分析** — 黄金是11年核心驱动，2015年是唯一双崩年份

## 关键依赖

```
akshare >= 1.12.0
pandas >= 2.0.0
numpy >= 1.24.0
scipy >= 1.11.0
scikit-learn >= 1.3.0
pyarrow >= 14.0.0
pyyaml >= 6.0
```
