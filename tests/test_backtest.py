"""回测引擎测试"""

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, "src")

from all_weather.backtest import (
    _compute_l1_deviation,
    _compute_rebalance_cost,
    _update_weights_after_drift,
    run_backtest,
)
from all_weather.settings import load_settings
from all_weather.universe import load_universe


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def settings():
    return load_settings("configs/all_weather.yaml")


@pytest.fixture
def universe(settings):
    return load_universe(settings)


@pytest.fixture
def synthetic_data():
    """生成合成测试数据（4个品种，500天）"""
    rng = np.random.default_rng(42)
    n = 500
    dates = pd.date_range("2018-01-02", periods=n, freq="B")

    # 模拟4个资产类别：股票(高波动)、债券(低波动)、黄金(中)、商品(高)
    returns = pd.DataFrame({
        "IF": rng.normal(0.0003, 0.012, n),   # 股票
        "T":  rng.normal(0.0001, 0.003, n),   # 债券
        "AU": rng.normal(0.0002, 0.007, n),   # 黄金
        "CU": rng.normal(0.0002, 0.011, n),   # 铜
    }, index=dates)

    # 构造价格序列（从100开始）
    prices = (1 + returns).cumprod() * 100

    return returns, prices


# ── 辅助函数测试 ──────────────────────────────────────────────────────────────

class TestHelperFunctions:
    def test_l1_deviation_same_weights(self):
        """完全相同的权重 → 偏差为0"""
        w = pd.Series({"A": 0.3, "B": 0.4, "C": 0.3})
        assert abs(_compute_l1_deviation(w, w)) < 1e-10

    def test_l1_deviation_opposite_extremes(self):
        """完全不同 → 偏差为2.0（100% A vs 100% B）"""
        w_cur = pd.Series({"A": 1.0, "B": 0.0})
        w_tgt = pd.Series({"A": 0.0, "B": 1.0})
        assert abs(_compute_l1_deviation(w_cur, w_tgt) - 2.0) < 1e-10

    def test_l1_deviation_with_new_symbol(self):
        """目标权重中有当前未持有的品种"""
        w_cur = pd.Series({"A": 0.6, "B": 0.4})
        w_tgt = pd.Series({"A": 0.4, "B": 0.3, "C": 0.3})
        dev = _compute_l1_deviation(w_cur, w_tgt)
        # |0.6-0.4| + |0.4-0.3| + |0-0.3| = 0.2+0.1+0.3 = 0.6
        assert abs(dev - 0.6) < 1e-10

    def test_weight_drift_update(self):
        """价格上涨后，该品种权重应增加"""
        w = pd.Series({"A": 0.5, "B": 0.5})
        ret = pd.Series({"A": 0.10, "B": 0.0})
        new_w = _update_weights_after_drift(w, ret)
        assert new_w["A"] > 0.5
        assert abs(new_w.sum() - 1.0) < 1e-10

    def test_weight_drift_sums_to_one(self):
        """漂移后权重之和应仍为1"""
        rng = np.random.default_rng(0)
        w = pd.Series({"A": 0.3, "B": 0.4, "C": 0.3})
        ret = pd.Series(rng.normal(0, 0.02, 3), index=["A", "B", "C"])
        new_w = _update_weights_after_drift(w, ret)
        assert abs(new_w.sum() - 1.0) < 1e-10


# ── 回测引擎测试 ──────────────────────────────────────────────────────────────

class TestRunBacktest:
    def test_equity_curve_starts_at_one(self, settings, universe, synthetic_data):
        """净值曲线应从1开始"""
        returns, prices = synthetic_data
        # 限制只用 IF, T 两个品种的子集
        sub_universe = {k: v for k, v in universe.items() if k in ["IF", "T"]}
        settings.raw["backtest"]["start_date"] = "2018-01-02"
        result = run_backtest(returns[["IF", "T"]], prices[["IF", "T"]], sub_universe, settings)
        assert abs(result.equity_curve.iloc[0] - 1.0) < 1e-6

    def test_portfolio_returns_length_matches_dates(self, settings, universe, synthetic_data):
        """组合收益率长度应与回测期间天数一致"""
        returns, prices = synthetic_data
        sub_universe = {k: v for k, v in universe.items() if k in ["IF", "T"]}
        settings.raw["backtest"]["start_date"] = "2018-01-02"
        result = run_backtest(returns[["IF", "T"]], prices[["IF", "T"]], sub_universe, settings)
        assert len(result.portfolio_returns) == len(returns)

    def test_rebalance_triggered_by_threshold(self, settings, universe, synthetic_data):
        """设置很低的阈值应触发更多次再平衡"""
        returns, prices = synthetic_data
        sub_universe = {k: v for k, v in universe.items() if k in ["IF", "T"]}
        settings.raw["backtest"]["start_date"] = "2018-01-02"

        settings.raw["backtest"]["rebalance_threshold"] = 0.05
        result_low = run_backtest(returns[["IF", "T"]], prices[["IF", "T"]], sub_universe, settings,
                                   strategy_name="low_threshold")

        settings.raw["backtest"]["rebalance_threshold"] = 0.30
        result_high = run_backtest(returns[["IF", "T"]], prices[["IF", "T"]], sub_universe, settings,
                                    strategy_name="high_threshold")

        assert result_low.n_rebalances > result_high.n_rebalances

    def test_weights_sum_to_one_after_rebalance(self, settings, universe, synthetic_data):
        """再平衡后持仓权重快照之和应为1"""
        returns, prices = synthetic_data
        sub_universe = {k: v for k, v in universe.items() if k in ["IF", "T"]}
        settings.raw["backtest"]["start_date"] = "2018-01-02"
        result = run_backtest(returns[["IF", "T"]], prices[["IF", "T"]], sub_universe, settings)

        if not result.weights_history.empty:
            for _, row in result.weights_history.iterrows():
                total = row.dropna().sum()
                assert abs(total - 1.0) < 0.01, f"权重之和={total:.4f}"

    def test_rebalance_log_has_cost(self, settings, universe, synthetic_data):
        """再平衡日志中的成本应为非负数"""
        returns, prices = synthetic_data
        sub_universe = {k: v for k, v in universe.items() if k in ["IF", "T"]}
        settings.raw["backtest"]["start_date"] = "2018-01-02"
        settings.raw["backtest"]["rebalance_threshold"] = 0.05
        result = run_backtest(returns[["IF", "T"]], prices[["IF", "T"]], sub_universe, settings)

        if not result.rebalance_log.empty:
            assert (result.rebalance_log["cost"] >= 0).all()

    def test_equity_curve_reflects_returns(self, settings, universe, synthetic_data):
        """净值曲线应与日收益率一致（累积乘积）"""
        returns, prices = synthetic_data
        sub_universe = {k: v for k, v in universe.items() if k in ["IF", "T"]}
        settings.raw["backtest"]["start_date"] = "2018-01-02"
        result = run_backtest(returns[["IF", "T"]], prices[["IF", "T"]], sub_universe, settings)

        expected_equity = (1 + result.portfolio_returns).cumprod()
        expected_equity = expected_equity / expected_equity.iloc[0]
        assert np.allclose(result.equity_curve.values, expected_equity.values, rtol=1e-5)
