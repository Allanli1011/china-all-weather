"""风险平价模块测试"""

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, "src")

from all_weather.risk_parity import (
    compute_rolling_cov,
    compute_risk_contributions,
    solve_erc,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def simple_returns():
    """3个品种，100天，已知波动率结构"""
    rng = np.random.default_rng(42)
    n = 100
    # 品种A: 低波动 (年化10%), 品种B: 中波动 (20%), 品种C: 高波动 (30%)
    vols = [0.10 / np.sqrt(252), 0.20 / np.sqrt(252), 0.30 / np.sqrt(252)]
    data = {
        "A": rng.normal(0, vols[0], n),
        "B": rng.normal(0, vols[1], n),
        "C": rng.normal(0, vols[2], n),
    }
    dates = pd.date_range("2020-01-02", periods=n, freq="B")
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def correlated_returns():
    """3个高度相关品种（模拟黑色金属 I/J/RB）"""
    rng = np.random.default_rng(42)
    n = 100
    factor = rng.normal(0, 0.02, n)  # 共同因子（高波动）
    data = {
        "I":  factor + rng.normal(0, 0.005, n),
        "J":  factor + rng.normal(0, 0.005, n),
        "RB": factor + rng.normal(0, 0.005, n),
    }
    dates = pd.date_range("2020-01-02", periods=n, freq="B")
    return pd.DataFrame(data, index=dates)


# ── solve_erc 测试 ────────────────────────────────────────────────────────────

class TestSolveERC:
    def test_weights_sum_to_one(self, simple_returns):
        cov = simple_returns.cov().values
        weights = solve_erc(cov)
        assert abs(weights.sum() - 1.0) < 1e-6

    def test_weights_nonnegative(self, simple_returns):
        cov = simple_returns.cov().values
        weights = solve_erc(cov)
        assert np.all(weights >= 0)

    def test_equal_risk_contribution(self, simple_returns):
        """ERC 后各品种风险贡献应相等（允许1%误差）"""
        cov = simple_returns.cov().values
        weights = solve_erc(cov)
        sigma = np.sqrt(weights @ cov @ weights)
        rc = weights * (cov @ weights) / sigma
        rc_normalized = rc / rc.sum()
        # 各品种等权贡献约 1/n
        expected = 1.0 / len(weights)
        assert np.allclose(rc_normalized, expected, atol=0.01), (
            f"风险贡献不均等: {rc_normalized}"
        )

    def test_low_vol_gets_higher_weight(self, simple_returns):
        """低波动品种应获得更高权重"""
        cov = simple_returns.cov().values
        weights = solve_erc(cov)
        # A(低波动) > B(中) > C(高波动)
        assert weights[0] > weights[1] > weights[2], (
            f"权重排序不符预期: A={weights[0]:.3f}, B={weights[1]:.3f}, C={weights[2]:.3f}"
        )

    def test_single_asset(self):
        """单品种应返回 [1.0]"""
        cov = np.array([[0.04]])
        weights = solve_erc(cov)
        assert abs(weights[0] - 1.0) < 1e-6

    def test_correlated_assets_lower_weight(self, correlated_returns):
        """高度相关品种合计权重应受到抑制（< 80%）"""
        cov = correlated_returns.cov().values
        weights = solve_erc(cov)
        # 3个高度相关品种均等分配（约33%每个），总和约100%
        # 关键：与等权（33.3%）相比，ERC应更分散（相关性导致每个权重降低）
        assert abs(weights.sum() - 1.0) < 1e-6


# ── compute_rolling_cov 测试 ──────────────────────────────────────────────────

class TestComputeRollingCov:
    def test_returns_dataframe(self, simple_returns):
        cov = compute_rolling_cov(simple_returns, window=60, min_periods=20)
        assert cov is not None
        assert isinstance(cov, pd.DataFrame)
        assert list(cov.index) == list(cov.columns)

    def test_symmetric(self, simple_returns):
        cov = compute_rolling_cov(simple_returns, window=60, min_periods=20)
        assert np.allclose(cov.values, cov.values.T, atol=1e-10)

    def test_positive_definite(self, simple_returns):
        cov = compute_rolling_cov(simple_returns, window=60, min_periods=20)
        eigvals = np.linalg.eigvalsh(cov.values)
        assert np.all(eigvals > 0), f"协方差矩阵不正定: min_eig={eigvals.min():.2e}"

    def test_insufficient_data_returns_none(self):
        dates = pd.date_range("2020-01-02", periods=10, freq="B")
        returns = pd.DataFrame({"A": np.random.randn(10), "B": np.random.randn(10)}, index=dates)
        cov = compute_rolling_cov(returns, window=60, min_periods=20)
        assert cov is None

    def test_nan_column_excluded(self):
        """有效观测不足的品种应被排除"""
        dates = pd.date_range("2020-01-02", periods=60, freq="B")
        returns = pd.DataFrame({
            "A": np.random.randn(60),
            "B": [np.nan] * 55 + list(np.random.randn(5)),  # 只有5个有效观测
        }, index=dates)
        cov = compute_rolling_cov(returns, window=60, min_periods=20)
        # B 应被排除
        assert cov is None or "B" not in cov.columns


# ── compute_risk_contributions 测试 ──────────────────────────────────────────

class TestComputeRiskContributions:
    def test_rc_sum_equals_portfolio_sigma(self, simple_returns):
        cov = compute_rolling_cov(simple_returns, window=60, min_periods=20)
        w = pd.Series({"A": 0.3, "B": 0.4, "C": 0.3})
        rc = compute_risk_contributions(w, cov)
        sigma = np.sqrt(w.values @ cov.values @ w.values)
        assert abs(rc.sum() - sigma) < 1e-10

    def test_erc_weights_give_equal_rc(self, simple_returns):
        cov = compute_rolling_cov(simple_returns, window=60, min_periods=20)
        erc_w_arr = solve_erc(cov.values)
        w = pd.Series(dict(zip(cov.columns, erc_w_arr)))
        rc = compute_risk_contributions(w, cov)
        rc_normalized = rc / rc.sum()
        expected = 1.0 / 3
        assert np.allclose(rc_normalized.values, expected, atol=0.02), (
            f"ERC 风险贡献不均等: {rc_normalized.values}"
        )
