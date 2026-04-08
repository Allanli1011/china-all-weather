"""收益率模块测试"""

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, "src")

from all_weather.returns import compute_daily_returns, detect_rollover_days


class TestDetectRolloverDays:
    def test_large_jump_detected(self):
        """单日大幅跳涨应被检测为展期"""
        dates = pd.date_range("2020-01-02", periods=10, freq="B")
        # 第5天有3.5%的跳跃（超过3%阈值）
        prices = [100, 100.5, 101, 101.5, 102, 105.5, 106, 106.5, 107, 107.5]
        df = pd.DataFrame({"close": prices}, index=dates)
        result = detect_rollover_days(df, jump_threshold=0.03)
        assert result.iloc[5] == True

    def test_normal_move_not_detected(self):
        """正常涨跌不应被误判为展期"""
        dates = pd.date_range("2020-01-02", periods=20, freq="B")
        prices = 100 * (1 + np.random.uniform(-0.015, 0.015, 20)).cumprod()
        df = pd.DataFrame({"close": prices}, index=dates)
        result = detect_rollover_days(df, jump_threshold=0.03)
        # 在正常市场下，误检率应很低
        assert result.sum() <= 3

    def test_first_row_always_false(self):
        """第一行无法计算收益率，应标记为 False"""
        dates = pd.date_range("2020-01-02", periods=5, freq="B")
        df = pd.DataFrame({"close": [100, 101, 102, 103, 110]}, index=dates)
        result = detect_rollover_days(df)
        assert result.iloc[0] == False

    def test_returns_series_with_same_index(self):
        """返回值应与输入 DataFrame 具有相同 index"""
        dates = pd.date_range("2020-01-02", periods=10, freq="B")
        df = pd.DataFrame({"close": range(100, 110)}, index=dates)
        result = detect_rollover_days(df)
        assert list(result.index) == list(df.index)


class TestComputeDailyReturns:
    def test_rollover_day_set_to_zero(self):
        """展期日收益率应被置为0"""
        dates = pd.date_range("2020-01-02", periods=5, freq="B")
        price_df = pd.DataFrame({"IF": [100.0, 101.0, 105.0, 105.5, 106.0]}, index=dates)
        rollover_mask = pd.Series([False, False, True, False, False], index=dates, name="is_rollover")
        returns_df, is_rollover_df = compute_daily_returns(price_df, {"IF": rollover_mask})
        # 展期日（第3行, index=2）收益率应为0
        assert returns_df["IF"].iloc[2] == 0.0
        # 非展期日收益率应正常
        assert abs(returns_df["IF"].iloc[1] - 0.01) < 1e-6

    def test_no_rollover_mask_normal_returns(self):
        """无展期掩码时应正常计算 pct_change"""
        dates = pd.date_range("2020-01-02", periods=5, freq="B")
        price_df = pd.DataFrame({"IF": [100.0, 102.0, 101.0, 103.0, 104.0]}, index=dates)
        returns_df, is_rollover_df = compute_daily_returns(price_df)
        expected_r1 = (102 - 100) / 100
        assert abs(returns_df["IF"].iloc[1] - expected_r1) < 1e-10

    def test_nan_preserved_for_unlisted_periods(self):
        """晚上市品种的早期 NaN 应被保留（不填充）"""
        dates = pd.date_range("2020-01-02", periods=6, freq="B")
        price_df = pd.DataFrame({
            "IF": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
            "NEW": [np.nan, np.nan, np.nan, 100.0, 101.0, 102.0],  # 晚上市
        }, index=dates)
        returns_df, _ = compute_daily_returns(price_df)
        # NEW 的前3行（含第4行的第一个收益率）应为 NaN
        assert pd.isna(returns_df["NEW"].iloc[0])
        assert pd.isna(returns_df["NEW"].iloc[3])  # 第一个有效收益率也为 NaN（pct_change 首行）
        # 第5行应有正常收益率
        assert not pd.isna(returns_df["NEW"].iloc[4])

    def test_multiple_symbols(self):
        """多品种情况下各品种独立计算"""
        dates = pd.date_range("2020-01-02", periods=4, freq="B")
        price_df = pd.DataFrame({
            "A": [100.0, 110.0, 121.0, 133.1],
            "B": [100.0, 99.0, 98.0, 97.0],
        }, index=dates)
        returns_df, _ = compute_daily_returns(price_df)
        assert abs(returns_df["A"].iloc[1] - 0.10) < 1e-10
        assert abs(returns_df["B"].iloc[1] - (-0.01)) < 1e-10
