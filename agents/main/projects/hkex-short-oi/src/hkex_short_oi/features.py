from __future__ import annotations

from collections import defaultdict
from math import sqrt
from typing import Dict, Iterable, List, Optional, Sequence

from .models import FeatureRow, ShortTurnoverRecord, MarketFeatureRow


def build_features(
    records: Iterable[ShortTurnoverRecord],
    short_windows: Sequence[int] = (20, 60),
) -> List[FeatureRow]:
    grouped: Dict[tuple, List[ShortTurnoverRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.market, record.session, record.code)].append(record)

    features: List[FeatureRow] = []
    for group_records in grouped.values():
        group_records.sort(key=lambda item: item.trade_date)
        previous_values: List[float] = []
        previous_ratios: List[float] = []

        for record in group_records:
            short_value_z: Dict[int, Optional[float]] = {}
            short_ratio_z: Dict[int, Optional[float]] = {}
            short_value_percentile: Dict[int, Optional[float]] = {}
            short_ratio_percentile: Dict[int, Optional[float]] = {}
            short_value_ma: Dict[int, Optional[float]] = {}
            short_ratio_ma: Dict[int, Optional[float]] = {}

            for window in short_windows:
                value_window = previous_values[-window:]
                ratio_window = previous_ratios[-window:]
                short_value_z[window] = _zscore(record.short_value, value_window)
                short_value_percentile[window] = _percentile_rank(record.short_value, value_window)
                short_value_ma[window] = _mean(value_window)

                ratio = record.short_ratio
                if ratio is None:
                    short_ratio_z[window] = None
                    short_ratio_percentile[window] = None
                    short_ratio_ma[window] = None
                else:
                    short_ratio_z[window] = _zscore(ratio, ratio_window)
                    short_ratio_percentile[window] = _percentile_rank(ratio, ratio_window)
                    short_ratio_ma[window] = _mean(ratio_window)

            features.append(
                FeatureRow(
                    record=record,
                    short_value_z=short_value_z,
                    short_ratio_z=short_ratio_z,
                    short_value_percentile=short_value_percentile,
                    short_ratio_percentile=short_ratio_percentile,
                    short_value_ma=short_value_ma,
                    short_ratio_ma=short_ratio_ma,
                )
            )

            previous_values.append(record.short_value)
            if record.short_ratio is not None:
                previous_ratios.append(record.short_ratio)

    features.sort(key=lambda item: (item.record.trade_date, item.record.market, item.record.session, item.record.code))
    return features


def build_market_features(
    records: Iterable[ShortTurnoverRecord],
    short_windows: Sequence[int] = (20, 60),
) -> List[MarketFeatureRow]:
    daily_aggregates: Dict[tuple, dict] = defaultdict(lambda: {"short": 0.0, "total": 0.0})
    for record in records:
        if record.is_non_hkd:
            continue
        key = (record.trade_date, record.market)
        daily_aggregates[key]["short"] += record.short_value
        daily_aggregates[key]["total"] += (record.total_value or 0.0)

    market_groups: Dict[str, List[dict]] = defaultdict(list)
    for (trade_date, market), values in daily_aggregates.items():
        if values["total"] > 0:
            ratio = values["short"] / values["total"]
        else:
            ratio = 0.0
        market_groups[market].append({
            "trade_date": trade_date,
            "short": values["short"],
            "total": values["total"],
            "ratio": ratio,
        })

    features: List[MarketFeatureRow] = []
    for market, group_records in market_groups.items():
        group_records.sort(key=lambda item: item["trade_date"])
        previous_ratios: List[float] = []

        for record in group_records:
            short_ratio_z: Dict[int, Optional[float]] = {}
            short_ratio_percentile: Dict[int, Optional[float]] = {}

            for window in short_windows:
                ratio_window = previous_ratios[-window:]
                short_ratio_z[window] = _zscore(record["ratio"], ratio_window)
                short_ratio_percentile[window] = _percentile_rank(record["ratio"], ratio_window)

            features.append(
                MarketFeatureRow(
                    trade_date=record["trade_date"],
                    market=market,
                    total_short_value=record["short"],
                    total_market_value=record["total"],
                    short_ratio=record["ratio"],
                    short_ratio_z=short_ratio_z,
                    short_ratio_percentile=short_ratio_percentile,
                )
            )

            previous_ratios.append(record["ratio"])

    features.sort(key=lambda item: (item.trade_date, item.market))
    return features


def _mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def _std(values: Sequence[float]) -> Optional[float]:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    if variance == 0:
        return 0.0
    return sqrt(variance)


def _zscore(value: float, history: Sequence[float]) -> Optional[float]:
    std = _std(history)
    if std is None:
        return None
    if std == 0:
        return 0.0 if value == history[-1] else 99.0
    mean = sum(history) / len(history)
    return (value - mean) / std


def _percentile_rank(value: float, history: Sequence[float]) -> Optional[float]:
    if not history:
        return None
    below_or_equal = sum(1 for item in history if item <= value)
    return below_or_equal / len(history)

