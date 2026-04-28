from __future__ import annotations

from statistics import median
from typing import Dict, Iterable, List, Sequence, Optional

from .models import FeatureRow, PriceBar, StrategyBacktestResult, StrategyBacktestStats
from .strategies import generate_signals


def run_event_backtest(
    features: Iterable[FeatureRow],
    price_bars: Iterable[PriceBar],
    horizons: Sequence[int] = (1, 3, 5, 10),
    min_short_value: float = 10_000_000,
    round_trip_cost: float = 0.003,
) -> StrategyBacktestResult:
    prices_by_code: Dict[str, List[PriceBar]] = {}
    for bar in price_bars:
        prices_by_code.setdefault(bar.code, []).append(bar)
    for bars in prices_by_code.values():
        bars.sort(key=lambda item: item.trade_date)

    events = []
    for feature in features:
        signals = generate_signals([feature], prices_by_code=prices_by_code, min_short_value=min_short_value)
        if not signals:
            continue
        bars = prices_by_code.get(feature.record.code, [])
        signal_idx = _bar_index_on_or_after(bars, feature.record.trade_date)
        if signal_idx is None:
            continue
        entry_idx = signal_idx + 1
        if entry_idx >= len(bars):
            continue
        entry_bar = bars[entry_idx]
        for signal in signals:
            for horizon in horizons:
                exit_idx = entry_idx + horizon - 1
                if exit_idx >= len(bars):
                    continue
                exit_bar = bars[exit_idx]
                
                is_short = "short" in signal.direction
                raw_return = exit_bar.close / entry_bar.open - 1
                strategy_return = -raw_return if is_short else raw_return
                
                # Calculate drawdown
                horizon_bars = bars[entry_idx:exit_idx+1]
                if is_short:
                    worst_price = max(b.high for b in horizon_bars)
                    worst_raw_return = worst_price / entry_bar.open - 1
                    drawdown = min(0.0, -worst_raw_return)
                else:
                    worst_price = min(b.low for b in horizon_bars)
                    worst_raw_return = worst_price / entry_bar.open - 1
                    drawdown = min(0.0, worst_raw_return)
                
                events.append(
                    {
                        "signal_date": feature.record.trade_date,
                        "entry_date": entry_bar.trade_date,
                        "exit_date": exit_bar.trade_date,
                        "code": signal.code,
                        "name": signal.name,
                        "strategy": signal.strategy,
                        "direction": signal.direction,
                        "horizon": horizon,
                        "entry_price": entry_bar.open,
                        "exit_price": exit_bar.close,
                        "raw_return": raw_return,
                        "strategy_return": strategy_return,
                        "net_return": strategy_return - round_trip_cost,
                        "drawdown": drawdown,
                        "score": signal.score,
                    }
                )

    by_strategy: Dict[str, Dict[int, List[dict]]] = {}
    for event in events:
        strategy = event["strategy"]
        horizon = event["horizon"]
        by_strategy.setdefault(strategy, {}).setdefault(horizon, [])
        by_strategy[strategy][horizon].append(event)

    stats: Dict[str, Dict[int, StrategyBacktestStats]] = {}
    for strategy, horizon_events in by_strategy.items():
        stats[strategy] = {}
        for horizon, h_events in horizon_events.items():
            stats[strategy][horizon] = _stats(h_events)

    return StrategyBacktestResult(by_strategy=stats, events=events)


def _bar_index_on_or_after(bars: List[PriceBar], target_date) -> Optional[int]:
    for idx, bar in enumerate(bars):
        if bar.trade_date >= target_date:
            return idx
    return None


def _stats(events: List[dict]) -> StrategyBacktestStats:
    if not events:
        return StrategyBacktestStats(0, None, None, None, None, None, None, None, None)
    returns = [e["strategy_return"] for e in events]
    net_returns = [e["net_return"] for e in events]
    drawdowns = [e["drawdown"] for e in events]
    scores = [e["score"] for e in events]
    
    wins = sum(1 for value in returns if value > 0)
    ic = _pearson_corr(scores, returns)
    
    return StrategyBacktestStats(
        count=len(returns),
        avg_return=sum(returns) / len(returns),
        median_return=median(returns),
        win_rate=wins / len(returns),
        min_return=min(returns),
        max_return=max(returns),
        avg_net_return=sum(net_returns) / len(net_returns),
        max_drawdown=min(drawdowns) if drawdowns else None,
        ic=ic,
    )

def _pearson_corr(x: List[float], y: List[float]) -> Optional[float]:
    if len(x) < 2 or len(x) != len(y):
        return None
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    num = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    den_x = sum((a - mean_x) ** 2 for a in x)
    den_y = sum((b - mean_y) ** 2 for b in y)
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / ((den_x * den_y) ** 0.5)
