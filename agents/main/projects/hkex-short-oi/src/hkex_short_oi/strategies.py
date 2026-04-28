from __future__ import annotations

from typing import Iterable, List, Optional, Dict

from .models import FeatureRow, StrategySignal, PriceBar


def generate_signals(
    features: Iterable[FeatureRow],
    prices_by_code: Optional[Dict[str, List[PriceBar]]] = None,
    min_short_value: float = 10_000_000,
) -> List[StrategySignal]:
    signals: List[StrategySignal] = []

    for row in features:
        record = row.record
        if record.short_value < min_short_value:
            continue

        z60 = _first_available(row.short_value_z.get(60), row.short_value_z.get(20))
        ratio_z60 = _first_available(row.short_ratio_z.get(60), row.short_ratio_z.get(20))
        stress_z = max(value for value in [z60, ratio_z60] if value is not None) if any(
            value is not None for value in [z60, ratio_z60]
        ) else None

        # Price confirmation variables
        bars = prices_by_code.get(record.code, []) if prices_by_code else []
        signal_idx = next((i for i, b in enumerate(bars) if b.trade_date == record.trade_date), None)
        
        has_breakdown = False
        has_reversal = False
        
        if signal_idx is not None and signal_idx >= 20:
            past_20_low = min(b.low for b in bars[signal_idx - 20:signal_idx])
            current_bar = bars[signal_idx]
            prev_bar = bars[signal_idx - 1]
            
            # Breakdown: Close below 20-day low
            has_breakdown = current_bar.close < past_20_low
            
            # Reversal: Close above previous day high OR a strong bullish candle
            is_bullish_candle = current_bar.close > current_bar.open and \
                (current_bar.close - current_bar.low) > (current_bar.high - current_bar.low) * 0.6
            has_reversal = (current_bar.close > prev_bar.high) or is_bullish_candle
            
        elif not prices_by_code:
            # If no price data was provided, we keep the watchlist behavior (both can fire)
            has_breakdown = True
            has_reversal = True

        if stress_z is not None and stress_z >= 2.0:
            if has_reversal:
                signals.append(
                    StrategySignal(
                        trade_date=record.trade_date,
                        code=record.code,
                        name=record.name,
                        strategy="crowded_reversal",
                        direction="long_reversal",
                        score=min(5.0, 2.5 + stress_z),
                        reasons=[
                            "Short-selling pressure is unusually high versus own history.",
                            "Price action confirms a potential reversal (bullish candle or close above prior high).",
                        ],
                        required_confirmation=[
                            "No fresh negative announcement",
                            "Liquidity above trade threshold",
                        ],
                    )
                )
            if has_breakdown:
                signals.append(
                    StrategySignal(
                        trade_date=record.trade_date,
                        code=record.code,
                        name=record.name,
                        strategy="pressure_breakdown",
                        direction="short_continuation",
                        score=min(5.0, 2.0 + stress_z),
                        reasons=[
                            "Short-selling pressure is elevated enough to confirm bearish sponsorship.",
                            "Price broke below 20-day low, follow-through risk is high.",
                        ],
                        required_confirmation=[
                            "Broad market pressure not easing",
                            "Borrow and execution constraints checked",
                        ],
                    )
                )

        z20 = row.short_value_z.get(20)
        if z20 is not None and z20 <= -1.5:
            is_stabilizing = False
            if signal_idx is not None and signal_idx >= 5:
                past_5_low = min(b.low for b in bars[signal_idx - 5:signal_idx])
                current_bar = bars[signal_idx]
                is_stabilizing = current_bar.close >= past_5_low
            elif not prices_by_code:
                is_stabilizing = True
                
            if is_stabilizing:
                signals.append(
                    StrategySignal(
                        trade_date=record.trade_date,
                        code=record.code,
                        name=record.name,
                        strategy="short_pressure_fade",
                        direction="pressure_normalization",
                        score=min(5.0, 2.0 + abs(z20)),
                        reasons=[
                            "Short-selling turnover has fallen materially versus recent history.",
                            "Price is holding above recent support, bearish pressure may be fading.",
                        ],
                        required_confirmation=[
                            "Short pressure stays below 20-day average for another session",
                        ],
                    )
                )

    signals.sort(key=lambda signal: (-signal.score, signal.code, signal.strategy))
    return signals


def _first_available(*values: Optional[float]) -> Optional[float]:
    for value in values:
        if value is not None:
            return value
    return None
