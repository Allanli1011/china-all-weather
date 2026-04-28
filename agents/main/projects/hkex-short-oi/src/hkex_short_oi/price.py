from __future__ import annotations

import json
from datetime import date, datetime, time, timezone
from typing import List, Optional

from .models import PriceBar
from .parsers import ParseError


def yahoo_symbol_from_hk_code(code: str) -> str:
    digits = "".join(ch for ch in code if ch.isdigit())
    if not digits:
        raise ValueError(f"Invalid HK code: {code}")
    return f"{digits[-4:].zfill(4)}.HK"


def yahoo_chart_url(code: str, start_date: date, end_date: date) -> str:
    symbol = yahoo_symbol_from_hk_code(code)
    period1 = _to_epoch(start_date)
    # Yahoo period2 is exclusive; add one day so end_date is included.
    period2 = _to_epoch(date.fromordinal(end_date.toordinal() + 1))
    return (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?period1={period1}&period2={period2}&interval=1d&events=history"
    )


def parse_yahoo_chart(text: str, code: str, source_url: str) -> List[PriceBar]:
    payload = json.loads(text)
    result = payload.get("chart", {}).get("result")
    error = payload.get("chart", {}).get("error")
    if error:
        raise ParseError(f"Yahoo chart error for {code}: {error}")
    if not result:
        raise ParseError(f"Yahoo chart returned no result for {code}")

    first = result[0]
    timestamps = first.get("timestamp") or []
    indicators = first.get("indicators", {})
    quote = (indicators.get("quote") or [{}])[0]
    adjclose = (indicators.get("adjclose") or [{}])[0].get("adjclose") or []

    bars: List[PriceBar] = []
    for idx, timestamp in enumerate(timestamps):
        open_price = _value_at(quote.get("open"), idx)
        high = _value_at(quote.get("high"), idx)
        low = _value_at(quote.get("low"), idx)
        close = _value_at(quote.get("close"), idx)
        if None in (open_price, high, low, close):
            continue
        volume = _value_at(quote.get("volume"), idx)
        bars.append(
            PriceBar(
                trade_date=datetime.fromtimestamp(timestamp, tz=timezone.utc).date(),
                code=code,
                source="yahoo",
                open=float(open_price),
                high=float(high),
                low=float(low),
                close=float(close),
                adj_close=_as_float(_value_at(adjclose, idx)),
                volume=int(volume) if volume is not None else None,
                turnover=None,
                source_url=source_url,
            )
        )
    return bars


def _to_epoch(value: date) -> int:
    return int(datetime.combine(value, time.min, tzinfo=timezone.utc).timestamp())


def _value_at(values, idx: int):
    if values is None or idx >= len(values):
        return None
    return values[idx]


def _as_float(value) -> Optional[float]:
    if value is None:
        return None
    return float(value)

