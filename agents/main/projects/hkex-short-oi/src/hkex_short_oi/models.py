from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ShortTurnoverRecord:
    trade_date: date
    market: str
    session: str
    code: str
    raw_code: str
    name: str
    short_shares: int
    short_value: float
    total_shares: Optional[int] = None
    total_value: Optional[float] = None
    currency: str = "HKD"
    is_non_hkd: bool = False
    source_url: str = ""
    fetched_at: Optional[datetime] = None

    @property
    def short_ratio(self) -> Optional[float]:
        if not self.total_value:
            return None
        return self.short_value / self.total_value


@dataclass(frozen=True)
class MarketSummary:
    trade_date: date
    market: str
    session: str
    section: str
    short_shares: Optional[int] = None
    short_values: Dict[str, float] = field(default_factory=dict)
    market_turnover_hkd: Optional[float] = None
    short_pct_market: Optional[float] = None


@dataclass(frozen=True)
class ShortTurnoverReport:
    trade_date: date
    market: str
    session: str
    source_url: str
    records: List[ShortTurnoverRecord]
    summaries: List[MarketSummary]
    fetched_at: Optional[datetime] = None


@dataclass(frozen=True)
class FeatureRow:
    record: ShortTurnoverRecord
    short_value_z: Dict[int, Optional[float]]
    short_ratio_z: Dict[int, Optional[float]]
    short_value_percentile: Dict[int, Optional[float]]
    short_ratio_percentile: Dict[int, Optional[float]]
    short_value_ma: Dict[int, Optional[float]]
    short_ratio_ma: Dict[int, Optional[float]]


@dataclass(frozen=True)
class MarketFeatureRow:
    trade_date: date
    market: str
    total_short_value: float
    total_market_value: float
    short_ratio: float
    short_ratio_z: Dict[int, Optional[float]]
    short_ratio_percentile: Dict[int, Optional[float]]


@dataclass(frozen=True)
class StrategySignal:
    trade_date: date
    code: str
    name: str
    strategy: str
    direction: str
    score: float
    reasons: List[str]
    required_confirmation: List[str]


@dataclass(frozen=True)
class ShortPositionRecord:
    report_date: date
    code: str
    raw_code: str
    name: str
    short_position_shares: int
    short_position_hkd: float
    source_url: str = ""
    fetched_at: Optional[datetime] = None


@dataclass(frozen=True)
class PriceBar:
    trade_date: date
    code: str
    source: str
    open: float
    high: float
    low: float
    close: float
    adj_close: Optional[float]
    volume: Optional[int]
    turnover: Optional[float]
    source_url: str = ""
    fetched_at: Optional[datetime] = None


@dataclass(frozen=True)
class StrategyBacktestStats:
    count: int
    avg_return: Optional[float]
    median_return: Optional[float]
    win_rate: Optional[float]
    min_return: Optional[float]
    max_return: Optional[float]
    avg_net_return: Optional[float] = None
    max_drawdown: Optional[float] = None
    ic: Optional[float] = None


@dataclass(frozen=True)
class StrategyBacktestResult:
    by_strategy: Dict[str, Dict[int, StrategyBacktestStats]]
    events: List[dict]
