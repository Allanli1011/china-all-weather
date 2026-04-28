from __future__ import annotations

import re
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

from .models import MarketSummary, ShortTurnoverRecord, ShortTurnoverReport


class ParseError(ValueError):
    """Raised when a source document cannot be parsed."""


class ReportUnavailable(ParseError):
    """Raised when HKEX has not published the requested report yet."""


MONTHS = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}

SHORT_ROW_RE = re.compile(r"^\s*(%?\s*\d{1,5})\s+(.+?)\s+([\d,]+)\s+([\d,]+)\s*$")
QUOTE_SHORT_ROW_RE = re.compile(
    r"^\s*(%?\s*\d{1,5})\s+(.+?)\s+([\d,]+)\s+([\d,]+)\s+([\d,]+)\s+([\d,]+)\s*$"
)
TRADE_DATE_RE = re.compile(
    r"(?:TRADING\s+DATE|DATE)\s*:\s*(\d{1,2})\s+([A-Z]{3})\s+(\d{4})",
    re.IGNORECASE,
)
SECTION_RE = re.compile(r"^\s*\(([ABC])\)\s+(.+)$")
SUMMARY_SHARES_RE = re.compile(r"Short Selling Turnover Total Shares \(SH\)\s*:\s*([\d,]+)")
SUMMARY_VALUE_RE = re.compile(
    r"Short Selling Turnover Total Value \(\$\)\s*:\s*([A-Z]{3})\s*([\d,]+)"
)
MARKET_TURNOVER_RE = re.compile(r"Total market turnover\s*:\s*HKD\s*([\d,]+)")
PCT_RE = re.compile(r"as % total turnover\s*:\s*([\d.]+)%")


def parse_int(value: str) -> int:
    return int(value.replace(",", "").strip())


def parse_money(value: str) -> float:
    return float(value.replace(",", "").strip())


def normalize_code(raw_code: str) -> Tuple[str, bool]:
    stripped = raw_code.strip()
    is_non_hkd = stripped.startswith("%")
    digits = re.sub(r"\D", "", stripped)
    if not digits:
        raise ParseError("Missing stock code")
    return digits.zfill(5), is_non_hkd


def parse_trade_date(text: str) -> date:
    match = TRADE_DATE_RE.search(text)
    if not match:
        raise ParseError("Could not find trade date")
    day, month_text, year = match.groups()
    month = MONTHS[month_text.upper()]
    return date(int(year), month, int(day))


def parse_short_turnover_page(
    text: str,
    market: str,
    session: str,
    source_url: str,
    fetched_at: Optional[datetime] = None,
) -> ShortTurnoverReport:
    if "will be available after" in text.lower():
        raise ReportUnavailable("HKEX report is not available yet")
    if "SHORTSELL REPORT" not in text:
        raise ParseError("Document does not look like a short-selling report")

    trade_date = parse_trade_date(text)
    records: List[ShortTurnoverRecord] = []
    summaries = _parse_market_summaries(text, trade_date, market, session)

    for line in text.splitlines():
        match = SHORT_ROW_RE.match(line)
        if not match:
            continue
        raw_code, name, short_shares, short_value = match.groups()
        code, is_non_hkd = normalize_code(raw_code)
        records.append(
            ShortTurnoverRecord(
                trade_date=trade_date,
                market=market,
                session=session,
                code=code,
                raw_code=raw_code.strip(),
                name=" ".join(name.split()),
                short_shares=parse_int(short_shares),
                short_value=parse_money(short_value),
                currency="NON_HKD" if is_non_hkd else "HKD",
                is_non_hkd=is_non_hkd,
                source_url=source_url,
                fetched_at=fetched_at,
            )
        )

    if not records:
        raise ParseError("No short-selling rows found")

    return ShortTurnoverReport(
        trade_date=trade_date,
        market=market,
        session=session,
        source_url=source_url,
        records=records,
        summaries=summaries,
        fetched_at=fetched_at,
    )


def parse_daily_quote_short_turnover(
    text: str,
    market: str,
    source_url: str,
    fetched_at: Optional[datetime] = None,
) -> ShortTurnoverReport:
    trade_date = parse_trade_date(text)
    section = _extract_daily_quote_short_section(text)
    records: List[ShortTurnoverRecord] = []

    for line in section.splitlines():
        match = QUOTE_SHORT_ROW_RE.match(line)
        if not match:
            continue
        raw_code, name, short_shares, short_value, total_shares, total_value = match.groups()
        code, is_non_hkd = normalize_code(raw_code)
        records.append(
            ShortTurnoverRecord(
                trade_date=trade_date,
                market=market,
                session="day",
                code=code,
                raw_code=raw_code.strip(),
                name=" ".join(name.split()),
                short_shares=parse_int(short_shares),
                short_value=parse_money(short_value),
                total_shares=parse_int(total_shares),
                total_value=parse_money(total_value),
                currency="NON_HKD" if is_non_hkd else "HKD",
                is_non_hkd=is_non_hkd,
                source_url=source_url,
                fetched_at=fetched_at,
            )
        )

    if not records:
        raise ParseError("No daily-quote short-selling rows found")

    summaries = _parse_market_summaries(section, trade_date, market, "day")
    return ShortTurnoverReport(
        trade_date=trade_date,
        market=market,
        session="day",
        source_url=source_url,
        records=records,
        summaries=summaries,
        fetched_at=fetched_at,
    )


def _extract_daily_quote_short_section(text: str) -> str:
    marker = "SHORT SELLING TURNOVER - DAILY REPORT"
    starts = [match.start() for match in re.finditer(re.escape(marker), text)]
    if not starts:
        raise ParseError("Could not find daily short-selling section")
    start = starts[-1]
    end_candidates = [
        text.find("PREVIOUS DAY'S ADJUSTED SHORT SELLING TURNOVER", start + 1),
        text.find("OVERSEAS TURNOVER HIGHLIGHTS", start + 1),
        text.find("OTHER INFORMATION", start + 1),
    ]
    end_candidates = [idx for idx in end_candidates if idx != -1]
    end = min(end_candidates) if end_candidates else len(text)
    return text[start:end]


def _parse_market_summaries(
    text: str,
    trade_date: date,
    market: str,
    session: str,
) -> List[MarketSummary]:
    summaries: List[MarketSummary] = []
    current_key: Optional[str] = None
    current_title = ""
    current_shares: Optional[int] = None
    current_values: Dict[str, float] = {}
    current_market_turnover: Optional[float] = None
    current_pct: Optional[float] = None

    def flush() -> None:
        nonlocal current_key, current_title, current_shares, current_values
        nonlocal current_market_turnover, current_pct
        if not current_key:
            return
        summaries.append(
            MarketSummary(
                trade_date=trade_date,
                market=market,
                session=session,
                section=current_title or current_key,
                short_shares=current_shares,
                short_values=dict(current_values),
                market_turnover_hkd=current_market_turnover,
                short_pct_market=current_pct,
            )
        )
        current_key = None
        current_title = ""
        current_shares = None
        current_values = {}
        current_market_turnover = None
        current_pct = None

    for line in text.splitlines():
        if line.lstrip().startswith("*Total No. of non-Designated"):
            flush()
            continue

        section_match = SECTION_RE.match(line)
        if section_match:
            flush()
            current_key, current_title = section_match.groups()
            current_title = " ".join(current_title.split())
            continue

        if not current_key:
            continue

        shares_match = SUMMARY_SHARES_RE.search(line)
        if shares_match:
            current_shares = parse_int(shares_match.group(1))
            continue

        value_match = SUMMARY_VALUE_RE.search(line)
        if value_match:
            currency, value = value_match.groups()
            current_values[currency.upper()] = parse_money(value)
            continue

        market_match = MARKET_TURNOVER_RE.search(line)
        if market_match:
            current_market_turnover = parse_money(market_match.group(1))
            continue

        pct_match = PCT_RE.search(line)
        if pct_match:
            current_pct = float(pct_match.group(1))
            continue

    flush()
    return summaries
