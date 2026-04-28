from __future__ import annotations

from datetime import datetime
from typing import Iterable, List, Optional

from .models import FeatureRow, MarketSummary, ShortPositionRecord, ShortTurnoverReport, StrategySignal


def build_monitor_report(
    reports: Iterable[ShortTurnoverReport],
    features: Iterable[FeatureRow],
    signals: Iterable[StrategySignal],
    summaries: Iterable[MarketSummary],
    sfc_positions: Iterable[ShortPositionRecord] = (),
    generated_at: Optional[datetime] = None,
    top_n: int = 20,
) -> str:
    report_list = list(reports)
    feature_list = list(features)
    signal_list = list(signals)
    summary_list = list(summaries)
    sfc_position_list = list(sfc_positions)
    generated_at = generated_at or datetime.now()

    latest_date = max((report.trade_date for report in report_list), default=None)
    lines: List[str] = [
        f"# HKEX Short-Selling Turnover Monitor - {latest_date or 'n/a'}",
        "",
        f"Generated: {generated_at.isoformat(timespec='seconds')}",
        "",
        "> Note: HKEX daily data is short-selling turnover, not short interest or open short positions.",
        "> Use SFC weekly aggregated reportable short positions as the position proxy when added.",
        "",
        "## Loaded Sources",
        "",
    ]

    for report in report_list:
        lines.append(
            f"- {report.market}/{report.session}: {len(report.records)} rows from {report.source_url}"
        )
    if not report_list:
        lines.append("- No reports loaded.")

    lines.extend(["", "## Market Summary", ""])
    if summary_list:
        lines.append("| Market | Session | Section | Short Value | Market Turnover HKD | Short % |")
        lines.append("|---|---:|---|---:|---:|---:|")
        for summary in summary_list:
            hkd_value = summary.short_values.get("HKD")
            lines.append(
                "| {market} | {session} | {section} | {short_value} | {market_turnover} | {pct} |".format(
                    market=summary.market,
                    session=summary.session,
                    section=summary.section,
                    short_value=_fmt_money(hkd_value),
                    market_turnover=_fmt_money(summary.market_turnover_hkd),
                    pct=_fmt_pct(summary.short_pct_market),
                )
            )
    else:
        lines.append("No market summaries parsed.")

    current_features = _latest_features(feature_list)
    lines.extend(["", f"## Top {top_n} Short-Selling Turnover", ""])
    lines.extend(_feature_table(current_features, top_n=top_n, sort_key="short_value"))

    ratio_features = [row for row in current_features if row.record.short_ratio is not None]
    if ratio_features:
        lines.extend(["", f"## Top {top_n} Short-Selling Ratio", ""])
        lines.extend(_feature_table(ratio_features, top_n=top_n, sort_key="short_ratio"))

    lines.extend(["", "## Latest SFC Short-Position Proxy", ""])
    lines.extend(_sfc_position_table(sfc_position_list, top_n=10))

    lines.extend(["", "## Strategy Watchlist", ""])
    if signal_list:
        lines.append("| Code | Name | Strategy | Direction | Score | Required Confirmation |")
        lines.append("|---|---|---|---|---:|---|")
        for signal in signal_list[:top_n]:
            lines.append(
                "| {code} | {name} | {strategy} | {direction} | {score:.2f} | {confirm} |".format(
                    code=signal.code,
                    name=signal.name,
                    strategy=signal.strategy,
                    direction=signal.direction,
                    score=signal.score,
                    confirm="; ".join(signal.required_confirmation),
                )
            )
    else:
        lines.append("No strategy signals yet. This is normal until enough history exists for z-scores.")

    lines.extend(
        [
            "",
            "## Next Checks",
            "",
            "- Add price/volume data before turning watchlist items into executable trades.",
            "- Add SFC weekly aggregated reportable short positions for true short-position context.",
            "- Treat non-HKD counters carefully; row values are preserved with a NON_HKD currency hint.",
            "",
        ]
    )
    return "\n".join(lines)


def _latest_features(features: List[FeatureRow]) -> List[FeatureRow]:
    if not features:
        return []
    latest_date = max(row.record.trade_date for row in features)
    return [row for row in features if row.record.trade_date == latest_date]


def _feature_table(features: List[FeatureRow], top_n: int, sort_key: str) -> List[str]:
    if sort_key == "short_ratio":
        sorted_rows = sorted(features, key=lambda row: row.record.short_ratio or 0.0, reverse=True)
    else:
        sorted_rows = sorted(features, key=lambda row: row.record.short_value, reverse=True)

    lines = [
        "| Code | Name | Market | Session | Short Value | Short Ratio | Z20 | Z60 |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for row in sorted_rows[:top_n]:
        lines.append(
            "| {code} | {name} | {market} | {session} | {value} | {ratio} | {z20} | {z60} |".format(
                code=row.record.code,
                name=row.record.name,
                market=row.record.market,
                session=row.record.session,
                value=_fmt_money(row.record.short_value),
                ratio=_fmt_ratio(row.record.short_ratio),
                z20=_fmt_number(row.short_value_z.get(20)),
                z60=_fmt_number(row.short_value_z.get(60)),
            )
        )
    if len(lines) == 2:
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
    return lines


def _sfc_position_table(records: List[ShortPositionRecord], top_n: int) -> List[str]:
    if not records:
        return ["No SFC positions loaded yet."]
    latest_date = max(record.report_date for record in records)
    latest = [record for record in records if record.report_date == latest_date]
    latest.sort(key=lambda record: record.short_position_hkd, reverse=True)
    lines = [
        f"Latest SFC report date: {latest_date}",
        "",
        "| Code | Name | Short Position Shares | Short Position HKD |",
        "|---|---|---:|---:|",
    ]
    for record in latest[:top_n]:
        lines.append(
            "| {code} | {name} | {shares} | {value} |".format(
                code=record.code,
                name=record.name,
                shares=f"{record.short_position_shares:,}",
                value=_fmt_money(record.short_position_hkd),
            )
        )
    return lines


def _fmt_money(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:,.0f}"


def _fmt_pct(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}%"


def _fmt_ratio(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1%}"


def _fmt_number(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"
