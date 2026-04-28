from __future__ import annotations

import csv
import re
from datetime import date, datetime
from io import StringIO
from typing import List, Optional

from .models import ShortPositionRecord
from .parsers import ParseError, normalize_code, parse_money


SFC_POSITIONS_PAGE_URL = (
    "https://www.sfc.hk/en/Regulatory-functions/Market/Short-position-reporting/"
    "Aggregated-reportable-short-positions-of-specified-shares"
)

CSV_LINK_RE = re.compile(r'href="([^"]+Short_Position_Reporting_Aggregated_Data_[^"]+\.csv[^"]*)"', re.I)


def find_sfc_csv_links(html: str) -> List[str]:
    links = []
    for match in CSV_LINK_RE.finditer(html):
        url = match.group(1).replace("&amp;", "&")
        if url.startswith("http"):
            links.append(url)
        else:
            links.append(f"https://www.sfc.hk{url}")
    return links


def parse_sfc_positions_csv(
    text: str,
    source_url: str,
    fetched_at: Optional[datetime] = None,
) -> List[ShortPositionRecord]:
    reader = csv.DictReader(StringIO(text))
    required = {
        "Date",
        "Stock Code",
        "Stock Name",
        "Aggregated Reportable Short Positions (Shares)",
        "Aggregated Reportable Short Positions (HK$)",
    }
    if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
        raise ParseError("SFC CSV is missing required columns")

    records: List[ShortPositionRecord] = []
    for row in reader:
        raw_code = row["Stock Code"].strip()
        code, _ = normalize_code(raw_code)
        records.append(
            ShortPositionRecord(
                report_date=_parse_sfc_date(row["Date"]),
                code=code,
                raw_code=raw_code,
                name=" ".join(row["Stock Name"].split()),
                short_position_shares=int(parse_money(row["Aggregated Reportable Short Positions (Shares)"])),
                short_position_hkd=parse_money(row["Aggregated Reportable Short Positions (HK$)"]),
                source_url=source_url,
                fetched_at=fetched_at,
            )
        )
    return records


def _parse_sfc_date(value: str) -> date:
    day, month, year = value.strip().split("/")
    return date(int(year), int(month), int(day))
