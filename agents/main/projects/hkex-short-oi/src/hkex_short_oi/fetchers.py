from __future__ import annotations

import time
import subprocess
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class FetchError(RuntimeError):
    """Raised when a source cannot be fetched."""


CURRENT_SHORT_URLS: Dict[Tuple[str, str], str] = {
    ("main", "morning"): "https://www.hkex.com.hk/eng/stat/smstat/ssturnover/ncms/mshtmain.htm",
    ("main", "day"): "https://www.hkex.com.hk/eng/stat/smstat/ssturnover/ncms/ashtmain.htm",
    ("gem", "morning"): "https://www.hkex.com.hk/eng/stat/smstat/ssturnover/ncms/mshtgem.htm",
    ("gem", "day"): "https://www.hkex.com.hk/eng/stat/smstat/ssturnover/ncms/ashtgem.htm",
}


@dataclass(frozen=True)
class FetchResult:
    url: str
    text: str


class HTTPTextClient:
    def __init__(self, timeout: int = 30, retries: int = 2, pause_seconds: float = 1.0) -> None:
        self.timeout = timeout
        self.retries = retries
        self.pause_seconds = pause_seconds

    def get(self, url: str) -> str:
        last_error = None
        for attempt in range(self.retries + 1):
            try:
                request = Request(
                    url,
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/123.0 Safari/537.36"
                        ),
                        "Accept": "text/html,text/plain,*/*",
                    },
                )
                with urlopen(request, timeout=self.timeout) as response:
                    charset = response.headers.get_content_charset() or "utf-8"
                    body = response.read()
                    return body.decode(charset, errors="replace")
            except (HTTPError, URLError, TimeoutError) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(self.pause_seconds * (attempt + 1))
        raise FetchError(f"Could not fetch {url}: {last_error}")


class CurlTextClient:
    def __init__(self, timeout: int = 20, retries: int = 0, pause_seconds: float = 1.0) -> None:
        self.timeout = timeout
        self.retries = retries
        self.pause_seconds = pause_seconds

    def get(self, url: str) -> str:
        last_error = None
        for attempt in range(self.retries + 1):
            result = subprocess.run(
                [
                    "curl",
                    "-L",
                    "-sS",
                    "--max-time",
                    str(self.timeout),
                    "-A",
                    "Mozilla/5.0",
                    url,
                ],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if result.returncode == 0 and result.stdout:
                return result.stdout
            last_error = result.stderr.strip() or f"curl exit {result.returncode}"
            if attempt < self.retries:
                time.sleep(self.pause_seconds * (attempt + 1))
        raise FetchError(f"Could not fetch {url}: {last_error}")


class HKEXFetcher:
    def __init__(self, client: HTTPTextClient) -> None:
        self.client = client

    def fetch_current_short_turnover(self, market: str, session: str) -> FetchResult:
        key = (market, session)
        if key not in CURRENT_SHORT_URLS:
            raise ValueError(f"Unsupported market/session: {market}/{session}")
        url = CURRENT_SHORT_URLS[key]
        return FetchResult(url=url, text=self.client.get(url))

    def fetch_daily_quote(self, market: str, trade_date: date) -> FetchResult:
        errors: List[str] = []
        for url in daily_quote_urls(market, trade_date):
            try:
                text = self.client.get(url)
            except FetchError as exc:
                errors.append(str(exc))
                continue
            if "The page requested may have been relocated" in text:
                errors.append(f"{url}: relocated/removed page")
                continue
            return FetchResult(url=url, text=text)
        raise FetchError("; ".join(errors) or f"No daily quotation URL worked for {market} {trade_date}")


class YahooPriceFetcher:
    def __init__(self, client: HTTPTextClient) -> None:
        self.client = client

    def fetch_chart(self, code: str, start_date: date, end_date: date) -> FetchResult:
        from .price import yahoo_chart_url

        url = yahoo_chart_url(code, start_date, end_date)
        return FetchResult(url=url, text=self.client.get(url))


class SFCFetcher:
    def __init__(self, client: HTTPTextClient) -> None:
        self.client = client

    def fetch_latest_positions_csv(self) -> FetchResult:
        from .sfc import SFC_POSITIONS_PAGE_URL, find_sfc_csv_links

        html = self.client.get(SFC_POSITIONS_PAGE_URL)
        links = find_sfc_csv_links(html)
        if not links:
            raise FetchError("Could not find SFC short-position CSV links")
        url = links[0]
        return FetchResult(url=url, text=self.client.get(url))


def daily_quote_urls(market: str, trade_date: date) -> List[str]:
    yymmdd = trade_date.strftime("%y%m%d")
    if market == "main":
        file_name = f"d{yymmdd}e.htm"
        paths = [file_name, f"e_{yymmdd}.htm"]
    elif market == "gem":
        file_name = f"e_G{yymmdd}.htm"
        paths = [f"GEM/{file_name}"]
    else:
        raise ValueError(f"Unsupported market: {market}")

    urls = []
    for path in paths:
        urls.append(f"https://www.hkex.com.hk/eng/stat/smstat/dayquot/{path}")
        urls.append(f"https://www.hkex.com.hk/eng/stat/smstat/dayquot_12m/{path}")
    return urls
