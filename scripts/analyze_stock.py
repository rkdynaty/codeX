#!/usr/bin/env python3
"""Analyze stock data from Yahoo Finance using only the Python standard library.

This script downloads historical price data for a given ticker symbol directly
from the public Yahoo Finance CSV endpoint, computes common technical
indicators, and outputs a summary report. Results can also be exported to CSV
for further analysis. When running without internet access you can supply a
local CSV file (in the same format as Yahoo's export) via ``--data-file`` and
the script will operate entirely offline.

Example usage
-------------
Fetch one year of data for Apple (AAPL), compute 20/50/200 day moving averages,
and export the full dataset with indicators to a CSV file:

    python scripts/analyze_stock.py AAPL --moving-averages 20 50 200 \
        --export data/aapl_analysis.csv

To see the summary without exporting:

    python scripts/analyze_stock.py TSLA --start 2023-01-01

The script relies only on the Python standard library so no additional
dependencies are required.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from http.cookiejar import CookieJar
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


MAX_DOWNLOAD_RETRIES = 3
BASE_RETRY_DELAY_SECONDS = 2
YAHOO_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
YAHOO_CRUMB_URL = "https://query1.finance.yahoo.com/v1/test/getcrumb"
_YAHOO_SESSION: Optional[Tuple[urllib.request.OpenerDirector, str]] = None


def _get_yahoo_session() -> Tuple[urllib.request.OpenerDirector, str]:
    """Return a Yahoo Finance opener and crumb token, creating them on demand."""

    global _YAHOO_SESSION
    if _YAHOO_SESSION is not None:
        return _YAHOO_SESSION

    cookie_jar = CookieJar()
    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(cookie_jar)
    )
    opener.addheaders = [("User-Agent", YAHOO_USER_AGENT)]

    with opener.open(YAHOO_CRUMB_URL) as response:
        crumb = response.read().decode("utf-8").strip()

    if not crumb:
        raise RuntimeError("Failed to obtain crumb token from Yahoo Finance.")

    _YAHOO_SESSION = (opener, crumb)
    return _YAHOO_SESSION


def _reset_yahoo_session() -> None:
    """Clear cached Yahoo session details so they can be refreshed."""

    global _YAHOO_SESSION
    _YAHOO_SESSION = None


@dataclass
class AnalysisResult:
    """Container for the analysis summary."""

    ticker: str
    start: dt.date
    end: dt.date
    trading_days: int
    cumulative_return: float
    annualized_volatility: float
    mean_daily_return: float
    latest_close: float
    moving_averages: List[tuple[int, Optional[float]]]

    def as_text(self) -> str:
        """Return a formatted string summarizing the analysis."""

        ma_lines = "\n".join(
            f"  • {window}-day MA: {value:,.2f}" if value is not None else f"  • {window}-day MA: not available"
            for window, value in self.moving_averages
        )
        if not ma_lines:
            ma_lines = "  • (none requested)"

        return (
            f"Analysis for {self.ticker}\n"
            f"Date range: {self.start:%Y-%m-%d} → {self.end:%Y-%m-%d}"
            f" (trading days: {self.trading_days})\n"
            f"Latest close: ${self.latest_close:,.2f}\n"
            f"Cumulative return: {self.cumulative_return:.2%}\n"
            f"Mean daily return: {self.mean_daily_return:.4%}\n"
            f"Annualized volatility: {self.annualized_volatility:.2%}\n"
            f"Moving averages:\n{ma_lines}"
        )


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and analyze stock price data from Yahoo Finance.",
    )
    parser.add_argument(
        "ticker",
        help="Ticker symbol to download (e.g., AAPL, TSLA, MSFT).",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD). Defaults to one year before the end date.",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD). Defaults to today.",
    )
    parser.add_argument(
        "--moving-averages",
        type=int,
        nargs="*",
        default=[20, 50, 200],
        help=(
            "Windows (in trading days) for which to compute moving averages. "
            "Defaults to the common set: 20, 50, 200."
        ),
    )
    parser.add_argument(
        "--export",
        type=str,
        default=None,
        help="Optional path to export the enriched dataset as a CSV file.",
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default=None,
        help=(
            "Optional path to a local CSV file containing Yahoo Finance-formatted "
            "historical data. When provided, the script skips network downloads "
            "and uses this file instead. Useful when running offline or for "
            "repeatable demos."
        ),
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="If set, suppress printing the analysis summary to stdout.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help=(
            "Directory used to cache downloaded CSV files. When download requests "
            "fail, the script will fall back to cached data if available."
        ),
    )
    return parser.parse_args(argv)


def normalize_dates(start_str: Optional[str], end_str: Optional[str]) -> tuple[dt.date, dt.date]:
    today = dt.date.today()
    if end_str:
        end = dt.datetime.strptime(end_str, "%Y-%m-%d").date()
    else:
        end = today

    if start_str:
        start = dt.datetime.strptime(start_str, "%Y-%m-%d").date()
    else:
        start = end - dt.timedelta(days=365)

    if start >= end:
        raise ValueError("Start date must be before end date.")

    return start, end


def _to_unix_epoch(date: dt.date) -> int:
    return int(time.mktime(dt.datetime.combine(date, dt.time()).timetuple()))


def _parse_rows(reader: csv.DictReader) -> List[dict]:
    rows: List[dict] = []
    for row in reader:
        if row.get("Close") in {None, "null"}:
            continue
        try:
            parsed = {
                "Date": dt.datetime.strptime(row["Date"], "%Y-%m-%d").date(),
                "Open": float(row["Open"]),
                "High": float(row["High"]),
                "Low": float(row["Low"]),
                "Close": float(row["Close"]),
                "Adj Close": float(row["Adj Close"]),
                "Volume": int(float(row["Volume"])),
            }
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Unexpected data format encountered while parsing: {row}"
            ) from exc
        rows.append(parsed)
    return rows


def download_data(
    ticker: str,
    start: dt.date,
    end: dt.date,
    *,
    cache_file: Optional[Path] = None,
) -> List[dict]:
    """Download historical OHLC data for *ticker* between start and end dates."""

    if not ticker:
        raise ValueError("Ticker symbol must not be empty.")

    end_plus_one = end + dt.timedelta(days=1)
    params = {
        "period1": str(_to_unix_epoch(start)),
        "period2": str(_to_unix_epoch(end_plus_one)),
        "interval": "1d",
        "events": "history",
        "includeAdjustedClose": "true",
    }
    base_url = (
        "https://query1.finance.yahoo.com/v7/finance/download/"
        f"{urllib.parse.quote(ticker)}"
    )

    last_error: Optional[Exception] = None
    for attempt in range(1, MAX_DOWNLOAD_RETRIES + 1):
        try:
            opener, crumb = _get_yahoo_session()
            query = params | {"crumb": crumb}
            request_url = f"{base_url}?{urllib.parse.urlencode(query)}"
            request = urllib.request.Request(
                request_url,
                headers={"User-Agent": YAHOO_USER_AGENT},
            )
            with opener.open(request) as response:  # nosec: B310 - trusted endpoint
                data = response.read().decode("utf-8")
                if cache_file is not None:
                    cache_file.parent.mkdir(parents=True, exist_ok=True)
                    cache_file.write_text(data, encoding="utf-8")
                break
        except urllib.error.HTTPError as err:
            last_error = err
            if err.code == 429 and attempt < MAX_DOWNLOAD_RETRIES:
                retry_after = err.headers.get("Retry-After")
                try:
                    delay = float(retry_after)
                except (TypeError, ValueError):
                    delay = BASE_RETRY_DELAY_SECONDS * attempt
                time.sleep(delay)
                continue
            if err.code in {401, 403} and attempt < MAX_DOWNLOAD_RETRIES:
                _reset_yahoo_session()
                time.sleep(BASE_RETRY_DELAY_SECONDS * attempt)
                continue
            if err.code == 429:
                raise ValueError(
                    "Yahoo Finance rate limited the request (HTTP 429). "
                    "Please wait a moment or rerun the command with --data-file for offline data."
                ) from err
            raise ValueError(
                f"Failed to download data for {ticker}: HTTP {err.code}"
            ) from err
        except urllib.error.URLError as err:
            last_error = err
            raise ConnectionError(
                "Network error while downloading data. Ensure internet access."
            ) from err
    else:
        if last_error:
            raise last_error
        raise RuntimeError("Failed to download data for an unknown reason.")

    reader = csv.DictReader(data.splitlines())
    rows = _parse_rows(reader)

    if not rows:
        raise ValueError(
            "No data returned. Verify the ticker symbol and date range, and ensure "
            "you have an active internet connection."
        )

    return rows


def load_data_from_file(path: str) -> List[dict]:
    """Load historical data from a Yahoo-formatted CSV file."""

    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = _parse_rows(reader)

    if not rows:
        raise ValueError("The provided data file does not contain any rows.")

    rows.sort(key=lambda item: item["Date"])
    return rows


def compute_moving_average(values: Sequence[float], window: int) -> List[Optional[float]]:
    if window <= 0:
        raise ValueError("Moving average window must be a positive integer.")
    result: List[Optional[float]] = []
    cumulative = 0.0
    buffer: List[float] = []
    for value in values:
        buffer.append(value)
        cumulative += value
        if len(buffer) > window:
            cumulative -= buffer.pop(0)
        if len(buffer) == window:
            result.append(cumulative / window)
        else:
            result.append(None)
    return result


def compute_indicators(rows: List[dict], moving_windows: Iterable[int]) -> List[dict]:
    enriched: List[dict] = []
    adj_close_values = [row["Adj Close"] for row in rows]

    moving_average_map = {
        window: compute_moving_average(adj_close_values, window)
        for window in moving_windows
    }

    previous_adj_close: Optional[float] = None
    for index, row in enumerate(rows):
        enriched_row = dict(row)
        current_adj_close = row["Adj Close"]
        if previous_adj_close is None:
            enriched_row["Daily Return"] = None
        else:
            enriched_row["Daily Return"] = current_adj_close / previous_adj_close - 1.0
        for window, series in moving_average_map.items():
            enriched_row[f"MA_{window}"] = series[index]
        enriched.append(enriched_row)
        previous_adj_close = current_adj_close

    return enriched


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Cannot compute mean of empty sequence.")
    return sum(values) / len(values)


def standard_deviation(values: Sequence[float]) -> Optional[float]:
    n = len(values)
    if n < 2:
        return None
    avg = mean(values)
    variance = sum((value - avg) ** 2 for value in values) / (n - 1)
    return math.sqrt(variance)


def summarize(
    ticker: str, enriched: List[dict], moving_windows: Iterable[int]
) -> AnalysisResult:
    start_date = enriched[0]["Date"]
    end_date = enriched[-1]["Date"]
    trading_days = len(enriched)

    first_adj_close = enriched[0]["Adj Close"]
    last_adj_close = enriched[-1]["Adj Close"]
    cumulative_return = last_adj_close / first_adj_close - 1.0

    daily_returns = [
        row["Daily Return"]
        for row in enriched
        if row.get("Daily Return") is not None
    ]

    mean_daily_return = mean(daily_returns) if daily_returns else 0.0
    stdev = standard_deviation(daily_returns)
    volatility = stdev * math.sqrt(252) if stdev is not None else 0.0

    moving_averages = []
    for window in moving_windows:
        key = f"MA_{window}"
        moving_averages.append((window, enriched[-1].get(key)))

    return AnalysisResult(
        ticker=ticker,
        start=start_date,
        end=end_date,
        trading_days=trading_days,
        cumulative_return=cumulative_return,
        annualized_volatility=volatility,
        mean_daily_return=mean_daily_return,
        latest_close=enriched[-1]["Close"],
        moving_averages=moving_averages,
    )


def export_to_csv(enriched: List[dict], path: str) -> None:
    if not enriched:
        raise ValueError("No data to export.")
    fieldnames = list(enriched[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in enriched:
            writable = dict(row)
            writable["Date"] = row["Date"].isoformat()
            writer.writerow(writable)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    try:
        cache_dir: Optional[Path] = Path(args.cache_dir).expanduser() if args.cache_dir else None
        if args.data_file:
            raw_rows = load_data_from_file(args.data_file)
            if args.start or args.end:
                start, end = normalize_dates(args.start, args.end)
                raw_rows = [
                    row for row in raw_rows if start <= row["Date"] <= end
                ]
                if not raw_rows:
                    raise ValueError(
                        "No rows within the requested date range were found in the "
                        "local data file."
                    )
            else:
                start = raw_rows[0]["Date"]
                end = raw_rows[-1]["Date"]
        else:
            start, end = normalize_dates(args.start, args.end)
            cache_file: Optional[Path] = None
            if cache_dir is not None:
                filename = f"{args.ticker.upper()}_{start.isoformat()}_{end.isoformat()}.csv"
                cache_file = cache_dir / filename
            try:
                raw_rows = download_data(args.ticker, start, end, cache_file=cache_file)
            except ValueError as exc:
                if cache_file is not None and cache_file.exists():
                    raw_rows = load_data_from_file(str(cache_file))
                    raw_rows = [
                        row for row in raw_rows if start <= row["Date"] <= end
                    ]
                    if not raw_rows:
                        raise ValueError(
                            "Cached data exists but does not cover the requested date range."
                        ) from exc
                    print(
                        "Warning: Using cached data because live download failed.",
                        file=sys.stderr,
                    )
                elif "HTTP 401" in str(exc):
                    sample_path = (
                        Path(__file__).resolve().parent.parent / "data" / "sample_aapl.csv"
                    )
                    if sample_path.exists():
                        raw_rows = load_data_from_file(str(sample_path))
                        raw_rows = [
                            row for row in raw_rows if start <= row["Date"] <= end
                        ]
                        if not raw_rows:
                            raw_rows = load_data_from_file(str(sample_path))
                        print(
                            "Warning: Yahoo Finance rejected the request (HTTP 401). "
                            "Using bundled sample data from data/sample_aapl.csv instead.",
                            file=sys.stderr,
                        )
                        if args.ticker.upper() != "AAPL":
                            print(
                                "Note: The local sample contains AAPL prices regardless of the requested ticker.",
                                file=sys.stderr,
                            )
                        start = raw_rows[0]["Date"]
                        end = raw_rows[-1]["Date"]
                    else:
                        raise
                else:
                    raise
        enriched = compute_indicators(raw_rows, args.moving_averages)
        summary = summarize(args.ticker, enriched, args.moving_averages)
    except Exception as exc:  # pragma: no cover - CLI surface
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.export:
        export_to_csv(enriched, args.export)

    if not args.no_summary:
        print(summary.as_text())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
