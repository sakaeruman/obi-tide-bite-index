"""JMA tide-prediction (suisan) fetcher for OBI v1 (Tokuyama / QA).

Source: 気象庁 潮位表データ (annual fixed-width text)
URL pattern: https://www.data.jma.go.jp/kaiyou/data/db/tide/suisan/txt/{YYYY}/{STN}.txt

NOTE on station code:
- The original spec referenced `hry{YYYYMM}TY.txt` (monthly observed-tide format
  under .../genbo/...). That endpoint is no longer publicly exposed as flat files,
  and `TY` is in fact Toyama (富山). Tokuyama (徳山) is station **QA** under JMA's
  suisan (tide prediction) annual text dataset. This implementation uses the
  suisan annual file as the canonical source and keeps a config-driven fallback
  list so the operator can swap codes without code changes.

Line format (fixed-width, 136 chars + LF, one line per day):
  [0:72]    24 hourly tide heights, each 3 chars (cm above station datum).
            Right-justified with leading spaces; negative values use a minus
            sign in the first char.
  [72:74]   year, last 2 digits (e.g. "24")
  [74:76]   month, right-justified (" 1" .. "12")
  [76:78]   day,   right-justified (" 1" .. "31")
  [78:80]   station code (2 chars, e.g. "QA")
  [80:108]  4 high-tide entries, 7 chars each: HHMM(4) + height(3).
            Padding "999"/"9999" when fewer than 4 high tides that day.
  [108:136] 4 low-tide entries, 7 chars each: HHMM(4) + height(3). Same padding.

This module exposes:
  fetch_year(year, station_code)   -> raw text (cached on disk per station/year)
  parse_year_text(text, year)      -> hourly DataFrame
  fetch_month(year, month, code)   -> DataFrame (datetime[JST], tide_cm)
  fetch_recent(days, station_code) -> DataFrame (today +/- days, month-rollover safe)
"""
from __future__ import annotations

import datetime as _dt
import logging
import logging.handlers
import sys
import time
import traceback
from pathlib import Path
from typing import List, Optional
from zoneinfo import ZoneInfo

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _THIS_FILE.parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / "tide_TY"  # name kept per spec
LOG_DIR = PROJECT_ROOT / "logs"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

JMA_URL_TEMPLATE = (
    "https://www.data.jma.go.jp/kaiyou/data/db/tide/suisan/txt/{year}/{code}.txt"
)
JST = ZoneInfo("Asia/Tokyo")

# Default code + fallback chain. Operator can override via config.yaml; the
# helper functions accept any string and try the configured chain in order.
DEFAULT_PRIMARY_CODE = "QA"          # Tokuyama 徳山 (confirmed against JMA selector)
DEFAULT_FALLBACK_CODES = ("TY", "TKY")  # legacy guesses kept per spec request

REQUEST_TIMEOUT_SEC = 30
USER_AGENT = "OBI-v1/0.1 (+okikamuro fishing index, contact: sakaeru)"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def _build_logger() -> logging.Logger:
    logger = logging.getLogger("obi.fetch_tide")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    file_handler = logging.handlers.RotatingFileHandler(
        LOG_DIR / "fetch_tide.log",
        maxBytes=1_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    return logger


log = _build_logger()


# ---------------------------------------------------------------------------
# Low-level fetch
# ---------------------------------------------------------------------------
def _annual_cache_path(year: int, station_code: str) -> Path:
    """Path to the cached raw annual text for a year/station."""
    return CACHE_DIR / f"{year:04d}_{station_code}.txt"


def _download_year(year: int, station_code: str) -> Optional[str]:
    """Download one year of raw JMA suisan text. Returns text or None on failure."""
    url = JMA_URL_TEMPLATE.format(year=year, code=station_code)
    headers = {"User-Agent": USER_AGENT}
    try:
        resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT_SEC)
    except requests.RequestException:
        log.error("network error fetching %s\n%s", url, traceback.format_exc())
        return None
    if resp.status_code == 404:
        log.warning("404 for %s (station code not valid for this year)", url)
        return None
    if resp.status_code != 200:
        log.error("HTTP %s for %s", resp.status_code, url)
        return None
    # JMA suisan txt is ASCII; encoding is safe as latin-1 or utf-8.
    try:
        text = resp.content.decode("utf-8")
    except UnicodeDecodeError:
        text = resp.content.decode("latin-1", errors="replace")
    if not text.strip():
        log.error("empty body for %s", url)
        return None
    return text


def fetch_year(
    year: int,
    station_code: str,
    fallback_codes: Optional[List[str]] = None,
    force_refresh: bool = False,
) -> Optional[str]:
    """Fetch raw annual text for a year, with on-disk cache and code-fallback chain.

    Cache policy:
      * Past years   -> cached forever (data is final).
      * Current year -> always refetched (`force_refresh` is ignored here; we
                        unconditionally re-download the current year so today's
                        predictions stay aligned with JMA's latest table).
    """
    today = _dt.datetime.now(JST).date()
    is_current_year = year == today.year

    codes_to_try: List[str] = [station_code]
    if fallback_codes:
        codes_to_try.extend(c for c in fallback_codes if c and c != station_code)

    # Try cached first (only for the requested primary code, only for past years).
    cache = _annual_cache_path(year, station_code)
    if cache.exists() and not is_current_year and not force_refresh:
        try:
            return cache.read_text(encoding="utf-8")
        except OSError:
            log.warning("cache read failed for %s; refetching", cache)

    last_error_code: Optional[str] = None
    for code in codes_to_try:
        text = _download_year(year, code)
        if text is None:
            last_error_code = code
            continue
        # Persist under the *primary* requested code so callers find it next time.
        try:
            _annual_cache_path(year, station_code).write_text(text, encoding="utf-8")
        except OSError:
            log.error("failed to write cache file\n%s", traceback.format_exc())
        if code != station_code:
            log.info("fetched %d using fallback code %s (primary %s failed)",
                     year, code, station_code)
        return text

    log.error("all station codes failed for year=%d (last tried=%s, chain=%s)",
              year, last_error_code, codes_to_try)
    return None


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def _parse_int_field(raw: str) -> int:
    """Parse a 3-char fixed-width tide value. Treat blanks and 999 sentinels as missing."""
    s = raw.strip()
    if not s:
        return -9999
    try:
        return int(s)
    except ValueError:
        return -9999


def parse_year_text(text: str, year: int) -> pd.DataFrame:
    """Parse a full JMA suisan annual file into an hourly DataFrame.

    Returns columns:
      datetime : tz-aware pandas Timestamp in Asia/Tokyo
      tide_cm  : int (predicted tide height in cm above station datum)
    Rows where the source had no value are dropped (treated as missing).
    """
    rows = []
    for raw_line in text.splitlines():
        # Strip only the trailing newline; preserve leading spaces (significant in fixed width).
        line = raw_line.rstrip("\r\n")
        if len(line) < 80:
            continue
        try:
            month = int(line[74:76].strip())
            day = int(line[76:78].strip())
        except ValueError:
            continue
        if not (1 <= month <= 12 and 1 <= day <= 31):
            continue
        try:
            date = _dt.date(year, month, day)
        except ValueError:
            continue
        for hour in range(24):
            start = hour * 3
            val = _parse_int_field(line[start:start + 3])
            if val == -9999:
                continue
            ts = _dt.datetime(date.year, date.month, date.day, hour, 0, 0, tzinfo=JST)
            rows.append((ts, val))
    if not rows:
        return pd.DataFrame(columns=["datetime", "tide_cm"])
    df = pd.DataFrame(rows, columns=["datetime", "tide_cm"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=False)
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def fetch_month(
    year: int,
    month: int,
    station_code: str = DEFAULT_PRIMARY_CODE,
    fallback_codes: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Hourly predicted tides for a single month at one JMA station."""
    if fallback_codes is None:
        fallback_codes = list(DEFAULT_FALLBACK_CODES)
    try:
        text = fetch_year(year, station_code, fallback_codes=fallback_codes)
        if text is None:
            return pd.DataFrame(columns=["datetime", "tide_cm"])
        df = parse_year_text(text, year)
        if df.empty:
            return df
        mask = (df["datetime"].dt.month == month) & (df["datetime"].dt.year == year)
        return df.loc[mask].reset_index(drop=True)
    except Exception:
        log.error("fetch_month(%d, %d, %s) failed\n%s",
                  year, month, station_code, traceback.format_exc())
        return pd.DataFrame(columns=["datetime", "tide_cm"])


def fetch_recent(
    days: int = 14,
    station_code: str = DEFAULT_PRIMARY_CODE,
    fallback_codes: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Hourly predicted tides for the window [today - days, today + days]. Month-rollover safe."""
    if fallback_codes is None:
        fallback_codes = list(DEFAULT_FALLBACK_CODES)
    try:
        today = _dt.datetime.now(JST).date()
        start = today - _dt.timedelta(days=days)
        end = today + _dt.timedelta(days=days)

        years = sorted({start.year, end.year, today.year})
        frames: List[pd.DataFrame] = []
        for y in years:
            text = fetch_year(y, station_code, fallback_codes=fallback_codes)
            if text is None:
                continue
            frames.append(parse_year_text(text, y))
        if not frames:
            return pd.DataFrame(columns=["datetime", "tide_cm"])
        df = pd.concat(frames, ignore_index=True)
        start_ts = pd.Timestamp(start, tz=JST)
        # end is exclusive at next day's 00:00 so we include the whole 'end' day.
        end_ts = pd.Timestamp(end + _dt.timedelta(days=1), tz=JST)
        df = df.loc[(df["datetime"] >= start_ts) & (df["datetime"] < end_ts)]
        return df.sort_values("datetime").reset_index(drop=True)
    except Exception:
        log.error("fetch_recent(days=%d, station=%s) failed\n%s",
                  days, station_code, traceback.format_exc())
        return pd.DataFrame(columns=["datetime", "tide_cm"])


# ---------------------------------------------------------------------------
# CLI driver
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    t0 = time.time()
    log.info("fetching recent 30 days for station=%s", DEFAULT_PRIMARY_CODE)
    df = fetch_recent(days=30, station_code=DEFAULT_PRIMARY_CODE)
    if df.empty:
        print("EMPTY DataFrame -- check logs/fetch_tide.log")
        sys.exit(1)
    print(f"rows={len(df)}  span={df['datetime'].min()} .. {df['datetime'].max()}")
    print(f"tide_cm: min={df['tide_cm'].min()}  max={df['tide_cm'].max()}  "
          f"mean={df['tide_cm'].mean():.1f}")
    print("head:")
    print(df.head(6).to_string(index=False))
    print("tail:")
    print(df.tail(6).to_string(index=False))
    log.info("done in %.2fs", time.time() - t0)
