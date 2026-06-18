"""JMA AMeDAS weather fetcher for OBI v1 (沖家室島近傍).

Source: 気象庁 AMeDAS (10-minute granularity, public JSON)
  - 地点別1時間ぶん: https://www.jma.go.jp/bosai/amedas/data/point/<station_id>/<yyyymmdd>_<HH>.json
  - 全地点最新時刻:   https://www.jma.go.jp/bosai/amedas/data/map/<yyyymmddhhmmss>.json
  - 地点メタ表:       https://www.jma.go.jp/bosai/amedas/const/amedastable.json

Station selection (実物のテーブルで確認済み):
  - 柳井 (Yanai)  station_id = "82056"  type=C   lat/lon ≈ 33.9117N, 132.1083E
  - 防府 (Hofu)   station_id = "82066"  type=C   lat/lon ≈ 34.0667N, 131.5583E
  Both are type "C": they report temp / humidity / wind / precip / sun.
  *** They do NOT report pressure. *** Only type "A" 気象官署 carry the
  `pressure` / `normalPressure` fields. So we keep `pressure_hpa` in the
  schema but leave it NaN for type-C stations; daily.py treats NaN/None as
  "use default 1.0" via obi.pressure_factor.

Note on "水温" in the original spec:
  AMeDAS does not measure sea surface temperature. Water temperature would
  come from a separate JMA dataset (海洋気象情報) which is out of scope for
  v1. This module returns air temperature (`temp_c`) only. daily.py / obi.py
  already use `temp` as a single input to temp_factor regardless of source.

Point JSON schema (per timestamp key):
  {
    "prefNumber": 82, "observationNumber": 56,
    "temp": [29.3, 0], "humidity": [53, 0],
    "windDirection": [5, 0], "wind": [3.6, 0],
    "pressure": [1010.1, 0], "normalPressure": [1010.0, 0],   # type A only
    ...
  }
  Each measurement is `[value, quality_flag]` (flag 0 = good).
  Top-level keys are JST timestamps formatted YYYYMMDDhhmmss.

Hour-bucket JSON file naming:
  Each `<yyyymmdd>_<HH>.json` actually covers a *3-hour* window starting at
  HH (so 18 ten-minute slots: HH:00 .. HH+2:50). Files are published only for
  HH in {00, 03, 06, 09, 12, 15, 18, 21} — that is, every 3 hours JST.
  Requesting any other HH returns 404. We step the cursor by 3h and the file
  for the current 3-hour window is published incrementally; older windows are
  always available within the same day.
"""
from __future__ import annotations

import datetime as _dt
import logging
import logging.handlers
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _THIS_FILE.parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
DATA_ROOT = PROJECT_ROOT / "data"
LOG_DIR.mkdir(parents=True, exist_ok=True)
DATA_ROOT.mkdir(parents=True, exist_ok=True)

JST = ZoneInfo("Asia/Tokyo")

POINT_URL = (
    "https://www.jma.go.jp/bosai/amedas/data/point/{station_id}/{yyyymmdd}_{hh:02d}.json"
)
TABLE_URL = "https://www.jma.go.jp/bosai/amedas/const/amedastable.json"

# 確定値（amedastable.json で実測確認: 2026-06)
STATION_YANAI = "82056"   # 柳井  (type C, no pressure)
STATION_HOFU = "82066"    # 防府  (type C, no pressure)
DEFAULT_STATION_ID = STATION_YANAI

REQUEST_TIMEOUT_SEC = 20
USER_AGENT = "OBI-v1/0.1 (+okikamuro fishing index; contact: sakaeru)"

# 一度に hours で要求できる最大値（ストレージ・APIマナー両面で）
MAX_HOURS = 72

# 出力スキーマ（カラム順）— 仕様で固定
SCHEMA_COLS = [
    "datetime",      # tz-aware Timestamp[Asia/Tokyo]
    "temp_c",        # float, AMeDAS気温 [°C]
    "pressure_hpa",  # float, type Aのみ。type Cは NaN
    "wind_dir",      # int(1..16) 風向（16方位、北=16, 静穏=0）
    "wind_speed",    # float, 風速 [m/s]
    "humidity",      # float, 相対湿度 [%]
]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def _build_logger() -> logging.Logger:
    logger = logging.getLogger("obi.fetch_weather")
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
        LOG_DIR / "fetch_weather.log",
        maxBytes=1_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    return logger


log = _build_logger()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _cache_dir(station_id: str) -> Path:
    d = DATA_ROOT / f"weather_{station_id}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_path_daily(station_id: str, date: _dt.date) -> Path:
    return _cache_dir(station_id) / f"{date.strftime('%Y%m%d')}.parquet"


def _extract_value(entry: dict, key: str) -> Optional[float]:
    """Return AMeDAS measurement value or None.

    Each measurement is `[value, quality_flag]`. Treat missing key, non-list,
    or any non-zero quality flag (= invalid / missing) as None.
    """
    v = entry.get(key)
    if not isinstance(v, list) or len(v) < 2:
        return None
    value, flag = v[0], v[1]
    if value is None:
        return None
    # JMA quality flag: 0 = normal. None or non-zero = invalid / missing.
    if flag is not None and flag != 0:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_timestamp_key(ts_key: str) -> Optional[_dt.datetime]:
    """JMA timestamp keys are JST in YYYYMMDDhhmmss."""
    if not (isinstance(ts_key, str) and len(ts_key) == 14 and ts_key.isdigit()):
        return None
    try:
        return _dt.datetime(
            int(ts_key[0:4]),
            int(ts_key[4:6]),
            int(ts_key[6:8]),
            int(ts_key[8:10]),
            int(ts_key[10:12]),
            int(ts_key[12:14]),
            tzinfo=JST,
        )
    except ValueError:
        return None


def _entry_to_row(ts: _dt.datetime, entry: dict) -> Dict[str, object]:
    """Convert one timestamped entry into a SCHEMA_COLS row."""
    wd = _extract_value(entry, "windDirection")
    return {
        "datetime": pd.Timestamp(ts),
        "temp_c": _extract_value(entry, "temp"),
        "pressure_hpa": _extract_value(entry, "pressure"),
        "wind_dir": int(wd) if wd is not None else None,
        "wind_speed": _extract_value(entry, "wind"),
        "humidity": _extract_value(entry, "humidity"),
    }


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in SCHEMA_COLS})


# ---------------------------------------------------------------------------
# Low-level fetch (one hour-bucket JSON)
# ---------------------------------------------------------------------------
def _fetch_hour_json(station_id: str, dt_hour: _dt.datetime) -> Optional[dict]:
    """Fetch the JSON file covering hour `dt_hour` (JST) at `station_id`.

    `dt_hour` must be tz-aware JST. Returns the parsed dict or None on failure.
    The current hour file may 404 briefly after publication; we log at INFO
    for that single case and at WARNING/ERROR otherwise.
    """
    if dt_hour.tzinfo is None:
        dt_hour = dt_hour.replace(tzinfo=JST)
    else:
        dt_hour = dt_hour.astimezone(JST)

    url = POINT_URL.format(
        station_id=station_id,
        yyyymmdd=dt_hour.strftime("%Y%m%d"),
        hh=dt_hour.hour,
    )
    headers = {"User-Agent": USER_AGENT}
    try:
        resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT_SEC)
    except requests.RequestException:
        log.error("network error fetching %s\n%s", url, traceback.format_exc())
        return None

    if resp.status_code == 404:
        # 3-hour bucket may not be published yet; treat softly when in future.
        now = _dt.datetime.now(JST)
        bucket_start_h = (now.hour // 3) * 3
        current_bucket = now.replace(hour=bucket_start_h, minute=0, second=0, microsecond=0)
        if dt_hour >= current_bucket:
            log.info("not yet published (404): %s", url)
        else:
            log.warning("404 for %s (no data this bucket)", url)
        return None
    if resp.status_code != 200:
        log.error("HTTP %s for %s", resp.status_code, url)
        return None

    try:
        return resp.json()
    except ValueError:
        log.error("malformed JSON at %s", url)
        return None


def _load_cache_for_date(station_id: str, date: _dt.date) -> Optional[pd.DataFrame]:
    p = _cache_path_daily(station_id, date)
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p)
    except Exception:  # noqa: BLE001  (parquet engines vary)
        log.warning("cache read failed for %s; ignoring", p)
        return None
    # Restore tz; parquet round-trips tz-naive in some engines.
    if "datetime" in df.columns and df["datetime"].dt.tz is None:
        df["datetime"] = df["datetime"].dt.tz_localize(JST)
    return df


def _write_cache_for_date(station_id: str, date: _dt.date, df: pd.DataFrame) -> None:
    if df.empty:
        return
    p = _cache_path_daily(station_id, date)
    try:
        df.to_parquet(p, index=False)
    except Exception:  # noqa: BLE001 — pyarrow / fastparquet may be missing
        log.warning(
            "parquet write failed (%s); install pyarrow or fastparquet to enable cache",
            p,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def fetch_recent_weather(
    station_id: str = DEFAULT_STATION_ID,
    hours: int = 24,
) -> pd.DataFrame:
    """Return AMeDAS observations for the last `hours` (10-min granularity).

    Columns (always present, in this order):
        datetime[JST], temp_c, pressure_hpa, wind_dir, wind_speed, humidity

    Behaviour:
        - tz-aware JST timestamps
        - Missing values are NaN (e.g. pressure_hpa for type-C stations)
        - On total API failure returns an EMPTY DataFrame with the schema —
          daily.py treats that as "use defaults" via obi.{temp,pressure}_factor
        - Per-day parquet cache under data/weather_<station>/<yyyymmdd>.parquet
          Today's slice is always refetched (data is still arriving); past
          days are reused from cache when complete (>= 144 rows = 24h × 6).
    """
    if hours <= 0:
        return _empty_frame()
    if hours > MAX_HOURS:
        log.warning("hours=%d capped to %d", hours, MAX_HOURS)
        hours = MAX_HOURS

    try:
        now = _dt.datetime.now(JST).replace(second=0, microsecond=0)
        # AMeDAS point JSON は3時間バケット。終端は now 以前で最大の {0,3,...,21}。
        end_bucket_h = (now.hour // 3) * 3
        end_bucket = now.replace(hour=end_bucket_h, minute=0)
        # 開始バケットは要求 hours を含むように切り下げ。3hバケット境界に合わせる。
        raw_start = now - _dt.timedelta(hours=hours)
        start_bucket_h = (raw_start.hour // 3) * 3
        start_bucket = raw_start.replace(hour=start_bucket_h, minute=0, second=0, microsecond=0)

        # Iterate every 3-hour bucket in [start_bucket, end_bucket]
        bucket_cursor = start_bucket
        per_day_rows: Dict[_dt.date, List[Dict[str, object]]] = {}
        per_day_cached: Dict[_dt.date, pd.DataFrame] = {}
        today = now.date()

        while bucket_cursor <= end_bucket:
            d = bucket_cursor.date()

            # Try day-cache reuse for non-today days (and skip if complete).
            if d != today and d not in per_day_cached:
                cached = _load_cache_for_date(station_id, d)
                if cached is not None and len(cached) >= 144:
                    per_day_cached[d] = cached
                    # Skip past this whole day in the loop.
                    bucket_cursor = _dt.datetime.combine(
                        d + _dt.timedelta(days=1), _dt.time(0, 0), tzinfo=JST
                    )
                    continue

            data = _fetch_hour_json(station_id, bucket_cursor)
            if data:
                day_bucket = per_day_rows.setdefault(d, [])
                for ts_key, entry in data.items():
                    ts = _parse_timestamp_key(ts_key)
                    if ts is None or not isinstance(entry, dict):
                        continue
                    day_bucket.append(_entry_to_row(ts, entry))
            bucket_cursor += _dt.timedelta(hours=3)

        # Assemble per-day dataframes + cache the *complete past* days.
        frames: List[pd.DataFrame] = list(per_day_cached.values())
        for d, rows in per_day_rows.items():
            if not rows:
                continue
            df_day = pd.DataFrame(rows, columns=SCHEMA_COLS)
            df_day = df_day.drop_duplicates(subset="datetime").sort_values("datetime")
            frames.append(df_day)
            # Cache only fully-formed past days (today is still incomplete).
            if d != today and len(df_day) >= 144:
                _write_cache_for_date(station_id, d, df_day)

        if not frames:
            return _empty_frame()

        df = pd.concat(frames, ignore_index=True)
        df = df.drop_duplicates(subset="datetime").sort_values("datetime")

        # Trim to the requested window (true wall-clock, not bucket boundary).
        cutoff = pd.Timestamp(now - _dt.timedelta(hours=hours))
        df = df.loc[df["datetime"] >= cutoff].reset_index(drop=True)
        return df[SCHEMA_COLS].reset_index(drop=True)

    except Exception:
        log.error(
            "fetch_recent_weather(station_id=%s, hours=%d) failed\n%s",
            station_id, hours, traceback.format_exc(),
        )
        return _empty_frame()


def compute_pressure_change_rate(
    df: pd.DataFrame,
    window_hours: int = 6,
) -> float:
    """Compute the average pressure trend (hPa/h) over the last `window_hours`.

    Linear regression slope of `pressure_hpa` vs. time, using only valid
    samples within the trailing window. Returns 0.0 when there is insufficient
    data (callers can treat 0.0 → pressure_factor = 1.2 cap, but obi.py
    already handles None gracefully — we never raise).
    """
    if df is None or df.empty or "pressure_hpa" not in df.columns:
        return 0.0
    if "datetime" not in df.columns:
        return 0.0
    if window_hours <= 0:
        return 0.0

    try:
        # Drop missing pressure rows (type-C stations -> all NaN -> empty).
        valid = df.dropna(subset=["pressure_hpa"]).copy()
        if valid.empty:
            return 0.0
        # Ensure tz-aware comparison.
        end_ts = valid["datetime"].max()
        if end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize(JST)
            valid["datetime"] = valid["datetime"].dt.tz_localize(JST)
        cutoff = end_ts - pd.Timedelta(hours=window_hours)
        window = valid.loc[valid["datetime"] >= cutoff]
        if len(window) < 2:
            return 0.0

        # Convert times to hours since the window start for the slope.
        t0 = window["datetime"].min()
        x = (window["datetime"] - t0).dt.total_seconds().to_numpy() / 3600.0
        y = window["pressure_hpa"].astype(float).to_numpy()
        if x.max() - x.min() < 1e-6:
            return 0.0
        # np.polyfit deg=1 returns [slope, intercept]
        import numpy as np  # local import keeps top-level imports minimal
        slope = float(np.polyfit(x, y, 1)[0])
        return slope
    except Exception:
        log.error(
            "compute_pressure_change_rate failed (window_hours=%d)\n%s",
            window_hours, traceback.format_exc(),
        )
        return 0.0


# ---------------------------------------------------------------------------
# CLI driver
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    t0 = time.time()
    station = DEFAULT_STATION_ID
    hours = 12
    log.info("fetching last %dh for station=%s (柳井)", hours, station)
    df = fetch_recent_weather(station_id=station, hours=hours)
    if df.empty:
        print("EMPTY DataFrame -- check logs/fetch_weather.log")
        sys.exit(1)

    print(
        f"rows={len(df)}  "
        f"span={df['datetime'].min()} .. {df['datetime'].max()}"
    )
    # 数値カラムのサマリ（欠損は除外）
    for col in ("temp_c", "pressure_hpa", "wind_speed", "humidity"):
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            print(f"{col:>14}: (no data)")
        else:
            print(f"{col:>14}: min={s.min():.1f}  max={s.max():.1f}  mean={s.mean():.1f}  n={len(s)}")

    dp = compute_pressure_change_rate(df, window_hours=6)
    print(f"pressure trend (last 6h): {dp:+.3f} hPa/h "
          f"({'no pressure series' if pd.to_numeric(df['pressure_hpa'], errors='coerce').dropna().empty else 'ok'})")

    print("head:")
    print(df.head(6).to_string(index=False))
    print("tail:")
    print(df.tail(6).to_string(index=False))
    log.info("done in %.2fs", time.time() - t0)
