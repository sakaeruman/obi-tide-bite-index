"""沖家室の天体計算モジュール（Skyfield使用、JST入出力）。"""
from __future__ import annotations

import logging
import math
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from pprint import pprint
from typing import Optional
from zoneinfo import ZoneInfo

from skyfield import almanac
from skyfield.api import Loader, wgs84

# ---------------------------------------------------------------------------
# 定数・設定
# ---------------------------------------------------------------------------
JST = ZoneInfo("Asia/Tokyo")
UTC = timezone.utc

# 沖家室島（デフォルト位置）
OKIKAMURO_LAT = 33.9167  # 33°55'N
OKIKAMURO_LON = 132.25   # 132°15'E

# Skyfieldデータの保存先（ベースディレクトリ直下 data/）
_BASE_DIR = Path(__file__).resolve().parent.parent
_DATA_DIR = _BASE_DIR / "data"
_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Loader: de421.bsp を data/ にキャッシュ（初回自動DL）
_loader = Loader(str(_DATA_DIR), verbose=False)

logger = logging.getLogger(__name__)

# 遅延ロード用キャッシュ
_eph = None
_ts = None


def _get_eph_ts():
    """ephemerisとtimescaleを遅延ロードしてキャッシュする。"""
    global _eph, _ts
    if _eph is None or _ts is None:
        try:
            _eph = _loader("de421.bsp")
            _ts = _loader.timescale()
        except Exception as e:
            logger.exception("Skyfield ephemeris/timescale ロード失敗: %s", e)
            raise
    return _eph, _ts


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------
def _to_jst(dt: datetime) -> datetime:
    """任意の datetime を JST(Asia/Tokyo) aware に変換する。"""
    if dt.tzinfo is None:
        # tz未指定はJSTとみなす（プロジェクトの入力前提）
        return dt.replace(tzinfo=JST)
    return dt.astimezone(JST)


def _to_utc(dt: datetime) -> datetime:
    """任意の datetime を UTC aware に変換する。"""
    if dt.tzinfo is None:
        # tz未指定はJSTとみなしてUTCへ
        return dt.replace(tzinfo=JST).astimezone(UTC)
    return dt.astimezone(UTC)


def _t_from_jst(dt: datetime):
    """JST(または任意tz) の datetime を Skyfield Time に変換する。"""
    _, ts = _get_eph_ts()
    dt_utc = _to_utc(dt)
    return ts.from_datetime(dt_utc)


def _skyfield_time_to_jst(t) -> datetime:
    """Skyfield Time を JST datetime に変換する。"""
    return t.utc_datetime().replace(tzinfo=UTC).astimezone(JST)


# ---------------------------------------------------------------------------
# 公開API
# ---------------------------------------------------------------------------
def compute_daily(
    d: date,
    lat: float = OKIKAMURO_LAT,
    lon: float = OKIKAMURO_LON,
) -> dict:
    """指定日(JST)・指定緯度経度の日出/日没・市民薄明・月出/月没・月相を返す。"""
    eph, ts = _get_eph_ts()
    observer = wgs84.latlon(lat, lon)

    # 当日 00:00 JST 〜 翌日 00:00 JST の範囲で探索
    jst_start = datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=JST)
    jst_end = jst_start + timedelta(days=1)
    t0 = ts.from_datetime(jst_start.astimezone(UTC))
    t1 = ts.from_datetime(jst_end.astimezone(UTC))

    result: dict = {
        "sunrise": None,
        "sunset": None,
        "civil_dawn": None,
        "civil_dusk": None,
        "moonrise": None,
        "moonset": None,
        "moon_phase_frac": None,
        "moon_illumination": None,
    }

    # --- 日出・日没 ---
    try:
        f_sun = almanac.sunrise_sunset(eph, observer)
        times, events = almanac.find_discrete(t0, t1, f_sun)
        for t, e in zip(times, events):
            jst_dt = _skyfield_time_to_jst(t)
            if e == 1 and result["sunrise"] is None:
                result["sunrise"] = jst_dt
            elif e == 0 and result["sunset"] is None:
                result["sunset"] = jst_dt
    except Exception as e:
        logger.exception("日出/日没の計算失敗: %s", e)

    # --- 市民薄明（civil twilight: -6°）---
    try:
        f_tw = almanac.dark_twilight_day(eph, observer)
        times, events = almanac.find_discrete(t0, t1, f_tw)
        # events: 0=Dark, 1=Astro, 2=Nautical, 3=Civil, 4=Day
        # civil_dawn: 2→3 への上昇遷移（朝、太陽が-6°を超える）
        # civil_dusk: 3→2 への下降遷移（夕、太陽が-6°を下回る）
        prev_e: Optional[int] = None
        prev_t = None
        # 直前状態を得るため少し前から走査
        t_pre = ts.from_datetime((jst_start - timedelta(hours=6)).astimezone(UTC))
        pre_times, pre_events = almanac.find_discrete(t_pre, t0, f_tw)
        if len(pre_events) > 0:
            prev_e = int(pre_events[-1])
        else:
            prev_e = int(f_tw(t0))

        for t, e in zip(times, events):
            e_int = int(e)
            jst_dt = _skyfield_time_to_jst(t)
            # 上昇: civil_dawn は「Civil(3) になる瞬間」
            if e_int == 3 and prev_e is not None and prev_e < 3 and result["civil_dawn"] is None:
                result["civil_dawn"] = jst_dt
            # 下降: civil_dusk は「Civil(3) から Nautical(2) に下がる瞬間」
            if e_int == 2 and prev_e is not None and prev_e >= 3 and result["civil_dusk"] is None:
                result["civil_dusk"] = jst_dt
            prev_e = e_int
            prev_t = t
    except Exception as e:
        logger.exception("市民薄明の計算失敗: %s", e)

    # --- 月出・月没 ---
    try:
        f_moon = almanac.risings_and_settings(eph, eph["Moon"], observer)
        times, events = almanac.find_discrete(t0, t1, f_moon)
        for t, e in zip(times, events):
            jst_dt = _skyfield_time_to_jst(t)
            if e == 1 and result["moonrise"] is None:
                result["moonrise"] = jst_dt
            elif e == 0 and result["moonset"] is None:
                result["moonset"] = jst_dt
    except Exception as e:
        logger.exception("月出/月没の計算失敗: %s", e)

    # --- 月相（new=0, full=0.5, next new=1.0）と月照度 ---
    try:
        # 当日正午JSTで評価
        noon_jst = datetime(d.year, d.month, d.day, 12, 0, 0, tzinfo=JST)
        t_noon = ts.from_datetime(noon_jst.astimezone(UTC))
        phase_deg = float(almanac.moon_phase(eph, t_noon).degrees)
        # phase_deg: 0=new, 90=first quarter, 180=full, 270=last quarter
        result["moon_phase_frac"] = (phase_deg % 360.0) / 360.0
        # 照度: (1 - cos(phase)) / 2
        result["moon_illumination"] = (1.0 - math.cos(math.radians(phase_deg))) / 2.0
    except Exception as e:
        logger.exception("月相/月照度の計算失敗: %s", e)

    return result


def moon_altitude(t: datetime, lat: float = OKIKAMURO_LAT, lon: float = OKIKAMURO_LON) -> float:
    """指定時刻(JST想定)・地点の月高度を度で返す（-90〜+90）。"""
    eph, ts = _get_eph_ts()
    observer = eph["Earth"] + wgs84.latlon(lat, lon)
    sf_t = _t_from_jst(t)
    astrometric = observer.at(sf_t).observe(eph["Moon"]).apparent()
    alt, _az, _dist = astrometric.altaz()
    return float(alt.degrees)


def twilight_factor(
    t: datetime,
    sunrise: datetime,
    sunset: datetime,
    window_min: int = 60,
) -> float:
    """日出/日没±window_minなら1.5、それ以外は1.0を返す（リサーチ仕様 B_twilight）。"""
    t_jst = _to_jst(t)
    sunrise_jst = _to_jst(sunrise) if sunrise is not None else None
    sunset_jst = _to_jst(sunset) if sunset is not None else None
    window = timedelta(minutes=window_min)
    if sunrise_jst is not None and abs(t_jst - sunrise_jst) <= window:
        return 1.5
    if sunset_jst is not None and abs(t_jst - sunset_jst) <= window:
        return 1.5
    return 1.0


def moon_factor(
    t: datetime,
    lat: float,
    lon: float,
    sunset: datetime,
    sunrise_next: datetime,
) -> float:
    """夜間かつ月高度>0なら 0.7+0.3*sin(alt_rad)*illum、それ以外1.0。"""
    t_jst = _to_jst(t)
    sunset_jst = _to_jst(sunset) if sunset is not None else None
    sunrise_next_jst = _to_jst(sunrise_next) if sunrise_next is not None else None

    # 夜間判定（sunset → 翌日の sunrise の間）
    is_night = False
    if sunset_jst is not None and sunrise_next_jst is not None:
        is_night = sunset_jst <= t_jst <= sunrise_next_jst
    if not is_night:
        return 1.0

    try:
        alt_deg = moon_altitude(t_jst, lat, lon)
    except Exception as e:
        logger.exception("月高度計算失敗 (%s): %s", t_jst, e)
        return 1.0

    if alt_deg <= 0:
        return 1.0

    # 月照度を取得（当日12時JST基準）
    eph, ts = _get_eph_ts()
    noon = datetime(t_jst.year, t_jst.month, t_jst.day, 12, 0, 0, tzinfo=JST)
    try:
        t_noon = ts.from_datetime(noon.astimezone(UTC))
        phase_deg = float(almanac.moon_phase(eph, t_noon).degrees)
        illum = (1.0 - math.cos(math.radians(phase_deg))) / 2.0
    except Exception as e:
        logger.exception("月照度計算失敗: %s", e)
        illum = 0.5

    return 0.7 + 0.3 * math.sin(math.radians(alt_deg)) * illum


# ---------------------------------------------------------------------------
# ドライバ
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    today = datetime.now(JST).date()
    out = compute_daily(today, OKIKAMURO_LAT, OKIKAMURO_LON)
    print(f"=== 沖家室 {today.isoformat()} の天体データ (JST) ===")
    pprint(out)
