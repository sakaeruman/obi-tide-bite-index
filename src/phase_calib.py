"""M2/S2 調和解析による潮位位相推定と OBI 位相補正の根拠出力.

目的
----
OBI v1 の流速 |v| は MSIL API 申請待ちで欠落しているため、`config.yaml` の
`tide_station.phase_shift_hours` (現在 1.5h, 暫定) は推測値である。
この `phase_calib.py` は、徳山(QA) の過去 60 日分の潮位を取得し、主要半日周潮
(M2 = 周期 12.421h, S2 = 周期 12.000h) を最小二乗で抽出して位相を推定する。
沖家室での流速ピークは観測がないため、

  * (a) 進行波領域なら 流速ピークは潮位ピーク/谷時刻に近い (位相差 0h 付近)
  * (b) 定常波領域なら 流速ピークは潮位の中間 (位相差 ±3.1h ≒ M2/4)
  * (c) 伊予灘西部の沖家室は遷移帯 → 典型 1.0 ~ 2.5h と推測

の知見と合わせて、`recommended_phase_shift_hours` を出力する (依然として暫定。
釣行ログが 10 件以上溜まったら ピーク照合で逆算する future-TODO を JSON に明記)。

成果物
------
`logs/phase_calib_<YYYYMMDD>.json`

注意
----
- 厳密な調和解析は U_tide 等の専用ライブラリを使うべきだが、ここでは scipy も
  使わず numpy だけで以下の線形最小二乗を解く::

      h(t) = A0 + sum_k [ Ck * cos(omega_k * t) + Sk * sin(omega_k * t) ]
      omega_k = 2*pi / T_k   (T_M2 = 12.421h, T_S2 = 12.000h)

  振幅 Ak = sqrt(Ck^2 + Sk^2), 位相 phi_k = atan2(Sk, Ck)。t の原点は窓の最初
  のタイムスタンプ。
- 沖家室の流速位相は観測不在のため "推定" であることを JSON にも README にも
  明記する。
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from fetch_tide import (
    DEFAULT_FALLBACK_CODES,
    DEFAULT_PRIMARY_CODE,
    fetch_recent,
)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _THIS_FILE.parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

JST = ZoneInfo("Asia/Tokyo")

# 主要分潮の周期 (時間)
PERIOD_M2_HOURS = 12.4206012   # 主太陰半日周潮
PERIOD_S2_HOURS = 12.0000000   # 主太陽半日周潮

# 沖家室 ~ 徳山 の概算距離 (config.yaml と整合)
DISTANCE_KM_TOKUYAMA_TO_SITE = 30.0

# 伊予灘西部・遷移帯における典型位相差 (h)
TRANSITION_PHASE_MIN_H = 1.0
TRANSITION_PHASE_MAX_H = 2.5
TRANSITION_PHASE_DEFAULT_H = 1.5


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def _build_logger() -> logging.Logger:
    logger = logging.getLogger("obi.phase_calib")
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
    return logger


log = _build_logger()


# ---------------------------------------------------------------------------
# Harmonic fit
# ---------------------------------------------------------------------------
def _fit_harmonics(
    t_hours: np.ndarray,
    h_cm: np.ndarray,
    periods_h: Tuple[float, ...],
) -> Tuple[float, Dict[float, Tuple[float, float]], float]:
    """h(t) = A0 + sum_k [C_k cos(ω_k t) + S_k sin(ω_k t)] を最小二乗で解く.

    Returns
    -------
    mean_cm : float
        定数項 A0 (平均潮位, cm)
    fit : dict[period_h -> (amplitude_cm, phase_rad)]
        各周期の振幅と位相. 位相は cos の基準, atan2(S, C) で計算.
        h(t) のピーク (=極大) は t = phase_rad / ω のとき.
    residual_std_cm : float
        観測 - フィット の標準偏差 (cm)
    """
    n = len(t_hours)
    if n < 4 * len(periods_h) + 2:
        raise ValueError("not enough samples for harmonic fit")

    cols = [np.ones(n)]
    for T in periods_h:
        omega = 2.0 * math.pi / T
        cols.append(np.cos(omega * t_hours))
        cols.append(np.sin(omega * t_hours))
    A = np.column_stack(cols)

    coeffs, *_ = np.linalg.lstsq(A, h_cm, rcond=None)
    mean_cm = float(coeffs[0])

    fit: Dict[float, Tuple[float, float]] = {}
    for i, T in enumerate(periods_h):
        C = float(coeffs[1 + 2 * i])
        S = float(coeffs[2 + 2 * i])
        amp = math.sqrt(C * C + S * S)
        phase_rad = math.atan2(S, C)
        fit[T] = (amp, phase_rad)

    pred = A @ coeffs
    resid = h_cm - pred
    residual_std = float(np.std(resid, ddof=1))
    return mean_cm, fit, residual_std


def _phase_rad_to_hours(phase_rad: float, period_h: float) -> float:
    """位相 (cos 基準, rad) → 「最初のピーク時刻」(窓開始からの時間, h, [0, T) ).

    h(t) = A cos(ω t - φ) のピークは t = φ / ω.
    fit が C cos + S sin 形式なので φ = atan2(S, C), ピーク時刻 = φ / ω.
    """
    omega = 2.0 * math.pi / period_h
    t_peak = phase_rad / omega
    # [0, period_h) に正規化
    t_peak = t_peak % period_h
    if t_peak < 0:
        t_peak += period_h
    return t_peak


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def estimate_phase(
    days: int = 60,
    station_code: str = DEFAULT_PRIMARY_CODE,
) -> Dict[str, object]:
    """徳山(QA) の潮位を取得して M2/S2 を抽出し, 位相補正の推奨値を出す."""
    df = fetch_recent(
        days=days,
        station_code=station_code,
        fallback_codes=list(DEFAULT_FALLBACK_CODES),
    )
    if df.empty:
        raise RuntimeError(
            f"fetch_recent({days}d, {station_code}) returned empty -- "
            "check logs/fetch_tide.log"
        )

    # 観測時刻 → 窓開始からの経過時間 (h)
    df = df.sort_values("datetime").reset_index(drop=True)
    t0 = df["datetime"].iloc[0]
    t_hours = (
        (df["datetime"] - t0).dt.total_seconds().to_numpy(dtype=float) / 3600.0
    )
    h_cm = df["tide_cm"].to_numpy(dtype=float)

    mean_cm, fit, resid_std = _fit_harmonics(
        t_hours, h_cm, periods_h=(PERIOD_M2_HOURS, PERIOD_S2_HOURS)
    )
    amp_m2, phase_m2 = fit[PERIOD_M2_HOURS]
    amp_s2, phase_s2 = fit[PERIOD_S2_HOURS]

    # 窓内における「最初の M2/S2 ピーク時刻」(参考表示用)
    peak_m2_hours_in_window = _phase_rad_to_hours(phase_m2, PERIOD_M2_HOURS)
    peak_s2_hours_in_window = _phase_rad_to_hours(phase_s2, PERIOD_S2_HOURS)

    # 沖家室 (=現地) の M2 流速ピーク位相は、観測がないため確定不能。
    # 進行波/定常波の遷移帯では typical 1.0 ~ 2.5h, 距離 30km からの伝播時間も
    # この帯に収まるため、暫定で TRANSITION_PHASE_DEFAULT_H を推奨する。
    recommended_phase_shift_h = TRANSITION_PHASE_DEFAULT_H

    notes = (
        "推奨 phase_shift_hours は暫定値です。沖家室の流速観測が無いため、"
        "(a)伊予灘西部の遷移帯 (進行波→定常波) における M2 位相差の典型値 "
        f"{TRANSITION_PHASE_MIN_H:.1f}〜{TRANSITION_PHASE_MAX_H:.1f}h と、"
        f"(b)徳山〜沖家室の距離 {DISTANCE_KM_TOKUYAMA_TO_SITE:.0f}km での "
        "伝播遅延から、中央値 1.5h を採用しています。"
        "(c)釣行ログが10件以上溜まったら、ログの『食ったタイミング』と "
        "OBI ピーク時刻を照合して位相補正を逆算してください "
        "(future-TODO: see future_todo)."
    )

    future_todo = {
        "trigger": "釣行ログ >= 10件 (data/catch_log.csv 等を作成して記録)",
        "method": (
            "各釣行で『食った時刻 (JST)』をログ → 同日の OBI 系列でピーク時刻を抽出 → "
            "(食った時刻 - 徳山潮位ピーク時刻) の中央値を取り、"
            "config.yaml の phase_shift_hours を更新する。"
            "OBI ピークと食ったタイミングのクロス相関 (lag, h) で逆算してもよい。"
        ),
        "expected_outcome": (
            "暫定 1.5h が 0.5 〜 3.0h の範囲で実測ベースに置換される見込み。"
        ),
    }

    result: Dict[str, object] = {
        "generated_at": _dt.datetime.now(JST).isoformat(timespec="seconds"),
        "station": station_code,
        "period_days": int(days),
        "sample_count": int(len(df)),
        "window": {
            "start": df["datetime"].iloc[0].isoformat(),
            "end": df["datetime"].iloc[-1].isoformat(),
        },
        "mean_tide_cm": round(mean_cm, 2),
        "M2": {
            "period_hours": PERIOD_M2_HOURS,
            "amplitude_cm": round(amp_m2, 2),
            "phase_rad": round(phase_m2, 4),
            "peak_hours_in_window": round(peak_m2_hours_in_window, 3),
        },
        "S2": {
            "period_hours": PERIOD_S2_HOURS,
            "amplitude_cm": round(amp_s2, 2),
            "phase_rad": round(phase_s2, 4),
            "peak_hours_in_window": round(peak_s2_hours_in_window, 3),
        },
        "fit_residual_std_cm": round(resid_std, 2),
        "site": {
            "name": "沖家室島",
            "distance_km_from_tokuyama": DISTANCE_KM_TOKUYAMA_TO_SITE,
        },
        "phase_band_hours": {
            "min": TRANSITION_PHASE_MIN_H,
            "max": TRANSITION_PHASE_MAX_H,
            "rationale": (
                "進行波/定常波の遷移帯における M2 位相差は典型 1.0〜2.5h、"
                f"徳山と沖家室の距離 {DISTANCE_KM_TOKUYAMA_TO_SITE:.0f}km から推定 1.5h(暫定)"
            ),
        },
        "recommended_phase_shift_hours": recommended_phase_shift_h,
        "is_provisional": True,
        "notes": notes,
        "future_todo": future_todo,
    }
    return result


def save_result(result: Dict[str, object], log_dir: Optional[Path] = None) -> Path:
    """結果を logs/phase_calib_<YYYYMMDD>.json に保存."""
    log_dir = log_dir or LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    today = _dt.datetime.now(JST).strftime("%Y%m%d")
    out_path = log_dir / f"phase_calib_{today}.json"
    out_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out_path


# ---------------------------------------------------------------------------
# CLI driver
# ---------------------------------------------------------------------------
def _print_summary(result: Dict[str, object], out_path: Path) -> None:
    m2 = result["M2"]
    s2 = result["S2"]
    win = result["window"]
    print("=" * 64)
    print(" OBI phase calibration (M2/S2 harmonic fit)")
    print("=" * 64)
    print(f"  station             : {result['station']}")
    print(f"  window              : {win['start']}  ..  {win['end']}")
    print(f"  samples             : {result['sample_count']}  "
          f"(period_days={result['period_days']})")
    print(f"  mean_tide_cm        : {result['mean_tide_cm']}")
    print(f"  M2  amplitude_cm    : {m2['amplitude_cm']}  "
          f"(period {m2['period_hours']:.3f}h)")
    print(f"  M2  phase_rad       : {m2['phase_rad']}  "
          f"(peak {m2['peak_hours_in_window']}h from window start)")
    print(f"  S2  amplitude_cm    : {s2['amplitude_cm']}  "
          f"(period {s2['period_hours']:.3f}h)")
    print(f"  S2  phase_rad       : {s2['phase_rad']}  "
          f"(peak {s2['peak_hours_in_window']}h from window start)")
    print(f"  fit residual std_cm : {result['fit_residual_std_cm']}")
    print("-" * 64)
    print(f"  推奨 phase_shift_hours : {result['recommended_phase_shift_hours']}  "
          f"(暫定: {result['is_provisional']})")
    print(f"  根拠帯                 : "
          f"{result['phase_band_hours']['min']}h "
          f"~ {result['phase_band_hours']['max']}h")
    print(f"  根拠                   : {result['phase_band_hours']['rationale']}")
    print("-" * 64)
    print("future_todo:")
    ft = result["future_todo"]
    print(f"  trigger          : {ft['trigger']}")
    print(f"  method           : {ft['method']}")
    print(f"  expected_outcome : {ft['expected_outcome']}")
    print("-" * 64)
    print(f"  saved -> {out_path}")
    print("=" * 64)


if __name__ == "__main__":
    t0 = time.time()
    try:
        result = estimate_phase(days=60, station_code=DEFAULT_PRIMARY_CODE)
    except Exception:
        log.error("phase calibration failed\n%s", traceback.format_exc())
        sys.exit(1)
    out_path = save_result(result)
    _print_summary(result, out_path)
    log.info("done in %.2fs", time.time() - t0)
