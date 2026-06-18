"""OBI v1 - 沖家室 食い時合指数 (Okikamuro Biting Index) コア計算モジュール.

設計式:
    OBI(t) = [ w1*f_v + w2*f_dh + w3*U + w4*B_twilight + w5*M_moon + w6*S_season ]
             * P_temp * P_pressure

v1 暫定:
    - 潮流速 |v| は取れないため w1=0、dh/dt を主軸 (w2 を拡大) に代用
    - 湧昇 U は地形依存のため暫定 0.5 固定 (v2 で MSIL と等深線で実計算)
    - 進行波/定常波の位相補正は v2 で実装 (config.phase_shift_hours は読み捨て可)

出力スコア:
    - raw_score を日内最大で min-max 正規化 → score_01 (0..1)
    - stars = round(score_01 * 5) (0..5)
"""

from __future__ import annotations

import logging
import math
from datetime import date as _date_t, datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 単機能ファクター
# ---------------------------------------------------------------------------


def compute_dh_dt(tide_df: pd.DataFrame) -> pd.DataFrame:
    """毎時潮位データに中心差分で dh/dt (cm/h) を追加して返す."""
    if tide_df is None or len(tide_df) == 0:
        raise ValueError("tide_df is empty")

    df = tide_df.copy().reset_index(drop=True)

    if "tide_cm" not in df.columns:
        raise KeyError("tide_df must contain 'tide_cm' column")

    tide = df["tide_cm"].astype(float).to_numpy()
    n = len(tide)
    dh = np.zeros(n, dtype=float)

    if n == 1:
        dh[0] = 0.0
    else:
        # 端点: 前進/後退差分
        dh[0] = tide[1] - tide[0]
        dh[-1] = tide[-1] - tide[-2]
        # 中央: 中心差分 (Δt = 2h なので /2)
        if n >= 3:
            dh[1:-1] = (tide[2:] - tide[:-2]) / 2.0

    df["dh_dt"] = dh
    return df


def dh_factor(dh_dt: float, dh_ref: float = 80.0) -> float:
    """|dh/dt|/dh_ref を [0,1] にクリップ. dh_ref は大潮時典型値 (暫定)."""
    if dh_dt is None or (isinstance(dh_dt, float) and math.isnan(dh_dt)):
        return 0.0
    if dh_ref <= 0:
        return 0.0
    return float(min(abs(dh_dt) / dh_ref, 1.0))


def temp_factor(
    T: Optional[float],
    dT_dt: Optional[float],
    T_opt: float = 20.0,
    T_range: float = 8.0,
) -> float:
    """水温/気温の最適度. T が無ければ 1.0."""
    if T is None or (isinstance(T, float) and math.isnan(T)):
        return 1.0
    if T_range <= 0:
        return 1.0

    base = math.exp(-(((T - T_opt) / T_range) ** 2))

    if dT_dt is None or (isinstance(dT_dt, float) and math.isnan(dT_dt)):
        penalty = 0.0
    else:
        penalty = 0.5 * min(abs(dT_dt) / 1.5, 1.0)

    return float(base * (1.0 - penalty))


def pressure_factor(dP_dt: Optional[float]) -> float:
    """気圧変化率による補正. dP_dt が無ければ 1.0."""
    if dP_dt is None or (isinstance(dP_dt, float) and math.isnan(dP_dt)):
        return 1.0
    return float(min(1.0, 1.2 - 0.05 * abs(dP_dt)))


def season_factor(month: int, species: str, table: Dict[str, Any]) -> float:
    """config の季節係数テーブルから取得. 欠損は 0.5."""
    if not table:
        return 0.5
    try:
        species_table = table.get(species)
        if species_table is None:
            return 0.5
        # キーは int / str 両対応
        if isinstance(species_table, dict):
            if month in species_table:
                return float(species_table[month])
            if str(month) in species_table:
                return float(species_table[str(month)])
        elif isinstance(species_table, (list, tuple)) and len(species_table) >= 12:
            return float(species_table[month - 1])
    except Exception:  # noqa: BLE001
        logger.exception("season_factor lookup failed for species=%s month=%s", species, month)
        return 0.5
    return 0.5


def upwelling_factor(species: str) -> float:
    """湧昇インデックス. v1 は地形依存のため暫定 0.5 固定 (v2 で実装)."""
    # TODO(v2): MSIL と等深線勾配から魚種別に動的計算
    _ = species
    return 0.5


# ---------------------------------------------------------------------------
# 魚種別 摂餌時刻プロファイル (日周性 / 夜行性 差別化)
# ---------------------------------------------------------------------------


def species_diurnal_factor(
    t: datetime,
    species: str,
    astro: Dict[str, Any],
    twilight_window_h: float = 1.5,
) -> float:
    """魚種別の摂餌時刻プロファイルによる修飾係数 (中央値 1.0).

    既存の B_twilight (全魚種共通の薄明ボーナス) とは独立に作用させ、
    日中/夜間/朝夕マヅメに対する魚種ごとの強弱を raw_score に乗算する.

    係数テーブル (v1 ハードコード, v2 で config 化):
        - madai  : 強い日周性 (朝夕マヅメ依存)。マヅメ +0.30、他はベース
        - aji    : 夜浮く・昼沈む (JAXA電子タグ調査)。夜 +0.20, 昼 -0.10
        - tachiuo: 強い夜行性。夜 +0.40, 昼 -0.20
        - sawara : 昼行性・高速捕食。昼 +0.20, 夜 -0.20

    時間帯判定:
        - 朝マヅメ: sunrise の前後 twilight_window_h
        - 夕マヅメ: sunset  の前後 twilight_window_h
        - 夜間   : sunset + window 〜 翌 sunrise - window
        - 日中   : sunrise + window 〜 sunset - window
    """
    # 魚種ごとの (twilight_bonus, day_delta, night_delta)
    # 戻り値は 1.0 + delta を返す
    table = {
        "madai":   {"twilight": 0.30, "day":  0.00, "night":  0.00},
        "aji":     {"twilight": 0.00, "day": -0.10, "night":  0.20},
        "tachiuo": {"twilight": 0.00, "day": -0.20, "night":  0.40},
        "sawara":  {"twilight": 0.00, "day":  0.20, "night": -0.20},
    }
    profile = table.get(species)
    if profile is None:
        return 1.0

    sunrise = astro.get("sunrise")
    sunset = astro.get("sunset")

    def _to_dt(x: Any) -> Optional[datetime]:
        if x is None:
            return None
        if isinstance(x, str):
            try:
                x = datetime.fromisoformat(x)
            except ValueError:
                return None
        if not isinstance(x, datetime):
            return None
        # tz 揃え
        if t.tzinfo is not None and x.tzinfo is None:
            x = x.replace(tzinfo=t.tzinfo)
        if t.tzinfo is None and x.tzinfo is not None:
            x = x.replace(tzinfo=None)
        return x

    sr = _to_dt(sunrise)
    ss = _to_dt(sunset)

    # sunrise/sunset 不明なら無修飾
    if sr is None or ss is None:
        return 1.0

    # 朝夕マヅメ窓に入っているか
    in_twilight = False
    for anchor in (sr, ss):
        diff_h = abs((t - anchor).total_seconds()) / 3600.0
        if diff_h < twilight_window_h:
            in_twilight = True
            break

    if in_twilight:
        delta = profile["twilight"]
    else:
        # 日中 = sunrise + window < t < sunset - window
        # 夜間 = それ以外
        is_day = (sr + timedelta(hours=twilight_window_h)) <= t <= (ss - timedelta(hours=twilight_window_h))
        delta = profile["day"] if is_day else profile["night"]

    factor = 1.0 + delta
    # 下限ガード (極端に 0 以下にならないよう)
    return float(max(0.1, factor))


# ---------------------------------------------------------------------------
# 天文ファクター (薄明・月)
# ---------------------------------------------------------------------------


def _twilight_bonus(
    hour_dt: datetime,
    astro_daily: Dict[str, Any],
    window_hours: float = 1.5,
) -> float:
    """日の出/日の入り前後 window_hours の三角窓で 0..1."""
    bonus = 0.0
    for key in ("sunrise", "sunset"):
        t = astro_daily.get(key)
        if t is None:
            continue
        if isinstance(t, str):
            try:
                t = datetime.fromisoformat(t)
            except ValueError:
                continue
        if not isinstance(t, datetime):
            continue
        # tz 揃え
        if hour_dt.tzinfo is not None and t.tzinfo is None:
            t = t.replace(tzinfo=hour_dt.tzinfo)
        if hour_dt.tzinfo is None and t.tzinfo is not None:
            t = t.replace(tzinfo=None)
        diff_h = abs((hour_dt - t).total_seconds()) / 3600.0
        if diff_h < window_hours:
            bonus = max(bonus, 1.0 - diff_h / window_hours)
    return float(bonus)


def _moon_factor(astro_daily: Dict[str, Any]) -> float:
    """月齢 + 月の出没への近接で 0..1 を返す簡易版."""
    illum = astro_daily.get("moon_illumination")
    if illum is None:
        phase = astro_daily.get("moon_phase")  # 0..1
        if phase is None:
            return 0.5
        # 0=新月, 0.5=満月. 満ち欠けの強さ → |cos(2π*phase)| ではなく月明かり寄せ
        try:
            illum = 0.5 * (1.0 - math.cos(2.0 * math.pi * float(phase)))
        except Exception:  # noqa: BLE001
            return 0.5
    try:
        return float(max(0.0, min(1.0, illum)))
    except Exception:  # noqa: BLE001
        return 0.5


# ---------------------------------------------------------------------------
# メイン: compute_obi
# ---------------------------------------------------------------------------


def _default_weights() -> Dict[str, float]:
    return {
        "w1": 0.0,   # v1 では潮流速が取れないため計上しない
        "w2": 0.50,  # dh/dt 主軸
        "w3": 0.15,  # 湧昇
        "w4": 0.15,  # 薄明
        "w5": 0.10,  # 月
        "w6": 0.10,  # 季節
    }


def compute_obi(
    hourly_df: pd.DataFrame,
    astro_daily: Dict[str, Any],
    lat: float,
    lon: float,
    weights: Optional[Dict[str, float]] = None,
    season_table: Optional[Dict[str, Any]] = None,
    species_list: Optional[List[str]] = None,
    date: Optional[_date_t] = None,
    weather: Optional[Dict[str, Optional[float]]] = None,
) -> pd.DataFrame:
    """毎時 × 魚種の OBI スコア表を返す.

    Args:
        weather: AMeDAS から取った直近の気象スナップショット (オプション).
            キー: temp (°C), dT_dt (°C/h), dP_dt (hPa/h).
            指定された値は hourly_df の同名列が無い／全て NaN のとき
            「日内一定」として broadcast される。temp_factor / pressure_factor
            が None を 1.0 として扱う仕様と整合.
    """
    if hourly_df is None or len(hourly_df) == 0:
        raise ValueError("hourly_df is empty")

    weights = {**_default_weights(), **(weights or {})}
    species_list = species_list or ["madai"]
    season_table = season_table or {}
    weather = weather or {}

    df = hourly_df.copy().reset_index(drop=True)
    if "datetime" not in df.columns:
        raise KeyError("hourly_df must contain 'datetime' column")
    df["datetime"] = pd.to_datetime(df["datetime"])

    if "dh_dt" not in df.columns:
        df = compute_dh_dt(df)

    # weather スナップショットを「列が無い／全 NaN のとき」だけ broadcast.
    # hourly_df 側に毎時データが入っていればそちらを優先する.
    def _broadcast(col: str, val: Optional[float]) -> None:
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return
        if col not in df.columns or df[col].dropna().empty:
            df[col] = float(val)

    _broadcast("temp", weather.get("temp"))
    _broadcast("dT_dt", weather.get("dT_dt"))
    _broadcast("dP_dt", weather.get("dP_dt"))

    # オプション列 (最終フォールバック: 何も無ければ NaN)
    if "temp" not in df.columns:
        df["temp"] = np.nan
    if "dT_dt" not in df.columns:
        # 1時間差分で簡易計算
        df["dT_dt"] = df["temp"].astype(float).diff().fillna(0.0)
    if "dP_dt" not in df.columns:
        df["dP_dt"] = np.nan

    # 緯度経度はログ・将来の地形係数用 (v1 ではログのみ)
    logger.debug("compute_obi at lat=%.4f lon=%.4f date=%s", lat, lon, date)

    target_date = date or df["datetime"].iloc[0].date()

    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        try:
            dt: datetime = r["datetime"].to_pydatetime() if hasattr(r["datetime"], "to_pydatetime") else r["datetime"]
        except Exception:  # noqa: BLE001
            logger.exception("datetime conversion failed for row=%s", r)
            continue

        tide_cm = float(r.get("tide_cm")) if "tide_cm" in r and pd.notna(r.get("tide_cm")) else float("nan")
        dh_dt_v = float(r["dh_dt"])
        temp_v = r.get("temp")
        temp_v = float(temp_v) if pd.notna(temp_v) else None
        dT_v = r.get("dT_dt")
        dT_v = float(dT_v) if pd.notna(dT_v) else None
        dP_v = r.get("dP_dt")
        dP_v = float(dP_v) if pd.notna(dP_v) else None

        f_dh = dh_factor(dh_dt_v)
        B = _twilight_bonus(dt, astro_daily)
        M = _moon_factor(astro_daily)
        P_temp = temp_factor(temp_v, dT_v)
        P_press = pressure_factor(dP_v)

        for sp in species_list:
            U = upwelling_factor(sp)
            S = season_factor(dt.month, sp, season_table)
            D_sp = species_diurnal_factor(dt, sp, astro_daily)

            raw = (
                weights["w2"] * f_dh
                + weights["w3"] * U
                + weights["w4"] * B
                + weights["w5"] * M
                + weights["w6"] * S
            ) * P_temp * P_press * D_sp

            rows.append(
                {
                    "datetime": dt,
                    "species": sp,
                    "tide_cm": tide_cm,
                    "dh_dt": dh_dt_v,
                    "f_dh": f_dh,
                    "B_twilight": B,
                    "M_moon": M,
                    "U": U,
                    "S_season": S,
                    "P_temp": P_temp,
                    "P_pressure": P_press,
                    "D_species": D_sp,
                    "raw_score": float(raw),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # 日内 (魚種ごと) 最大で min-max 正規化
    out["date"] = pd.to_datetime(out["datetime"]).dt.date
    out["score_01"] = 0.0
    for (_, _sp), g in out.groupby(["date", "species"]):
        mx = g["raw_score"].max()
        if mx and mx > 0:
            out.loc[g.index, "score_01"] = g["raw_score"] / mx
        else:
            out.loc[g.index, "score_01"] = 0.0
    out["stars"] = out["score_01"].apply(lambda v: int(round(float(v) * 5)))

    # 引数 date が指定されていればその日に絞る
    if date is not None:
        out = out[out["date"] == target_date].copy()

    out = out.drop(columns=["date"]).reset_index(drop=True)
    return out[
        [
            "datetime",
            "species",
            "tide_cm",
            "dh_dt",
            "f_dh",
            "B_twilight",
            "M_moon",
            "U",
            "S_season",
            "P_temp",
            "P_pressure",
            "D_species",
            "raw_score",
            "score_01",
            "stars",
        ]
    ]


# ---------------------------------------------------------------------------
# ドライバ (動作確認用)
# ---------------------------------------------------------------------------


def _demo() -> pd.DataFrame:
    """仮データで OBI を計算してプレビュー."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    base = datetime(2026, 6, 19, 0, 0, 0)
    hours = [base + timedelta(hours=h) for h in range(24)]
    # 半日周潮 + 1日周潮の合成で擬似潮位
    tide = [
        150.0
        + 80.0 * math.sin(2 * math.pi * h / 12.42)
        + 20.0 * math.sin(2 * math.pi * h / 24.0)
        for h in range(24)
    ]
    temps = [22.0 + 2.0 * math.sin(2 * math.pi * (h - 6) / 24.0) for h in range(24)]
    press = [1013.0 + 0.5 * math.sin(2 * math.pi * h / 24.0) for h in range(24)]

    hourly = pd.DataFrame(
        {
            "datetime": hours,
            "tide_cm": tide,
            "temp": temps,
            "pressure": press,
            "dP_dt": [0.0] + [press[i] - press[i - 1] for i in range(1, 24)],
        }
    )

    astro_daily = {
        "sunrise": datetime(2026, 6, 19, 5, 10),
        "sunset": datetime(2026, 6, 19, 19, 25),
        "moon_illumination": 0.45,
    }

    season_table = {
        "madai":   {3: 0.6, 4: 0.8, 5: 1.0, 6: 0.9, 7: 0.6, 10: 0.7, 11: 0.6},
        "aji":     {6: 0.9, 7: 1.0, 8: 1.0, 9: 0.8},
        "tachiuo": {6: 0.7, 7: 0.9, 8: 1.0, 9: 1.0, 10: 0.8},
        "sawara":  {4: 0.8, 5: 1.0, 6: 0.9, 7: 0.7, 10: 0.8, 11: 0.9},
    }

    result = compute_obi(
        hourly_df=hourly,
        astro_daily=astro_daily,
        lat=33.9167,
        lon=132.25,
        weights=None,
        season_table=season_table,
        species_list=["madai", "aji", "tachiuo", "sawara"],
        date=_date_t(2026, 6, 19),
    )
    return result


if __name__ == "__main__":
    df = _demo()
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.width", 220)

    # 1) 魚種別 raw_score の代表時刻サマリ (差別化チェック)
    print("=" * 80)
    print("species 差別化チェック: 代表時刻ごとの raw_score / D_species")
    print("=" * 80)
    # 代表時刻: 03:00 (深夜), 05:00 (朝マヅメ近傍), 12:00 (日中), 19:00 (夕マヅメ近傍), 22:00 (夜間)
    pick_hours = [3, 5, 12, 19, 22]
    df["hour"] = pd.to_datetime(df["datetime"]).dt.hour
    summary = (
        df[df["hour"].isin(pick_hours)]
        .pivot_table(
            index="hour",
            columns="species",
            values=["raw_score", "D_species"],
            aggfunc="first",
        )
        .round(3)
    )
    print(summary.to_string())
    print()

    # 2) 魚種ごとに raw_score 合計と最大時刻 (ピーク) を出す
    print("=" * 80)
    print("species 別: raw_score 合計・最大値・ピーク時刻")
    print("=" * 80)
    for sp, g in df.groupby("species"):
        g = g.sort_values("datetime")
        peak_row = g.loc[g["raw_score"].idxmax()]
        print(
            f"  {sp:8s}  sum={g['raw_score'].sum():.3f}  "
            f"max={peak_row['raw_score']:.3f} @ {peak_row['datetime'].strftime('%H:%M')}  "
            f"stars_peak={int(peak_row['stars'])}"
        )
    print()

    # 3) 詳細表
    print("=" * 80)
    print("詳細 (全行)")
    print("=" * 80)
    print(df.drop(columns=["hour"]).to_string(index=False))
