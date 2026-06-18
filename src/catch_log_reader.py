"""釣行ログ (catch_log/*.csv) を OBI レポートと突き合わせるモジュール.

主な関数:
    - load_log(csv_path)          : テンプレ CSV を pandas で読む
    - compare_with_obi(log_df, obi_dir="out")
                                   : 各釣行レコードに対し、その時間帯の
                                     OBI スコア（★0-5）と CPUE(count/h) を並べる
    - spearman_correlation(comparison_df, species)
                                   : OBI スコアと CPUE の順位相関を計算
                                     ベンチマーク: タイドグラフBI 第三者検証 r=0.46

ドライバ(__main__):
    sample.csv を読んで比較レポートを print する.

注意:
    OBI v1 の md レポート (out/obi_YYYYMMDD.md) を回帰するのではなく、
    再パースして比較する設計。obi.py 側の内部スキーマに依存しないので、
    将来 OBI を v2 に差し替えても CSV ↔ md 比較は壊れない。
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

# CSV の英語 species 列 → OBI md の日本語列名
SPECIES_LABEL: Dict[str, str] = {
    "madai": "マダイ",
    "aji": "アジ",
    "saba": "サバ",
    "sawara": "サワラ",
    "mebaru": "メバル",
    "yazu": "ヤズ",
    "tachi": "タチウオ",
    "tachiuo": "タチウオ",
}

# タイドグラフBI 第三者検証のベンチマーク値
BENCHMARK_R: float = 0.46

# 星 → 0..5 整数
_STAR_FULL = "★"
_STAR_EMPTY = "☆"


# ---------------------------------------------------------------------------
# CSV ロード
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS: List[str] = [
    "date",
    "time_start",
    "time_end",
    "spot_name",
    "spot_lat",
    "spot_lon",
    "depth_m",
    "tide_phase",
    "species",
    "count",
    "total_kg",
    "max_size_cm",
    "rig",
    "notes",
]


def load_log(csv_path: str) -> pd.DataFrame:
    """釣行ログ CSV を読み、型を整えた DataFrame を返す.

    - date は date 型
    - time_start / time_end は time 型
    - count は int、total_kg / max_size_cm は float (NA 許容)
    - species は小文字に正規化
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"catch log not found: {csv_path}")

    df = pd.read_csv(path, dtype=str).fillna("")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"catch log missing columns: {missing}")

    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d").dt.date
    df["time_start"] = df["time_start"].apply(_parse_time)
    df["time_end"] = df["time_end"].apply(_parse_time)
    df["species"] = df["species"].str.strip().str.lower()
    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0).astype(int)
    df["total_kg"] = pd.to_numeric(df["total_kg"], errors="coerce")
    df["max_size_cm"] = pd.to_numeric(df["max_size_cm"], errors="coerce")
    df["spot_lat"] = pd.to_numeric(df["spot_lat"], errors="coerce")
    df["spot_lon"] = pd.to_numeric(df["spot_lon"], errors="coerce")
    df["depth_m"] = pd.to_numeric(df["depth_m"], errors="coerce")

    return df


def _parse_time(value: str) -> Optional[time]:
    if not value:
        return None
    try:
        return datetime.strptime(value.strip(), "%H:%M").time()
    except ValueError:
        logger.warning("invalid time literal: %s", value)
        return None


# ---------------------------------------------------------------------------
# OBI md レポートのパース
# ---------------------------------------------------------------------------
def _md_path_for(target: date, obi_dir: Path) -> Path:
    return obi_dir / f"obi_{target.strftime('%Y%m%d')}.md"


def _stars_to_int(cell: str) -> Optional[int]:
    if not isinstance(cell, str):
        return None
    s = cell.strip()
    if not s:
        return None
    full = s.count(_STAR_FULL)
    if full == 0 and _STAR_EMPTY not in s:
        # 数字フォールバック（将来 OBI 出力が変わった時用）
        try:
            return int(s)
        except ValueError:
            return None
    return full


def _parse_obi_md(md_path: Path) -> Optional[pd.DataFrame]:
    """OBI v1 の md レポートから時刻×魚種の星 DF を返す.

    返り値の列: hour (0-23), 列名は魚種日本語 (マダイ/アジ/タチウオ/サワラ/…)
    各セルは int 0..5.
    """
    if not md_path.exists():
        logger.warning("OBI md not found: %s", md_path)
        return None

    text = md_path.read_text(encoding="utf-8")
    # 「| 時刻 | 潮位cm | ... |」 から始まり、次の空行までを切り出す
    m = re.search(r"^\|\s*時刻\s*\|.+?\n((?:\|.+\n)+)", text, flags=re.MULTILINE)
    if not m:
        logger.warning("hourly table not found in %s", md_path)
        return None

    header_line = re.search(r"^\|\s*時刻\s*\|.+", text, flags=re.MULTILINE).group(0)
    headers = [h.strip() for h in header_line.strip().strip("|").split("|")]

    rows: List[Dict[str, object]] = []
    for line in m.group(1).splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != len(headers):
            continue
        # 区切り行 (---|---|...) を除外
        if all(set(c) <= set("-: ") for c in cells):
            continue
        rec: Dict[str, object] = {}
        for h, c in zip(headers, cells):
            rec[h] = c
        rows.append(rec)

    if not rows:
        return None

    df = pd.DataFrame(rows)
    # 時刻 "HH:MM" → hour int
    df["hour"] = df["時刻"].apply(
        lambda v: int(v.split(":")[0]) if isinstance(v, str) and ":" in v else None
    )
    df = df.dropna(subset=["hour"]).copy()
    df["hour"] = df["hour"].astype(int)

    species_cols = [
        c for c in df.columns if c not in {"時刻", "潮位cm", "dh/dt", "hour"}
    ]
    for col in species_cols:
        df[col] = df[col].apply(_stars_to_int)

    return df[["hour", *species_cols]]


# ---------------------------------------------------------------------------
# 釣行 × OBI 比較
# ---------------------------------------------------------------------------
def _resolve_obi_dir(obi_dir: str) -> Path:
    p = Path(obi_dir)
    if p.is_absolute() and p.exists():
        return p
    # スクリプト位置基準で 04_OBI/ を解決
    project_root = Path(__file__).resolve().parent.parent
    candidate = project_root / obi_dir
    return candidate


def _hours_overlap(start: time, end: time) -> List[int]:
    """time_start 〜 time_end をカバーする hour リスト.

    終了が翌日に跨る場合は wrap して 0..23 を返す.
    """
    if start is None or end is None:
        return []
    s = datetime.combine(date.today(), start)
    e = datetime.combine(date.today(), end)
    if e <= s:
        e += timedelta(days=1)
    hours: List[int] = []
    cur = s.replace(minute=0, second=0, microsecond=0)
    while cur < e:
        hours.append(cur.hour)
        cur += timedelta(hours=1)
    # uniq 保持順
    seen = set()
    uniq: List[int] = []
    for h in hours:
        if h not in seen:
            seen.add(h)
            uniq.append(h)
    return uniq


def _duration_hours(start: time, end: time) -> float:
    if start is None or end is None:
        return 0.0
    s = datetime.combine(date.today(), start)
    e = datetime.combine(date.today(), end)
    if e <= s:
        e += timedelta(days=1)
    return (e - s).total_seconds() / 3600.0


def compare_with_obi(
    log_df: pd.DataFrame, obi_dir: str = "out"
) -> pd.DataFrame:
    """各釣行レコードについて、その時間帯の OBI スコアと CPUE を並べた DF を返す.

    返り値の列:
        date, time_start, time_end, spot_name, species, species_label,
        count, hours, cpue,            # CPUE = count / hours
        obi_score_mean, obi_score_max  # 当該時間帯の hour 平均/最大 (★0-5)
        obi_md                          # 参照した md パス（無ければ "")
    """
    obi_root = _resolve_obi_dir(obi_dir)

    md_cache: Dict[date, Optional[pd.DataFrame]] = {}
    out_rows: List[Dict[str, object]] = []

    for _, row in log_df.iterrows():
        d: date = row["date"]
        if d not in md_cache:
            md_cache[d] = _parse_obi_md(_md_path_for(d, obi_root))
        obi_df = md_cache[d]

        sp_en = str(row["species"]).lower()
        sp_jp = SPECIES_LABEL.get(sp_en, sp_en)
        hours = _hours_overlap(row["time_start"], row["time_end"])
        dur = _duration_hours(row["time_start"], row["time_end"])
        cpue = (row["count"] / dur) if dur > 0 else float("nan")

        obi_mean: Optional[float] = None
        obi_max: Optional[int] = None
        if obi_df is not None and sp_jp in obi_df.columns and hours:
            sub = obi_df[obi_df["hour"].isin(hours)][sp_jp]
            sub = sub.dropna()
            if not sub.empty:
                obi_mean = float(sub.mean())
                obi_max = int(sub.max())

        out_rows.append(
            {
                "date": d,
                "time_start": row["time_start"],
                "time_end": row["time_end"],
                "spot_name": row["spot_name"],
                "species": sp_en,
                "species_label": sp_jp,
                "count": int(row["count"]),
                "hours": round(dur, 2),
                "cpue": round(cpue, 3) if pd.notna(cpue) else None,
                "obi_score_mean": (
                    round(obi_mean, 2) if obi_mean is not None else None
                ),
                "obi_score_max": obi_max,
                "obi_md": str(_md_path_for(d, obi_root)),
            }
        )

    return pd.DataFrame(out_rows)


# ---------------------------------------------------------------------------
# 順位相関
# ---------------------------------------------------------------------------
def spearman_correlation(
    comparison_df: pd.DataFrame, species: str
) -> dict:
    """OBI スコア（hour 平均）と CPUE の Spearman 順位相関を計算.

    返り値:
        {
            "species": str,
            "n": サンプル数,
            "r": 相関係数 (float, NaN なら None),
            "p_value": p値,
            "benchmark_r": 0.46,
            "exceeds_benchmark": bool,
            "verdict": "実用" / "未達" / "サンプル不足",
        }
    """
    sp = species.lower()
    df = comparison_df[comparison_df["species"] == sp].dropna(
        subset=["obi_score_mean", "cpue"]
    )
    n = len(df)

    if n < 3:
        return {
            "species": sp,
            "n": n,
            "r": None,
            "p_value": None,
            "benchmark_r": BENCHMARK_R,
            "exceeds_benchmark": False,
            "verdict": "サンプル不足",
        }

    res = spearmanr(df["obi_score_mean"], df["cpue"])
    # SciPy のバージョン差を吸収
    r = float(getattr(res, "correlation", res[0]))
    p = float(getattr(res, "pvalue", res[1]))

    if pd.isna(r):
        return {
            "species": sp,
            "n": n,
            "r": None,
            "p_value": None,
            "benchmark_r": BENCHMARK_R,
            "exceeds_benchmark": False,
            "verdict": "サンプル不足",
        }

    exceeds = r >= BENCHMARK_R
    return {
        "species": sp,
        "n": n,
        "r": round(r, 3),
        "p_value": round(p, 4),
        "benchmark_r": BENCHMARK_R,
        "exceeds_benchmark": bool(exceeds),
        "verdict": "実用" if exceeds else "未達",
    }


# ---------------------------------------------------------------------------
# ドライバ
# ---------------------------------------------------------------------------
def _print_report(csv_path: Path, obi_dir: str) -> None:
    print(f"== catch log: {csv_path}")
    print(f"== obi dir : {_resolve_obi_dir(obi_dir)}")

    log = load_log(str(csv_path))
    print(f"\n[1] ログ件数: {len(log)}")
    print(log.to_string(index=False))

    cmp_df = compare_with_obi(log, obi_dir=obi_dir)
    print("\n[2] OBI 突き合わせ:")
    print(cmp_df.to_string(index=False))

    print("\n[3] Spearman 順位相関 (OBI★平均 vs CPUE)")
    print(f"    ベンチマーク: タイドグラフBI 第三者検証 r={BENCHMARK_R}")
    for sp in sorted(cmp_df["species"].unique()):
        result = spearman_correlation(cmp_df, sp)
        mark = "OK" if result["exceeds_benchmark"] else "..."
        print(
            f"  [{mark}] {sp:<8s} n={result['n']:<3d} "
            f"r={result['r']} p={result['p_value']} "
            f"→ {result['verdict']}"
        )


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    sample = project_root / "catch_log" / "sample.csv"
    _print_report(sample, obi_dir="out")
