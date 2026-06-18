"""OBI v1 メインオーケストレーター.

処理フロー:
    1. config.yaml 読み込み
    2. fetch_tide.fetch_recent でターゲット日±2日の予測潮位を取得
    3. obi.compute_dh_dt で dh/dt 付与
    4. astro.compute_daily で日出/日没/月相など取得
    5. obi.compute_obi で魚種×時刻のスコア計算
    6. render.render_markdown / render.render_png で out/ に書き出し

CLI:
    python -m src.daily                    # 明日 (JST) の OBI
    python -m src.daily --date 2026-06-19  # 指定日
    python -m src.daily --species madai,aji
"""
from __future__ import annotations

import argparse
import logging
import logging.handlers
import sys
import time
import traceback
from datetime import date as date_t, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# パッケージ内 import
# ---------------------------------------------------------------------------
# `python -m src.daily` でも `python src/daily.py` でも動くよう両対応.
try:
    from . import astro, fetch_tide, obi, render
    from . import fetch_weather, discord_notify
except ImportError:  # pragma: no cover - スクリプト直叩き用フォールバック
    _HERE = Path(__file__).resolve().parent
    sys.path.insert(0, str(_HERE.parent))
    from src import astro, fetch_tide, obi, render  # type: ignore
    from src import fetch_weather, discord_notify  # type: ignore

# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------
JST = ZoneInfo("Asia/Tokyo")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# 魚種コード -> 日本語表示名 (render.py の DEFAULT_FISH_ORDER と整合)
SPECIES_LABEL: Dict[str, str] = {
    "madai": "マダイ",
    "aji": "アジ",
    "tachiuo": "タチウオ",
    "sawara": "サワラ",
}

# fetch_tide が実際に動くステーションコードのフォールバック.
# config.yaml の station_code が古い (TY=富山) でも QA=徳山 で拾える.
TIDE_CODE_FALLBACK: Tuple[str, ...] = ("QA", "TY", "TKY")


# ---------------------------------------------------------------------------
# ロギング
# ---------------------------------------------------------------------------
def _build_logger(target_date: date_t) -> logging.Logger:
    """logs/daily_YYYYMMDD.log + コンソールの両方に出力するロガーを返す."""
    logger = logging.getLogger("obi.daily")
    # 既存ハンドラを掃除して冪等に
    for h in list(logger.handlers):
        logger.removeHandler(h)
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log_path = LOG_DIR / f"daily_{target_date.strftime('%Y%m%d')}.log"
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# 設定ロード
# ---------------------------------------------------------------------------
def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """dict を再帰マージ. override 側を優先."""
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Path = CONFIG_PATH) -> Dict[str, Any]:
    """config.yaml を読んで dict で返す.

    同じディレクトリに config.local.yaml があれば後勝ちでマージする.
    秘密情報 (Discord webhook URL 等) を公開リポと分離する仕組み.
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    local_path = path.parent / "config.local.yaml"
    if local_path.exists():
        try:
            with open(local_path, "r", encoding="utf-8") as f:
                local_cfg = yaml.safe_load(f) or {}
            cfg = _deep_merge(cfg, local_cfg)
        except Exception:
            sys.stderr.write(f"config.local.yaml の読み込みに失敗 (無視): {local_path}\n")
            traceback.print_exc(file=sys.stderr)
    return cfg


# ---------------------------------------------------------------------------
# 引数パース
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """argparse で --date / --species をパースする."""
    p = argparse.ArgumentParser(
        prog="obi.daily",
        description="OBI v1 日次計算: 翌日 (JST) の食い時合指数を出力",
    )
    p.add_argument(
        "--date",
        type=str,
        default=None,
        help="対象日 YYYY-MM-DD (省略時は明日 JST)",
    )
    p.add_argument(
        "--species",
        type=str,
        default=None,
        help="対象魚種カンマ区切り 例: madai,aji (省略時は config の target_species)",
    )
    p.add_argument(
        "--no-open",
        action="store_true",
        help="完了後にブラウザでHTMLを自動オープンしない（cron用）",
    )
    return p.parse_args(argv)


def _resolve_date(arg_date: Optional[str]) -> date_t:
    """--date 引数を date に変換. 省略時は明日 JST."""
    if arg_date:
        return datetime.strptime(arg_date, "%Y-%m-%d").date()
    return (datetime.now(JST) + timedelta(days=1)).date()


def _resolve_species(arg_species: Optional[str], cfg: Dict[str, Any]) -> List[str]:
    """--species 引数または config から魚種リストを決定."""
    if arg_species:
        return [s.strip() for s in arg_species.split(",") if s.strip()]
    fallback = cfg.get("target_species") or list(SPECIES_LABEL.keys())
    return list(fallback)


# ---------------------------------------------------------------------------
# パイプライン各ステップ
# ---------------------------------------------------------------------------
def _step_fetch_tide(
    cfg: Dict[str, Any],
    target_date: date_t,
    logger: logging.Logger,
) -> pd.DataFrame:
    """ターゲット日±2日の予測潮位を取得して JST 列で返す."""
    station_code = (cfg.get("tide_station") or {}).get("code") or "QA"
    fallback = [c for c in TIDE_CODE_FALLBACK if c != station_code]

    # fetch_recent は「今日」基準に days 幅で返すので、今日との差を吸収して広めに取る.
    today = datetime.now(JST).date()
    delta = abs((target_date - today).days)
    days = max(delta + 2, 3)

    logger.info(
        "潮位データ取得: station=%s fallback=%s days=±%d (today=%s, target=%s)",
        station_code, fallback, days, today, target_date,
    )
    df = fetch_tide.fetch_recent(
        days=days,
        station_code=station_code,
        fallback_codes=fallback,
    )
    if df.empty:
        raise RuntimeError("潮位データの取得に失敗 (空 DataFrame)")
    logger.info(
        "潮位データ取得 OK: %d 行 span=%s..%s",
        len(df), df["datetime"].min(), df["datetime"].max(),
    )
    return df


def _step_dh_dt(tide_df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """dh/dt を中心差分で付与."""
    df = obi.compute_dh_dt(tide_df)
    logger.info(
        "dh/dt 計算 OK: |dh/dt| min=%.2f max=%.2f mean=%.2f",
        df["dh_dt"].abs().min(),
        df["dh_dt"].abs().max(),
        df["dh_dt"].abs().mean(),
    )
    return df


def _step_fetch_weather(
    cfg: Dict[str, Any],
    logger: logging.Logger,
) -> Dict[str, Optional[float]]:
    """AMeDAS から直近気象を取得し {temp, dT_dt, dP_dt} を返す.

    失敗時は全 None を返す (daily.py は続行する).
    柳井 (82056) は type C のため pressure は常に NaN -> dP_dt=None になる.
    """
    snapshot: Dict[str, Optional[float]] = {"temp": None, "dT_dt": None, "dP_dt": None}
    try:
        amedas_cfg = cfg.get("amedas") or {}
        station_id = str(amedas_cfg.get("station_id") or fetch_weather.DEFAULT_STATION_ID)
        station_name = amedas_cfg.get("name") or "柳井"
        logger.info("アメダス取得: station_id=%s name=%s", station_id, station_name)

        df = fetch_weather.fetch_recent_weather(station_id=station_id, hours=12)
        if df is None or df.empty:
            logger.warning("アメダス取得が空 (続行、補正は 1.0)")
            return snapshot

        # 最新の有効気温
        temp_series = pd.to_numeric(df["temp_c"], errors="coerce").dropna()
        if not temp_series.empty:
            snapshot["temp"] = float(temp_series.iloc[-1])

            # 直近 1h 変化率: 1時間前の値と差分 (1時間=6サンプル想定だが安全のため時間ベース)
            try:
                tail = df.dropna(subset=["temp_c"]).copy()
                tail["datetime"] = pd.to_datetime(tail["datetime"])
                end_t = tail["datetime"].max()
                ref_cut = end_t - pd.Timedelta(hours=1)
                prev = tail.loc[tail["datetime"] <= ref_cut]
                if not prev.empty:
                    t_prev = float(prev["temp_c"].iloc[-1])
                    dt_h = (end_t - prev["datetime"].iloc[-1]).total_seconds() / 3600.0
                    if dt_h > 0:
                        snapshot["dT_dt"] = (snapshot["temp"] - t_prev) / dt_h
            except Exception:
                logger.debug("dT_dt 計算失敗 (無視)\n%s", traceback.format_exc())

        # 気圧トレンド (type-C 局では常に 0.0)
        dp = fetch_weather.compute_pressure_change_rate(df, window_hours=6)
        # 有効な気圧サンプルが 1 行も無ければ dP_dt は None のまま (1.0 補正)
        press_series = pd.to_numeric(df["pressure_hpa"], errors="coerce").dropna()
        if not press_series.empty:
            snapshot["dP_dt"] = float(dp)
        else:
            logger.info("アメダス局に気圧無し (type C) -> dP_dt は補正なし (1.0)")

        logger.info(
            "アメダス取得 OK: temp=%s dT_dt=%s dP_dt=%s (rows=%d)",
            snapshot["temp"], snapshot["dT_dt"], snapshot["dP_dt"], len(df),
        )
        return snapshot
    except Exception:
        logger.error("アメダス取得で例外 (続行、補正は 1.0)\n%s", traceback.format_exc())
        return {"temp": None, "dT_dt": None, "dP_dt": None}


def _step_astro(
    cfg: Dict[str, Any],
    target_date: date_t,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """天体イベントを Skyfield で計算."""
    site = cfg.get("site") or {}
    lat = float(site.get("lat", astro.OKIKAMURO_LAT))
    lon = float(site.get("lon", astro.OKIKAMURO_LON))
    daily = astro.compute_daily(target_date, lat=lat, lon=lon)
    logger.info(
        "天体計算 OK: sunrise=%s sunset=%s moon_illum=%s",
        daily.get("sunrise"),
        daily.get("sunset"),
        daily.get("moon_illumination"),
    )
    return daily


def _filter_target_day(df: pd.DataFrame, target_date: date_t) -> pd.DataFrame:
    """対象日の 0-23 時 24 行に絞る."""
    out = df.copy()
    out["datetime"] = pd.to_datetime(out["datetime"])
    mask = out["datetime"].dt.date == target_date
    return out.loc[mask].reset_index(drop=True)


def _step_compute_obi(
    hourly_df: pd.DataFrame,
    astro_daily: Dict[str, Any],
    cfg: Dict[str, Any],
    target_date: date_t,
    species: List[str],
    logger: logging.Logger,
    weather: Optional[Dict[str, Optional[float]]] = None,
) -> pd.DataFrame:
    """compute_obi を叩いて long 形式 (datetime, species, ..., stars) を返す."""
    site = cfg.get("site") or {}
    lat = float(site.get("lat", astro.OKIKAMURO_LAT))
    lon = float(site.get("lon", astro.OKIKAMURO_LON))

    w_cfg = cfg.get("weights") or {}
    weights = {
        "w1": float(w_cfg.get("w1_flow", 0.0)),
        "w2": float(w_cfg.get("w2_dh", 0.50)),
        "w3": float(w_cfg.get("w3_currentratio", 0.15)),
        "w4": float(w_cfg.get("w4_twilight", 0.15)),
        "w5": float(w_cfg.get("w5_moon", 0.10)),
        "w6": float(w_cfg.get("w6_season", 0.10)),
    }
    season_table = cfg.get("season") or {}

    day_df = _filter_target_day(hourly_df, target_date)
    if day_df.empty:
        raise RuntimeError(f"対象日 {target_date} の潮位データが見つかりません")

    result = obi.compute_obi(
        hourly_df=day_df,
        astro_daily=astro_daily,
        lat=lat,
        lon=lon,
        weights=weights,
        season_table=season_table,
        species_list=species,
        date=target_date,
        weather=weather,
    )
    logger.info(
        "OBI 計算 OK: %d 行 (魚種=%d 時間=%d)",
        len(result), len(species), len(result) // max(len(species), 1),
    )
    return result


# ---------------------------------------------------------------------------
# render 形式への変換
# ---------------------------------------------------------------------------
def _to_render_frame(
    obi_long: pd.DataFrame,
    astro_daily: Dict[str, Any],
    species_order: List[str],
) -> pd.DataFrame:
    """long (species 列) → wide (魚種=列) に変換し、attrs に天体情報を埋める.

    render.render_markdown / render_png は wide 形式 (日本語魚種列) を想定しているため.
    """
    df = obi_long.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["hour"] = df["datetime"].dt.hour

    # 魚種ラベル
    df["species_label"] = df["species"].map(SPECIES_LABEL).fillna(df["species"])
    label_order = [SPECIES_LABEL.get(s, s) for s in species_order]

    # tide_cm / dh_dt は時刻側のメタ. species ごとに重複してるので 1 本だけ抜く.
    meta = (
        df[["hour", "tide_cm", "dh_dt"]]
        .drop_duplicates(subset=["hour"])
        .sort_values("hour")
        .reset_index(drop=True)
    )

    # 魚種スコア (0..1) を pivot
    wide = df.pivot_table(
        index="hour",
        columns="species_label",
        values="score_01",
        aggfunc="first",
    )
    # 列順を整える (config に無い魚種が来ても末尾に追加)
    cols = [c for c in label_order if c in wide.columns]
    extra = [c for c in wide.columns if c not in cols]
    wide = wide[cols + extra]
    wide = wide.reset_index()

    merged = meta.merge(wide, on="hour", how="left").sort_values("hour").reset_index(drop=True)

    # attrs に天体サマリ
    def _fmt_time(v: Any) -> str:
        if isinstance(v, datetime):
            return v.astimezone(JST).strftime("%H:%M") if v.tzinfo else v.strftime("%H:%M")
        return "--:--"

    merged.attrs["sunrise"] = _fmt_time(astro_daily.get("sunrise"))
    merged.attrs["sunset"] = _fmt_time(astro_daily.get("sunset"))
    merged.attrs["moon_phase"] = astro_daily.get("moon_phase_frac")
    merged.attrs["moon_illumination"] = astro_daily.get("moon_illumination")
    return merged


# ---------------------------------------------------------------------------
# サマリ表示
# ---------------------------------------------------------------------------
def _print_console_summary(
    render_df: pd.DataFrame,
    target_date: date_t,
    species_order: List[str],
) -> None:
    """コンソールに当日の TOP3 時刻を魚種別で出す."""
    print(f"\n=== OBI v1 サマリ {target_date.isoformat()} ===")
    sun = render_df.attrs.get("sunrise", "--:--")
    sunset = render_df.attrs.get("sunset", "--:--")
    illum = render_df.attrs.get("moon_illumination")
    illum_s = f"{illum:.2f}" if isinstance(illum, (int, float)) else "--"
    print(f"日出 {sun} / 日没 {sunset} / 月照度 {illum_s}")
    for sp in species_order:
        label = SPECIES_LABEL.get(sp, sp)
        if label not in render_df.columns:
            continue
        series = pd.to_numeric(render_df[label], errors="coerce")
        if series.dropna().empty:
            continue
        top = series.nlargest(3)
        parts = []
        for idx in top.index:
            hour = int(render_df.loc[idx, "hour"])
            score = float(top[idx])
            stars = int(round(score * 5))
            parts.append(f"{hour:02d}:00({stars}★ / {score:.2f})")
        print(f"  {label}: " + " / ".join(parts))


# ---------------------------------------------------------------------------
# 出力先パス
# ---------------------------------------------------------------------------
def _output_paths(cfg: Dict[str, Any], target_date: date_t) -> Tuple[Path, Path, Path]:
    """out/obi_YYYYMMDD.{md,png,html} のパスを返す."""
    out_cfg = cfg.get("output") or {}
    out_dir = PROJECT_ROOT / (out_cfg.get("dir") or "out")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = target_date.strftime("%Y%m%d")
    md = out_dir / f"obi_{stamp}.md"
    png = out_dir / f"obi_{stamp}.png"
    html = out_dir / f"obi_{stamp}.html"
    return md, png, html


# ---------------------------------------------------------------------------
# エントリポイント
# ---------------------------------------------------------------------------
def run(argv: Optional[List[str]] = None) -> int:
    """CLI エントリポイント. 終了コードを返す."""
    args = parse_args(argv)

    # 設定ロード (ここはコケたら何も出来ないので即死)
    try:
        cfg = load_config()
    except Exception as e:
        sys.stderr.write(f"config.yaml の読み込みに失敗: {e}\n")
        traceback.print_exc(file=sys.stderr)
        return 2

    target_date = _resolve_date(args.date)
    species = _resolve_species(args.species, cfg)

    logger = _build_logger(target_date)
    logger.info("=== OBI v1 daily start target=%s species=%s ===", target_date, species)

    t0 = time.time()
    tide_df: Optional[pd.DataFrame] = None
    hourly_df: Optional[pd.DataFrame] = None
    astro_daily: Dict[str, Any] = {}
    obi_long: Optional[pd.DataFrame] = None
    render_df: Optional[pd.DataFrame] = None
    md_path: Optional[Path] = None
    png_path: Optional[Path] = None

    # --- 1. 潮位取得 (致命的) ---
    try:
        tide_df = _step_fetch_tide(cfg, target_date, logger)
    except Exception:
        logger.error("潮位取得で致命エラー\n%s", traceback.format_exc())
        return 3

    # --- 2. dh/dt ---
    try:
        hourly_df = _step_dh_dt(tide_df, logger)
    except Exception:
        logger.error("dh/dt 計算で致命エラー\n%s", traceback.format_exc())
        return 4

    # --- 3. 天体 (失敗しても続行: 空 dict で代用) ---
    try:
        astro_daily = _step_astro(cfg, target_date, logger)
    except Exception:
        logger.error("天体計算でエラー (続行)\n%s", traceback.format_exc())
        astro_daily = {}

    # --- 3b. アメダス気象 (失敗しても続行: 全 None で代用) ---
    weather_snap: Dict[str, Optional[float]] = _step_fetch_weather(cfg, logger)

    # --- 4. OBI 本体 ---
    try:
        obi_long = _step_compute_obi(
            hourly_df, astro_daily, cfg, target_date, species, logger,
            weather=weather_snap,
        )
    except Exception:
        logger.error("OBI 計算で致命エラー\n%s", traceback.format_exc())
        return 5

    # --- 5. wide 変換 ---
    try:
        render_df = _to_render_frame(obi_long, astro_daily, species)
    except Exception:
        logger.error("wide 変換でエラー\n%s", traceback.format_exc())
        return 6

    md_path, png_path, html_path = _output_paths(cfg, target_date)

    # --- 6. Markdown 出力 (続行可) ---
    try:
        render.render_markdown(render_df, target_date, str(md_path))
        logger.info("Markdown 出力 OK: %s", md_path)
    except Exception:
        logger.error("Markdown 出力でエラー (続行)\n%s", traceback.format_exc())
        md_path = None

    # --- 7. PNG 出力 (続行可) ---
    try:
        render.render_png(render_df, target_date, str(png_path))
        logger.info("PNG 出力 OK: %s", png_path)
    except Exception:
        logger.error("PNG 出力でエラー (続行)\n%s", traceback.format_exc())
        png_path = None

    # --- 7b. HTML 出力 (続行可) PNGを埋め込み・スマホで開ける単一ファイル ---
    try:
        render.render_html(render_df, target_date, str(html_path),
                           png_path=str(png_path) if png_path else None)
        logger.info("HTML 出力 OK: %s", html_path)
    except Exception:
        logger.error("HTML 出力でエラー (続行)\n%s", traceback.format_exc())
        html_path = None

    # --- 7c. 「最新版」ショートカット用コピー ---
    # 目的: 毎日ファイル名が変わると iPhone のホーム画面追加で参照が切れる。
    # そこで「常に最新を指す固定パス」を 2 箇所に作る:
    #   (a) 04_OBI/out/latest.html   ← 04_OBI 内で latest を見たい人向け
    #   (b) Claude office/最新OBI.html ← iCloud Drive のトップ、iPhone Files で一番アクセスしやすい
    latest_local = None
    latest_icloud = None
    if html_path and Path(html_path).exists():
        try:
            import shutil
            # (a) out/latest.html
            latest_local = Path(html_path).parent / "latest.html"
            shutil.copy2(html_path, latest_local)
            logger.info("最新版コピー OK (ローカル): %s", latest_local)
        except Exception:
            logger.error("最新版コピー失敗 (ローカル, 続行)\n%s", traceback.format_exc())
            latest_local = None

        try:
            # (b) iCloud Drive ルート (Claude office/) 直下に「最新OBI.html」
            # PROJECT_ROOT は 04_OBI/ なので 3 つ上がれば Claude office/ になる:
            #   04_OBI / 一本釣りAI / 1_プロジェクト / Claude office
            icloud_root = PROJECT_ROOT.parent.parent.parent
            latest_icloud = icloud_root / "最新OBI.html"
            import shutil as _shutil
            _shutil.copy2(html_path, latest_icloud)
            logger.info("最新版コピー OK (iCloud): %s", latest_icloud)
        except Exception:
            logger.error("最新版コピー失敗 (iCloud, 続行)\n%s", traceback.format_exc())
            latest_icloud = None

    # --- 8. コンソールサマリ ---
    try:
        _print_console_summary(render_df, target_date, species)
    except Exception:
        logger.error("コンソールサマリで失敗 (無視)\n%s", traceback.format_exc())

    # --- 8b. Discord 通知 (URL 未設定なら skip。daily 本処理は止めない) ---
    try:
        webhook_url = (cfg.get("discord") or {}).get("webhook_url")
        # null/空文字は無効扱い
        if isinstance(webhook_url, str) and webhook_url.strip():
            sent = discord_notify.send_obi_summary(
                obi_df=obi_long,
                astro_daily=astro_daily,
                target_date=target_date,
                webhook_url=webhook_url,
            )
            logger.info("Discord通知: sent=%s", sent)
        else:
            # 環境変数 OBI_DISCORD_WEBHOOK のフォールバックを試す
            sent = discord_notify.send_obi_summary(
                obi_df=obi_long,
                astro_daily=astro_daily,
                target_date=target_date,
            )
            logger.info("Discord通知 (env fallback): sent=%s", sent)
    except Exception:
        logger.error("Discord通知で例外 (続行)\n%s", traceback.format_exc())

    elapsed = time.time() - t0
    logger.info("=== OBI v1 daily done in %.2fs ===", elapsed)

    # --- 9. 出力ファイルパスを最後にプリント ---
    print("\n--- 出力ファイル ---")
    if md_path:
        print(f"MD  : {md_path}")
    else:
        print("MD  : (失敗)")
    if png_path:
        print(f"PNG : {png_path}")
    else:
        print("PNG : (失敗)")
    if html_path:
        print(f"HTML: {html_path}")
    else:
        print("HTML: (失敗)")
    if latest_local:
        print(f"最新: {latest_local}")
    if latest_icloud:
        print(f"📱 iPhone から見るなら: {latest_icloud}")
        print(f"   (iCloud Drive → Claude office → 最新OBI.html)")

    # --- 10. ブラウザで自動オープン (no_open=Trueでスキップ) ---
    if html_path and not getattr(args, "no_open", False):
        try:
            import subprocess, sys as _sys
            if _sys.platform == "darwin":
                subprocess.run(["open", str(html_path)], check=False)
            elif _sys.platform.startswith("linux"):
                subprocess.run(["xdg-open", str(html_path)], check=False)
            elif _sys.platform.startswith("win"):
                import os as _os
                _os.startfile(str(html_path))  # type: ignore[attr-defined]
        except Exception:
            logger.debug("ブラウザ自動オープンに失敗 (無視)\n%s", traceback.format_exc())

    return 0


def main() -> None:
    """`python -m src.daily` 用エントリ."""
    sys.exit(run())


if __name__ == "__main__":
    main()
