"""Discord 通知モジュール (OBI v1).

役割:
    朝5時の GitHub Actions / cron / 手動実行のあとに
    「今日の OBI ★5、最高 XX:XX 〇〇」を Discord webhook で投げる。

設計方針:
    - webhook URL が未設定なら **何もせず False を返す**（エラーにしない）
    - 例外で daily.py 本処理を絶対に止めない（必ず try/except で握って False 返却）
    - 秘密情報 (webhook URL) は config.yaml には書かず、
      環境変数 OBI_DISCORD_WEBHOOK か config.local.yaml で渡す前提

使い方:
    from src import discord_notify
    discord_notify.send_obi_summary(obi_df, astro_daily, target_date)

    # テスト送信:
    python -m src.discord_notify --test
"""
from __future__ import annotations

import logging
import os
import sys
from datetime import date as date_t, datetime
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

import pandas as pd
import requests

logger = logging.getLogger("obi.discord")

JST = ZoneInfo("Asia/Tokyo")

# 環境変数名（Actions の secrets / シェルの export で渡す想定）
ENV_WEBHOOK = "OBI_DISCORD_WEBHOOK"

# Discord 側仕様
DISCORD_TIMEOUT_SEC = 10
DISCORD_OK_STATUS = (200, 204)

# 魚種コード -> 表示名 (daily.py の SPECIES_LABEL と同じ)
SPECIES_LABEL: Dict[str, str] = {
    "madai": "マダイ",
    "aji": "アジ",
    "tachiuo": "タチウオ",
    "sawara": "サワラ",
}


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------
def _fmt_time(v: Any) -> str:
    """datetime っぽいものを HH:MM に整形. 取れなければ '--:--'."""
    if isinstance(v, datetime):
        if v.tzinfo is not None:
            v = v.astimezone(JST)
        return v.strftime("%H:%M")
    if isinstance(v, str) and len(v) >= 5:
        # "2026-06-19T05:00:00+09:00" や "05:00" などをざっくり拾う
        try:
            dt = datetime.fromisoformat(v)
            if dt.tzinfo is not None:
                dt = dt.astimezone(JST)
            return dt.strftime("%H:%M")
        except ValueError:
            return v[-5:] if ":" in v[-5:] else "--:--"
    return "--:--"


def _fmt_illum(v: Any) -> str:
    """月照度を 0.22 形式に整形. 数値以外は '--'."""
    if isinstance(v, (int, float)):
        return f"{float(v):.2f}"
    return "--"


def _pick_best(obi_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """obi_long DataFrame から最高スコアの 1 行を抜く.

    期待する列: datetime, species, score_01, stars
    どれか欠けていれば None.
    """
    if obi_df is None or len(obi_df) == 0:
        return None
    needed = {"datetime", "species", "score_01"}
    if not needed.issubset(set(obi_df.columns)):
        logger.warning(
            "obi_df に必要列が無い (need=%s, have=%s)",
            needed, list(obi_df.columns),
        )
        return None

    df = obi_df.copy()
    df["score_01"] = pd.to_numeric(df["score_01"], errors="coerce")
    df = df.dropna(subset=["score_01"])
    if df.empty:
        return None

    idx = df["score_01"].idxmax()
    row = df.loc[idx]
    dt = pd.to_datetime(row["datetime"])
    if dt.tzinfo is None:
        # naive は JST と見なす（fetch_tide が JST 前提のため）
        dt = dt.tz_localize(JST)
    else:
        dt = dt.tz_convert(JST)

    species_code = str(row["species"])
    species_label = SPECIES_LABEL.get(species_code, species_code)
    score = float(row["score_01"])
    if "stars" in row and pd.notna(row["stars"]):
        stars = int(row["stars"])
    else:
        stars = int(round(score * 5))
    stars = max(0, min(5, stars))

    return {
        "time": dt.strftime("%H:%M"),
        "species_label": species_label,
        "species_code": species_code,
        "score": score,
        "stars": stars,
    }


def _build_message(
    obi_df: pd.DataFrame,
    astro_daily: Dict[str, Any],
    target_date: date_t,
    pages_url: str,
) -> str:
    """Discord に投げる本文を組み立てる."""
    best = _pick_best(obi_df)
    if best is None:
        # スコアが取れなかった場合も最低限のメッセージは出す
        head_line = "今日のおすすめ: -- -- (スコア計算なし)"
    else:
        star_str = "★" * best["stars"] + "☆" * (5 - best["stars"])
        head_line = (
            f"今日のおすすめ: {best['time']} {best['species_label']} {star_str}"
        )

    sun_line = (
        f"日出{_fmt_time(astro_daily.get('sunrise'))} "
        f"/ 日没{_fmt_time(astro_daily.get('sunset'))} "
        f"/ 月照度{_fmt_illum(astro_daily.get('moon_illumination'))}"
    )
    detail_line = f"詳細: {pages_url}"

    title = f"🎣 沖家室 OBI {target_date.isoformat()}"
    return "\n".join([title, head_line, sun_line, detail_line])


# ---------------------------------------------------------------------------
# 本処理
# ---------------------------------------------------------------------------
def send_obi_summary(
    obi_df: pd.DataFrame,
    astro_daily: Dict[str, Any],
    target_date: date_t,
    pages_url: str = "https://sakaeruman.github.io/obi-tide-bite-index/",
    webhook_url: Optional[str] = None,
) -> bool:
    """OBI のサマリを Discord webhook に投げる.

    Args:
        obi_df: obi.compute_obi の戻り値 (long形式, columns=datetime,species,score_01,stars,...)
        astro_daily: astro.compute_daily の戻り値 (sunrise/sunset/moon_illumination 等)
        target_date: 対象日 (JST)
        pages_url: GitHub Pages の公開 URL
        webhook_url: 明示指定. None なら環境変数 OBI_DISCORD_WEBHOOK を見る.

    Returns:
        True: 送信成功 (HTTP 200/204)
        False: 未設定でスキップ / 失敗 (どちらも例外は投げない)
    """
    # --- webhook URL 解決 ---
    if not webhook_url:
        webhook_url = os.environ.get(ENV_WEBHOOK)
    if not webhook_url:
        logger.info("Discord通知スキップ(webhook未設定)")
        return False

    # --- メッセージ組み立て (失敗しても daily.py を止めない) ---
    try:
        content = _build_message(obi_df, astro_daily, target_date, pages_url)
    except Exception as e:
        logger.error("Discord メッセージ生成で例外: %s", e, exc_info=True)
        return False

    # --- 送信 ---
    try:
        resp = requests.post(
            webhook_url,
            json={"content": content},
            timeout=DISCORD_TIMEOUT_SEC,
        )
    except Exception as e:
        logger.error("Discord webhook POST 失敗: %s", e, exc_info=True)
        return False

    if resp.status_code in DISCORD_OK_STATUS:
        logger.info("Discord通知送信 OK (status=%d)", resp.status_code)
        return True

    # 失敗時はステータスと本文の冒頭だけログに残す（webhook URL は出さない）
    body_head = (resp.text or "")[:200].replace("\n", " ")
    logger.error(
        "Discord通知失敗: status=%d body=%s",
        resp.status_code, body_head,
    )
    return False


def send_test(webhook_url: Optional[str] = None) -> bool:
    """疎通テスト用. 簡易メッセージを投げる."""
    if not webhook_url:
        webhook_url = os.environ.get(ENV_WEBHOOK)
    if not webhook_url:
        logger.info("Discord通知スキップ(webhook未設定)")
        return False

    today = datetime.now(JST).strftime("%Y-%m-%d %H:%M")
    content = (
        f"🎣 OBI 疎通テスト ({today} JST)\n"
        "このメッセージが見えれば webhook 設定 OK。"
    )
    try:
        resp = requests.post(
            webhook_url,
            json={"content": content},
            timeout=DISCORD_TIMEOUT_SEC,
        )
    except Exception as e:
        logger.error("Discord テスト送信失敗: %s", e, exc_info=True)
        return False

    if resp.status_code in DISCORD_OK_STATUS:
        logger.info("Discord テスト送信 OK (status=%d)", resp.status_code)
        return True

    body_head = (resp.text or "")[:200].replace("\n", " ")
    logger.error(
        "Discord テスト送信失敗: status=%d body=%s",
        resp.status_code, body_head,
    )
    return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _main(argv: Optional[list] = None) -> int:
    """`python -m src.discord_notify --test` 用."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    p = argparse.ArgumentParser(
        prog="obi.discord_notify",
        description="OBI Discord 通知 (疎通テスト用 CLI)",
    )
    p.add_argument("--test", action="store_true", help="疎通テストメッセージを送る")
    p.add_argument("--webhook", type=str, default=None, help="webhook URL を明示指定")
    args = p.parse_args(argv)

    if not args.test:
        p.print_help()
        return 0

    ok = send_test(webhook_url=args.webhook)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(_main())
