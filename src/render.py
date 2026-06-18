"""OBI v1 レンダリング: DataFrame から Markdown / PNG を生成する."""
from __future__ import annotations

import logging
import os
import traceback
from datetime import date as date_type, datetime, timedelta
from typing import Iterable, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 内部ユーティリティ
# ---------------------------------------------------------------------------

# 表示順を固定するための魚種既定リスト
DEFAULT_FISH_ORDER: List[str] = ["マダイ", "アジ", "タチウオ", "サワラ"]

# Markdown に出す固定列（魚種以外）
META_COLS_FOR_TABLE: List[str] = ["tide_cm", "dh_dt"]
META_COL_LABELS = {"tide_cm": "潮位cm", "dh_dt": "dh/dt"}


def _ensure_dir(path: str) -> None:
    """出力先ディレクトリを作成する."""
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)


def _to_date(date: Union[str, date_type, datetime]) -> date_type:
    """date / datetime / 文字列 を date に正規化する."""
    if isinstance(date, datetime):
        return date.date()
    if isinstance(date, date_type):
        return date
    if isinstance(date, str):
        # YYYY-MM-DD または YYYYMMDD を許容
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
            try:
                return datetime.strptime(date, fmt).date()
            except ValueError:
                continue
    raise ValueError(f"date を解釈できません: {date!r}")


def _score_to_stars(score: float, max_stars: int = 5) -> str:
    """0-1 の連続スコアを ★(0-5) 表記に丸める."""
    if score is None or (isinstance(score, float) and np.isnan(score)):
        return "-"
    s = float(score)
    # スコアが ★ 単位（0..max_stars）の場合と 0..1 の場合の両方を許容
    if s > max_stars:
        s = max_stars
    if s <= 1.0 and s >= 0.0:
        n = int(round(s * max_stars))
    else:
        n = int(round(s))
    n = max(0, min(max_stars, n))
    return "★" * n + "☆" * (max_stars - n)


def _detect_fish_columns(obi_df: pd.DataFrame) -> List[str]:
    """DataFrame の列名から魚種列を抽出する.

    既定順序にある名前を優先し、残りはアルファベット順で追加する.
    """
    cols = list(obi_df.columns)
    # メタ列・既定の補助列を除外
    excluded = {
        "time", "hour", "datetime", "date",
        "tide_cm", "tide", "level", "h", "dh_dt", "dhdt",
        "score", "U", "B", "M", "S",
    }
    fish_cols = [c for c in cols if c not in excluded and not c.startswith("_")]
    ordered: List[str] = []
    for f in DEFAULT_FISH_ORDER:
        if f in fish_cols:
            ordered.append(f)
    for c in fish_cols:
        if c not in ordered:
            ordered.append(c)
    return ordered


def _format_time_label(idx, row, date_obj: date_type) -> str:
    """行から HH:MM ラベルを生成する."""
    # 優先順: time 列 -> hour 列 -> datetime/index
    if "time" in row and pd.notna(row["time"]):
        v = row["time"]
        if isinstance(v, str):
            return v[:5]
        if hasattr(v, "strftime"):
            return v.strftime("%H:%M")
    if "hour" in row and pd.notna(row["hour"]):
        try:
            return f"{int(row['hour']):02d}:00"
        except Exception:
            pass
    if isinstance(idx, (pd.Timestamp, datetime)):
        return idx.strftime("%H:%M")
    if isinstance(idx, (int, np.integer)):
        return f"{int(idx):02d}:00"
    return str(idx)


def _classify_tide_range(tide_series: pd.Series) -> str:
    """潮位差から大潮/中潮/小潮の暫定ラベルを返す（v1 暫定）."""
    if tide_series is None or tide_series.empty:
        return "判定不能"
    rng = float(tide_series.max() - tide_series.min())
    if rng >= 200:
        return f"大潮相当（潮位差 {rng:.0f}cm）"
    if rng >= 120:
        return f"中潮相当（潮位差 {rng:.0f}cm）"
    return f"小潮相当（潮位差 {rng:.0f}cm）"


def _extract_summary(obi_df: pd.DataFrame) -> dict:
    """DataFrame.attrs などから天体・潮汐サマリを取り出す."""
    attrs = getattr(obi_df, "attrs", {}) or {}
    return {
        "sunrise": attrs.get("sunrise", "--:--"),
        "sunset": attrs.get("sunset", "--:--"),
        "moon_phase": attrs.get("moon_phase"),
        "moon_illumination": attrs.get("moon_illumination"),
    }


def _format_phase(phase: Optional[float]) -> str:
    if phase is None or (isinstance(phase, float) and np.isnan(phase)):
        return "--"
    try:
        return f"{float(phase):.2f}"
    except Exception:
        return "--"


def _format_illum(illum: Optional[float]) -> str:
    if illum is None or (isinstance(illum, float) and np.isnan(illum)):
        return "--"
    try:
        return f"{float(illum):.2f}"
    except Exception:
        return "--"


def _top3_per_fish(obi_df: pd.DataFrame, fish_cols: List[str], date_obj: date_type) -> List[str]:
    """魚種別 TOP3 時間帯を Markdown 箇条書きにする."""
    lines: List[str] = []
    for fish in fish_cols:
        if fish not in obi_df.columns:
            continue
        series = pd.to_numeric(obi_df[fish], errors="coerce")
        if series.dropna().empty:
            lines.append(f"- {fish}: データなし")
            continue
        top = series.nlargest(3)
        parts = []
        for idx in top.index:
            label = _format_time_label(idx, obi_df.loc[idx], date_obj)
            parts.append(f"{label}({_score_to_stars(float(top[idx]))})")
        lines.append(f"- {fish}: " + " / ".join(parts))
    return lines


# ---------------------------------------------------------------------------
# Markdown 出力
# ---------------------------------------------------------------------------

def render_markdown(
    obi_df: pd.DataFrame,
    date: Union[str, date_type, datetime],
    output_path: str,
) -> str:
    """OBI スコア表を Markdown として書き出し、書き出し先パスを返す."""
    try:
        date_obj = _to_date(date)
        _ensure_dir(output_path)

        summary = _extract_summary(obi_df)
        fish_cols = _detect_fish_columns(obi_df)

        # ---- ヘッダ ----
        lines: List[str] = []
        lines.append(f"# 沖家室 食い時合指数 OBI v1 — {date_obj.isoformat()}")
        lines.append("")
        lines.append("## 天体・潮汐サマリ")
        lines.append(f"- 日出: {summary['sunrise']} / 日没: {summary['sunset']}")
        moon_phase_s = _format_phase(summary.get("moon_phase"))
        moon_illum_s = _format_illum(summary.get("moon_illumination"))
        lines.append(f"- 月相: {moon_phase_s}(満ち欠け0-1) / 月照度: {moon_illum_s}")

        tide_series = None
        if "tide_cm" in obi_df.columns:
            tide_series = pd.to_numeric(obi_df["tide_cm"], errors="coerce").dropna()
        lines.append(f"- 潮回り判定: {_classify_tide_range(tide_series) if tide_series is not None else '潮位データなし'}")
        lines.append("")

        # ---- スコア表 ----
        lines.append("## 1時間ごとのスコア（★0-5）")
        header_cells = ["時刻"]
        meta_cols_present = [c for c in META_COLS_FOR_TABLE if c in obi_df.columns]
        for mc in meta_cols_present:
            header_cells.append(META_COL_LABELS.get(mc, mc))
        header_cells.extend(fish_cols)

        lines.append("| " + " | ".join(header_cells) + " |")
        lines.append("|" + "|".join(["---"] * len(header_cells)) + "|")

        for idx, row in obi_df.iterrows():
            cells: List[str] = [_format_time_label(idx, row, date_obj)]
            for mc in meta_cols_present:
                val = row.get(mc)
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    cells.append("-")
                else:
                    try:
                        if mc == "tide_cm":
                            cells.append(f"{float(val):.0f}")
                        else:
                            cells.append(f"{float(val):+.2f}")
                    except Exception:
                        cells.append(str(val))
            for fish in fish_cols:
                cells.append(_score_to_stars(row.get(fish)))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

        # ---- 推奨時間帯 ----
        lines.append("## 推奨時間帯（魚種別TOP3）")
        lines.extend(_top3_per_fish(obi_df, fish_cols, date_obj))
        lines.append("")

        # ---- 注意書き ----
        lines.append("## 注意（v1の限界）")
        lines.append("- 暫定: dh/dtを主軸、進行波/定常波の位相補正未実装")
        lines.append("- 海上保安庁MSILの潮流速 |v| 未統合（w1=0）")
        lines.append("- 採用しないもの: Solunor、潮回りラベル（リサーチで否定）")
        lines.append("")

        text = "\n".join(lines)
        with open(output_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        logger.info("Markdown を書き出しました: %s", output_path)
        return output_path

    except Exception:
        # ログにスタックを残し、続行できるよう例外は再送出しない
        _dump_traceback("render_markdown")
        raise


# ---------------------------------------------------------------------------
# HTML 出力（スマホで読みやすい・PNG埋め込み・単一ファイル）
# ---------------------------------------------------------------------------

def _stars_visual(score, max_stars: int = 5):
    """0-1 のスコアから (★塗り数, ★全表示HTML) を返す."""
    if score is None or (isinstance(score, float) and np.isnan(score)):
        return 0, '<span class="star-empty">—</span>'
    s = float(score)
    if s > max_stars:
        s = max_stars
    if 0.0 <= s <= 1.0:
        n = int(round(s * max_stars))
    else:
        n = int(round(s))
    n = max(0, min(max_stars, n))
    filled = "★" * n
    empty = "☆" * (max_stars - n)
    return n, f'<span class="star-filled">{filled}</span><span class="star-empty">{empty}</span>'


def _score_color(score):
    """スコアに応じた背景色（淡）を返す."""
    if score is None or (isinstance(score, float) and np.isnan(score)):
        return "#f6f6f4"
    s = float(score)
    if s > 1.0:
        s = s / 5.0
    if s >= 0.85:
        return "#fde2c2"  # ★5 オレンジ強め
    if s >= 0.65:
        return "#fef0d8"
    if s >= 0.45:
        return "#f6f3e6"
    if s >= 0.25:
        return "#eef0ee"
    return "#f6f6f4"


def render_html(
    obi_df: pd.DataFrame,
    date,
    output_path: str,
    png_path: Optional[str] = None,
) -> str:
    """OBI スコアをスマホでも読みやすい単一ファイルHTMLとして書き出す."""
    import base64
    import html as html_mod

    try:
        date_obj = _to_date(date)
        _ensure_dir(output_path)

        summary = _extract_summary(obi_df)
        fish_cols = _detect_fish_columns(obi_df)

        # PNG を base64 埋め込み（あれば）
        png_data_uri = ""
        if png_path and os.path.exists(png_path):
            try:
                with open(png_path, "rb") as fp:
                    b64 = base64.b64encode(fp.read()).decode("ascii")
                png_data_uri = f"data:image/png;base64,{b64}"
            except Exception:
                png_data_uri = ""

        # 全体のベストTOP1（魚種横断で最高スコアの時刻と魚種）
        best_overall = None  # (time_str, fish, score, stars_n)
        for idx, row in obi_df.iterrows():
            t_label = _format_time_label(idx, row, date_obj)
            for fish in fish_cols:
                sc = row.get(fish)
                if sc is None or (isinstance(sc, float) and np.isnan(sc)):
                    continue
                if best_overall is None or float(sc) > best_overall[2]:
                    n, _ = _stars_visual(sc)
                    best_overall = (t_label, fish, float(sc), n)

        # 魚種別TOP3 を構造化（render_markdownの_top3_per_fishは文字列のみ）
        per_fish_top3 = {}
        for fish in fish_cols:
            scores = obi_df[fish].dropna() if fish in obi_df.columns else pd.Series(dtype=float)
            if scores.empty:
                per_fish_top3[fish] = []
                continue
            top = scores.sort_values(ascending=False).head(3)
            per_fish_top3[fish] = [
                (_format_time_label(idx, obi_df.loc[idx], date_obj), float(top.loc[idx]))
                for idx in top.index
            ]

        # 潮回り判定
        tide_series = None
        if "tide_cm" in obi_df.columns:
            tide_series = pd.to_numeric(obi_df["tide_cm"], errors="coerce").dropna()
        tide_range_label = _classify_tide_range(tide_series) if tide_series is not None else "潮位データなし"

        moon_phase_s = _format_phase(summary.get("moon_phase"))
        moon_illum_s = _format_illum(summary.get("moon_illumination"))

        # ---- HTML 組み立て ----
        title = f"OBI {date_obj.strftime('%Y-%m-%d')} 沖家室"
        meta_lines = [
            f"日出 {summary.get('sunrise', '?')} / 日没 {summary.get('sunset', '?')}",
            f"月相 {moon_phase_s} / 月照度 {moon_illum_s}",
            f"潮回り: {tide_range_label}",
        ]

        # 推奨ハイライト
        if best_overall:
            t, fish, sc, n = best_overall
            stars_filled = "★" * n + "☆" * (5 - n)
            best_html = f"""
        <div class="best">
            <div class="best-label">今日のおすすめ</div>
            <div class="best-time">{html_mod.escape(t)}</div>
            <div class="best-detail">{html_mod.escape(fish)}　<span class="best-stars">{stars_filled}</span></div>
        </div>"""
        else:
            best_html = ""

        # 魚種別TOP3カード
        cards_html_parts = []
        for fish in fish_cols:
            tops = per_fish_top3.get(fish, [])
            if not tops:
                continue
            items = "".join(
                f'<li><span class="t">{html_mod.escape(t)}</span>'
                f'<span class="s">{_stars_visual(sc)[1]}</span></li>'
                for t, sc in tops
            )
            cards_html_parts.append(
                f'<div class="card"><h3>{html_mod.escape(fish)}</h3><ul>{items}</ul></div>'
            )
        cards_html = "\n".join(cards_html_parts)

        # 時刻別スコア表
        rows_html_parts = []
        meta_cols_present = [c for c in META_COLS_FOR_TABLE if c in obi_df.columns]
        # ヘッダ
        header_cells = ["<th>時刻</th>"]
        for mc in meta_cols_present:
            label = META_COL_LABELS.get(mc, mc)
            header_cells.append(f"<th>{html_mod.escape(label)}</th>")
        for fish in fish_cols:
            header_cells.append(f"<th>{html_mod.escape(fish)}</th>")
        header_html = "<tr>" + "".join(header_cells) + "</tr>"

        for idx, row in obi_df.iterrows():
            cells = [f'<td class="time">{html_mod.escape(_format_time_label(idx, row, date_obj))}</td>']
            for mc in meta_cols_present:
                val = row.get(mc)
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    cells.append('<td>—</td>')
                else:
                    try:
                        if mc == "tide_cm":
                            cells.append(f'<td>{float(val):.0f}</td>')
                        else:
                            cells.append(f'<td>{float(val):+.1f}</td>')
                    except Exception:
                        cells.append(f'<td>{html_mod.escape(str(val))}</td>')
            # 最大スコアの魚種を行内で軽くハイライト
            row_scores = {f: row.get(f) for f in fish_cols if f in obi_df.columns}
            valid = {k: v for k, v in row_scores.items() if v is not None and not (isinstance(v, float) and np.isnan(v))}
            row_best_fish = max(valid, key=lambda k: float(valid[k])) if valid else None
            for fish in fish_cols:
                sc = row.get(fish)
                bg = _score_color(sc)
                n, vis = _stars_visual(sc)
                cls = "fish best-in-row" if fish == row_best_fish and n >= 4 else "fish"
                cells.append(f'<td class="{cls}" style="background:{bg}">{vis}</td>')
            rows_html_parts.append("<tr>" + "".join(cells) + "</tr>")
        table_html = "<table><thead>" + header_html + "</thead><tbody>" + "\n".join(rows_html_parts) + "</tbody></table>"

        # PNG セクション
        if png_data_uri:
            png_section = f'<div class="chart"><img src="{png_data_uri}" alt="OBI heatmap"></div>'
        else:
            png_section = ""

        meta_html = "<ul class='meta-list'>" + "".join(f"<li>{html_mod.escape(m)}</li>" for m in meta_lines) + "</ul>"

        # スタイル＆HTML本体（モバイルファースト）
        html_doc = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{html_mod.escape(title)}</title>
<style>
  :root {{
    --bg: #f8f6f1;
    --ink: #2c4a5e;
    --accent: #c84b31;
    --line: #d8d4c8;
    --card: #ffffff;
    --muted: #6b7785;
  }}
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; padding: 0; background: var(--bg); color: var(--ink);
       font-family: "Hiragino Sans", "Yu Gothic", "Noto Sans CJK JP", system-ui, sans-serif;
       font-feature-settings: "palt"; line-height: 1.55; }}
  .container {{ max-width: 760px; margin: 0 auto; padding: 16px; }}
  header h1 {{ font-size: 20px; margin: 0 0 4px; }}
  header .sub {{ color: var(--muted); font-size: 13px; }}
  .meta-list {{ list-style: none; padding: 0; margin: 12px 0; font-size: 13px; color: var(--muted); }}
  .meta-list li {{ display: inline-block; margin-right: 14px; }}
  .best {{ background: var(--accent); color: #fff; padding: 16px; border-radius: 10px;
           margin: 12px 0 18px; text-align: center; }}
  .best-label {{ font-size: 12px; opacity: 0.85; letter-spacing: 0.1em; }}
  .best-time {{ font-size: 32px; font-weight: 700; margin: 4px 0; letter-spacing: 0.04em; }}
  .best-detail {{ font-size: 16px; }}
  .best-stars {{ font-size: 20px; letter-spacing: 2px; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 10px; margin-bottom: 18px; }}
  .card {{ background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; }}
  .card h3 {{ font-size: 14px; margin: 0 0 6px; color: var(--ink); }}
  .card ul {{ list-style: none; padding: 0; margin: 0; font-size: 13px; }}
  .card li {{ display: flex; justify-content: space-between; padding: 2px 0; }}
  .card .t {{ color: var(--muted); }}
  .star-filled {{ color: var(--accent); letter-spacing: 1px; }}
  .star-empty {{ color: #ccc8b8; letter-spacing: 1px; }}
  section h2 {{ font-size: 16px; margin: 24px 0 8px; border-bottom: 1px solid var(--line); padding-bottom: 4px; }}
  .table-wrap {{ overflow-x: auto; -webkit-overflow-scrolling: touch; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
  th, td {{ padding: 6px 4px; text-align: center; border-bottom: 1px solid var(--line); white-space: nowrap; }}
  th {{ background: var(--card); position: sticky; top: 0; font-weight: 600; color: var(--muted); font-size: 11px; }}
  td.time {{ color: var(--muted); font-variant-numeric: tabular-nums; }}
  td.fish {{ font-size: 11px; }}
  td.best-in-row {{ font-weight: 700; }}
  .chart {{ margin: 18px 0; }}
  .chart img {{ width: 100%; height: auto; border: 1px solid var(--line); border-radius: 6px; background: #fff; }}
  footer {{ font-size: 11px; color: var(--muted); margin-top: 24px; padding-top: 12px;
            border-top: 1px solid var(--line); line-height: 1.7; }}
  footer a {{ color: var(--ink); }}
</style>
</head>
<body>
  <div class="container">
    <header>
      <h1>沖家室 食い時合指数</h1>
      <div class="sub">OBI v1 — {html_mod.escape(date_obj.strftime("%Y年%m月%d日"))}</div>
    </header>
    {meta_html}
    {best_html}

    <section>
      <h2>魚種別 ベスト時間帯</h2>
      <div class="cards">{cards_html}</div>
    </section>

    <section>
      <h2>1時間ごとのスコア</h2>
      <div class="table-wrap">{table_html}</div>
    </section>

    {f'<section><h2>潮位 × 魚種ヒートマップ</h2>{png_section}</section>' if png_data_uri else ''}

    <footer>
      <div>※ 暫定 v1：dh/dt（潮位の動き）を主軸。海上保安庁 MSIL の流速 |v| は未統合（w2 暫定拡大）。</div>
      <div>※ 進行波/定常波の位相補正は地点校正前（phase_shift_hours=1.5 ハードコード）。</div>
      <div>※ 採用しないもの：Solunar、潮回りラベル（リサーチで否定）。</div>
      <div>潮位データ：気象庁 徳山（QA, 約30km）／天体：Skyfield（沖家室緯度経度ピンポイント）</div>
    </footer>
  </div>
</body>
</html>
"""
        with open(output_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(html_doc)
        logger.info("HTML を書き出しました: %s", output_path)
        return output_path

    except Exception:
        _dump_traceback("render_html")
        raise


# ---------------------------------------------------------------------------
# PNG 出力（ヒートマップ）
# ---------------------------------------------------------------------------

def render_png(
    obi_df: pd.DataFrame,
    date: Union[str, date_type, datetime],
    output_path: str,
) -> str:
    """OBI スコアを 24h × 魚種数 のヒートマップ PNG として書き出す."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # 日本語フォントは環境依存なので sans-serif に寄せて安全側へ
        plt.rcParams["font.family"] = [
            "Hiragino Sans", "Yu Gothic", "YuGothic",
            "IPAexGothic", "Noto Sans CJK JP", "DejaVu Sans", "sans-serif",
        ]
        plt.rcParams["axes.unicode_minus"] = False

        date_obj = _to_date(date)
        _ensure_dir(output_path)

        fish_cols = _detect_fish_columns(obi_df)
        if not fish_cols:
            raise ValueError("魚種列が見つかりません")

        # スコア行列（時刻 × 魚種）
        score_df = obi_df[fish_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        # 表記正規化: 0..1 と 0..5 のどちらでも色域 0..1 で扱う
        arr = score_df.to_numpy(dtype=float)
        if arr.size and arr.max() > 1.5:
            arr = arr / 5.0
        arr = np.clip(arr, 0.0, 1.0)

        # 行ラベル（時刻）
        time_labels = [
            _format_time_label(idx, obi_df.loc[idx], date_obj) for idx in obi_df.index
        ]

        # 描画
        n_hours, n_fish = arr.shape
        fig_w = max(4.0, 0.6 * n_fish + 2.0)
        fig_h = max(5.0, 0.28 * n_hours + 1.5)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        im = ax.imshow(arr, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)

        ax.set_xticks(np.arange(n_fish))
        ax.set_xticklabels(fish_cols, rotation=0)
        ax.set_yticks(np.arange(n_hours))
        ax.set_yticklabels(time_labels, fontsize=8)
        ax.set_xlabel("魚種")
        ax.set_ylabel("時刻")
        ax.set_title(f"OBI v1 ヒートマップ {date_obj.isoformat()}")

        # セルにスコア（★換算）を重ねる
        for i in range(n_hours):
            for j in range(n_fish):
                v = arr[i, j]
                ax.text(
                    j, i, f"{int(round(v * 5))}",
                    ha="center", va="center",
                    color="white" if v < 0.55 else "black",
                    fontsize=8,
                )

        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("スコア (0-1)")

        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        logger.info("PNG を書き出しました: %s", output_path)
        return output_path

    except Exception:
        _dump_traceback("render_png")
        raise


# ---------------------------------------------------------------------------
# 失敗時のロギング補助
# ---------------------------------------------------------------------------

def _dump_traceback(tag: str) -> None:
    """logs/ ディレクトリにスタックトレースを書き出す."""
    try:
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        logs_dir = os.path.join(base, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(logs_dir, f"render_error_{tag}_{stamp}.log")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(traceback.format_exc())
        logger.error("エラーログを保存しました: %s", path)
    except Exception:
        logger.exception("ロガーへのフォールバックも失敗")


# ---------------------------------------------------------------------------
# ドライバ（サンプルデータで両形式を出力）
# ---------------------------------------------------------------------------

def _make_sample_df(date_obj: date_type) -> pd.DataFrame:
    """24h × 4魚種のダミー DataFrame を作成する."""
    hours = np.arange(24)
    # 適当な潮位カーブ（半日周期）
    tide = 150 + 90 * np.sin(2 * np.pi * (hours - 3) / 12.42)
    dh_dt = np.gradient(tide)

    rng = np.random.default_rng(42)

    def base_curve(peak_hours: Iterable[int]) -> np.ndarray:
        y = np.zeros(24)
        for ph in peak_hours:
            y += np.exp(-((hours - ph) ** 2) / 6.0)
        y += 0.15 * rng.random(24)
        y = y / y.max()
        return np.clip(y, 0.0, 1.0)

    df = pd.DataFrame({
        "hour": hours,
        "tide_cm": tide,
        "dh_dt": dh_dt,
        "マダイ": base_curve([5, 18]),
        "アジ":   base_curve([4, 19]),
        "タチウオ": base_curve([19, 20, 4]),
        "サワラ": base_curve([6, 16]),
    })
    df.attrs.update({
        "sunrise": "05:08",
        "sunset": "19:25",
        "moon_phase": 0.42,
        "moon_illumination": 0.85,
    })
    return df


def _main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.join(base, "out")
    os.makedirs(out_dir, exist_ok=True)

    target = date_type.today() + timedelta(days=1)
    df = _make_sample_df(target)

    md_path = os.path.join(out_dir, f"obi_{target.strftime('%Y%m%d')}.md")
    png_path = os.path.join(out_dir, f"obi_{target.strftime('%Y%m%d')}.png")

    render_markdown(df, target, md_path)
    render_png(df, target, png_path)
    logger.info("サンプル出力完了: %s / %s", md_path, png_path)


if __name__ == "__main__":
    _main()
