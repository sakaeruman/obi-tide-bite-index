# 釣行ログ × OBI 突き合わせ

OBI v1 が出すスコア（★0-5）が、実際の釣果と相関しているかを検証するための釣行記録置き場。

## 概要

- `template.csv`: 新規ログを書くときのヘッダ＋1行サンプル。コピーして使う
- `sample.csv` : 動作確認用ダミー（2026-06-19 の架空釣行 5 件）
- 読み込み・比較は `src/catch_log_reader.py` で行う

## カラム仕様

| カラム | 型 | 必須 | 例 | 備考 |
|---|---|---|---|---|
| date         | YYYY-MM-DD | ◯ | 2026-06-19 | OBI レポート `obi_YYYYMMDD.md` と突き合わせるキー |
| time_start   | HH:MM      | ◯ | 05:00 | 24h 表記 |
| time_end     | HH:MM      | ◯ | 06:30 | 翌日に跨ぐ場合は自動 wrap |
| spot_name    | 自由記述   | ◯ | 西の高根 | |
| spot_lat     | 小数点4桁  | -  | 33.9170 | 任意 |
| spot_lon     | 小数点4桁  | -  | 132.2510 | 任意 |
| depth_m      | int        | -  | 30 | 水深 |
| tide_phase   | 満ち/引き/止 | - | 引き | 主観でOK |
| species      | 英小文字   | ◯ | aji | `madai / aji / saba / sawara / mebaru / yazu / tachi` |
| count        | int (匹)   | ◯ | 5 | 0 でも OK（ボウズ記録） |
| total_kg     | float      | -  | 1.2 | |
| max_size_cm  | float      | -  | 22 | |
| rig          | 自由記述   | -  | 枝5号 針14号 白カブラ | |
| notes        | 自由記述   | -  | 薄曇り 風南西2m | 気象・潮況・所感 |

species は CSV では英語、内部で日本語ラベル（マダイ/アジ/タチウオ/サワラ など）に対応付けて
OBI md の列と突き合わせる。

## 使い方

### 1. ログを追加する

```bash
cp catch_log/template.csv catch_log/2026_06.csv   # 月別などお好みで
# エディタで開いて行を足す
```

複数ファイルに分けてもよいし、1 ファイル（例 `master.csv`）に追記し続けてもよい。
`catch_log_reader.load_log()` は単一ファイルを読む想定なので、分けた場合は集約してから渡す。

### 2. OBI スコアと突き合わせる

```bash
cd "1_プロジェクト/一本釣りAI/04_OBI"
python -m src.catch_log_reader
```

ドライバは `catch_log/sample.csv` を読み、`out/obi_YYYYMMDD.md` から該当日の
スコア表を再パースして次を表示する:

1. 読み込んだログ
2. 釣行 × OBI 突き合わせ表（hour 平均 ★ と CPUE = count/h）
3. 魚種ごとの Spearman 順位相関  
   ベンチマーク: タイドグラフBI 第三者検証 **r = 0.46** を超えれば「実用」表示

### 3. 自前ログで回す

```python
from src.catch_log_reader import load_log, compare_with_obi, spearman_correlation

log = load_log("catch_log/2026_06.csv")
cmp = compare_with_obi(log, obi_dir="out")
print(spearman_correlation(cmp, "madai"))
```

## 設計メモ

- OBI v1 の内部スキーマには触らず、生成済みの `out/obi_YYYYMMDD.md` を
  正規表現でパースして時刻 × 魚種の★を取り出す。OBI 側を v2 に差し替えても
  md フォーマットを保てばこのモジュールは壊れない
- 釣行時間が複数 hour に跨る場合は、その範囲の★の平均と最大を取る
- CPUE（Catch Per Unit Effort）= `count / (time_end - time_start)` を単位「匹/時」で計算
- サンプル数 n < 3 のときは `verdict: サンプル不足` を返す
- 翌日に跨ぐ夜釣り（21:00→01:00 等）は wrap 加算で扱う

## 注意

- 公開リポ（`obi-tide-bite-index`）には **ログ CSV を含めない**。
  個人の釣果は秘匿情報として `.gitignore` で除外する想定（README 公開時は別途設定が必要）
- 場所名・緯度経度は自分用メモ。共有時は粒度を落とす
