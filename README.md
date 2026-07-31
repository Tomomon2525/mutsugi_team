# PTCG AI Battle Challenge — ローカル開発環境

Kaggle コンペ [The Pokémon Company - PTCG AI Battle Challenge Simulation](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle) 向けの作業リポジトリである。

対戦エンジン `cabt` は `kaggle-environments` パッケージに同梱されており、macOS arm64 用の共有ライブラリ (`libcg.dylib`) も含まれる。追加のダウンロードなしにローカルで対戦が回る状態にしてある。

## 動かす前に

クローンしただけでは動かない。以下を先に済ませること。

### 1. Python 環境

`.venv/` は git 管理外である。手元で作り直す。Kaggle 側のランタイムに合わせて 3.11 を使う。

```bash
uv venv --python 3.11 .venv          # uv がなければ brew install uv
uv pip install --python .venv/bin/python -r requirements.txt
```

ここまでで `tools/evaluate.py` と `tools/replay.py` は動く。エンジンは pip 経由で入るため、コンペページからのダウンロードは要らない。

### 2. Kaggle の認証（提出・データ取得をする場合）

コンペページで規約に同意したうえで、アカウント設定から API トークンを発行し `~/.kaggle/kaggle.json` に置く。パーミッションは 600 にする。これがないと `kaggle` コマンドは一切通らない。

```bash
chmod 600 ~/.kaggle/kaggle.json
.venv/bin/kaggle competitions list -s pokemon-tcg   # 疎通確認
```

### 3. GitHub の認証（push をする場合）

```bash
gh auth login
```

リポジトリは private である。**コードをチーム外へ非公開に共有することは Kaggle のルールで禁止されている**ため、コラボレータの追加はチームメンバーに限ること。

## 構成

```
ptcg-abc/
├── .venv/                    Python 3.11 仮想環境
├── requirements.txt
├── agents/
│   └── baseline/
│       ├── main.py           エージェント本体
│       └── deck.csv          60 枚の card ID
├── tools/
│   ├── evaluate.py           ローカル対戦で勝率を測る
│   ├── build_submission.py   検証対戦 → submission.tar.gz
│   ├── probe_obs.py          observation をダンプして Enum を逆引きする
│   └── replay.py             公式ビジュアライザで HTML リプレイを出力
└── scratch/                  一時ファイル置き場 (git 管理外)
```

## 使い方

```bash
cd ~/Desktop/Kaggle/ptcg-abc

# ベースライン vs ランダム、20 戦を 6 並列で
.venv/bin/python tools/evaluate.py agents/baseline random -n 20 -j 6

# observation の中身を見る
.venv/bin/python tools/probe_obs.py --summary --out scratch/obs.jsonl

# リプレイを HTML で書き出してブラウザで開く
.venv/bin/python tools/replay.py agents/baseline random -o scratch/replay.html
open scratch/replay.html

# 提出物を作る (自己対戦の検証つき)
.venv/bin/python tools/build_submission.py agents/baseline
```

計測はおよそ 15 試合/秒/コア。並列数を上げれば数百戦の勝率比較も現実的な時間で終わる。

## ブランチ運用

**`main` は提出物のブランチである。** ここに入っているエージェントが、Kaggle に出す（あるいは出した）ものと一致している状態を保つ。

各自の試行は自分のブランチで行い、`main` へ直接コミットしない。

```bash
git switch -c feat/<名前>-<やること>     # 例: feat/kawamura-score-option
# ... 作業 ...
git push -u origin feat/kawamura-score-option
```

`main` へ入れる条件は 2 つ。

1. `tools/build_submission.py` の検証対戦が通ること
2. `tools/evaluate.py` で現行 `main` のエージェントと戦わせ、勝率が落ちていないこと

デッキやスコアリングを変えると勝率は簡単に上下する。試行数が少ないと差が誤差に埋もれるので、比較は最低でも 100 戦は回したい。

```bash
.venv/bin/python tools/evaluate.py agents/mine agents/baseline -n 200 -j 8
```

エージェントを増やす場合は `agents/<名前>/` を切り、`main.py` と `deck.csv` を置く。既存のディレクトリを書き換えるより、並べて勝率を比較できるほうが扱いやすい。

## エージェントの契約

`kaggle_environments/envs/cabt/cabt.py` の `interpreter` が実際の仕様である。要点は 3 つ。

1. `obs["select"]` が `None` の初回呼び出しでは、60 枚の card ID リストを返す。枚数が違えば `INVALID` になり即座に敗北する。
2. 以降は `obs["select"]["option"]` に対するインデックスのリストを返す。個数は `minCount` 以上 `maxCount` 以下。
3. 例外を投げるか範囲外を返すと `INVALID` 扱いで、相手に報酬 +1 が入る。`main.py` の `legal_fallback` はこれを防ぐための保険である。

observation の主なキーは `logs`（前回の選択以降のイベント列）、`current`（盤面）、`select`（選択要求）、`search_begin_input`（公式 SDK の探索 API に渡すバイナリ）。

### ローダの癖

`kaggle_environments` は `main.py` を読んで `exec(code, {})` する。そのため次の 2 点に注意する。

- 実行時に `__file__` が存在しない。`deck.csv` の場所は `co_filename` から復元している（`main.py` の `_here()`）。
- エージェントとして採用されるのは、モジュール内で**最後に定義された callable**。`agent` より後に関数やクラスを足すと壊れる。

## 未取得のもの

公式ドキュメント <https://matsuoinstitute.github.io/cabt/> には `cabt.api` として `all_card_data()` や `search_begin()` / `search_step()` / `search_end()` が載っているが、`kaggle-environments` 同梱の `cg` モジュールには含まれていない。カードのメタデータと先読み探索を使うには、コンペページ配布の SDK を別途取得する必要がある。PyPI に `cabt` パッケージは存在しない。

Kaggle CLI は `requirements.txt` に含めてある。認証を通せば以下が使える。

```bash
.venv/bin/kaggle competitions download -c pokemon-tcg-ai-battle
.venv/bin/kaggle competitions submit pokemon-tcg-ai-battle \
  -f agents/baseline/submission.tar.gz -m "baseline"
```

## 当面の課題

`score_option()` が定数を返す状態なので、ベースラインは実質「先頭から maxCount 個選ぶ」だけの挙動である。ランダム相手には 9 割勝つが、それは相手が弱すぎるからにすぎない。

- `select.type` と `context` の整数が何を意味するかを `probe_obs.py` の出力から特定する
- デッキを差し替えて勝率を比較する（既存の分析では、エージェントの精緻さよりデッキ選択のほうが Elo への寄与が大きいとされる）
- `search_*` API が入手できれば、1 手先読みまたは MCTS に進める
