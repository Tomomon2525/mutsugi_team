"""PTCG AI Battle Challenge - baseline agent.

Kaggle 側は submission.tar.gz を展開し、main.py 内で最後に定義された callable を
エージェントとして呼び出す。したがって `agent` はこのファイルの末尾で定義する。

エージェント契約 (kaggle_environments/envs/cabt/cabt.py の interpreter より):
  - obs["select"] が None の初手 -> 60 枚の card ID リスト (デッキ) を返す
  - それ以外 -> obs["select"]["option"] のインデックス列を返す
    要素数は minCount 以上 maxCount 以下。範囲外・不正なら status=INVALID で敗北。
"""

import csv
import os
import random
import sys


def _here() -> str:
    """このファイルが置かれたディレクトリ。

    kaggle_environments は main.py を read して exec(code, {}) するため、
    実行時に __file__ が存在しない。compile 時に渡されたパスが co_filename に
    残るので、そちらから復元する。
    """
    for cand in (globals().get("__file__"), sys._getframe().f_code.co_filename):
        if cand and os.path.isfile(cand):
            return os.path.dirname(os.path.abspath(cand))
    # 最後の砦: loader が sys.path に追加したエージェントディレクトリを探す
    for p in reversed(sys.path):
        if p and os.path.isfile(os.path.join(p, "deck.csv")):
            return os.path.abspath(p)
    return os.getcwd()


DECK_PATH = os.path.join(_here(), "deck.csv")


def load_deck(path: str = DECK_PATH) -> list[int]:
    """deck.csv から 60 枚の card ID を読む。1 行 1 ID、空行とコメントは無視。"""
    ids: list[int] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            for cell in row:
                cell = cell.strip()
                if not cell or cell.startswith("#"):
                    continue
                ids.append(int(cell))
    if len(ids) != 60:
        raise ValueError(f"deck must contain 60 cards, got {len(ids)}: {path}")
    return ids


DECK = load_deck()


def score_option(option: dict, obs: dict) -> float:
    """1 つの選択肢を評価する。ここを育てるのが本体。

    option は {"type": int, ...} 形式で、type やカード情報の意味は
    公式 SDK (cabt.api の Enum) 側に定義がある。現状は素の整数なので、
    tools/probe_obs.py でダンプしながら意味を埋めていく。
    """
    return 0.0


def choose(obs: dict) -> list[int]:
    sel = obs["select"]
    options = sel["option"]
    lo = int(sel.get("minCount") or 0)
    hi = int(sel.get("maxCount") or 0)
    hi = min(hi, len(options))
    lo = min(lo, hi)

    ranked = sorted(range(len(options)), key=lambda i: score_option(options[i], obs), reverse=True)
    # 上限まで取る挙動が常に正しいとは限らない。効果によっては最小枚数が有利。
    return ranked[:hi] if hi > 0 else ranked[:lo]


def legal_fallback(obs: dict) -> list[int]:
    """choose() が例外を投げても必ず合法手を返すための保険。"""
    sel = obs.get("select") or {}
    n = len(sel.get("option") or [])
    hi = min(int(sel.get("maxCount") or 0), n)
    lo = min(int(sel.get("minCount") or 0), hi)
    k = hi if hi > 0 else lo
    return random.sample(range(n), k) if n else []


def agent(obs: dict) -> list[int]:
    if obs.get("select") is None:
        return list(DECK)
    try:
        picked = choose(obs)
        sel = obs["select"]
        n = len(sel["option"])
        lo, hi = int(sel.get("minCount") or 0), min(int(sel.get("maxCount") or 0), n)
        picked = [i for i in dict.fromkeys(int(i) for i in picked) if 0 <= i < n]
        if not (lo <= len(picked) <= hi):
            raise ValueError("selection count out of range")
        return picked
    except Exception:
        return legal_fallback(obs)
