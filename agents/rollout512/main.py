"""ロールアウト方式のエージェント。

各選択肢について「そこから終局まで適当に打つ」を何回か繰り返し、勝率が最も高い
選択肢を選ぶ。評価関数は持たない。勝敗そのものを価値とする。

学習はしない。1 手ごとに探索し、終わったら捨てる。強さは以下で決まる。
  - PTCG_BUDGET          1 手あたりのロールアウト総数
  - PTCG_DETERMINIZATION 隠れ情報の推定を何通り試すか
"""

import os
import random
import sys


def _here() -> str:
    for cand in (globals().get("__file__"), sys._getframe().f_code.co_filename):
        if cand and os.path.isfile(cand):
            return os.path.dirname(os.path.abspath(cand))
    for p in reversed(sys.path):
        if p and os.path.isfile(os.path.join(p, "deck.csv")):
            return os.path.abspath(p)
    return os.getcwd()


sys.path.insert(0, _here())

import ptcg  # noqa: E402
import search  # noqa: E402

DECK = ptcg.load_deck(os.path.join(_here(), "deck.csv"))

# 既定値は実測に基づく。64 -> 54.8%、256 -> 64.0% (対 agents/baseline)。
# 1 手あたり平均 218ms、1 試合の思考時間 3〜20 秒。cabt.json の持ち時間は
# 1 エピソード 600 秒なので余裕がある。
BUDGET = int(os.environ.get("PTCG_BUDGET512", "512"))
DETERMINIZATIONS = int(os.environ.get("PTCG_DETERMINIZATION", "2"))
MIN_PER_OPTION = 2

_searcher: search.Searcher | None = None


def searcher() -> search.Searcher:
    global _searcher
    if _searcher is None:
        _searcher = search.Searcher()
    return _searcher


def choose(obs: dict) -> list[int]:
    sel = obs["select"]
    options = sel["option"]
    n = len(options)
    hi = min(int(sel.get("maxCount") or 0), n)
    lo = min(int(sel.get("minCount") or 0), hi)

    # 選択の余地がない場面では探索しない。試合の 2 割はここに該当する。
    if n <= 1 or hi != 1:
        return list(range(hi if hi > 0 else lo))

    per_option = max(MIN_PER_OPTION, BUDGET // n)
    my_index = obs["current"]["yourIndex"]
    score = [0.0] * n
    played = [0] * n

    s = searcher()
    try:
        for _ in range(DETERMINIZATIONS):
            root = s.begin(obs, DECK)
            if root is None:
                break
            try:
                for i in range(n):
                    for _ in range(max(1, per_option // DETERMINIZATIONS)):
                        child = s.step(root.search_id, [i])
                        if child is None:
                            break
                        # playout は通過したノードを child ごと解放する
                        r = s.playout(child, my_index)
                        if r is None:
                            continue
                        score[i] += r
                        played[i] += 1
            finally:
                s.release(root.search_id)
    finally:
        s.end()

    if not any(played):
        return [0]

    # 未評価の選択肢を最下位に落とすため、試行 0 は -inf 扱いにする
    rates = [score[i] / played[i] if played[i] else float("-inf") for i in range(n)]
    return [max(range(n), key=lambda i: rates[i])]


def legal_fallback(obs: dict) -> list[int]:
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
