"""ロールアウト方式のエージェント。

各選択肢について「そこから終局まで適当に打つ」を何回か繰り返し、勝率が最も高い
選択肢を選ぶ。評価関数は持たない。勝敗そのものを価値とする。

学習はしない。1 手ごとに探索し、終わったら捨てる。

回数ではなく時間で予算を切る。cabt は 1 エピソードにつきエージェント 1 体あたり
600 秒の持ち時間を与え、残量を obs["remainingOverageTime"] で毎手知らせてくる。
デッキによって 1 ロールアウトの長さが 7 倍以上変わるうえ、Kaggle の実行環境は
手元より 8〜9 倍遅い。固定回数だと、この二つが重なった時点で持ち時間を使い切る。

  PTCG_MAX_SLICE   1 手に使う秒数の上限
  PTCG_RESERVE     使い切らずに残す秒数
  PTCG_HORIZON     残り何手ぶんに割るとみなすか
  PTCG_TIME_POOL   remainingOverageTime が無い環境 (ローカル評価) での持ち時間
"""

import os
import random
import sys
import time


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

# 1 手の持ち時間は (残り - RESERVE) / HORIZON。残りが減れば自動的に細くなるので、
# 何手かかる試合でも RESERVE を割り込まない。HORIZON は実測 (75〜120 手) より
# やや小さく取り、序盤に厚く配る。
MAX_SLICE = float(os.environ.get("PTCG_MAX_SLICE", "5.0"))
MIN_SLICE = float(os.environ.get("PTCG_MIN_SLICE", "0.2"))
RESERVE = float(os.environ.get("PTCG_RESERVE", "45"))
HORIZON = float(os.environ.get("PTCG_HORIZON", "70"))
TIME_POOL = float(os.environ.get("PTCG_TIME_POOL", "600"))
# ローカルの kaggle_environments は runTimeout の 2000 秒をそのまま
# remainingOverageTime として渡してくる。Kaggle 本番の 600 秒を模したい場合は
# PTCG_TIME_POOL を明示すると、observation 側の値を無視して自前で数える。
FORCE_POOL = "PTCG_TIME_POOL" in os.environ

DETERMINIZATIONS = int(os.environ.get("PTCG_DETERMINIZATION", "2"))
MAX_ROLLOUTS = int(os.environ.get("PTCG_MAX_ROLLOUTS", "4096"))

_searcher: search.Searcher | None = None
# remainingOverageTime が来ない環境では自前で持ち時間を減らして模倣する
_pool = TIME_POOL


def searcher() -> search.Searcher:
    global _searcher
    if _searcher is None:
        _searcher = search.Searcher()
    return _searcher


def slice_seconds(obs: dict) -> float:
    rem = None if FORCE_POOL else obs.get("remainingOverageTime")
    if rem is None:
        rem = _pool
    return max(MIN_SLICE, min(MAX_SLICE, (float(rem) - RESERVE) / HORIZON))


def choose(obs: dict) -> list[int]:
    sel = obs["select"]
    options = sel["option"]
    n = len(options)
    hi = min(int(sel.get("maxCount") or 0), n)
    lo = min(int(sel.get("minCount") or 0), hi)

    # 選択の余地がない場面では探索しない。試合の 2 割はここに該当する。
    if n <= 1 or hi != 1:
        return list(range(hi if hi > 0 else lo))

    deadline = time.monotonic() + slice_seconds(obs)
    my_index = obs["current"]["yourIndex"]
    score = [0.0] * n
    played = [0] * n
    total = 0

    s = searcher()
    try:
        for d in range(DETERMINIZATIONS):
            # 2 通り目以降は、時間が余っている場合だけ引き直す
            if d and time.monotonic() >= deadline:
                break
            root = s.begin(obs, DECK)
            if root is None:
                break
            try:
                while total < MAX_ROLLOUTS:
                    ran = 0
                    for i in range(n):
                        # 全選択肢を 1 度は試す。未試行を残すと比較にならない。
                        if time.monotonic() >= deadline and played[i]:
                            continue
                        child = s.step(root.search_id, [i])
                        if child is None:
                            continue
                        # playout は通過したノードを child ごと解放する
                        r = s.playout(child, my_index)
                        total += 1
                        ran += 1
                        if r is None:
                            continue
                        score[i] += r
                        played[i] += 1
                    if ran == 0 or time.monotonic() >= deadline:
                        break
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
    global _pool
    if obs.get("select") is None:
        _pool = TIME_POOL
        return list(DECK)
    t0 = time.monotonic()
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
    finally:
        _pool -= time.monotonic() - t0
