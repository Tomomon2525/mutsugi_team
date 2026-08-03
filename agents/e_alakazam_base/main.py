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

import json
import math
import os
import random
import sys
import time
import traceback


def _here() -> str:
    for cand in (globals().get("__file__"), sys._getframe().f_code.co_filename):
        if cand and os.path.isfile(cand):
            return os.path.dirname(os.path.abspath(cand))
    for p in reversed(sys.path):
        if p and os.path.isfile(os.path.join(p, "deck.csv")):
            return os.path.abspath(p)
    return os.getcwd()


def _load_shared() -> dict:
    """自分のディレクトリにある共有モジュールを、自分専用の名前で読み込む。

    対戦の両側が同じプロセスで動くため、`import policy` と書くと sys.modules を
    奪い合う。過去バージョンを凍結して対戦相手にすると、先に読み込まれたほうの
    policy.py が両者に適用され、片方が壊れる。

    exec_module が終わった時点で、各モジュールの globals は兄弟モジュールの実体を
    直接掴んでいる。したがって、読み込みの間だけ素の名前を差し替え、終わったら
    元に戻せば、エージェントごとに独立した組を持てる。
    """
    import importlib.util

    here = _here()
    tag = "%08x" % (abs(hash(here)) & 0xFFFFFFFF)
    order = ("enums", "ptcg", "policy", "search")
    saved = {n: sys.modules.get(n) for n in order}
    mods: dict = {}
    try:
        for name in order:
            # 提出物では main.py と同じ階層に並ぶ。ローカル評価ではリポジトリの
            # shared/ を tools が sys.path へ足しているので、そちらから拾う。
            # 他のエージェントのディレクトリ (deck.csv がある) は候補から外す。
            # 凍結した過去バージョンが sys.path の先頭に自分を差し込むため、
            # 除外しないと相手のモジュールを掴んでしまう。
            cands = [here] + [
                d for d in sys.path
                if d and not os.path.isfile(os.path.join(d, "deck.csv"))
            ]
            path = next(
                (p for p in (os.path.join(d, name + ".py") for d in cands)
                 if os.path.isfile(p)),
                None,
            )
            if path is None:
                continue
            spec = importlib.util.spec_from_file_location(f"{name}__{tag}", path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            sys.modules[name] = mod  # 兄弟が素の名前で import できるようにする
            spec.loader.exec_module(mod)
            mods[name] = mod
    finally:
        for n, old in saved.items():
            if old is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = old
    return mods


_SHARED = _load_shared()
policy = _SHARED["policy"]
ptcg = _SHARED["ptcg"]
search = _SHARED["search"]

DECK = ptcg.load_deck(os.path.join(_here(), "deck.csv"))


def _config() -> dict:
    """エージェントごとの設定。対戦の両側が同じプロセスで動くため、環境変数では
    片側だけ変えられない。A/B 比較はこのファイルの有無で切り替える。

      {"policy": false}   ロールアウトを完全ランダムに戻す
      {"depth": 40}       40 手で打ち切って盤面評価に切り替える
    """
    try:
        with open(os.path.join(_here(), "config.json")) as f:
            return json.load(f)
    except Exception:
        return {}


CONFIG = _config()
USE_POLICY = CONFIG.get("policy")
DEPTH = CONFIG.get("depth")
PROFILE = policy.Profile(CONFIG, DECK)

# 候補を均等に試すと、明らかに悪い手にも同じ回数を使ってしまう。UCB1 で
# 平均の高い候補に寄せつつ、試行回数の少ない候補も拾う。
USE_UCB = bool(CONFIG.get("ucb", os.environ.get("PTCG_UCB", "1") not in ("0", "", "off")))
UCB_C = float(CONFIG.get("ucb_c", os.environ.get("PTCG_UCB_C", "0.7")))

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

# kaggle_environments はエージェントの標準出力も例外も握り潰す。探索が何回
# 失敗しているか、フォールバックに落ちているかを知る手段が他にないので、
# PTCG_TRACE を指定したときだけ 1 手 1 行の JSON を書く。エージェントは
# ジョブごとに別プロセスで動くため、追記の混線を避けて PID で分ける。
TRACE = os.environ.get("PTCG_TRACE")
_stat: dict = {}
_game = 0


def trace(rec: dict) -> None:
    if not TRACE:
        return
    try:
        with open(f"{TRACE}.{os.getpid()}", "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass

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


def pick_arm(score: list[float], played: list[int], tried: list[int], total: int) -> int | None:
    """次にロールアウトする候補。全部が展開に失敗していれば None。"""
    n = len(tried)
    for i in range(n):
        if not tried[i]:
            return i  # 未試行を残したまま比較しない
    if not USE_UCB:
        return min(range(n), key=lambda i: tried[i])

    logt = math.log(max(2.0, float(sum(tried))))
    best, best_v = None, float("-inf")
    for i in range(n):
        # 何度試しても展開できない候補は見切る (実測では発生していない)
        if played[i] == 0 and tried[i] >= 4:
            continue
        mean = (score[i] / played[i] + 1.0) / 2.0 if played[i] else 0.5
        v = mean + UCB_C * math.sqrt(logt / tried[i])
        if v > best_v:
            best, best_v = i, v
    return best


def choose(obs: dict) -> list[int]:
    sel = obs["select"]
    options = sel["option"]
    n = len(options)
    hi = min(int(sel.get("maxCount") or 0), n)
    lo = min(int(sel.get("minCount") or 0), hi)

    _stat.update(n=n, lo=lo, hi=hi, searched=0, begin_none=0, step_none=0,
                 playout_none=0, rollouts=0, evaluated=0)

    # 選択の余地がない場面では探索しない。試合の 2 割はここに該当する。
    # 複数枚を選ぶ場面 (全体の 2.8%) は先頭から取らず、方策の順位で選ぶ。
    if n <= 1 or hi != 1:
        if USE_POLICY is False:
            return list(range(hi if hi > 0 else lo))
        return policy.picks(obs, random, eps=0.0, jitter=0.0, prof=PROFILE)

    _stat["searched"] = 1
    deadline = time.monotonic() + slice_seconds(obs)
    my_index = obs["current"]["yourIndex"]
    score = [0.0] * n
    played = [0] * n
    tried = [0] * n
    total = 0

    s = searcher()
    try:
        for d in range(DETERMINIZATIONS):
            # 2 通り目以降は、時間が余っている場合だけ引き直す
            if d and time.monotonic() >= deadline:
                break
            root = s.begin(obs, DECK)
            if root is None:
                _stat["begin_none"] += 1
                break
            try:
                while total < MAX_ROLLOUTS:
                    i = pick_arm(score, played, tried, total)
                    if i is None:
                        break
                    tried[i] += 1
                    child = s.step(root.search_id, [i])
                    if child is None:
                        _stat["step_none"] += 1
                        continue
                    # playout は通過したノードを child ごと解放する
                    r = s.playout(child, my_index, use_policy=USE_POLICY,
                                  depth=DEPTH, profile=PROFILE)
                    total += 1
                    if r is not None:
                        score[i] += r
                        played[i] += 1
                    else:
                        _stat["playout_none"] += 1
                    # tried で判定する。展開に失敗し続ける候補があっても抜けられる。
                    if time.monotonic() >= deadline and all(tried):
                        break
            finally:
                s.release(root.search_id)
    finally:
        s.end()

    _stat["rollouts"] = total
    _stat["evaluated"] = sum(1 for p in played if p)
    if not any(played):
        return [0]

    # UCB は良さそうな候補に試行を寄せるので、平均値で選ぶと「2 回試して両方勝った」
    # 候補が最上位に来る。試行回数を第一基準にし、平均は同数のときの決め手にする。
    # 均等配分 (UCB なし) の場合は試行回数が並ぶので、実質は平均で決まる。
    rates = [score[i] / played[i] if played[i] else float("-inf") for i in range(n)]
    return [max(range(n), key=lambda i: (played[i], rates[i]))]


def legal_fallback(obs: dict) -> list[int]:
    sel = obs.get("select") or {}
    n = len(sel.get("option") or [])
    hi = min(int(sel.get("maxCount") or 0), n)
    lo = min(int(sel.get("minCount") or 0), hi)
    k = hi if hi > 0 else lo
    return random.sample(range(n), k) if n else []


def agent(obs: dict) -> list[int]:
    global _pool, _game
    if obs.get("select") is None:
        _pool = TIME_POOL
        # プロセスは複数試合で使い回されるので、試合ごとに番号を振る
        _game += 1
        return list(DECK)
    t0 = time.monotonic()
    _stat.clear()
    err = None
    picked: list[int] | None = None
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
        err = traceback.format_exc(limit=3)
        picked = legal_fallback(obs)
        return picked
    finally:
        dt = time.monotonic() - t0
        _pool -= dt
        if TRACE:
            sel = obs.get("select") or {}
            cur = obs.get("current") or {}
            rec = dict(_stat)
            rec.update(g=_game, turn=cur.get("turn"), side=cur.get("yourIndex"),
                       t=round(dt, 4), slice=round(slice_seconds(obs), 3),
                       pool=round(_pool, 1), picked=picked,
                       sel_type=sel.get("type"), sel_ctx=sel.get("context"))
            try:
                # 「攻撃できるのにしなかった」「きぜつを逃した」を後から数えるため、
                # 機会の有無と、実際に取ったかどうかを両方残す
                atk, ko = policy.attack_options(obs)
                got = set(picked or [])
                rec.update(atk_avail=len(atk), ko_avail=len(ko),
                           atk_taken=int(bool(got & atk)),
                           ko_taken=int(bool(got & ko)),
                           in_play=policy.in_play_ids(obs))
            except Exception:
                pass
            if err:
                rec["error"] = err[-400:]
            trace(rec)
