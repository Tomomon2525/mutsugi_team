"""cabt エンジンの探索 API の ctypes ラッパ。

`kaggle_environments` 同梱の libcg は AgentStart / SearchBegin / SearchStep /
SearchEnd / SearchRelease を公開しているが、cg/sim.py が宣言していないだけである。
ここで宣言して使えるようにする。

使い方:

    s = Searcher()
    root = s.begin(obs, my_deck=DECK)          # 隠れ情報を推定して探索を開始
    child = s.step(root.search_id, [option_i]) # 1 手進める
    result = s.playout(child.search_id, my_index)
    s.end()                                    # 確保した局面をまとめて解放

隠れ情報 (相手の手札・山札・サイド、自分の山札・サイド) は SearchBegin の引数として
具体的なカード ID を渡す必要がある。存在しない ID を渡すとエラーになる。
"""

import collections
import ctypes
import json
import os
import random

import policy

# ロールアウトの打ち方。
#   PTCG_POLICY=0  完全ランダムに戻す (A/B 比較用)
#   PTCG_DEPTH=n   n 手で打ち切り、policy.evaluate で採点する (0 で打ち切らない)
#
# 既定はルールベース方策 + 60 手で打ち切り。同一デッキ・同一時間予算の 60 戦で、
# 方策は完全ランダムに 75.0%、打ち切りは終局まで回す版に 73.3% で勝つ。
# ロールアウトは平均 149 手あるので、打ち切ると標本数がおよそ 2〜3 倍になる。
USE_POLICY = os.environ.get("PTCG_POLICY", "1") not in ("0", "", "off")
DEPTH = int(os.environ.get("PTCG_DEPTH", "60"))

_bound = False


def _bind() -> None:
    global _bound
    if _bound:
        return
    from kaggle_environments.envs.cabt.cg.sim import lib

    lib.AgentStart.restype = ctypes.c_void_p
    lib.SearchBegin.restype = ctypes.c_char_p
    lib.SearchBegin.argtypes = (
        [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
        + [ctypes.POINTER(ctypes.c_int)] * 6
        + [ctypes.c_int]
    )
    lib.SearchStep.restype = ctypes.c_char_p
    lib.SearchStep.argtypes = [ctypes.c_void_p, ctypes.c_longlong, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    lib.SearchEnd.argtypes = [ctypes.c_void_p]
    lib.SearchRelease.argtypes = [ctypes.c_void_p, ctypes.c_longlong]
    _bound = True
    return lib


def _lib():
    from kaggle_environments.envs.cabt.cg.sim import lib

    _bind()
    return lib


def _iarr(ids: list[int]) -> "ctypes.Array":
    ids = ids or [0]
    return (ctypes.c_int * len(ids))(*ids)


class Node:
    __slots__ = ("search_id", "obs")

    def __init__(self, search_id: int, obs: dict):
        self.search_id = search_id
        self.obs = obs

    @property
    def select(self) -> dict | None:
        return self.obs.get("select")

    @property
    def result(self) -> int:
        """決着していれば勝者の playerIndex、引き分けなら 2、未決着なら -1。"""
        cur = self.obs.get("current")
        return cur.get("result", -1) if cur else -1


# ---------------------------------------------------------------- 隠れ情報の推定


def visible_ids(player: dict) -> list[int]:
    """そのプレイヤーについて中身が見えているカードの ID を全部集める。"""
    ids: list[int] = []
    for c in player.get("hand") or []:
        if c:
            ids.append(c["id"])
    for zone in ("active", "bench"):
        for p in player.get(zone) or []:
            if not p:
                continue
            ids.append(p["id"])
            for key in ("energyCards", "tools", "preEvolution"):
                for c in p.get(key) or []:
                    if c:
                        ids.append(c["id"])
    for c in player.get("discard") or []:
        if c:
            ids.append(c["id"])
    return ids


def hidden_pool(deck_list: list[int], player: dict) -> list[int]:
    """デッキリストから見えているカードを引いた残り。山札・サイド・手札の候補になる。"""
    pool = collections.Counter(deck_list)
    pool.subtract(collections.Counter(visible_ids(player)))
    out: list[int] = []
    for cid, n in pool.items():
        if n > 0:
            out.extend([cid] * n)
    return out


def _take(pool: list[int], n: int, fallback: list[int], rng: random.Random) -> list[int]:
    """pool から n 枚取り出す。足りなければ fallback を循環させて埋める。"""
    if n <= 0:
        return []
    if len(pool) >= n:
        picked = rng.sample(pool, n)
        # 取ったぶんは pool から除く (重複カードを二重に割り当てないため)
        remain = collections.Counter(pool)
        remain.subtract(collections.Counter(picked))
        pool[:] = [c for c, k in remain.items() for _ in range(max(0, k))]
        return picked
    picked = list(pool)
    pool.clear()
    filler = (fallback * ((n // max(1, len(fallback))) + 1))[: n - len(picked)]
    return picked + filler


def _basic_pokemon(deck_list: list[int]) -> list[int]:
    """たねポケモンの ID。相手のバトル場が伏せられている場合の穴埋めに使う。"""
    try:
        import ptcg

        out = [cid for cid in dict.fromkeys(deck_list) if (ptcg.card(cid) or {}).get("basic")]
        if out:
            return out
    except Exception:
        pass
    return list(dict.fromkeys(deck_list))


class Searcher:
    """1 プロセスにつき 1 個だけ作る。AgentStart が確保した領域を使い回す。"""

    def __init__(self) -> None:
        self.lib = _lib()
        self.ptr = ctypes.c_void_p(self.lib.AgentStart())
        if not self.ptr.value:
            raise RuntimeError("AgentStart に失敗した")

    # ------------------------------------------------------------ 探索の開始と前進

    def begin(
        self,
        obs: dict,
        my_deck: list[int],
        enemy_deck: list[int] | None = None,
        manual_coin: bool = False,
        rng: random.Random = random,
    ) -> Node | None:
        """現局面から探索を開始する。隠れ情報はここで 1 通りに決め打つ (determinization)。

        enemy_deck を省略した場合は「相手も自分と同じデッキ」と仮定する。
        """
        cur = obs["current"]
        mi = cur["yourIndex"]
        me, you = cur["players"][mi], cur["players"][1 - mi]
        enemy_deck = enemy_deck or my_deck

        my_pool = hidden_pool(my_deck, me)
        en_pool = hidden_pool(enemy_deck, you)

        my_prize = _take(my_pool, len(me.get("prize") or []), my_deck, rng)
        my_rest = _take(my_pool, me.get("deckCount") or 0, my_deck, rng)
        en_prize = _take(en_pool, len(you.get("prize") or []), enemy_deck, rng)
        en_hand = _take(en_pool, you.get("handCount") or 0, enemy_deck, rng)
        en_rest = _take(en_pool, you.get("deckCount") or 0, enemy_deck, rng)

        # 相手のバトル場が伏せられている場合のみ中身を要求される。ポケモン以外を
        # 渡すとエンジン側で errorCode=2 になるので、たねポケモンで埋める。
        active = you.get("active") or []
        en_active: list[int] = []
        if any(p is None for p in active):
            basics = _basic_pokemon(enemy_deck)
            en_active = [basics[i % len(basics)] for i in range(len(active))]

        blob = obs["search_begin_input"].encode("ascii")
        raw = self.lib.SearchBegin(
            self.ptr,
            blob,
            len(blob),
            _iarr(my_rest),
            _iarr(my_prize),
            _iarr(en_rest),
            _iarr(en_prize),
            _iarr(en_hand),
            _iarr(en_active),
            1 if manual_coin else 0,
        )
        return self._node(raw)

    def step(self, search_id: int, picks: list[int]) -> Node | None:
        raw = self.lib.SearchStep(self.ptr, int(search_id), _iarr(picks), len(picks) or 1)
        return self._node(raw)

    def _node(self, raw) -> Node | None:
        if not raw:
            return None
        j = json.loads(raw.decode())
        if j.get("error"):
            return None
        st = j.get("state") or {}
        obs = st.get("observation")
        if obs is None or "searchId" not in st:
            return None
        return Node(st["searchId"], obs)

    # ------------------------------------------------------------ ロールアウト

    def playout(
        self,
        node: Node,
        my_index: int,
        max_steps: int = 2000,
        use_policy: bool | None = None,
        depth: int | None = None,
        profile: "policy.Profile | None" = None,
    ) -> float | None:
        """終局まで打つ。1 勝ち / 0 引き分け / -1 負け。決着しなければ None。

        PTCG_DEPTH を指定した場合は途中で打ち切り、policy.evaluate の値を返す。
        評価値は ±0.7 に収まるので、終局の ±1 を上回ることはない。

        通過したノードは即座に解放する。渡された node も解放するので、呼び出し側は
        これ以降 node を触らないこと。

        エンジンは 1 手ごとに State (std::array<Card,128> を含む数十 KB) を確保し、
        SearchEnd までは再利用されない。1 回のロールアウトが 90 手前後あるため、
        ここで解放しないと 1 手の思考で数百 MB を消費する。
        """
        # 対戦の両側が同じプロセスで動くので、環境変数では片側だけ設定を変えられない。
        # A/B 比較のため、呼び出し側 (エージェントごとの config.json) から上書きできる。
        use_policy = USE_POLICY if use_policy is None else use_policy
        depth = DEPTH if depth is None else depth
        profile = policy.DEFAULT if profile is None else profile
        steps = 0
        try:
            while steps < max_steps:
                r = node.result
                if r >= 0:
                    if r == my_index:
                        return 1
                    if r == 1 - my_index:
                        return -1
                    return 0
                if depth and steps >= depth:
                    # 学習データは手番側の視点だけで作る。相手の手札は観測では
                    # 隠れていて、手番でない側のベクトルは手札由来の特徴が
                    # 必ず 0 になるためである。打ち切りが相手の手番に当たった
                    # ときは、相手視点で評価して符号を返し、推論を学習と揃える。
                    # 手書きの式は差分だけで出来ているので、この変更で値は動かない
                    mover = (node.obs.get("current") or {}).get(
                        "yourIndex", my_index)
                    v = policy.evaluate(node.obs, mover, profile)
                    return v if mover == my_index else -v
                sel = node.select
                if not sel or not sel.get("option"):
                    return None
                if use_policy:
                    picks = policy.picks(node.obs, random, prof=profile)
                else:
                    n = len(sel["option"])
                    hi = min(int(sel.get("maxCount") or 0), n) or 1
                    lo = min(int(sel.get("minCount") or 0), hi)
                    picks = random.sample(range(n), hi if hi > 0 else lo)
                if not picks:
                    return None
                nxt = self.step(node.search_id, picks)
                if nxt is None:
                    return None
                self.release(node.search_id)
                node = nxt
                steps += 1
            return None
        finally:
            self.release(node.search_id)

    # ------------------------------------------------------------ 後始末

    def release(self, search_id: int) -> None:
        self.lib.SearchRelease(self.ptr, int(search_id))

    def end(self) -> None:
        """この探索で確保した局面をまとめて解放する。1 手ごとに必ず呼ぶ。"""
        self.lib.SearchEnd(self.ptr)
