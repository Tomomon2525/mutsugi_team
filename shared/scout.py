"""相手のデッキを推定する。

探索の決定化 (search.begin) は、省略すると「相手も自分と同じデッキ」を仮定する。
公開ログを数えると環境の 4 割強は別のデッキなので、その間ずっと嘘の山札で
ロールアウトしていたことになる。場に出ているイワパレスは見えていても、
相手がこの先に何を引くかの想定が丸ごと違う。

見えている情報は多い。相手の場とトラッシュは全部見える。そこに出たカードの
集合を尤度に、公開ログの採用率を事前分布にすれば、素朴なベイズで足りる。

  P(デッキ | 見えたカード) ∝ 採用率 × Π P(そのカードがそのデッキに入っている)

「入っていない」を 0 にすると、集計に無い変種が出た瞬間に全部の候補が消える。
小さい値に留めて、証拠が積み上がるほど効くようにしてある。
"""

import collections
import json
import os

# 見えたカードがそのデッキに入っていなかったときの尤度。0 にはしない
MISS = 0.03
# これを下回る事前分布のデッキは候補から外す。裾の 1 戦ずつまで見る意味は薄い
MIN_SHARE = 0.002

_TABLE: list | None = None


def _load() -> list:
    """field.json を探して読む。main.py と同じ階層に置く前提である。"""
    here = os.path.dirname(os.path.abspath(__file__))
    for d in (here, os.path.dirname(here),
              os.path.join(os.path.dirname(here), "shared")):
        p = os.path.join(d, "field.json")
        if os.path.isfile(p):
            with open(p) as f:
                rows = json.load(f)
            out = []
            for r in rows:
                if r.get("share", 0.0) < MIN_SHARE:
                    continue
                deck = r["deck"]
                out.append((float(r["share"]), collections.Counter(deck), deck,
                            r.get("name", "")))
            return out
    return []


def table() -> list:
    global _TABLE
    if _TABLE is None:
        _TABLE = _load()
    return _TABLE


def visible(you: dict) -> collections.Counter:
    """相手側で中身が分かっているカード。場とトラッシュ。"""
    seen: collections.Counter = collections.Counter()
    for zone in ("active", "bench"):
        for x in you.get(zone) or ():
            if x and x.get("id") is not None:
                seen[x["id"]] += 1
    for c in you.get("discard") or ():
        if c and c.get("id") is not None:
            seen[c["id"]] += 1
    return seen


def guess(you: dict) -> tuple[list[int] | None, float, str]:
    """(推定したデッキリスト, 確信度, 名前)。候補が無ければ (None, 0, "")。

    確信度は最尤のデッキの事後確率。1 位と 2 位が拮抗している間は低く出る。
    """
    rows = table()
    if not rows:
        return None, 0.0, ""
    seen = visible(you)

    best = None
    logs = []
    for share, cnt, deck, name in rows:
        # log をそのまま足す。桁が落ちるので指数化の前に最大値を引く
        s = _log(share)
        for cid, n in seen.items():
            have = cnt.get(cid, 0)
            if have >= n:
                continue
            # 足りないぶんだけ罰する。1 枚だけ多い変種と、そもそも別のデッキを
            # 同じ扱いにしない
            s += (n - have) * _log(MISS)
        logs.append(s)
        if best is None or s > best:
            best = s

    tot = 0.0
    top = (-1.0, None, "")
    for (share, cnt, deck, name), s in zip(rows, logs):
        w = _exp(s - best)
        tot += w
        if w > top[0]:
            top = (w, deck, name)
    return top[1], (top[0] / tot if tot else 0.0), top[2]


def _log(x: float) -> float:
    from math import log

    return log(x) if x > 0 else -60.0


def _exp(x: float) -> float:
    from math import exp

    return exp(x) if x > -60.0 else 0.0
