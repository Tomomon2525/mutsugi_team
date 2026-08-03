"""ロールアウト中の方策と、途中局面の評価。

完全ランダムなロールアウトは、1 標本あたりの情報量が小さい。実測では 1 手の思考で
1 候補あたり 8 回程度しか試行できておらず、勝敗を ±1 で平均した標準誤差が 0.35 になる。
候補間の差はそれよりずっと小さいので、根の比較がほぼ雑音になっていた。

対策は 2 つある。ランダムより少しましな手を打たせて 1 標本の分散を下げること
(`picks`)、そして終局まで回さず途中で打ち切って採点し、標本数を稼ぐこと (`evaluate`)。

方策は貪欲にしない。EPS の確率で一様ランダムを混ぜる。決め打ちにすると、評価の誤りが
そのまま探索の盲点になるためである。相手側の手番でも同じ方策を使う。

カードの価値表 (`CARD_VALUE`) はオーロンゲデッキ向けの上書きで、載っていないカードは
種別からの既定値で評価する。他のデッキでも動くが、精度は落ちる。
方針の根拠は docs/grimmsnarl.md にある。
"""

import os

import ptcg

EPS = float(os.environ.get("PTCG_EPS", "0.25"))

# OptionType ごとの基礎点。攻撃と進化を上に、番の終了を最下位に置く。
BASE = {
    13: 90,   # Attack
    9: 80,    # Evolve
    10: 55,   # Ability
    7: 50,    # Play
    8: 45,    # Attach
    6: 40,    # Energy
    3: 35,    # Card
    0: 20,    # Number
    1: 25,    # Yes
    2: 15,    # No
    12: 8,    # Retreat
    14: 2,    # End
}

# 個別カードの価値。docs/grimmsnarl.md の優先順位に対応する。
CARD_VALUE = {
    648: 100,  # Marnie's Grimmsnarl ex   勝ち筋そのもの
    1079: 85,  # Rare Candy               最速着地に必要
    1086: 80,  # Buddy-Buddy Poffin       たね切れの主因を潰す
    1231: 75,  # Dawn                     進化ラインを 1 枚で揃える
    1182: 70,  # Boss's Orders
    646: 70,   # Marnie's Impidimp
    647: 62,   # Marnie's Morgrem
    1219: 60,  # Team Rocket's Petrel
    1227: 58,  # Lillie's Determination
    1259: 55,  # Spikemuth Gym
    112: 55,   # Munkidori
    1152: 50,  # Poké Pad
    1097: 45,  # Night Stretcher
    7: 42,     # Basic {D} Energy
    1080: 40,  # Unfair Stamp
    104: 32,   # Froslass
    860: 28,   # Snorunt
    1122: 28,  # Pokégear 3.0
    1137: 12,  # Tool Scrapper
}

_DEFAULT_BY_TYPE = {0: 55, 1: 45, 2: 50, 3: 45, 4: 40, 5: 40}


def card_value(cid: int | None) -> float:
    if cid is None:
        return 30.0
    v = CARD_VALUE.get(cid)
    if v is not None:
        return float(v)
    c = ptcg.card(cid)
    if c is None:
        return 30.0
    if c["cardType"] == 0:  # Pokemon
        if c.get("ex") or c.get("megaEx"):
            return 90.0
        if c.get("stage2"):
            return 75.0
        if c.get("stage1"):
            return 60.0
        return 50.0
    return float(_DEFAULT_BY_TYPE.get(c["cardType"], 35))


def _first(seq):
    for x in seq or []:
        if x:
            return x
    return None


def _in_play(player: dict, area: int | None, index: int | None) -> dict | None:
    if area == 4:
        lst = player.get("active") or []
    elif area == 5:
        lst = player.get("bench") or []
    else:
        return None
    return lst[index] if index is not None and 0 <= index < len(lst) else None


def _card_of(opt: dict, sel: dict, me: dict) -> dict | None:
    """選択肢が指しているカード。手札・山札・トラッシュのどれでも引く。"""
    area, index = opt.get("area"), opt.get("index")
    if index is None:
        return None
    if area == 1:  # 山札はサーチ時だけ select 側に実体が載る
        deck = sel.get("deck") or []
        return deck[index] if 0 <= index < len(deck) else None
    if area == 2 or area is None:
        hand = me.get("hand") or []
        return hand[index] if 0 <= index < len(hand) else None
    if area == 3:
        pile = me.get("discard") or []
        return pile[index] if 0 <= index < len(pile) else None
    return _in_play(me, area, index)


def score(opt: dict, sel: dict, cur: dict, me: dict, you: dict) -> float:
    t = opt.get("type")
    s = float(BASE.get(t, 30))

    if t == 13:  # Attack
        a = ptcg.attack(opt.get("attackId"))
        if a:
            dmg = a.get("damage") or 0
            s += dmg / 8.0
            tgt = _first(you.get("active"))
            if tgt and dmg and dmg >= (tgt.get("hp") or 0):
                s += 80  # きぜつを取れる攻撃は他の何より優先する
        return s

    if t in (7, 9, 3, 6):  # Play / Evolve / Card / Energy
        c = _card_of(opt, sel, me)
        cid = c.get("id") if c else None
        s += card_value(cid) * (1.0 if t != 3 else 0.8)
        if t == 9 and cid == 648:
            s += 40  # Punk Up でエネルギー 5 枚が付くので、進化そのものが加速になる
        if t == 7:
            s += _play_bonus(cid, me, you)
        return s

    if t == 8:  # Attach
        tgt = _in_play(me, opt.get("inPlayArea"), opt.get("inPlayIndex"))
        if tgt is not None:
            n = len(tgt.get("energies") or [])
            if opt.get("inPlayArea") == 4:
                s += 12  # バトル場が先。ベンチに貯めても今のターンには効かない
            if n < 2:
                s += 10
            else:
                s -= 6 * (n - 1)  # 3 枚目以降は腐る
        return s

    if t == 12:  # Retreat
        act = _first(me.get("active"))
        if act and act.get("maxHp") and (act.get("hp") or 0) * 3 <= act["maxHp"]:
            s += 25  # 瀕死なら下げる価値がある
        return s

    return s


def _play_bonus(cid: int | None, me: dict, you: dict) -> float:
    """局面によって価値が大きく動くカードだけ補正する。"""
    bench = [p for p in (me.get("bench") or []) if p]
    hand_ids = [c["id"] for c in (me.get("hand") or []) if c]

    if cid == 1086:  # Buddy-Buddy Poffin
        return 35.0 if len(bench) < 3 else -45.0
    if cid == 1079:  # Rare Candy
        return 35.0 if 648 in hand_ids else -60.0
    if cid == 1227:  # Lillie's Determination
        if len(hand_ids) <= 3:
            return 30.0
        return -25.0  # 手札が厚いうちに切ると、揃いかけた進化ラインごと流れる
    if cid == 1182:  # Boss's Orders
        for p in (you.get("bench") or []):
            if p and (p.get("hp") or 0) <= 180:
                return 25.0
        return -10.0
    return 0.0


def picks(obs: dict, rng, eps: float | None = None, jitter: float = 6.0) -> list[int]:
    """この局面で打つ手。eps の確率で一様ランダムに落とす。

    根で使うときは eps=0.0, jitter=0.0 を渡して、雑音を入れずに順位だけで決める。
    """
    if eps is None:
        eps = EPS
    sel = obs.get("select") or {}
    options = sel.get("option") or []
    n = len(options)
    if n == 0:
        return []
    hi = min(int(sel.get("maxCount") or 0), n) or 1
    lo = min(int(sel.get("minCount") or 0), hi)
    k = hi if hi > 0 else lo
    if k >= n:
        return list(range(n))
    if eps > 0 and rng.random() < eps:
        return rng.sample(range(n), k)

    cur = obs.get("current") or {}
    players = cur.get("players") or []
    mi = cur.get("yourIndex", 0)
    if len(players) < 2:
        return rng.sample(range(n), k)
    me, you = players[mi], players[1 - mi]

    # 同点の候補が並ぶ場面が多いので、乱数を足して順位を崩す
    ranked = sorted(
        range(n),
        key=lambda i: -(score(options[i], sel, cur, me, you) + rng.random() * jitter),
    )
    return ranked[:k]


def evaluate(obs: dict, my_index: int) -> float:
    """未決着の局面を -0.7〜0.7 で採点する。終局の ±1 より必ず内側に収める。"""
    cur = obs.get("current") or {}
    players = cur.get("players") or []
    if len(players) < 2:
        return 0.0
    me, you = players[my_index], players[1 - my_index]

    my_left = len(me.get("prize") or [])
    op_left = len(you.get("prize") or [])
    v = 0.45 * (op_left - my_left) / 6.0

    def board(p: dict) -> tuple[float, int]:
        hp = mx = 0
        cnt = 0
        for zone in ("active", "bench"):
            for x in p.get(zone) or []:
                if not x:
                    continue
                cnt += 1
                hp += x.get("hp") or 0
                mx += x.get("maxHp") or 0
        return (hp / mx if mx else 0.0), cnt

    my_hp, my_n = board(me)
    op_hp, op_n = board(you)
    v += 0.20 * (my_hp - op_hp)
    # 場のポケモンが尽きるとその時点で負けるので、枚数差は勝敗に直結する
    v += 0.15 * max(-1.0, min(1.0, (my_n - op_n) / 3.0))
    if my_n <= 1:
        v -= 0.15
    if op_n <= 1:
        v += 0.15
    # 手札は自分側しか実体が見えない場面があるので、枚数だけを使う
    my_hand = me.get("handCount") or len(me.get("hand") or [])
    op_hand = you.get("handCount") or len(you.get("hand") or [])
    v += 0.05 * max(-1.0, min(1.0, (my_hand - op_hand) / 6.0))
    return max(-0.7, min(0.7, v))
