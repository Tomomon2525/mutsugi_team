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
from math import exp as _exp

import features
import ptcg

EPS = float(os.environ.get("PTCG_EPS", "0.25"))


class Profile:
    """方策の設定。エージェントごとの config.json から作る。

    対戦の両側が同じプロセスで動くため、グローバル変数では片側だけ設定を変えられない。
    A/B 比較も、リーグ用の相手を作り分けることも、この入れ物を通して行う。

      {"random_rate": 0.15, "attack_weight": 1.2, "setup_weight": 0.8}
    """

    __slots__ = ("eps", "context_signs", "foe_target", "attack_weight",
                 "setup_weight", "resource_weight", "name", "traits", "value_w",
                 "mlp")

    def __init__(self, cfg: dict | None = None, deck: list[int] | None = None):
        cfg = cfg or {}
        # デッキ固有の性質。設定ファイルではなくデッキの中身から導く。
        # 指定漏れや取り違えが起きないようにするため。
        self.traits = deck_traits(deck) if cfg.get("deck_traits", True) else {}
        self.name = cfg.get("strategy_profile", "standard")
        self.eps = float(cfg.get("random_rate", EPS))
        # 選択文脈による符号の切り替え。False で以前の「常に正」に戻す
        self.context_signs = bool(cfg.get("context_signs", True))
        # 相手のカードを指す選択の専用ルール。False で以前の挙動に戻す
        self.foe_target = bool(cfg.get("foe_target", True))
        self.attack_weight = float(cfg.get("attack_weight", 1.0))
        self.setup_weight = float(cfg.get("setup_weight", 1.0))
        self.resource_weight = float(cfg.get("resource_weight", 1.0))
        # 学習した評価関数の重み。config.json に直接書く。別ファイルにすると
        # 提出物への同梱漏れが起きるため、設定の中に持たせる。
        w = cfg.get("value_weights")
        self.value_w = [float(x) for x in w] if w and len(w) == features.N else None
        # 非線形版。特徴の数が合わなければ黙って無視する。特徴を足したあとに
        # 古い重みを読み込んで静かに壊れるのを避けるため、長さで弾く。
        m = cfg.get("value_mlp")
        self.mlp = None
        if m and len(m.get("w1") or ()) == features.N - 1:
            # 推論は 1 手の思考で 500 回前後走る。行優先で持つと積和が素直に書ける
            self.mlp = ([[float(x) for x in row] for row in m["w1"]],
                        [float(x) for x in m["b1"]],
                        [float(x) for x in m["w2"]],
                        float(m["b2"]))


def deck_traits(deck: list[int] | None) -> dict:
    """デッキの中身から、汎用ルールと向きが逆になる性質を拾う。

    どのデッキにも通用する一般則だけで動かすと、勝ち筋と正面から矛盾する相手が出る。
    根拠は docs/meta_decks.md にある。

      hand_hoard        手札の枚数がそのまま打点になる技を持つ。手札を吐き出すと弱くなる
      discard_energy    トラッシュのエネルギーを回収して使う。捨てることが利益になる
    """
    if not deck:
        return {}
    ids = set(deck)
    out = {}
    if 743 in ids:  # Alakazam の Powerful Hand は手札 1 枚につきダメカン 2 個
        out["hand_hoard"] = True
    if 678 in ids:  # Mega Lucario ex の Aura Jab はトラッシュから闘エネを 3 枚回収する
        out["discard_energy"] = True
    return out


DEFAULT = Profile()

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
    649: 68,   # Marnie's Morpeko        Punk Up の 5 枚をそのまま打点に変える
    1152: 50,  # Poké Pad
    1097: 45,  # Night Stretcher
    7: 42,     # Basic {D} Energy
    1080: 40,  # Unfair Stamp
    104: 32,   # Froslass
    860: 28,   # Snorunt
    1122: 28,  # Pokégear 3.0
    1137: 12,  # Tool Scrapper
    # --- 環境デッキ側の主要カード。相手をルールベース以上の強さで動かすために置く。
    #     ここが既定値のままだと相手が勝ち筋を実行できず、学習相手にならない。
    678: 100,  # Mega Lucario ex     HP340、Mega Brave 270
    121: 100,  # Dragapult ex        HP320、Phantom Dive 200 + ベンチ 60
    743: 95,   # Alakazam            手札枚数 × 20 打点
    742: 62,   # Kadabra
    741: 68,   # Abra
    120: 62,   # Drakloak
    119: 68,   # Dreepy
    677: 68,   # Riolu               Mega Lucario ex の進化元
    674: 55,   # Hariyama
    673: 45,   # Makuhita
    676: 45,   # Solrock
    675: 45,   # Lunatone
    140: 60,   # Fezandipiti ex
    1071: 55,  # Meowth ex
    66: 50,    # Dudunsparce
    305: 45,   # Dunsparce
    235: 40,   # Budew
    343: 35,   # Shaymin
    1121: 60,  # Ultra Ball
    1225: 58,  # Hilda
    1213: 55,  # Judge
    1198: 55,  # Crispin
    1141: 55,  # Premium Power Pro
    1142: 50,  # Fighting Gong
    1229: 50,  # Wally's Compassion
    1197: 48,  # Xerosic's Machinations
    1184: 45,  # Lana's Aid
    1120: 40,  # Crushing Hammer
    1081: 38,  # Enhanced Hammer
    1266: 40,  # Nighttime Mine
    1246: 38,  # Jamming Tower
    1129: 30,  # Sacred Ash
    1123: 25,  # Switch
    1159: 25,  # Hero's Cape
}

# 手札が 1 枚減る以外に代償の無いサーチ札。抱えていても何も生まない。
# Lillie's (手札を山札に戻す) と Unfair Stamp (使用条件つき) は状況で
# 良し悪しが変わるため入れない。
FREE_SEARCH = frozenset({
    1086,  # Buddy-Buddy Poffin   たね 2 体をベンチへ
    1152,  # Poké Pad             ルールボックス無しのポケモンを 1 枚
    1122,  # Pokégear 3.0         上 7 枚からサポートを 1 枚
    1219,  # Team Rocket's Petrel トレーナーを 1 枚
    1231,  # Dawn                 進化ラインを 1 枚ずつ
})

# そのうち、序盤は何より先に切るもの。手札を見てから他の手を決めたほうが
# 選択の幅が広がる。攻撃すると番が終わるので、サーチを先に済ませるのは常に正しい。
SEARCH_FIRST = frozenset({1086, 1152, 1122})
EARLY_TURNS = 4

# 他のどの Play よりも上に来る値。オーロンゲ ex への進化 (220) だけは上に置く。
# 3 枚の間の順序はカード価値で決める。同点にすると選択肢の並び順で決まってしまう
SEARCH_FIRST_SCORE = 200.0

_DEFAULT_BY_TYPE = {0: 55, 1: 45, 2: 50, 3: 45, 4: 40, 5: 40}

# SelectContext ごとの、カード価値をどちら向きに使うか。
# 「山札から手札に加える」と「手札から捨てる」では、同じカードでも良し悪しが逆になる。
# 向きの分からない文脈では価値表を使わない。誤った知識を当てるより、探索に委ねる。
GAIN = frozenset({
    1,   # SetupActivePokemon
    2,   # SetupBenchPokemon
    3,   # Switch      (自分の場を指す場合。相手を指す場合は playerIndex で分岐する)
    4,   # ToActive
    5,   # ToBench
    6,   # ToField
    7,   # ToHand
    16,  # RemoveDamageCounter
    17,  # Heal
    22,  # AttachTo
})
LOSE = frozenset({
    8,   # Discard
    9,   # ToDeck
    10,  # ToDeckBottom
    11,  # ToPrize
    26,  # DiscardEnergyCard
    27,  # DiscardToolCard
    29,  # DiscardCardOrAttachedCard
    30,  # DiscardEnergy
    32,  # ToDeckEnergy
})


def direction(context) -> int:
    """1 なら価値の高いカードを選ぶ、-1 なら低いカードを選ぶ、0 なら価値表を使わない。"""
    if context in GAIN:
        return 1
    if context in LOSE:
        return -1
    return 0


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


def effective_damage(attack_id: int | None, a: dict, me: dict, you: dict) -> int:
    """バトル場のポケモンがその技を撃ったときの打点。実体は features 側にある。

    同じ計算を評価関数の特徴でも使うため、二重に書かないよう一箇所に寄せた。
    """
    return features.damage_of(attack_id, a, _first(me.get("active")) or {}, me, you)


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


def score(opt: dict, sel: dict, cur: dict, me: dict, you: dict,
          prof: "Profile" = DEFAULT) -> float:
    t = opt.get("type")
    s = float(BASE.get(t, 30))

    if t == 13:  # Attack
        a = ptcg.attack(opt.get("attackId"))
        if a:
            dmg = effective_damage(opt.get("attackId"), a, me, you)
            s += prof.attack_weight * dmg / 8.0
            tgt = _first(you.get("active"))
            if tgt and dmg and dmg >= (tgt.get("hp") or 0):
                s += 80 * prof.attack_weight  # きぜつを取れる攻撃は他の何より優先する
        return s

    if t in (7, 9):  # Play / Evolve
        # 手札から出す・進化させるのは常に自陣を強くする行動なので向きは正
        c = _card_of(opt, sel, me)
        cid = c.get("id") if c else None
        if (t == 7 and cid in SEARCH_FIRST
                and (cur.get("turn") or 0) <= EARLY_TURNS
                and not (cid == 1086 and _bench_full(me))):
            return SEARCH_FIRST_SCORE + 0.1 * card_value(cid)
        s += prof.setup_weight * card_value(cid)
        if t == 9 and cid == 648:
            s += 40  # Punk Up でエネルギー 5 枚が付くので、進化そのものが加速になる
        if t == 7:
            s += _play_bonus(cid, me, you, cur.get("turn") or 0)
            if prof.traits.get("hand_hoard"):
                # 手札枚数が打点になるデッキでは、殴れる状態が整っているのに
                # カードを使うと自分で打点を削ることになる。準備中は減点しない。
                act = _first(me.get("active"))
                if act and (act.get("energies") or []) and act.get("id") == 743:
                    s -= 25
        return s

    if t in (3, 6):  # Card / Energy
        # 選択肢は playerIndex で持ち主が分かる。相手のカードを指す選択 (ダメージの
        # 対象、ボスの指令で引きずり出す先) は、自分のカードとは評価の向きが違う。
        owner = opt.get("playerIndex")
        foe = (prof.foe_target and owner is not None
               and owner != cur.get("yourIndex", 0))
        c = _card_of(opt, sel, you if foe else me)
        cid = c.get("id") if c else None
        if foe:
            hp = (c or {}).get("hp")
            s += 0.5 * card_value(cid)
            if hp is not None:
                # 落としやすいものを狙う。残り HP が低いほど取りやすい
                s += max(0.0, 40.0 - hp / 8.0)
            return s
        d = direction(sel.get("context")) if prof.context_signs else 1
        if d < 0 and prof.traits.get("discard_energy"):
            c2 = ptcg.card(cid)
            if c2 and c2.get("cardType") == 4:  # BasicEnergy
                # Aura Jab がトラッシュから回収するので、落とすこと自体が仕込みになる
                d = 1
        if d:
            s += d * card_value(cid) * 0.8 * prof.resource_weight
        return s

    if t == 8:  # Attach
        tgt = _in_play(me, opt.get("inPlayArea"), opt.get("inPlayIndex"))
        if tgt is not None:
            n = len(tgt.get("energies") or [])
            if opt.get("inPlayArea") == 4:
                s += 12  # バトル場が先。ベンチに貯めても今のターンには効かない
            if tgt.get("id") == 112:
                # Adrena-Brain は {D} が 1 個でも付いていれば使える。ダメカンを
                # 3 個動かせるので 1 個目の価値が高い。攻撃役ではないので 2 個目は無駄
                s += 35.0 if n == 0 else -30.0
            elif tgt.get("id") == 649:
                # Spiky Wheel は闇エネルギー 1 個につき 40 増える。5 個で 220 になり、
                # HP210 の ex を一撃で取れる。他のポケモンと逆に、貯めるほど良い。
                s += 8 * min(5, n + 1)
            elif n < 2:
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


def must_avoid(obs: dict) -> set:
    """探索に選ばせてはいけない選択肢の index。

    ベンチが空のまま番を終えると、バトル場の 1 体を落とされた時点で負ける。
    サイドを 1 枚も取られないまま試合が終わるので、取り返しがつかない。
    本番のリプレイ (89704936) では、手札に Munkidori と Buddy-Buddy Poffin を
    抱えたまま End を選び、次の番に倒されて負けていた。

    たねポケモンはベンチが空いている限り何枚でも出せる。出さずに番を終える
    理由が無いので、その場面の End だけを禁じる。ロールアウトの勝率平均が
    雑音に埋もれても、この手には落ちないようにする。
    """
    sel = obs.get("select") or {}
    cur = obs.get("current") or {}
    players = cur.get("players") or []
    if len(players) < 2:
        return set()
    me = players[cur.get("yourIndex", 0)]
    options = sel.get("option") or []
    end = {i for i, o in enumerate(options) if o.get("type") == 14}
    if not end or len(end) >= len(options):
        return set()

    empty_bench = not any(x for x in (me.get("bench") or []))
    early = (cur.get("turn") or 0) <= 2
    for o in options:
        if o.get("type") != 7:
            continue
        c = _card_of(o, sel, me)
        cid = (c or {}).get("id")
        if cid is None:
            continue
        # ベンチが空なのにたねを出さずに終える
        if empty_bench and (ptcg.card(cid) or {}).get("basic"):
            return end
        # 最初の番に、代償の無いサーチ札を抱えたまま終える
        if early and cid in SEARCH_FIRST and not (cid == 1086 and _bench_full(me)):
            return end
    return set()


def attack_options(obs: dict) -> tuple[set, set]:
    """(攻撃できる選択肢, そのうち相手をきぜつさせられるもの) の index 集合。

    「攻撃できるのにしなかった」「きぜつを取れるのに逃した」を数えるために使う。
    勝率だけでは、どこが弱いのかが分からない (docs/design.md 9 節)。
    """
    sel = obs.get("select") or {}
    cur = obs.get("current") or {}
    players = cur.get("players") or []
    if len(players) < 2:
        return set(), set()
    mi = cur.get("yourIndex", 0)
    me, you = players[mi], players[1 - mi]
    tgt = _first(you.get("active"))
    hp = (tgt or {}).get("hp") or 0
    atk: set = set()
    ko: set = set()
    for i, o in enumerate(sel.get("option") or []):
        if o.get("type") != 13:
            continue
        atk.add(i)
        aid = o.get("attackId")
        a = ptcg.attack(aid)
        if a and hp and effective_damage(aid, a, me, you) >= hp:
            ko.add(i)
    return atk, ko


def in_play_ids(obs: dict) -> list[int]:
    """自分の場 (バトル場 + ベンチ) にいるポケモンの ID。"""
    cur = obs.get("current") or {}
    players = cur.get("players") or []
    if not players:
        return []
    me = players[cur.get("yourIndex", 0)]
    out = []
    for zone in ("active", "bench"):
        for p in me.get(zone) or []:
            if p:
                out.append(p.get("id"))
    return out


def _bench_full(me: dict) -> bool:
    return len([p for p in (me.get("bench") or []) if p]) >= 3


def _play_bonus(cid: int | None, me: dict, you: dict, turn: int = 0) -> float:
    """局面によって価値が大きく動くカードだけ補正する。"""
    bench = [p for p in (me.get("bench") or []) if p]
    hand_ids = [c["id"] for c in (me.get("hand") or []) if c]

    if cid in FREE_SEARCH and turn <= EARLY_TURNS and cid != 1086:
        return 30.0  # 序盤は盤面を作る速度がそのまま勝率になる
    if cid == 1086:  # Buddy-Buddy Poffin
        return 35.0 if len(bench) < 3 else -45.0
    if cid == 1079:  # Rare Candy
        # 手札に 2 進化がいるかで判断する。デッキ固有の ID で判定すると、
        # 同じアメを使う相手デッキ (フーディン等) が常に減点になり、
        # 学習相手として成立しなくなる。
        if any((ptcg.card(h) or {}).get("stage2") for h in hand_ids):
            return 35.0
        return -60.0
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


def picks(obs: dict, rng, eps: float | None = None, jitter: float = 6.0,
          prof: "Profile" = DEFAULT) -> list[int]:
    """この局面で打つ手。eps の確率で一様ランダムに落とす。

    根で使うときは eps=0.0, jitter=0.0 を渡して、雑音を入れずに順位だけで決める。
    """
    if eps is None:
        eps = prof.eps
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
        key=lambda i: -(score(options[i], sel, cur, me, you, prof) + rng.random() * jitter),
    )
    return ranked[:k]


def evaluate(obs: dict, my_index: int, prof: "Profile" = DEFAULT) -> float:
    """未決着の局面を -0.7〜0.7 で採点する。終局の ±1 より必ず内側に収める。

    重みを与えられていれば学習したモデルを使う。無ければ手書きの式に落ちる。
    """
    if prof is not None and (prof.mlp is not None or prof.value_w is not None):
        x = features.vector(obs, my_index)
        if prof.mlp is not None:
            w1, b1, w2, b2 = prof.mlp
            z = b2
            for j in range(len(b1)):
                h = b1[j]
                for i in range(1, features.N):   # 先頭は定数項なので入力に入れない
                    xi = x[i]
                    if xi:
                        h += w1[i - 1][j] * xi
                if h > 0.0:
                    z += w2[j] * h
        else:
            w = prof.value_w
            z = 0.0
            for i in range(features.N):
                z += w[i] * x[i]
        # ロジスティック回帰の出力 p を [-1, 1] に写す。z が大きく振れても
        # 飽和するので、上下の切り詰めは形式的な保険にすぎない。
        p = 1.0 / (1.0 + _exp(-max(-30.0, min(30.0, z))))
        return max(-0.7, min(0.7, 1.4 * p - 0.7))

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
