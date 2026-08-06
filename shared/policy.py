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
    104: 72,   # Froslass    ベンチから毎ターン特性持ち全員にダメカンを配る
    860: 55,   # Snorunt     ユキメノコへの唯一の経路
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

# ベンチに置いたままでないと仕事をしないポケモン。バトル場に出すと特性が止まり、
# そのうえ落とされてサイドを渡す。本番のリプレイ 89696303 では、イタズラコゾウが
# 手札にあるのに開始時のバトル場にユキワラシを選んでいた。
#
#   860 ユキワラシ / 104 ユキメノコ   Freezing Shroud は場に居るだけで、毎回の
#       ポケモンチェックで特性を持つポケモン全員にダメカンを 1 個乗せる。
#       ユキメノコ自身は対象外。HP90 でバトル場に立つ理由が無い
#   112 マシマシラ   Adrena-Brain はベンチから使える。超エネを入れていないので
#       バトル場に出しても技が撃てず、ただの的になる
#
# ユキワラシとユキメノコは must_avoid で禁じる。マシマシラは、他に出せるものが
# 無い場面もあるため減点にとどめる。
BENCH_ONLY = {860: 250.0, 104: 300.0, 112: 90.0}

# バトル場に出す順の悪さ。数字が大きいほど強く避ける。全部が対象になったら
# 誰かを出すしかないので、悪いほうから順に外していく。
#
# ユキワラシとユキメノコしか残っていない場面で、両方まとめて禁じると禁止が
# 丸ごと無効になり、ユキメノコが出てしまうことがあった。ユキメノコは特性が
# 本体なので、失うと勝ち筋が消える。ユキワラシは進化前で、出しても損が小さい。
# 数字が大きいほど強く避ける。全部が対象になったら誰かを出すしかないので、
# 悪いほうから順に外していく。
#   112 マシマシラ  特性はベンチから使える。超エネを入れていないので技が撃てず、
#                   出したところでサイドを渡すだけ
#   104 ユキメノコ  特性が本体。落とすと勝ち筋が消える。ただしベンチにもう 1 体
#                   いるなら、出しても場からは消えないので許容する
#   860 ユキワラシ  進化前なので損は小さい
_BENCH_TIER = {112: 3, 104: 3, 860: 1}
HARD_BENCH = frozenset(_BENCH_TIER)


def bench_tier(cid: int | None, me: dict) -> int:
    t = _BENCH_TIER.get(cid, 0)
    if cid == 104 and sum(1 for x in features.in_play(me)
                          if x.get("id") == 104) >= 2:
        return 2      # 替えが利く。マシマシラを出すよりはまし
    return t
# 代償なしで、1 ターンに 1 回だけ使える特性。攻撃を選ぶと番が終わるので、
# 攻撃より先に使わないとその番のぶんが丸ごと消える。リプレイと自己対戦の
# どちらでも、使える手番の 4 割しか使えていなかった。
#   112 マシマシラ Adrena-Brain  ダメカンを 3 個まで相手へ移す
FREE_ABILITY = {112: 150.0}

# 個数を選ぶ場面のうち、多いほうが常に良いもの。マシマシラは 3 個動かせるのに
# 1 個や 2 個で済ませている場面が実際にあった (89692194)。
MORE_IS_BETTER = frozenset({
    38,  # DrawCount
    40,  # RemoveDamageCounterCount   Adrena-Brain が動かす個数
})

TO_ACTIVE = frozenset({
    1,   # SetupActivePokemon
    3,   # Switch
    4,   # ToActive
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


# 相手のポケモンにダメージを置く場面。Shadow Bullet のベンチへの飛び火と、
# Adrena-Brain で運ぶダメカンが該当する。どちらも 30 (ダメカン 3 個) である
DAMAGE_TO_FOE = frozenset({13, 15})
SPLASH = 30


def _splash_value(tgt: dict, me: dict, you: dict) -> float:
    """相手のどのポケモンに 30 を置くか。

    これまでは「HP が低いもの」だけで選んでいた。取れるサイドの枚数と、
    次の攻撃の射程に入るかを見る。飛び火そのものには弱点が乗らないので、
    30 は素の値で扱う。射程の判定には弱点を適用する (そちらは通常の攻撃)。
    """
    hp = tgt.get("hp") or 0
    pv = features.prize_value(tgt)
    if hp <= SPLASH:
        return 90.0 * pv                    # これで落ちる。他より優先する
    after = hp - SPLASH
    act = _first(me.get("active"))
    if act:
        fake = {"active": [tgt], "bench": (), "hand": None, "discard": (),
                "deckCount": 0}
        reach = features.best_attack(act, me, fake)
        # 既に届く相手に置いても射程は変わらない。届かないものを届かせる
        # ことに意味がある
        if reach > 0 and hp > reach >= after:
            # 次の攻撃で落とせる射程に入る。瀕死のものほど確実に取れる
            return 40.0 + 15.0 * pv + max(0.0, 30.0 - after / 10.0)
    return 12.0 * pv + max(0.0, 25.0 - hp / 10.0)


def attach_value(tgt: dict, me: dict) -> float:
    """闇エネルギー 1 個をこのポケモンに付ける価値。

    手貼りと Punk Up の付け先で同じ判断をするので、1 箇所にまとめてある。
    """
    n = len(tgt.get("energies") or [])
    cid = tgt.get("id")

    if cid == 112:
        # Adrena-Brain は {D} が 1 個でも付いていれば使える。ダメカンを 3 個
        # 動かせるので 1 個目の価値が高い。攻撃役ではないので 2 個目は無駄。
        # オーロンゲ ex は Punk Up が山札から最大 5 個付けるため、手貼りの 1 回は
        # こちらに使うほうが得になる。
        if n:
            return -30.0
        armed = sum(1 for x in features.in_play(me)
                    if x.get("id") == 112
                    and any(e == features.DARK for e in (x.get("energies") or ())))
        return max(8.0, 90.0 - 20.0 * armed)
    if cid == 649:
        # Spiky Wheel は闇エネルギー 1 個につき 40 増える。5 個で 220 になり、
        # HP210 の ex を一撃で取れる。他のポケモンと逆に、貯めるほど良い。
        return 8.0 * min(5, n + 1)
    if cid == 648:
        # Shadow Bullet は {D}{D}。エネルギーは Punk Up が山札から持ってくるので、
        # 手貼りは基本しない。ただし進化できない番が続くと撃てないまま止まるため、
        # 2 個目までは保険として弱めに加点する。3 個目以降は無駄でしかない
        return 20.0 if n < 2 else -60.0
    if cid in (104, 860):
        return -25.0  # 技を撃たせるつもりが無い。特性はエネルギーを要求しない
    if n < 2:
        return 10.0
    return -6.0 * (n - 1)  # 3 枚目以降は腐る


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
            act = _first(me.get("active"))
            if tgt and dmg and dmg >= (tgt.get("hp") or 0):
                # 落とせる攻撃は他の何より優先する。取れるサイドの枚数で重みを
                # 変える。ex は 2 枚、メガ ex は 3 枚 (リプレイで実測)
                got = features.prize_value(tgt)
                s += (50 + 35 * got) * prof.attack_weight
                if got >= len(you.get("prize") or ()):
                    s += 200.0   # これで決まる
            else:
                if dmg <= 0:
                    # 通らない相手に殴りかかっても番が終わるだけ。無効化の特性を
                    # 持つ相手や、打点 0 の技がここに来る
                    s -= 45.0
                if act and tgt:
                    # 落とせないまま殴ると、返しでこちらが落ちる。自分が ex なら
                    # 渡すサイドが重いので、殴らずに下がる手と比べさせる
                    back = features.best_attack(tgt, you, me)
                    if back > 0 and back >= (act.get("hp") or 0):
                        s -= 12.0 * features.prize_value(act)
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
            s += _play_bonus(cid, me, you, cur.get("turn") or 0, cur)
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
            s += 0.5 * card_value(cid)
            if c is not None and sel.get("context") in DAMAGE_TO_FOE:
                s += _splash_value(c, me, you)
            elif (c or {}).get("hp") is not None:
                # 落としやすいものを狙う。残り HP が低いほど取りやすい
                s += max(0.0, 40.0 - c["hp"] / 8.0)
            return s
        if d_ctx := direction(sel.get("context")):
            if d_ctx > 0 and sel.get("context") not in TO_ACTIVE:
                # 山札やトラッシュから持ってくる先。手札から出すときと同じ順で
                # 揃えないと、サーチが勝ち筋の部品を素通りする
                s += missing_bonus(cid, me)
                s += candy_bonus(cid, me)
                s += search_bonus(cid, me)
        if sel.get("context") in (16, 17) and c is not None:
            # ダメカンを取る先・回復する先。残りが薄いものから直す。
            # Adrena-Brain では、乗っている数がそのまま動かせる数になる
            hp, mx = c.get("hp") or 0, c.get("maxHp") or 0
            if mx:
                s += 45.0 * (1.0 - hp / mx)
        if sel.get("context") in TO_ACTIVE:
            s -= BENCH_ONLY.get(cid, 0.0)
            if cid == 648 and c is not None and features.best_attack(c, me, you) <= 0:
                # 撃てないオーロンゲ ex をバトル場に置いても、サイドを 2 枚
                # 差し出すだけになる。殴れないなら盾はベロバーに任せる
                s -= 45.0
        elif sel.get("context") == 21 and c is not None and cid is not None:
            # Punk Up の付け先。option type が 3 で来るので Attach の分岐に
            # 入らず、これまで全ての候補が同点だった
            return s + attach_value(c, me)
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
            if opt.get("inPlayArea") == 4:
                s += 12  # バトル場が先。ベンチに貯めても今のターンには効かない
            s += attach_value(tgt, me)
        return s

    if t == 10:  # Ability
        # 特性の選択肢は area / index で来る。inPlayArea ではない
        v = FREE_ABILITY.get((_card_of(opt, sel, me) or {}).get("id"))
        return v if v is not None else s

    if t == 0:  # Number
        n = opt.get("number")
        if n is not None and sel.get("context") in MORE_IS_BETTER:
            s += 12.0 * n
        return s

    if t == 12:  # Retreat
        act = _first(me.get("active"))
        if act and act.get("maxHp") and (act.get("hp") or 0) * 3 <= act["maxHp"]:
            s += 25  # 瀕死なら下げる価値がある
        if act and act.get("id") in BENCH_ONLY:
            # 相手のボスの指令などで引きずり出された場合。戻すのを最優先にする
            s += 60 if act.get("id") in HARD_BENCH else 30
        # 次の番に落とされる ex を、サイドの軽いポケモンと入れ替える。
        # 落とされる前提なら、渡す枚数が少ないほうを前に置く
        tgt = _first(you.get("active"))
        if act and tgt:
            back = features.best_attack(tgt, you, me)
            mine = features.prize_value(act)
            if back > 0 and back >= (act.get("hp") or 0) and mine >= 2:
                cheap = min((features.prize_value(x)
                             for x in (me.get("bench") or ()) if x), default=mine)
                if cheap < mine:
                    s += 30.0 * (mine - cheap)
        return s

    return s


# ベンチが空のときに、それを埋められる札。ポフィンは直接ベンチへ出し、
# 残りはたねを手札に持ってくるので、同じ番のうちに出せる。
# ペトレルはトレーナーしか取れないので入れない (二段構えになり、確実性が落ちる)
BENCH_REFILL = frozenset({
    1086,  # Buddy-Buddy Poffin      たね 2 体をベンチへ
    1152,  # Poké Pad                ルールボックス無しのポケモンを 1 枚
    1231,  # Dawn                    たね・1 進化・2 進化を 1 枚ずつ
    1259,  # Spikemuth Gym           マリィのポケモンを 1 枚
})


# 攻撃を選ぶと番が終わる。手札を切る、エネルギーを付ける、特性を使う、進化する、
# どれも番を終えないので、殴る前に済ませるのが常に正しい。
# 本番のリプレイ 90139220 では、手札が 12 枚まで膨れてポフィン 2 枚・ポケパッド
# 2 枚・ユキメノコ・スタジアム 2 枚を抱えたまま、毎ターン殴るだけだった。
# 攻撃は 192 点で、どのトレーナーよりも高いためである。
ATTACK_FLOOR = 90.0


def pending_before_attack(sel: dict, cur: dict, me: dict, you: dict,
                          prof: "Profile" = DEFAULT,
                          skip: frozenset = frozenset(),
                          no_supporter: bool = False) -> bool:
    """先に済ませておくべき手が残っているか。

    skip に入れたカードは「先にやること」から外す。手札を山札に戻す札は、
    自分自身が判定に混ざると永久に順番が回ってこない。
    """
    for o in sel.get("option") or ():
        t = o.get("type")
        if t == 9:
            return True   # 進化。番を終えず、盤面が強くなるだけ
        if t == 10 and (_card_of(o, sel, me) or {}).get("id") in FREE_ABILITY:
            return True   # 代償のない特性。1 ターン 1 回で、使わない理由が無い
        if t == 8 and score(o, sel, cur, me, you, prof) >= BASE[8]:
            return True   # エネルギーを付ける。1 ターン 1 回、腐らせる意味が無い
        if t == 7:
            cid = (_card_of(o, sel, me) or {}).get("id")
            if cid in skip:
                continue
            if no_supporter and (ptcg.card(cid) or {}).get("cardType") == 3:
                continue   # サポートは 1 ターン 1 枚。抱えて待つ価値がある
            if score(o, sel, cur, me, you, prof) >= ATTACK_FLOOR:
                return True   # 明らかに得なトレーナー。減点が乗ったものは拾わない
    return False


# 迷う理由が無い進化。ロールアウトの勝率平均は、じわじわ効く特性を拾えない。
# 実際のリプレイでは、方策が 1 位に置いていても探索がひっくり返していた。
# ユキメノコに進化が選べた 58 局面のうち、進化したのは 6 局面しかない。
#
#   860 ユキワラシ → 104 ユキメノコ  HP 70→90 で、場に居るだけで毎回の
#       ポケモンチェックで特性持ち全員にダメカンが乗る。進化を遅らせて
#       得することが無い
FORCED_EVOLVE = frozenset({104})


def must_take(obs: dict) -> set:
    """探索を通さずに即決してよい選択肢の index。

    must_avoid の裏返しである。禁じ手と同じく、勝率平均が雑音に埋もれても
    この手だけは落とさないようにする。思考時間も浮く。
    """
    sel = obs.get("select") or {}
    cur = obs.get("current") or {}
    players = cur.get("players") or []
    if len(players) < 2:
        return set()
    me = players[cur.get("yourIndex", 0)]
    out = set()
    options = sel.get("option") or ()
    for i, o in enumerate(options):
        if o.get("type") != 9:
            continue
        if (_card_of(o, sel, me) or {}).get("id") in FORCED_EVOLVE:
            out.add(i)
    if out:
        return out

    # エネ 0 のマシマシラへの手貼り。1 個付けば毎ターン 3 個のダメカンを運べる。
    # Main の中ではトレーナーのほうが点数が高く、方策の順位で 1 位に来るのは
    # リプレイで 22% しかなかった。手貼りは 1 ターン 1 回なので、後回しにすると
    # そのまま流れる。
    #
    # 例外は、オーロンゲ ex がバトル場でエネ 1 個のとき。あと 1 個で
    # Shadow Bullet が撃てるので、そちらが優先になる。
    act = _first(me.get("active"))
    grim_needs = (act and act.get("id") == 648
                  and len(act.get("energies") or ()) == 1)
    if not grim_needs:
        for i, o in enumerate(options):
            if o.get("type") != 8:
                continue
            tgt = _in_play(me, o.get("inPlayArea"), o.get("inPlayIndex"))
            if (tgt and tgt.get("id") == 112
                    and not (tgt.get("energies") or ())):
                return {i}

    # 山札サーチで、今すぐ 2 進化を飛ばせるふしぎなアメ。方策は 1 位に置いて
    # いたが探索が覆していた (リプレイで 13 回中 0 回しか取っていない)
    if direction(sel.get("context")) > 0 and int(sel.get("maxCount") or 0) == 1:
        hand = [c["id"] for c in (me.get("hand") or ()) if c]
        play = [x.get("id") for x in features.in_play(me)]
        if (648 in hand and (646 in play or 649 in play)
                and hand.count(RARE_CANDY) == 0):
            for i, o in enumerate(options):
                if (_card_of(o, sel, me) or {}).get("id") == RARE_CANDY:
                    return {i}
    return out


def must_avoid(obs: dict) -> set:
    """探索に選ばせてはいけない選択肢の index。

    ベンチが空のまま番を終えると、バトル場の 1 体を落とされた時点で負ける。
    サイドを 1 枚も取られないまま試合が終わるので、取り返しがつかない。
    本番のリプレイ (89704936) では、手札に Munkidori と Buddy-Buddy Poffin を
    抱えたまま End を選び、次の番に倒されて負けていた。

    たねポケモンはベンチが空いている限り何枚でも出せる。出さずに番を終える
    理由が無いので、その場面の End だけを禁じる。ロールアウトの勝率平均が
    雑音に埋もれても、この手には落ちないようにする。

    もう一つ、ユキワラシとユキメノコをバトル場に出す手も禁じる。この 2 枚は
    ベンチに居ることで仕事をする。減点だけでは探索がひっくり返してしまうため、
    他に出せるものがある限り候補から外す。
    """
    sel = obs.get("select") or {}
    cur = obs.get("current") or {}
    players = cur.get("players") or []
    if len(players) < 2:
        return set()
    mi = cur.get("yourIndex", 0)
    me = players[mi]
    options = sel.get("option") or []

    if sel.get("context") in TO_ACTIVE:
        tier: dict = {}
        for i, o in enumerate(options):
            if o.get("type") != 3:
                continue
            owner = o.get("playerIndex")
            if owner is not None and owner != mi:
                continue  # 相手の場を指す選択 (ボスの指令など) は別の話
            t = bench_tier((_card_of(o, sel, me) or {}).get("id"), me)
            if t:
                tier[i] = t
        # まずは全部外す。それだと出せる相手が居なくなる場合に限って、
        # 損の小さいものから順に戻す
        for lo in (1, 2, 3):
            bad = {i for i, t in tier.items() if t >= lo}
            if bad and len(bad) < len(options):
                return bad
        return set()

    you = players[1 - mi]

    atk = {i for i, o in enumerate(options) if o.get("type") == 13}
    if atk and len(atk) < len(options):
        if pending_before_attack(sel, cur, me, you):
            return atk

    # 手札を山札に戻してから引く札は、出しておきたいものを全部出したあとに
    # 切る。スタジアムを抱えたままアンフェアスタンプを使うと、そのまま
    # 山札に戻ってしまう。自分自身は判定から外す (2 枚あっても同じ)
    shuf = {i for i, o in enumerate(options)
            if o.get("type") == 7
            and (_card_of(o, sel, me) or {}).get("id") in SHUFFLE_DRAW}
    if shuf and len(shuf) < len(options):
        if pending_before_attack(sel, cur, me, you, skip=SHUFFLE_DRAW):
            return shuf

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
        # ベンチが空なのに、埋める手を持ったまま終える。バトル場の 1 体を
        # 落とされた時点で負けるので、たね本体だけでなく、たねをベンチへ出す札と
        # たねを手札に持ってくる札も対象にする
        if empty_bench and ((ptcg.card(cid) or {}).get("basic")
                            or cid in BENCH_REFILL):
            return end
        # 最初の番に、代償の無いサーチ札を抱えたまま終える
        if early and cid in SEARCH_FIRST and not (cid == 1086 and _bench_full(me)):
            return end

    # 必須の枠を潰すベンチ出しは選ばせない。減点だけだと探索がひっくり返す
    tight = set()
    for i, o in enumerate(options):
        if o.get("type") != 7:
            continue
        cid = (_card_of(o, sel, me) or {}).get("id")
        card = ptcg.card(cid) or {}
        if card.get("cardType") == 0 and squeezes_bench(cid, me):
            tight.add(i)
        elif cid == 1086 and squeezes_bench(1086, me, need=2):
            tight.add(i)
    if tight and len(tight) < len(options):
        return tight

    # 明らかに得な手を残したまま番を渡さない。カードは手札にある間は何もしない。
    # リプレイでは、番を終えた 89 局面のうち 38 局面 (43%) が 90 点以上の手を
    # 抱えたままだった。ポケパッド 7、ポフィン 5、ペトレル 5、コゾウ 6 など
    if pending_before_attack(sel, cur, me, you, no_supporter=True):
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
    # 上限は observation の benchMax に入っている。決め打ちにすると、増やす
    # 手段を持つデッキで判断がずれる
    return len([p for p in (me.get("bench") or []) if p]) >= (me.get("benchMax") or 5)


# ループの部品。ユキメノコがダメカンを配る側、マシマシラが運ぶ側で、
# 片方だけ場に居ても何も起きない。まだ場に無いものを最優先で並べる。
# ユキワラシとユキメノコは同じ枠として数える (進化前でもそのうち化ける)
RARE_CANDY = 1079


def search_bonus(cid: int | None, me: dict) -> float:
    """山札やトラッシュから 1 枚取るときの、局面ごとの上乗せ。

    静的な価値表だけで選ぶと、いつでも同じ順で取ってしまう。実際には
    「たねが居ないならポフィン、オーロンゲを作れるならアメ」のように、
    そのとき欠けているものが最優先になる。ここは盤面を見て決める。
    """
    if cid is None:
        return 0.0
    hand = [c["id"] for c in (me.get("hand") or ()) if c]
    play = [x.get("id") for x in features.in_play(me)]
    basics = sum(1 for i in play if (ptcg.card(i) or {}).get("basic"))
    line_base = 646 in play or 649 in play          # 進化元のたね
    candy = RARE_CANDY in hand

    if cid == 1086:                                  # なかよしポフィン
        if _bench_full(me):
            return -60.0
        if 1086 in hand:
            return -20.0                             # 既に 1 枚ある
        return 70.0 if basics <= 1 else 20.0
    if cid == 648:                                   # オーロンゲ ex
        if 648 in hand:
            return -40.0                             # 既に持っている。2 枚目は要らない
        if 647 in play or (candy and line_base):
            return 70.0                              # 今すぐ乗せられる
        if line_base:
            return 25.0
        return 0.0                                   # 進化元が無いと置物
    if cid == 647:                                   # オソマツ
        if 646 in play and not candy:
            return 45.0                              # アメが無いなら手で繋ぐ
        return 5.0
    if cid == 646:                                   # イタズラコゾウ
        if not line_base and 647 not in play and 648 not in play:
            return 55.0                              # ラインが場に無い
        return 5.0
    if cid == 1152:                                  # ポケパッド
        return 35.0 if basics <= 1 else 10.0
    if cid == 1231:                                  # Dawn
        if 648 not in play and (647 not in play or not line_base):
            return 40.0                              # ラインが欠けている
        return 5.0
    if cid == 1219:                                  # ペトレル
        return 20.0                                  # 何にでも化けるので腐りにくい
    if cid == 1097:                                  # ナイトスターチャー
        pile = [c.get("id") for c in (me.get("discard") or ()) if c]
        if any(i in pile for i in (648, 104, 860, 646)):
            return 35.0                              # 勝ち筋の部品が落ちている
        return 0.0
    return 0.0


def candy_bonus(cid: int | None, me: dict) -> float:
    """山札からふしぎなアメを取る価値。

    2 進化を飛ばせるので、オーロンゲ ex の着地が 1 ターン早くなる。ただし
    2 進化が手に入る見込みが無い場面では、ただの死に札である。デッキに 2 枚しか
    無いので、抱えすぎも避ける。
    """
    if cid != RARE_CANDY:
        return 0.0
    hand = [c["id"] for c in (me.get("hand") or ()) if c]
    if hand.count(RARE_CANDY):
        return -30.0   # 既に 1 枚ある。デッキに 2 枚しかないので抱え込まない
    play = [x.get("id") for x in features.in_play(me)]
    if 648 in hand and (646 in play or 649 in play):
        return 70.0   # 今すぐ飛べる
    if 646 in play or 646 in hand:
        return 35.0   # 進化元は用意できている
    return 10.0


# 目指す盤面。バトル場を含めた数で、ベンチ上限 5 なので合計 6 体まで置ける。
#
#   序盤    ギモー 3 / マシマシラ 2 / ユキワラシ 1          合計 6
#   完成後  オーロンゲ ex 1 / マシマシラ 2 / ユキメノコ 2   合計 5
#
# ユキワラシとユキメノコは同じ枠として数える。進化前でもそのうち化ける。
EARLY_TARGET = (((646,), 3), ((112,), 2), ((860, 104), 1))
LATE_TARGET = (((648,), 1), ((112,), 2), ((860, 104), 2))

# 目標に足りないときの加点。埋める順序がこの大小で決まる
WANT_SCORE = {112: 80.0, 860: 70.0, 104: 70.0, 646: 55.0, 648: 60.0}
LOOP_PIECES = frozenset(WANT_SCORE)


# 最低限これだけは場に居てほしい枠。ここが埋まるまでは、ベンチをそのぶん
# 空けておく。埋めてしまうと、後から引いても出せない。
#   ユキメノコの線 (ダメカンを配る)  マシマシラ (運ぶ)  オーロンゲの線 (殴る)
ESSENTIAL = (
    (860, 104),
    (112,),
    (646, 647, 648),
)


def reserved_slots(me: dict) -> int:
    """まだ埋まっていない必須枠の数。このぶんベンチを空けておく。"""
    ids = [x.get("id") for x in features.in_play(me)]
    return sum(1 for g in ESSENTIAL if not any(i in ids for i in g))


def bench_room(me: dict) -> int:
    """ベンチの空き数。"""
    filled = len([p for p in (me.get("bench") or []) if p])
    return max(0, (me.get("benchMax") or 5) - filled)


def fills_essential(cid: int | None, me: dict) -> bool:
    """そのカードが、まだ空いている必須枠を埋めるか。"""
    if cid is None:
        return False
    ids = [x.get("id") for x in features.in_play(me)]
    for g in ESSENTIAL:
        if cid in g and not any(i in ids for i in g):
            return True
    return False


# ポフィンが出せるのは HP70 以下のたね。マシマシラ (HP110) は対象外なので、
# 埋められる必須枠はユキメノコの線とオーロンゲの線の 2 つだけ
POFFIN_FILLS = (860, 646)


def squeezes_bench(cid: int | None, me: dict, need: int = 1) -> bool:
    """それを出すとベンチが足りなくなるか。必須枠のぶんは残す。

    need はそのカードが要求する枠の数。ポフィンは 2 体まとめて出す。
    """
    if fills_essential(cid, me):
        return False
    keep = reserved_slots(me)
    if cid == 1086:
        # ポフィン自身が必須枠を埋められるぶんは、空けておく必要が無い
        keep = max(0, keep - sum(1 for c in POFFIN_FILLS if fills_essential(c, me)))
    return bench_room(me) - need < keep


def board_target(me: dict) -> tuple:
    """今どちらの形を目指すか。オーロンゲ ex が立ったら後半の形に切り替える。"""
    return (LATE_TARGET
            if any(x.get("id") == 648 for x in features.in_play(me))
            else EARLY_TARGET)


def missing_bonus(cid: int | None, me: dict) -> float:
    """目指す盤面に対して、その枠がまだ埋まっていないなら加点する。

    手札から出す場合とサーチの両方で使う。同じ順序で揃えないと、サーチが
    勝ち筋の部品を素通りする。
    """
    if cid not in LOOP_PIECES:
        return 0.0
    ids = [x.get("id") for x in features.in_play(me)]

    # 進化は体数を増やさない。枠が埋まっていても、ユキワラシが立っている限り
    # ユキメノコは取りに行く。ここを枠の判定に混ぜると、ユキワラシ 1 体で
    # 目標を満たしたことになって進化先を一生引かない
    if cid == 104 and 860 in ids and 104 not in ids:
        return 70.0

    for group, want in board_target(me):
        if cid not in group:
            continue
        have = sum(ids.count(g) for g in group)
        if have >= want:
            return -20.0          # もう足りている。これ以上はベンチの無駄
        if group == (860, 104):
            # ユキメノコだけ持ってきても場に出せない。まずユキワラシを取る
            return 70.0 if cid == 860 else 30.0
        return WANT_SCORE[cid]
    return -20.0                  # 今の形では要らない枠


# その札を切ると山札が何枚減るか。ナイトスターチャーはトラッシュから拾うので
# 入れない
DECK_COST = {
    1086: 2,   # Buddy-Buddy Poffin      たね 2 体
    1231: 3,   # Dawn                    たね・1 進化・2 進化を 1 枚ずつ
    1152: 1,   # Poké Pad
    1219: 1,   # Team Rocket's Petrel
    1122: 1,   # Pokégear 3.0
    1259: 1,   # Spikemuth Gym
}

# 手札を山札に戻してから引く札。切った後の山札枚数がそのまま計算できる。
# 山札が薄いほど危ないのではなく、「引く枚数に足りるか」で決まる
SHUFFLE_DRAW = {
    1227: 6,   # Lillie's Determination  サイドがちょうど 6 枚なら 8 枚
    1080: 5,   # Unfair Stamp
}

DECK_EATERS = frozenset(DECK_COST) | frozenset(SHUFFLE_DRAW)


def deck_after(cid: int | None, me: dict) -> int:
    """その札を切った後に山札に残る枚数。"""
    d = me.get("deckCount") or 0
    n = SHUFFLE_DRAW.get(cid)
    if n is None:
        return max(0, d - DECK_COST.get(cid, 0))
    if cid == 1227 and len(me.get("prize") or ()) == 6:
        n = 8
    # 自分自身はトラッシュへ行くので手札から 1 枚引く
    hand = max(0, (me.get("handCount") or len(me.get("hand") or ())) - 1)
    return max(0, d + hand - n)


def _play_bonus(cid: int | None, me: dict, you: dict, turn: int = 0,
                cur: dict | None = None) -> float:
    """局面によって価値が大きく動くカードだけ補正する。"""
    v = _play_bonus_base(cid, me, you, turn) + missing_bonus(cid, me)
    v += _stadium_bonus(cid, cur, me)
    # ベンチを埋め切ると、必須の枠を後から引いても出せない
    if (ptcg.card(cid) or {}).get("cardType") == 0 and squeezes_bench(cid, me):
        v -= 150.0
    elif cid == 1086 and squeezes_bench(1086, me, need=2):
        v -= 150.0   # ポフィンは 2 体まとめて出す
    if cid in DECK_EATERS:
        # 今の残り枚数ではなく、切った後に何枚残るかで判断する。リーリエと
        # スタンプは手札を山札に戻すので、薄い山札でも通ることがある一方、
        # 手札が細いと 1 枚で山札切れまで行く
        v -= 250.0 * features.deck_ruin(deck_after(cid, me))
    return v


STADIUM = frozenset({1259})   # スパイクタウンのジム


def _stadium_bonus(cid: int | None, cur: dict | None, me: dict) -> float:
    """スタジアムは張り替えられる。相手のものが出ているなら、まず剥がす。"""
    if cid not in STADIUM or cur is None:
        return 0.0
    field = cur.get("stadium") or []
    if not field:
        return 45.0   # 場が空いている。スパイクタウンのジムは毎ターン
                      # マリィのポケモンを引けるので、早いほど回数が増える
    owner = (field[0] or {}).get("playerIndex")
    mine = owner == cur.get("yourIndex", 0)
    if mine and (field[0] or {}).get("id") == cid:
        return -80.0  # 同じものを張り直しても何も起きない。手札を捨てるだけ
    return 80.0       # 相手のものを剥がしたうえで自分のが立つ。二重に得


def _play_bonus_base(cid: int | None, me: dict, you: dict, turn: int = 0) -> float:
    bench = [p for p in (me.get("bench") or []) if p]
    hand_ids = [c["id"] for c in (me.get("hand") or []) if c]

    if cid in FREE_SEARCH and turn <= EARLY_TURNS and cid != 1086:
        return 30.0  # 序盤は盤面を作る速度がそのまま勝率になる
    if cid == 1086:  # Buddy-Buddy Poffin
        return 35.0 if not _bench_full(me) else -45.0
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
        # 目の前を落とせるなら引きずり出さない。入れ替えた先が固いと、
        # 取れたはずのきぜつがそのまま消える。リプレイ 90346222 では、
        # アラカザム (HP140) を 180 で落とせる場面で 2 回これをやって、
        # HP200 のフェザンディペティ ex に入れ替わり、攻撃せず番を終えていた
        act = _first(me.get("active"))
        tgt = _first(you.get("active"))
        if act and tgt:
            dmg = features.best_attack(act, me, you)
            if dmg > 0 and dmg >= (tgt.get("hp") or 0):
                return -80.0
        for p in (you.get("bench") or []):
            if p and (p.get("hp") or 0) <= 180:
                return 25.0
        return -10.0
    return 0.0


def picks(obs: dict, rng, eps: float | None = None, jitter: float = 6.0,
          prof: "Profile" = DEFAULT, banned: set | None = None) -> list[int]:
    """この局面で打つ手。eps の確率で一様ランダムに落とす。

    根で使うときは eps=0.0, jitter=0.0 を渡して、雑音を入れずに順位だけで決める。
    banned を渡すと、その index を候補から外す。全部が対象なら無視する。
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

    pool = [i for i in range(n) if not banned or i not in banned] or list(range(n))
    # 同点の候補が並ぶ場面が多いので、乱数を足して順位を崩す
    ranked = sorted(
        pool,
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
    # 山札が尽きると、次の番の最初に引けずにその場で負ける。今までこの式には
    # 山札の項が無く、残り 0 枚の局面と 40 枚の局面が同じ点数だった
    v -= 0.55 * features.deck_ruin_of(me)
    v += 0.55 * features.deck_ruin_of(you)
    return max(-0.7, min(0.7, v))
