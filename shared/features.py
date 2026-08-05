"""局面を固定長のベクトルにする。評価関数の学習と推論で同じものを使う。

学習側と推論側で別々に特徴を書くと、必ずどこかでずれる。ずれても勝率が少し
落ちるだけなので気づけない。ここ 1 箇所に集約する。

制約が二つある。

  速さ   ロールアウトの打ち切りごとに呼ばれる。1 手の思考で 500 回前後になるので、
         1 回あたり数十マイクロ秒に収める。カード表の参照は辞書引きなので許容範囲。
  可視性 相手の手札は None で来る (handCount だけ見える)。探索木の中では見える
         こともあるが、見えるときだけ使うと学習時と推論時で分布が変わる。
         相手側の特徴は、公開情報だけで作る。

前半は汎用の盤面統計、後半はオーロンゲデッキ固有の項目である。汎用の 17 個だけで
学習した線形モデルは手書きの式と互角 (400 戦 51.7%) にとどまった。勝ち筋そのものを
表す項目が無かったためで、後半はそこを埋める。

値はおおむね [-1, 1] に収まるよう割ってある。
"""

import ptcg

GRIMMSNARL = 648      # Marnie's Grimmsnarl ex   2 進化、勝ち筋
MORGREM = 647         # Marnie's Morgrem         1 進化
IMPIDIMP = 646        # Marnie's Impidimp        たね
MORPEKO = 649         # Marnie's Morpeko         付いた闇エネ 1 個につき +40
RARE_CANDY = 1079
DARK = 7              # EnergyIndex.Darkness

NAMES = (
    # --- 汎用の盤面統計
    "bias",
    "prize_diff",       # サイドの残り枚数差。勝敗そのものに一番近い
    "hp_ratio_diff",    # 場全体の HP 残存率の差
    "count_diff",       # 場のポケモン数の差
    "my_last_one",      # 自分の場が 1 体。次に取られると負ける
    "op_last_one",
    "hand_diff",
    "act_hp_diff",      # バトル場の HP 残存率の差
    "act_energy_diff",
    "board_energy_diff",
    "deck_diff",        # 山札の残り。切れると負ける
    "discard_diff",
    "act_size_diff",    # バトル場の最大 HP。ex かどうかの代理
    "turn",
    "op_reach",         # 相手のサイドが 1 枚。次で負ける
    "my_reach",
    "to_move",          # 手番が自分なら +1
    # --- 勝ち筋の成立
    "grim_in_play",     # オーロンゲ ex が場にいる
    "grim_active",      # それがバトル場にいる
    "morgrem_in_play",  # 進化元 (モルペコではない方) が場にいる
    "impidimp_in_play",
    "evolve_path",      # 手札から今すぐオーロンゲ ex にできる
    "punk_up_ready",    # その進化で Punk Up が何個の基本闇エネを持ってこられるか
    "punk_up_done",     # 既に撃った形跡。場の闇エネが 4 個以上でオーロンゲ ex がいる
    "line_pieces",      # 進化ラインの部品が手札と場に何種類あるか
    # --- 攻撃の準備
    "grim_can_attack",      # オーロンゲ ex が今のエネルギーで撃てる
    "grim_attack_next",     # あと 1 枚付ければ撃てる
    "morpeko_energy",       # モルペコに付いている闇エネの数。打点に直結する
    "attackers_ready",      # 技を撃てる自分のポケモンの数
    "bench_attacker",       # ベンチに撃てるポケモンがいる (後続がいる)
    # --- 打ち合いの見通し
    "my_ko_next",       # 自分の次の攻撃で相手のバトル場を落とせる
    "op_ko_next",       # 相手の次の攻撃で自分のバトル場を落とされる
    "my_hits",          # 相手のバトル場を落とすのに要る攻撃回数
    "op_hits",
    "fragile_bench",    # 倒されるとサイドを 2 枚以上渡すベンチの数
    # --- 山札切れ
    "my_deck_low",      # 山札の残りが薄いほど 1 に近づく (2 乗して凸にしてある)
    "op_deck_low",
)

N = len(NAMES)


# ---------------------------------------------------------------- 打点

# 相手のバトル場に立つと、条件を満たす攻撃を丸ごと無効にする特性。カードプールを
# "Prevent all damage" で洗って、こちらの攻撃役に刺さるものだけを拾った。
# 判定は攻撃側のカードを見る。ベンチへの飛び火 (Shaymin など) は damage_of が
# バトル場への打点しか返さないので、ここでは扱わない。
def _prevented(atk: dict, dmg: int, target: dict) -> bool:
    tid = (target or {}).get("id")
    if tid is None:
        return False
    is_ex = bool(atk.get("ex") or atk.get("megaEx"))
    if tid in (345, 330):  # Crustle / Sylveon: 相手の {ex} からは受けない
        return is_ex
    if tid == 117:  # Cornerstone Mask Ogerpon ex: 特性を持つポケモンからは受けない
        return bool(atk.get("skills"))
    if tid == 83:  # Farigiraf ex: たねの {ex} からは受けない
        return is_ex and bool(atk.get("basic"))
    if tid == 158:  # Drednaw: 200 以上のダメージを受けない
        return dmg >= 200
    return False


def damage_of(attack_id: int | None, a: dict, poke: dict, me: dict, you: dict) -> int:
    """その技を poke が撃ったときに、相手のバトル場に実際に入る打点。

    カードデータの damage は基礎値なので、そのまま使うと Spiky Wheel (20) のような
    技が最下位に沈む。打点が動く技だけ個別に計算する。

    そのうえで、相手のバトル場が無効化の特性を持っていれば 0 に落とす。ロールアウトは
    本物のエンジンを通るので勝手に 0 になるが、評価関数と方策はこの関数の値で動く。
    通らない相手に殴りかかる手を高く見積もったままだと、探索の入口で候補が歪む。
    """
    dmg = _base_damage(attack_id, a, poke, me, you)
    if dmg > 0 and _prevented(ptcg.card(poke.get("id")) or {}, dmg,
                              first(you.get("active"))):
        return 0
    return dmg


def _base_damage(attack_id: int | None, a: dict, poke: dict,
                 me: dict, you: dict) -> int:
    dmg = a.get("damage") or 0
    if attack_id == 938:  # Spiky Wheel: 付いている {D} 1 個につき +40
        d = sum(1 for e in (poke.get("energies") or []) if e == DARK)
        return dmg + 40 * d
    if attack_id == 120:  # Myriad Leaf Shower: 両者のバトル場のエネルギー 1 個につき +30
        n = 0
        for p in (me, you):
            act = first(p.get("active"))
            n += len(act.get("energies") or []) if act else 0
        return dmg + 30 * n
    if attack_id == 339:  # Psychic (Alakazam): 相手のバトル場のエネルギー 1 個につき +50
        act = first(you.get("active"))
        return dmg + 50 * (len(act.get("energies") or []) if act else 0)
    if attack_id == 1072:  # Powerful Hand (Alakazam): 自分の手札 1 枚につきダメカン 2 個
        return dmg + 20 * (me.get("handCount") or len(me.get("hand") or []))
    if attack_id == 183:  # Cruel Arrow (Fezandipiti ex): 相手 1 体に 100 (ベンチ可)
        return 100
    if attack_id == 980:  # Cosmic Beam (Solrock): ベンチに Lunatone がいなければ不発
        if not any((p or {}).get("id") == 675 for p in (me.get("bench") or [])):
            return 0
        return dmg
    return dmg


def first(seq):
    for x in seq or []:
        if x:
            return x
    return None


def cost_met(attached: list, cost: list, spare: int = 0) -> bool:
    """付いているエネルギーで技のコストを払えるか。

    コストの 0 は無色で、何でもよい。spare は「あと何枚付けられるとみなすか」で、
    次の番に撃てるかを見るときに 1 を渡す。付ける色は選べるので万能札として数える。
    """
    pool = list(attached or [])
    for t in cost or ():
        if t == 0:
            continue
        if t in pool:
            pool.remove(t)
        elif spare > 0:
            spare -= 1
        else:
            return False
    colorless = sum(1 for t in (cost or ()) if t == 0)
    return len(pool) + spare >= colorless


def best_attack(poke: dict, me: dict, you: dict, spare: int = 0) -> int:
    """撃てる技のうち最大の打点。1 つも撃てなければ -1。"""
    if not poke:
        return -1
    card = ptcg.card(poke.get("id")) or {}
    best = -1
    for aid in card.get("attacks") or ():
        a = ptcg.attack(aid)
        if not a:
            continue
        if not cost_met(poke.get("energies"), a.get("energies"), spare):
            continue
        d = damage_of(aid, a, poke, me, you)
        if d > best:
            best = d
    return best


def in_play(p: dict):
    for zone in ("active", "bench"):
        for x in p.get(zone) or ():
            if x:
                yield x


# ---------------------------------------------------------------- 汎用

def _side(p: dict) -> tuple:
    hp = mx = energy = cnt = 0
    for x in in_play(p):
        cnt += 1
        hp += x.get("hp") or 0
        mx += x.get("maxHp") or 0
        energy += len(x.get("energies") or ())

    act = first(p.get("active"))
    a_hp = (act.get("hp") or 0) / act["maxHp"] if act and act.get("maxHp") else 0.0
    a_en = len(act.get("energies") or ()) if act else 0
    a_mx = (act.get("maxHp") or 0) if act else 0

    return (
        len(p.get("prize") or ()),
        hp / mx if mx else 0.0,
        cnt,
        p.get("handCount") or len(p.get("hand") or ()),
        a_hp,
        a_en,
        energy,
        p.get("deckCount") or 0,
        len(p.get("discard") or ()),
        a_mx,
    )


def _clip(v: float) -> float:
    return -1.0 if v < -1.0 else (1.0 if v > 1.0 else v)


# ---------------------------------------------------------------- 固有

# 山札がこの枚数を切ると危ない、という目安。番の最初に引けないとその時点で負ける
DECK_SAFE = 8


def deck_low(p: dict) -> float:
    """山札切れへの近さ。残り 0 で 1.0、DECK_SAFE 以上で 0.0。学習の入力用。

    2 乗して凸にしてある。ここは素直な形にしておき、どれくらい効くかは
    学習に任せる。手書きの判断は deck_ruin のほうを使う。
    """
    d = p.get("deckCount") or 0
    x = max(0.0, (DECK_SAFE - d)) / DECK_SAFE
    return x * x


def deck_ruin(p: dict) -> float:
    """山札切れの危なさ。手書きの評価と方策が使う判断。

    deck_low より下側を鋭くしてある。残り 2 枚まで来ると、引き直す手立てが
    ほぼ無く、勝敗としてはほぼ決まっている。緩やかに増える形だと、探索が
    「あと 1 枚くらい削っても」を繰り返して落ちる。
    """
    d = p.get("deckCount") or 0
    if d >= DECK_SAFE:
        return 0.0
    if d <= 2:
        return 1.0 - 0.12 * d          # 2→0.76  1→0.88  0→1.00
    x = (DECK_SAFE - d) / DECK_SAFE
    return 0.9 * x * x                 # 3→0.35  5→0.13  7→0.01


def _dark_left_in_deck(me: dict) -> int:
    """山札に残っている基本闇エネの推定枚数。

    デッキの 10 枚から、見えている場所にある分を引く。相手の山札は数えられないが、
    この特徴は自分側にしか使わない。
    """
    used = sum(1 for c in (me.get("hand") or ()) if c and c.get("id") == DARK)
    used += sum(1 for c in (me.get("discard") or ()) if c and c.get("id") == DARK)
    for x in in_play(me):
        used += sum(1 for e in (x.get("energies") or ()) if e == DARK)
    return max(0, 10 - used)


def _deck_side(me: dict, you: dict) -> list[float]:
    hand_ids = [c["id"] for c in (me.get("hand") or ()) if c]
    play_ids = [x.get("id") for x in in_play(me)]

    grim_play = GRIMMSNARL in play_ids
    act = first(me.get("active"))
    grim_active = bool(act and act.get("id") == GRIMMSNARL)
    morgrem = MORGREM in play_ids
    impidimp = IMPIDIMP in play_ids

    # 手札のオーロンゲ ex を今すぐ立てられるか。モルペコからは 1 進化を挟むので、
    # ふしぎなアメが要る。Punk Up は山札から闇エネを引くので、山札が尽きたら撃てない。
    # 撃てるか否かではなく、何個持ってこられるかを見る。Punk Up は最大 5 個まで
    # 付けるので、山札に 1 個しか残っていない盤面と 5 個ある盤面では価値が違う。
    path = bool(GRIMMSNARL in hand_ids
                and (morgrem or (impidimp and RARE_CANDY in hand_ids)))
    punk_ready = min(5, _dark_left_in_deck(me)) / 5.0 if path else 0.0

    board_energy = sum(len(x.get("energies") or ()) for x in in_play(me))
    punk_done = bool(grim_play and board_energy >= 4)

    pieces = sum(1 for cid in (IMPIDIMP, MORGREM, GRIMMSNARL)
                 if cid in hand_ids or cid in play_ids)

    grim = next((x for x in in_play(me) if x.get("id") == GRIMMSNARL), None)
    # 0 打点は「撃てる」に数えない。無効化の特性を持つ相手に殴りかかっても
    # 番が終わるだけで、脅威としては成立していない。
    grim_now = bool(grim and best_attack(grim, me, you) > 0)
    grim_next = bool(grim and not grim_now and best_attack(grim, me, you, 1) > 0)

    morpeko_e = 0
    for x in in_play(me):
        if x.get("id") == MORPEKO:
            morpeko_e = max(morpeko_e,
                            sum(1 for e in (x.get("energies") or ()) if e == DARK))

    # ここは >= 0 のまま。マリィのイタズラコゾウの Filch は打点 0 なので、> 0 に
    # すると「エネの付いたコゾウ」が後続から外れる。ミラーでの分布まで変わって
    # 学習済みの評価関数が古くなるため、無効化の対応とは分けて扱う
    ready = sum(1 for x in in_play(me) if best_attack(x, me, you) >= 0)
    bench_ready = any(best_attack(x, me, you) >= 0
                      for x in (me.get("bench") or ()) if x)

    return [
        1.0 if grim_play else 0.0,
        1.0 if grim_active else 0.0,
        1.0 if morgrem else 0.0,
        1.0 if impidimp else 0.0,
        1.0 if path else 0.0,
        punk_ready,
        1.0 if punk_done else 0.0,
        pieces / 3.0,
        1.0 if grim_now else 0.0,
        1.0 if grim_next else 0.0,
        min(5, morpeko_e) / 5.0,
        _clip(ready / 3.0),
        1.0 if bench_ready else 0.0,
    ]


def _trade(me: dict, you: dict) -> list[float]:
    """打ち合いの見通し。何回殴れば落とせるかを両側で見る。"""
    my_act = first(me.get("active"))
    op_act = first(you.get("active"))
    my_dmg = best_attack(my_act, me, you)
    op_dmg = best_attack(op_act, you, me)
    my_hp = (my_act.get("hp") or 0) if my_act else 0
    op_hp = (op_act.get("hp") or 0) if op_act else 0

    def hits(dmg: int, hp: int) -> float:
        if dmg <= 0 or hp <= 0:
            return 1.0          # 撃てない側は「無限回」なので上限に張り付ける
        return _clip((-(-hp // dmg) - 1) / 3.0)

    return [
        1.0 if (my_dmg > 0 and op_hp and my_dmg >= op_hp) else 0.0,
        1.0 if (op_dmg > 0 and my_hp and op_dmg >= my_hp) else 0.0,
        hits(my_dmg, op_hp),
        hits(op_dmg, my_hp),
    ]


# ---------------------------------------------------------------- 本体

def vector(obs: dict, my_index: int) -> list[float]:
    cur = obs.get("current") or {}
    players = cur.get("players") or ()
    if len(players) < 2:
        return [0.0] * N
    me, you = players[my_index], players[1 - my_index]
    m = _side(me)
    o = _side(you)

    out = [
        1.0,
        (o[0] - m[0]) / 6.0,
        m[1] - o[1],
        _clip((m[2] - o[2]) / 3.0),
        1.0 if m[2] <= 1 else 0.0,
        1.0 if o[2] <= 1 else 0.0,
        _clip((m[3] - o[3]) / 6.0),
        m[4] - o[4],
        _clip((m[5] - o[5]) / 4.0),
        _clip((m[6] - o[6]) / 6.0),
        _clip((m[7] - o[7]) / 20.0),
        _clip((o[8] - m[8]) / 20.0),
        (m[9] - o[9]) / 340.0,
        _clip((cur.get("turn") or 0) / 30.0),
        1.0 if o[0] <= 1 else 0.0,
        1.0 if m[0] <= 1 else 0.0,
        1.0 if cur.get("yourIndex") == my_index else -1.0,
    ]
    out.extend(_deck_side(me, you))
    out.extend(_trade(me, you))
    # 倒されるとサイドを 2 枚以上渡すベンチ。ex は 2 枚、メガ ex は 3 枚
    fragile = 0
    for x in (me.get("bench") or ()):
        if not x:
            continue
        c = ptcg.card(x.get("id")) or {}
        if c.get("ex") or c.get("megaEx"):
            fragile += 1
    out.append(_clip(fragile / 3.0))
    out.append(deck_low(me))
    out.append(deck_low(you))
    return out
