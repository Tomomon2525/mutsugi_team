"""局面を固定長のベクトルにする。評価関数の学習と推論で同じものを使う。

学習側と推論側で別々に特徴を書くと、必ずどこかでずれる。ずれても勝率が少し
落ちるだけなので気づけない。ここ 1 箇所に集約する。

制約が二つある。

  速さ   ロールアウトの打ち切りごとに呼ばれる。1 手の思考で 500 回前後になるので、
         カードテキストの参照のような重い処理は入れない。
  可視性 相手の手札は None で来る (handCount だけ見える)。探索木の中では見える
         こともあるが、見えるときだけ使うと学習時と推論時で分布が変わる。
         どちらでも同じ値になるものだけを使う。

値はおおむね [-1, 1] に収まるよう割ってある。線形モデルの重みが同じ桁に
そろうので、正則化と収束の両方で扱いやすい。
"""

NAMES = (
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
)

N = len(NAMES)


def _side(p: dict) -> tuple:
    """1 プレイヤーぶんの生の集計。2 回書かないためにまとめる。"""
    hp = mx = energy = cnt = 0
    for zone in ("active", "bench"):
        for x in p.get(zone) or []:
            if not x:
                continue
            cnt += 1
            hp += x.get("hp") or 0
            mx += x.get("maxHp") or 0
            energy += len(x.get("energies") or [])

    act = None
    for x in p.get("active") or []:
        if x:
            act = x
            break
    a_hp = (act.get("hp") or 0) / act["maxHp"] if act and act.get("maxHp") else 0.0
    a_en = len(act.get("energies") or []) if act else 0
    a_mx = (act.get("maxHp") or 0) if act else 0

    return (
        len(p.get("prize") or []),
        hp / mx if mx else 0.0,
        cnt,
        p.get("handCount") or len(p.get("hand") or []),
        a_hp,
        a_en,
        energy,
        p.get("deckCount") or 0,
        len(p.get("discard") or []),
        a_mx,
    )


def _clip(v: float) -> float:
    return -1.0 if v < -1.0 else (1.0 if v > 1.0 else v)


def vector(obs: dict, my_index: int) -> list[float]:
    cur = obs.get("current") or {}
    players = cur.get("players") or []
    if len(players) < 2:
        return [0.0] * N
    m = _side(players[my_index])
    o = _side(players[1 - my_index])

    return [
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
