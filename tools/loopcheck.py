"""ユキメノコ + マシマシラのループが実際に回っているかを数える。

  PTCG_SEARCH=0 .venv/bin/python tools/loopcheck.py agents/d4_frost \
      --opponents agents/field/weights.json -n 2000 -j 6

勝率だけを見ても、勝ち筋のどこで止まっているかは分からない。このデッキは
次の 4 つが全部そろって初めて機能する。どこが欠けているかを分けて数える。

  ユキワラシがベンチに出る
  ユキメノコに進化する
  マシマシラに闇エネが 1 個乗る
  毎ターン Adrena-Brain を使い、3 個動かす

バトル場に出してはいけない 2 枚が出てしまった回数も数える。
"""

import argparse
import collections
import concurrent.futures as cf
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "shared"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from collect import load_agent  # noqa: E402

SNORUNT, FROSLASS, MUNKIDORI, GRIMMSNARL, DARK = 860, 104, 112, 648, 7

_STATE: dict = {}


def setup(dirs: tuple) -> dict:
    if "features" not in _STATE:
        import features

        _STATE["features"] = features
        _STATE["mods"] = {}
    for d in dirs:
        if d not in _STATE["mods"]:
            _STATE["mods"][d] = load_agent(d, "a%04x" % (abs(hash(d)) & 0xFFFF))
    return _STATE


def play(job):
    i, ours, opp, swap = job
    st = setup((ours, opp))
    features = st["features"]
    mods = st["mods"]
    from kaggle_environments import make

    my_index = 1 if swap else 0
    s = collections.Counter()
    turns_seen: set = set()
    ability_turns: dict = {}   # turn -> [使える, 使った]

    def wrap(mod):
        def fn(obs):
            sel = obs.get("select")
            if sel is not None:
                cur = obs.get("current") or {}
                if cur.get("yourIndex", 0) == my_index:
                    look(cur, sel)
            act = mod.agent(obs)
            if sel is not None:
                cur = obs.get("current") or {}
                if cur.get("yourIndex", 0) == my_index:
                    after(cur, sel, act)
            return act
        return fn

    def look(cur, sel):
        me = (cur.get("players") or [{}])[my_index]
        turn = cur.get("turn") or 0
        board = list(features.in_play(me))
        ids = [x.get("id") for x in board]
        if turn not in turns_seen:
            turns_seen.add(turn)
            s["turns"] += 1
            if SNORUNT in ids or FROSLASS in ids:
                s["turn_line"] += 1
            if FROSLASS in ids:
                s["turn_froslass"] += 1
            if GRIMMSNARL in ids:
                s["turn_grim"] += 1
            armed = sum(1 for x in board if x.get("id") == MUNKIDORI
                        and any(e == DARK for e in (x.get("energies") or ())))
            s["armed_total"] += armed
            if armed:
                s["turn_armed"] += 1
            act = features.first(me.get("active"))
            if act and act.get("id") in (SNORUNT, FROSLASS):
                s["active_frost"] += 1
            if act and act.get("id") == MUNKIDORI:
                s["active_munki"] += 1

            # ユキメノコが場に出ない理由の切り分け
            hand = [c.get("id") for c in (me.get("hand") or ()) if c]
            disc = [c.get("id") for c in (me.get("discard") or ()) if c]
            if SNORUNT in ids and FROSLASS not in ids:
                s["snorunt_alone"] += 1
                if FROSLASS in hand:
                    s["could_evolve"] += 1   # 手札にいるのに進化していない
            if SNORUNT not in ids and FROSLASS not in ids:
                s["line_absent"] += 1
                if SNORUNT in hand:
                    s["snorunt_in_hand"] += 1  # 手札にいるのに出していない
            if disc.count(FROSLASS) or disc.count(SNORUNT):
                s["line_in_discard"] += 1

        # 進化の選択肢が実際に出ているか。出ていないなら規則で止められている
        # (出したターンには進化できない) 側であって、選ばなかったのではない
        import policy
        for o in sel.get("option") or ():
            if o.get("type") != 9:
                continue
            if (policy._card_of(o, sel, me) or {}).get("id") == FROSLASS:
                s["evolve_offered"] += 1
                break

        # Adrena-Brain が選択肢に出た手番
        for o in sel.get("option") or ():
            if o.get("type") != 10:
                continue
            p = _at(me, o, sel)
            if p is not None and p.get("id") == MUNKIDORI:
                ability_turns.setdefault(turn, [0, 0])[0] = 1
                break

    def after(cur, sel, act):
        turn = cur.get("turn") or 0
        me = (cur.get("players") or [{}])[my_index]
        idx = act if isinstance(act, list) else []
        for i2 in idx:
            opts = sel.get("option") or []
            if not (0 <= i2 < len(opts)):
                continue
            o = opts[i2]
            if o.get("type") == 10:
                p = _at(me, o, sel)
                if p is not None and p.get("id") == MUNKIDORI:
                    ability_turns.setdefault(turn, [0, 0])[1] = 1
            if sel.get("context") == 40 and o.get("type") == 0:
                s["move_calls"] += 1
                s["move_counters"] += int(o.get("number") or 0)

    def _at(me, o, sel=None):
        import policy
        p = policy._in_play(me, o.get("inPlayArea"), o.get("inPlayIndex"))
        if p is None and sel is not None:
            p = policy._card_of(o, sel, me)
        return p

    m0 = mods[opp] if swap else mods[ours]
    m1 = mods[ours] if swap else mods[opp]
    env = make("cabt")
    env.run([wrap(m0), wrap(m1)])
    last = env.steps[-1]
    r = last[0].get("reward") or 0
    ok = all(x["status"] in ("DONE", "ACTIVE", "INACTIVE") for x in last)

    s["games"] = 1
    if ok and r:
        s["decided"] = 1
        s["wins"] = int((0 if r > 0 else 1) == my_index)
    for _, (avail, used) in ability_turns.items():
        s["ab_avail"] += avail
        s["ab_used"] += used
    return s, os.path.basename(opp)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("agent")
    ap.add_argument("--opponents", default=None)
    ap.add_argument("-n", "--games", type=int, default=500)
    ap.add_argument("-j", "--jobs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ours = os.path.abspath(args.agent)
    if args.opponents:
        import random

        field = json.load(open(args.opponents))
        rng = random.Random(args.seed)
        opps = rng.choices([os.path.join(ROOT, w["dir"]) for w in field],
                           weights=[w["share"] for w in field], k=args.games)
    else:
        opps = [ours] * args.games

    jobs = [(i, ours, opps[i], bool(i % 2)) for i in range(args.games)]
    tot = collections.Counter()
    per: dict = {}
    t0 = time.time()

    with cf.ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for s, tag in ex.map(play, jobs):
            tot.update(s)
            p = per.setdefault(tag, collections.Counter())
            p.update(s)

    turns = max(1, tot["turns"])
    dec = max(1, tot["decided"])
    print(f"\n{tot['games']} 戦 / {time.time() - t0:.0f} 秒  "
          f"勝率 {tot['wins'] / dec:.1%}  自分の手番 {tot['turns']} 回\n")
    print("勝ち筋の部品が場にあった手番の割合")
    print(f"  ユキワラシかユキメノコ  {tot['turn_line'] / turns:6.1%}")
    print(f"  ユキメノコ (進化後)     {tot['turn_froslass'] / turns:6.1%}")
    print(f"  オーロンゲ ex           {tot['turn_grim'] / turns:6.1%}")
    print(f"  闇エネの乗ったマシマシラ {tot['turn_armed'] / turns:6.1%}  "
          f"(平均 {tot['armed_total'] / turns:.2f} 体)")
    print("\nAdrena-Brain")
    av = max(1, tot["ab_avail"])
    print(f"  使えた手番 {tot['ab_avail']} ({tot['ab_avail'] / turns:.1%})  "
          f"うち使った {tot['ab_used']} ({tot['ab_used'] / av:.1%})")
    mc = max(1, tot["move_calls"])
    print(f"  動かした個数 平均 {tot['move_counters'] / mc:.2f} / 3  "
          f"({tot['move_calls']} 回)")
    print("\nユキメノコが場にいない手番の内訳")
    print(f"  ユキワラシだけ          {tot['snorunt_alone'] / turns:6.1%}"
          f"  (うち手札にユキメノコ {tot['could_evolve'] / max(1, tot['snorunt_alone']):.1%})")
    print(f"  ラインが場に無い        {tot['line_absent'] / turns:6.1%}"
          f"  (うち手札にユキワラシ {tot['snorunt_in_hand'] / max(1, tot['line_absent']):.1%})")
    print(f"  どちらかがトラッシュ    {tot['line_in_discard'] / turns:6.1%}")

    print(f"  進化の選択肢が出ていた手番  {tot['evolve_offered']}")

    print("\nバトル場に出してはいけないものが出ていた手番")
    print(f"  ユキワラシ・ユキメノコ  {tot['active_frost']} ({tot['active_frost'] / turns:.2%})")
    print(f"  マシマシラ              {tot['active_munki']} ({tot['active_munki'] / turns:.2%})")

    if len(per) > 1:
        print("\n相手デッキごとの勝率")
        for k, c in sorted(per.items(), key=lambda t: -t[1]["decided"]):
            d = max(1, c["decided"])
            print(f"  {k:<34} {c['wins']:>4}/{c['decided']:<5} {c['wins'] / d:6.1%}")


if __name__ == "__main__":
    main()
