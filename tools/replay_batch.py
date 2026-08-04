"""複数のリプレイをまとめて解析する。負けた試合の共通点を探すための道具。

  .venv/bin/python tools/replay_batch.py ~/Downloads/8969*.json --team mutsugi

1 試合ずつ読むなら tools/replay_read.py。こちらは相手デッキ別の成績、基本動作の
取りこぼし、持ち時間の使い方、サイドの取り合いがどのターンで壊れたかを横断で出す。

自分の側は --team で決める。mirror (両側が自分) の試合は相手の傾向を知る材料に
ならないので分けて数える。
"""

import argparse
import collections
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "shared"))

GRIMMSNARL = 648


def archetype(deck: list[int], ptcg) -> str:
    """デッキの見出し。進化の終点と ex だけ並べれば型は区別できる。"""
    c = collections.Counter(deck)
    out = []
    for cid, n in c.most_common():
        card = ptcg.card(cid) or {}
        if card.get("cardType") != 0:
            continue
        if (card.get("hp") or 0) >= 200 or card.get("stage2"):
            out.append(f"{n}x{card.get('name')}")
    return " / ".join(out[:2]) or "?"


def analyze(path: str, team: str, ptcg, policy) -> dict | None:
    with open(path) as f:
        ep = json.load(f)
    steps = ep.get("steps") or []
    names = (ep.get("info") or {}).get("TeamNames") or []
    if team not in names:
        return None
    mine = names.index(team)
    mirror = names.count(team) == 2

    decks = []
    if len(steps) > 1:
        for ag in steps[1]:
            a = ag.get("action")
            if isinstance(a, list) and len(a) == 60:
                decks.append(a)

    turns: dict = {}
    grim = None
    rem = []
    prize_hist = []
    last = None

    for i, step in enumerate(steps):
        ag = step[mine]
        obs = ag.get("observation") or {}
        cur = obs.get("current") or {}
        ps = cur.get("players") or []
        if len(ps) == 2:
            pz = (len(ps[mine].get("prize") or []), len(ps[1 - mine].get("prize") or []))
            if pz != last:
                prize_hist.append((cur.get("turn"), pz))
                last = pz
        if ag.get("status") != "ACTIVE":
            continue
        r = obs.get("remainingOverageTime")
        if r is not None:
            rem.append(r)
        sel = obs.get("select")
        # action は 1 段あとに入る (tools/replay_read.py と同じ)
        nxt = steps[i + 1] if i + 1 < len(steps) else None
        act = nxt[mine].get("action") if nxt else None
        if not sel or not isinstance(act, list) or len(act) == 60:
            continue
        turn = cur.get("turn")
        try:
            atk, ko = policy.attack_options(obs)
        except Exception:
            atk, ko = set(), set()
        t = turns.setdefault(turn, {"atk": 0, "atk_t": 0, "ko": 0, "ko_t": 0})
        got = set(act)
        t["atk"] |= bool(atk)
        t["ko"] |= bool(ko)
        t["atk_t"] |= bool(got & atk)
        t["ko_t"] |= bool(got & ko)
        try:
            if grim is None and GRIMMSNARL in policy.in_play_ids(obs):
                grim = turn
        except Exception:
            pass

    rewards = ep.get("rewards") or [0, 0]
    atk_c = [v for v in turns.values() if v["atk"]]
    ko_c = [v for v in turns.values() if v["ko"]]
    return {
        "file": os.path.basename(path),
        "episode": (ep.get("info") or {}).get("EpisodeId"),
        "mirror": mirror,
        "result": (rewards[mine] or 0),
        "opponent": names[1 - mine] if len(names) > 1 else "?",
        "op_deck": archetype(decks[1 - mine], ptcg) if len(decks) == 2 else "?",
        "turns": len(turns),
        "grim": grim,
        "atk_chance": len(atk_c),
        "atk_taken": sum(1 for v in atk_c if v["atk_t"]),
        "ko_chance": len(ko_c),
        "ko_taken": sum(1 for v in ko_c if v["ko_t"]),
        "time_used": (rem[0] - rem[-1]) if len(rem) > 1 else None,
        "time_left": rem[-1] if rem else None,
        "prize_hist": prize_hist,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--team", required=True)
    args = ap.parse_args()

    import policy
    import ptcg

    files: list[str] = []
    for p in args.paths:
        files.extend(sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p])

    rows = []
    for p in files:
        try:
            r = analyze(p, args.team, ptcg, policy)
        except Exception as e:
            print(f"  ! {os.path.basename(p)}: {type(e).__name__} {e}")
            continue
        if r:
            rows.append(r)
    if not rows:
        sys.exit("該当する試合がない")

    real = [r for r in rows if not r["mirror"]]
    mir = [r for r in rows if r["mirror"]]
    print(f"{len(rows)} 試合 (うち mirror {len(mir)} 試合は相手の傾向には使えない)")

    def line(r):
        res = "勝ち" if r["result"] > 0 else ("負け" if r["result"] < 0 else "分け")
        g = f"T{r['grim']}" if r["grim"] is not None else "立たず"
        return (f"  {r['episode']} {res}  {r['turns']:>2}手番  ex成立 {g:<6} "
                f"攻撃 {r['atk_taken']}/{r['atk_chance']}  きぜつ {r['ko_taken']}/{r['ko_chance']}  "
                f"残り時間 {r['time_left']:.0f}s  {r['op_deck']}")

    print("\n[試合ごと]")
    for r in rows:
        print(line(r))

    if real:
        print("\n[相手デッキ別]")
        by: dict = collections.defaultdict(lambda: [0, 0])
        for r in real:
            e = by[r["op_deck"]]
            e[0] += 1
            e[1] += int(r["result"] > 0)
        for deck, (n, w) in sorted(by.items(), key=lambda t: -t[1][0]):
            print(f"  {n:>3} 戦 {w} 勝  {deck}")

    print("\n[基本動作の合計]")
    a_c = sum(r["atk_chance"] for r in rows)
    a_t = sum(r["atk_taken"] for r in rows)
    k_c = sum(r["ko_chance"] for r in rows)
    k_t = sum(r["ko_taken"] for r in rows)
    if a_c:
        print(f"  攻撃できた手番 {a_c}  うち攻撃した {a_t} ({a_t / a_c:.0%})")
    if k_c:
        print(f"  きぜつを取れた手番 {k_c}  うち取った {k_t} ({k_t / k_c:.0%})")
    g = [r["grim"] for r in rows if r["grim"] is not None]
    if g:
        print(f"  オーロンゲ ex 成立  中央値 T{sorted(g)[len(g) // 2]}  "
              f"立たなかった {sum(1 for r in rows if r['grim'] is None)} 試合")

    t = [r["time_left"] for r in rows if r["time_left"] is not None]
    if t:
        print(f"  持ち時間の残り  中央値 {sorted(t)[len(t) // 2]:.0f}s  "
              f"最小 {min(t):.0f}s  最大 {max(t):.0f}s")

    lost = [r for r in rows if r["result"] < 0]
    if lost:
        print("\n[負けた試合のサイドの動き]  (自分, 相手)")
        for r in lost:
            h = " ".join(f"T{t}:{a}-{b}" for t, (a, b) in r["prize_hist"])
            print(f"  {r['episode']}  {h}")


if __name__ == "__main__":
    main()
