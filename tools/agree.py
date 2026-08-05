"""本番のリプレイで、方策の 1 位と実際の選択がどれだけ食い違ったかを数える。

  .venv/bin/python tools/agree.py
  .venv/bin/python tools/agree.py --team mutsugi

探索は方策の順位を事前分布として使うだけで、最後はロールアウトの勝率平均で
決める。ゆっくり効く手 (ユキメノコの進化、ふしぎなアメ、Punk Up の付け先) は
打ち切りまでのロールアウトに現れないため、方策が 1 位に置いても覆される。
個別に must_take で塞いできたが、どこで何回起きているかを一度まとめて見る。

食い違いが多い場所が、次に塞ぐべき候補になる。逆に食い違いが少ない場所は、
探索と方策が同じ結論に達しているので、手を入れても変わらない。
"""

import argparse
import collections
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "shared"))

import enums  # noqa: E402
import policy  # noqa: E402
import ptcg  # noqa: E402

TYPE_NAME = {
    0: "Number", 3: "Card", 6: "Energy", 7: "Play", 8: "Attach", 9: "Evolve",
    10: "Ability", 12: "Retreat", 13: "Attack", 14: "End", 1: "Yes", 2: "No",
}


def ctx_name(c) -> str:
    try:
        return enums.SelectContext(c).name
    except Exception:
        return str(c)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", default=None, help="この名前の側だけ数える")
    ap.add_argument("--top", type=int, default=14)
    args = ap.parse_args()

    by_type: dict = collections.defaultdict(lambda: [0, 0])
    by_ctx: dict = collections.defaultdict(lambda: [0, 0])
    lost: collections.Counter = collections.Counter()
    n_dec = n_same = 0

    for path in sorted(glob.glob(os.path.join(ROOT, "Replay", "*.json"))):
        rep = json.load(open(path))
        steps = rep["steps"]
        names = [(rep.get("info") or {}).get("TeamNames", [None, None])[i]
                 for i in (0, 1)]

        for si, st in enumerate(steps):
            for side in (0, 1):
                if args.team and names[side] and args.team not in names[side]:
                    continue
                obs = st[side].get("observation") or {}
                sel = obs.get("select")
                if not sel:
                    continue
                opts = sel.get("option") or []
                if len(opts) < 2:
                    continue
                if int(sel.get("maxCount") or 0) != 1:
                    continue
                cur = obs.get("current") or {}
                ps = cur.get("players") or []
                if len(ps) < 2:
                    continue
                mi = cur.get("yourIndex", 0)
                me, you = ps[mi], ps[1 - mi]
                nxt = steps[si + 1] if si + 1 < len(steps) else None
                act = nxt[side].get("action") if nxt else None
                if not act:
                    continue

                best = max(range(len(opts)),
                           key=lambda i: policy.score(opts[i], sel, cur, me, you))
                same = best in act
                n_dec += 1
                n_same += same

                t = opts[best].get("type")
                by_type[t][0] += 1
                by_type[t][1] += same
                c = sel.get("context")
                by_ctx[c][0] += 1
                by_ctx[c][1] += same

                if not same:
                    card = policy._card_of(opts[best], sel, me)
                    nm = (ptcg.card((card or {}).get("id")) or {}).get("name")
                    lost[(TYPE_NAME.get(t, t), nm or "-")] += 1

    print(f"選択肢が 2 つ以上ある決定 {n_dec}、方策の 1 位が選ばれた {n_same} "
          f"({n_same / max(1, n_dec):.0%})\n")

    print("選択肢の種類ごと")
    for t, (n, ok) in sorted(by_type.items(), key=lambda kv: -kv[1][0]):
        print(f"  {TYPE_NAME.get(t, t):<10} {ok:>4}/{n:<5} {ok / n:5.0%}")

    print("\n文脈ごと")
    for c, (n, ok) in sorted(by_ctx.items(), key=lambda kv: -kv[1][0])[:10]:
        print(f"  {ctx_name(c):<24} {ok:>4}/{n:<5} {ok / n:5.0%}")

    print(f"\n覆された 1 位 (上位 {args.top})")
    for (t, nm), v in lost.most_common(args.top):
        print(f"  {v:>4}  {t:<9} {nm}")


if __name__ == "__main__":
    main()
