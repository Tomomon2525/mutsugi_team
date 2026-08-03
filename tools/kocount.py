"""どのポケモンが何回きぜつしたかを数える。

エンジンは「きぜつ」のログを出さないので、終局時のトラッシュから逆算する。
このデッキは自分からポケモンを捨てる手段を持たないため、トラッシュにいる
ポケモン = きぜつしたポケモン とみなせる。

  .venv/bin/python tools/kocount.py agents/d0_grimmsnarl agents/d0_grimmsnarl -n 20 -j 6
"""

import argparse
import collections
import concurrent.futures as cf
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "shared"))

import ptcg  # noqa: E402


def resolve(spec: str) -> str:
    p = os.path.abspath(spec)
    if os.path.isdir(p):
        p = os.path.join(p, "main.py")
    if not os.path.isfile(p):
        sys.exit(f"agent not found: {spec}")
    return p


def play(args) -> dict:
    a0, a1 = args
    from kaggle_environments import make

    env = make("cabt")
    env.run([a0, a1])
    last = env.steps[-1]
    reward = last[0].get("reward") or 0
    # 勝者側の observation には両者の盤面が見えている
    view = 0 if reward >= 0 else 1
    players = last[view]["observation"]["current"]["players"]

    ko: collections.Counter = collections.Counter()
    inplay: collections.Counter = collections.Counter()
    for p in players:
        for c in p.get("discard") or []:
            if c and (ptcg.card(c["id"]) or {}).get("cardType") == 0:
                ko[c["id"]] += 1
        for zone in ("active", "bench"):
            for x in p.get(zone) or []:
                if x:
                    inplay[x["id"]] += 1
    return {"ko": dict(ko), "inplay": dict(inplay)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("agent0")
    ap.add_argument("agent1")
    ap.add_argument("-n", "--games", type=int, default=20)
    ap.add_argument("-j", "--jobs", type=int, default=1)
    args = ap.parse_args()

    a0, a1 = resolve(args.agent0), resolve(args.agent1)
    jobs = [(a1, a0) if i % 2 else (a0, a1) for i in range(args.games)]

    ko: collections.Counter = collections.Counter()
    survive: collections.Counter = collections.Counter()

    def record(r: dict) -> None:
        ko.update(r["ko"])
        survive.update(r["inplay"])

    if args.jobs > 1:
        with cf.ProcessPoolExecutor(max_workers=args.jobs) as ex:
            for f in cf.as_completed([ex.submit(play, j) for j in jobs]):
                record(f.result())
    else:
        for j in jobs:
            record(play(j))

    total_ko = sum(ko.values())
    print(f"{args.games} 試合 / きぜつ計 {total_ko} 体 (両者合計)\n")
    print(f"{'カード':<28}{'きぜつ':>6}{'割合':>8}{'終局時に場':>12}{'1試合あたり':>12}")
    ids = set(ko) | set(survive)
    for cid in sorted(ids, key=lambda c: -ko.get(c, 0)):
        n = ko.get(cid, 0)
        c = ptcg.card(cid)
        ex = " ex" if c and c.get("ex") else ""
        print(f"{(ptcg.name(cid) + ex):<28}{n:>6}{n / max(1, total_ko):>8.1%}"
              f"{survive.get(cid, 0):>12}{n / args.games:>12.2f}")

    # ex は 2 枚、それ以外は 1 枚のサイドを渡す
    prizes = sum(n * (2 if (ptcg.card(cid) or {}).get("ex") else 1) for cid, n in ko.items())
    print(f"\n渡したサイド計 {prizes} 枚 / きぜつ {total_ko} 体 "
          f"= 1 きぜつあたり {prizes / max(1, total_ko):.2f} 枚")


if __name__ == "__main__":
    main()
