"""負け方を分類する。

敗北条件は 3 つある。
  1. 相手にサイドを取り切られる
  2. 気絶したあと場にポケモンが 1 匹もいない
  3. 自分の番の最初に山札を引けない (山札切れ)

エンジンは終局理由をログに出さないので、終局時の盤面から逆算する。勝者側の
observation には敗者の残りサイド枚数と山札枚数が見えるので、

  残りサイド 6 かつ 山札あり  -> 場のポケモンが尽きた (相手はサイドを 1 枚も取っていない)
  残りサイド 6 かつ 山札 0    -> 山札切れ
  残りサイド 0                -> サイドを取り切られた
  それ以外                    -> 途中で場のポケモンが尽きた or 山札切れ

マリガン (最初の手札にたねポケモンが無くて引き直す) は LogType.HasBasicPokemon
の hasBasicPokemon=False として全ステップのログに残る。

  .venv/bin/python tools/endings.py agents/d0_grimmsnarl agents/d0_grimmsnarl -n 20 -j 6
"""

import argparse
import collections
import concurrent.futures as cf
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "shared"))


def resolve(spec: str) -> str:
    if spec in ("random", "first"):
        return spec
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

    mull = collections.Counter()
    seen = set()
    for st in env.steps:
        for e in st[0]["observation"].get("logs") or []:
            if e.get("type") == 1 and not e.get("hasBasicPokemon"):
                key = (id(st), e.get("playerIndex"), len(seen))
                seen.add(key)
                mull[e.get("playerIndex")] += 1

    out = {"reward": reward, "steps": len(env.steps), "mulligan": dict(mull),
           "statuses": [s["status"] for s in last]}
    if reward != 0:
        win = 0 if reward > 0 else 1
        lose = 1 - win
        players = last[win]["observation"]["current"]["players"]
        # 勝利条件は「自分のサイドを 6 枚取り切る」なので勝者側の残り枚数を見る
        win_left = len(players[win].get("prize") or [])
        lose_left = len(players[lose].get("prize") or [])
        out["loser"] = lose
        out["prize_left"] = win_left
        out["loser_prize_left"] = lose_left
        out["deck_left"] = players[lose].get("deckCount") or 0
        if win_left == 0:
            out["cause"] = "サイド6枚取り切り"
        elif out["deck_left"] == 0:
            out["cause"] = "相手の山札切れ"
        else:
            out["cause"] = f"相手の場のポケモン全滅 (勝者サイド残り{win_left})"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("agent0")
    ap.add_argument("agent1")
    ap.add_argument("-n", "--games", type=int, default=20)
    ap.add_argument("-j", "--jobs", type=int, default=1)
    args = ap.parse_args()

    a0, a1 = resolve(args.agent0), resolve(args.agent1)
    jobs = [(a1, a0) if i % 2 else (a0, a1) for i in range(args.games)]

    causes: collections.Counter = collections.Counter()
    prize_left: collections.Counter = collections.Counter()
    mulligans: collections.Counter = collections.Counter()
    draws = 0
    steps: list[int] = []

    def record(i: int, r: dict) -> None:
        nonlocal draws
        steps.append(r["steps"])
        for c in (r.get("mulligan") or {}).values():
            mulligans[c] += 1
        if r["reward"] == 0:
            draws += 1
            return
        causes[r["cause"]] += 1
        prize_left[r["prize_left"]] += 1

    if args.jobs > 1:
        with cf.ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(play, j): i for i, j in enumerate(jobs)}
            for f in cf.as_completed(futs):
                record(futs[f], f.result())
    else:
        for i, j in enumerate(jobs):
            record(i, play(j))

    n = sum(causes.values())
    print(f"{args.games} 試合 (決着 {n} / 引き分け {draws})  平均 {sum(steps)/len(steps):.0f} step")
    print("\n敗因")
    for c, k in causes.most_common():
        print(f"  {c:<28} {k:>4}  {k/max(1,n):>6.1%}")
    print("\n勝者の残りサイド枚数 (0 = 取り切って勝ち、6 = 一度も取らずに勝ち)")
    for p in sorted(prize_left):
        print(f"  残り{p}枚 {prize_left[p]:>4}  {prize_left[p]/max(1,n):>6.1%}")
    if mulligans:
        print("\nマリガン回数の分布 (プレイヤー単位):", dict(sorted(mulligans.items())))


if __name__ == "__main__":
    main()
