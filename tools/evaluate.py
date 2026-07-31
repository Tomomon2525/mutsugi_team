"""ローカル対戦評価。

例:
  .venv/bin/python tools/evaluate.py agents/baseline random -n 50
  .venv/bin/python tools/evaluate.py agents/baseline agents/other -n 100 -j 8
"""

import argparse
import collections
import concurrent.futures as cf
import os
import sys
import time

BUILTIN = {"random", "first"}

# 提出物では shared/*.py が main.py と同じ階層に置かれるが、ローカル評価では
# エージェントのディレクトリから直接読む。`import ptcg` を両方で通すために
# リポジトリの shared/ を sys.path に足しておく。
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "shared"))


def resolve(spec: str) -> str:
    if spec in BUILTIN:
        return spec
    p = os.path.abspath(spec)
    if os.path.isdir(p):
        p = os.path.join(p, "main.py")
    if not os.path.isfile(p):
        sys.exit(f"agent not found: {spec}")
    return p


def play(args) -> tuple[int, list[str]]:
    """1 試合。戻り値は (player0 視点の reward, 両者の status)。

    エンジンの乱数は native 側にあり seed を固定できない。同じ組み合わせでも
    毎回違う試合になるので、勝率の比較には試合数が要る。
    """
    a0, a1 = args
    from kaggle_environments import make

    env = make("cabt")
    env.run([a0, a1])
    last = env.steps[-1]
    return last[0].reward or 0, [last[0].status, last[1].status]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("agent0")
    ap.add_argument("agent1")
    ap.add_argument("-n", "--games", type=int, default=20)
    ap.add_argument("-j", "--jobs", type=int, default=1, help="並列プロセス数")
    ap.add_argument("--no-swap", action="store_true", help="先後の入れ替えをしない")
    args = ap.parse_args()

    a0, a1 = resolve(args.agent0), resolve(args.agent1)
    jobs = []
    for i in range(args.games):
        swap = (not args.no_swap) and (i % 2 == 1)
        jobs.append((a1, a0) if swap else (a0, a1))

    t0 = time.time()
    if args.jobs > 1:
        with cf.ProcessPoolExecutor(max_workers=args.jobs) as ex:
            raw = list(ex.map(play, jobs))
    else:
        raw = [play(j) for j in jobs]

    # player0 視点の reward を agent0 視点に直す
    results = []
    bad = collections.Counter()
    for i, (r, statuses) in enumerate(raw):
        swap = (not args.no_swap) and (i % 2 == 1)
        results.append(-r if swap else r)
        for side, st in enumerate(statuses):
            if st not in ("DONE", "ACTIVE", "INACTIVE"):
                who = args.agent1 if (swap ^ (side == 1)) else args.agent0
                bad[f"{who}:{st}"] += 1

    w = results.count(1)
    l = results.count(-1)
    d = len(results) - w - l
    rate = w / len(results) if results else 0.0
    print(f"{args.agent0} vs {args.agent1}")
    print(f"  {len(results)} games / {time.time() - t0:.1f}s")
    print(f"  win {w}  lose {l}  draw {d}  -> winrate {rate:.1%}")
    if bad:
        print("  ! 異常終了 (提出すると即敗北になる):")
        for k, n in bad.most_common():
            print(f"      {k}  x{n}")


if __name__ == "__main__":
    main()
