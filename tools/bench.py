"""並列数ごとの実行速度を測る。Mac と Colab のどちらで大量対戦を回すかを決めるため。

判断基準は 1 戦あたりの速度ではなく、1 時間あたりの完了対戦数である。
CPU 数を超えてプロセスを増やすと、かえって遅くなる。

  .venv/bin/python tools/bench.py -j 1,2,4,6,8 -n 6

Colab では PTCG_* を本番相当ではなく短めに設定して構わない。ここで測るのは
エンジンとエージェントの実行速度であって、対戦の強さではない。
"""

import argparse
import concurrent.futures as cf
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "shared"))


def play(pair) -> int:
    from kaggle_environments import make

    env = make("cabt")
    env.run(list(pair))
    return len(env.steps)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-a", "--agent", default="agents/d0_grimmsnarl")
    ap.add_argument("-j", "--jobs", default="1,2,4", help="カンマ区切りの並列数")
    ap.add_argument("-n", "--games", type=int, default=6, help="各並列数で回す対戦数")
    args = ap.parse_args()

    p = os.path.abspath(args.agent)
    if os.path.isdir(p):
        p = os.path.join(p, "main.py")
    if not os.path.isfile(p):
        sys.exit(f"agent not found: {args.agent}")

    try:
        cpu = len(os.sched_getaffinity(0))  # Linux
    except AttributeError:
        cpu = os.cpu_count() or 0
    print(f"論理 CPU {cpu} 個  agent={args.agent}  各条件 {args.games} 戦\n")
    print(f"{'並列数':>6}{'所要秒':>10}{'戦/時':>10}{'1戦あたり秒':>14}")

    best = (0, 0.0)
    for j in [int(x) for x in args.jobs.split(",") if x.strip()]:
        jobs = [(p, p)] * args.games
        t0 = time.time()
        if j > 1:
            with cf.ProcessPoolExecutor(max_workers=j) as ex:
                list(ex.map(play, jobs))
        else:
            for job in jobs:
                play(job)
        dt = time.time() - t0
        per_hour = args.games / dt * 3600
        print(f"{j:>6}{dt:>10.1f}{per_hour:>10.0f}{dt / args.games:>14.1f}")
        if per_hour > best[1]:
            best = (j, per_hour)

    print(f"\n最良: 並列 {best[0]} で {best[1]:.0f} 戦/時")


if __name__ == "__main__":
    main()
