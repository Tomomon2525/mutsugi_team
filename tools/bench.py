"""並列数ごとの実行速度と思考量を測る。大量対戦をどこで回すかを決めるため。

戦/時だけで判断してはいけない。思考時間は壁時計で切っているので、コア数を超えて
プロセスを詰め込んでも 1 戦の所要時間はあまり延びない。代わりに各エージェントが
同じ時間で回せるロールアウト数が減る。つまり戦/時は上がるが、エージェントは弱くなる。

そこで PTCG_TRACE を使い、1 手あたりのロールアウト数も同時に測る。
**思考量を保った上での戦/時**で判断すること。

  .venv/bin/python tools/bench.py -j 1,2,4,6,8 -n 6

ここで測るのは実行速度であって対戦の強さではないので、PTCG_* は短めでよい。
ただし並列数ごとに同じ設定で回すこと。
"""

import argparse
import concurrent.futures as cf
import glob
import json
import os
import sys
import tempfile
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
    print(f"{'並列数':>6}{'所要秒':>10}{'戦/時':>9}{'1戦あたり秒':>13}"
          f"{'ロールアウト/手':>17}{'思考量の維持率':>16}")

    rows = []
    base_rollouts = None
    for j in [int(x) for x in args.jobs.split(",") if x.strip()]:
        tdir = tempfile.mkdtemp(prefix=f"bench_j{j}_")
        os.environ["PTCG_TRACE"] = os.path.join(tdir, "t.jsonl")
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

        # 探索した手だけを対象に、1 手あたりのロールアウト数を平均する
        tot = cnt = 0
        for path in glob.glob(os.path.join(tdir, "t.jsonl.*")):
            for line in open(path):
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("searched"):
                    tot += r.get("rollouts") or 0
                    cnt += 1
        ro = tot / cnt if cnt else 0.0
        if base_rollouts is None:
            base_rollouts = ro
        keep = ro / base_rollouts if base_rollouts else 0.0
        rows.append((j, per_hour, ro, keep))
        print(f"{j:>6}{dt:>10.1f}{per_hour:>9.0f}{dt / args.games:>13.1f}"
              f"{ro:>17.0f}{keep:>15.0%}")
    os.environ.pop("PTCG_TRACE", None)

    # 思考量が並列1の 90% を下回る条件は、速いのではなくエージェントが弱いだけ
    ok = [r for r in rows if r[3] >= 0.90]
    if ok:
        b = max(ok, key=lambda r: r[1])
        print(f"\n思考量を保てる範囲での最良: 並列 {b[0]} で {b[1]:.0f} 戦/時 "
              f"(1 手あたり {b[2]:.0f} ロールアウト)")
    else:
        print("\nどの並列数でも思考量が落ちている。並列 1 で測り直すこと。")
    bad = [r for r in rows if r[3] < 0.90]
    if bad:
        print("  次の条件は戦/時が上がっても思考量が落ちている: "
              + ", ".join(f"並列{r[0]} ({r[3]:.0%})" for r in bad))


if __name__ == "__main__":
    main()
