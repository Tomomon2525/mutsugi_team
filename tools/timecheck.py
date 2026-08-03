"""1 試合を実際に回して、エージェントが 1 手ごとに何秒使ったかを測る。

kaggle_environments はエージェントの標準出力を握り潰すので、思考時間は外から
測るしかない。ここでは env.logs に残る duration を読む。

  .venv/bin/python tools/timecheck.py agents/d0_grimmsnarl agents/d0_grimmsnarl -n 3
  PTCG_TIME_POOL=70 .venv/bin/python tools/timecheck.py agents/d0_grimmsnarl ...
"""

import argparse
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("agent0")
    ap.add_argument("agent1")
    ap.add_argument("-n", "--games", type=int, default=1)
    args = ap.parse_args()

    from kaggle_environments import make

    a0, a1 = resolve(args.agent0), resolve(args.agent1)

    for g in range(args.games):
        env = make("cabt")
        env.run([a0, a1])
        for side in (0, 1):
            d = [
                row[side]["duration"]
                for row in env.logs
                if len(row) > side and (row[side] or {}).get("duration")
            ]
            thinking = [x for x in d if x > 0.01]
            if not d:
                print(f"game {g} p{side}: duration が取れない")
                continue
            print(
                f"game {g} p{side}: 手数 {len(d):>3}  思考手 {len(thinking):>3}  "
                f"合計 {sum(d):>7.1f}s  最大 {max(d):>5.2f}s  "
                f"思考平均 {sum(thinking) / max(1, len(thinking)):>5.2f}s"
            )
        print(f"  reward {env.steps[-1][0]['reward']} / status {[s['status'] for s in env.steps[-1]]}")


if __name__ == "__main__":
    main()
