"""1 試合を回して HTML リプレイを書き出す。cabt には公式ビジュアライザが同梱されている。

  .venv/bin/python tools/replay.py agents/baseline random -o scratch/replay.html
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import resolve  # noqa: E402  (shared/ の sys.path 追加も evaluate 側で行われる)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("agent0")
    ap.add_argument("agent1")
    ap.add_argument("-o", "--out", default="scratch/replay.html")
    args = ap.parse_args()

    from kaggle_environments import make

    env = make("cabt")
    env.run([resolve(args.agent0), resolve(args.agent1)])

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(env.render(mode="html", width=1000, height=700))

    last = env.steps[-1]
    print(f"statuses={[a.status for a in last]} rewards={[a.reward for a in last]}")
    print(f"wrote {out}")
    print(f"  open {out}")


if __name__ == "__main__":
    main()
