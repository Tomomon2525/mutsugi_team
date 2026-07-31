"""1 試合を回して observation を JSONL に落とす。select.type / context / option.type
などの整数 Enum が何を指すのかを、実データから逆引きするための道具。

  .venv/bin/python tools/probe_obs.py --out scratch/obs.jsonl
  .venv/bin/python tools/probe_obs.py --summary
"""

import argparse
import collections
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "agents", "baseline"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="observation を書き出す JSONL パス")
    ap.add_argument("--summary", action="store_true", help="select.type / context の出現数を集計")
    ap.add_argument("-n", "--games", type=int, default=1, help="回す試合数")
    args = ap.parse_args()

    from kaggle_environments import make
    from main import DECK

    records: list[dict] = []
    counter: collections.Counter = collections.Counter()

    def probe_agent(obs: dict) -> list[int]:
        if obs.get("select") is None:
            return list(DECK)
        sel = obs["select"]
        records.append({"select": sel, "current": obs.get("current"), "logs": obs.get("logs")})
        counter[(sel.get("type"), sel.get("context"))] += 1
        n = len(sel["option"])
        hi = min(int(sel.get("maxCount") or 0), n)
        return random.sample(range(n), hi)

    for _ in range(args.games):
        env = make("cabt")
        env.run([probe_agent, probe_agent])

    print(f"games: {args.games}  selections: {len(records)}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {args.out}")

    if args.summary:
        print(f"{'select.type':>12} {'context':>8} {'count':>6}")
        for (t, c), n in counter.most_common():
            print(f"{t:>12} {c:>8} {n:>6}")


if __name__ == "__main__":
    main()
