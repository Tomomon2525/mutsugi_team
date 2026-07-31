"""1 試合を回して observation を観察する。

  .venv/bin/python tools/probe_obs.py --summary -n 5 --out scratch/obs.jsonl
  .venv/bin/python tools/probe_obs.py --explain 20        # 選択肢を人間可読で表示
"""

import argparse
import collections
import copy
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "shared"))
sys.path.insert(0, os.path.join(ROOT, "agents", "baseline"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="observation を書き出す JSONL パス")
    ap.add_argument("--summary", action="store_true", help="select.type / context の出現数を集計")
    ap.add_argument("--explain", type=int, default=0, metavar="N", help="先頭 N 個の選択場面を人間可読で表示")
    ap.add_argument("-n", "--games", type=int, default=1, help="回す試合数")
    args = ap.parse_args()

    from kaggle_environments import make
    from main import DECK

    import ptcg

    records: list[dict] = []
    counter: collections.Counter = collections.Counter()
    samples: list[dict] = []
    explained = 0

    def probe_agent(obs: dict) -> list[int]:
        nonlocal explained
        if obs.get("select") is None:
            return list(DECK)
        sel = obs["select"]
        records.append({"select": sel, "current": obs.get("current"), "logs": obs.get("logs")})
        counter[(sel.get("type"), sel.get("context"))] += 1
        if explained < args.explain:
            # kaggle_environments はエージェント実行中の出力を横取りするので、ここでは溜めるだけ。
            # observation は各ステップで作り直されるが、念のため複製しておく。
            explained += 1
            samples.append(copy.deepcopy({"select": sel, "current": obs["current"]}))
        n = len(sel["option"])
        hi = min(int(sel.get("maxCount") or 0), n)
        return random.sample(range(n), hi)

    for _ in range(args.games):
        env = make("cabt")
        env.run([probe_agent, probe_agent])

    for i, s_ in enumerate(samples, 1):
        cur = s_["current"]
        print(f"\n--- #{i}  turn={cur['turn']} player={cur['yourIndex']} ---")
        print(ptcg.describe_select(s_["select"], {"current": cur}))

    print(f"\ngames: {args.games}  selections: {len(records)}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {args.out}")

    if args.summary:
        print(f"{'select.type':>18} {'context':>20} {'count':>6}")
        for (t, c), n in counter.most_common():
            st = ptcg.enum_name(ptcg.SelectType, t)
            sc = ptcg.enum_name(ptcg.SelectContext, c)
            print(f"{st:>18} {sc:>20} {n:>6}")


if __name__ == "__main__":
    main()
