"""相手デッキの推定が、実際のリプレイでどこまで当たるかを測る。

  .venv/bin/python tools/scoutcheck.py

リプレイには両者のデッキリストが最初の action として入っているので、正解が
分かる。ターンごとに、推定したリストが正解と何枚一致するかを出す。

「どのデッキか当てる」よりも「山札の中身をどれだけ正しく決め打てるか」が
本題なので、集計はデッキ名の一致ではなく枚数の一致で取る。集計に無い変種を
引いたときも、近いリストを当てられていれば決定化としては十分に働く。
"""

import collections
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "shared"))

import scout  # noqa: E402


def overlap(a: list[int], b: list[int]) -> int:
    ca, cb = collections.Counter(a), collections.Counter(b)
    return sum(min(n, cb.get(c, 0)) for c, n in ca.items())


def main() -> None:
    rows = scout.table()
    print(f"推定表 {len(rows)} 件\n")

    by_turn: dict = {}
    for p in sorted(glob.glob(os.path.join(ROOT, "Replay", "*.json"))):
        d = json.load(open(p))
        steps = d["steps"]
        truth = {}
        for side in (0, 1):
            a = steps[1][side].get("action")
            if isinstance(a, list) and len(a) == 60:
                truth[side] = a
        if len(truth) < 2:
            continue

        seen_turn: set = set()
        last = None
        for st in steps:
            for side in (0, 1):
                obs = st[side].get("observation") or {}
                cur = obs.get("current") or {}
                ps = cur.get("players") or []
                if not obs.get("select") or len(ps) < 2:
                    continue
                mi = cur.get("yourIndex", 0)
                you = ps[1 - mi]
                turn = cur.get("turn") or 0
                key = (side, turn)
                if key in seen_turn:
                    continue
                seen_turn.add(key)
                deck, conf, name = scout.guess(you)
                if deck is None:
                    continue
                n = overlap(deck, truth[1 - mi])
                base = overlap(truth[mi], truth[1 - mi])  # 自分のデッキを仮定した場合
                r = by_turn.setdefault(min(turn, 20), [0, 0, 0, 0.0])
                r[0] += 1
                r[1] += n
                r[2] += base
                r[3] += conf
                last = (os.path.basename(p), turn, name, conf, n, base)
        if last:
            print(f"  {last[0]}  最終 T{last[1]}  推定 {last[2]}  "
                  f"確信度 {last[3]:.2f}  一致 {last[4]}/60 (自分のデッキなら {last[5]}/60)")

    print("\nターン別  (60 枚のうち何枚が正しく当たるか)")
    print("  turn   件数   推定  自分のデッキ仮定   確信度")
    for t in sorted(by_turn):
        n, hit, base, conf = by_turn[t]
        print(f"  {t:>4} {n:>6} {hit / n:>6.1f} {base / n:>13.1f} {conf / n:>12.2f}")


if __name__ == "__main__":
    main()
