"""公開エピソードからデッキ相性を集計する。

deck_census.py がデッキ単位で数えるのに対し、こちらはアーキタイプ単位でまとめ、
どのアーキタイプがどれに強いかの行列を出す。

アーキタイプはポケモンの構成で決める。トレーナーズやエネルギーの枚数は人によって
数枚違うが、ポケモンの並びが同じなら同じ型とみなしてよい。

  .venv/bin/python tools/deck_matchup.py data/pokemon-tcg-ai-battle-episodes-2026-07-30.zip
  .venv/bin/python tools/deck_matchup.py <zip> --top 8 --limit 2000
"""

import argparse
import collections
import csv
import json
import os
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "shared"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from deck_census import extract  # noqa: E402


def archetype(deck: list[int], ptcg) -> tuple:
    """ポケモンの構成をアーキタイプの鍵にする。"""
    counts = collections.Counter(deck)
    mons = {c: n for c, n in counts.items() if (ptcg.card(c) or {}).get("cardType") == 0}
    return tuple(sorted(mons.items()))


def label(key: tuple, ptcg) -> str:
    parts = [f"{n}x{ptcg.name(c)}" for c, n in sorted(key, key=lambda kv: (-kv[1], kv[0]))]
    return " / ".join(parts[:4])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("zip_path")
    ap.add_argument("--top", type=int, default=8, help="行列に載せるアーキタイプ数")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="scratch/matchup.csv")
    args = ap.parse_args()

    import ptcg

    use: collections.Counter = collections.Counter()
    # (自分, 相手) -> [勝ち, 試合]
    head: dict[tuple, list[int]] = collections.defaultdict(lambda: [0, 0])
    best_list: dict[tuple, tuple[int, list[int]]] = {}
    read = 0

    with zipfile.ZipFile(args.zip_path) as zf:
        entries = [n for n in zf.namelist() if n.endswith(".json")]
        for name in entries:
            if args.limit and read >= args.limit:
                break
            try:
                with zf.open(name) as f:
                    ep = json.load(f)
            except Exception:
                continue
            got = extract(ep)
            if got is None:
                continue
            decks, rewards, _ = got
            read += 1
            keys = [archetype(d, ptcg) for d in decks]
            for side in (0, 1):
                k, o = keys[side], keys[1 - side]
                use[k] += 1
                head[(k, o)][1] += 1
                if (rewards[side] or 0) > 0:
                    head[(k, o)][0] += 1
                # そのアーキタイプの代表となるデッキリストを 1 本残す
                cur = best_list.get(k)
                if cur is None or use[k] > cur[0]:
                    best_list[k] = (use[k], decks[side])

    print(f"読めた {read} 件 / アーキタイプ {len(use)} 種")
    tops = [k for k, _ in use.most_common(args.top)]
    total = sum(use.values())

    # 環境全体を相手にしたときの期待勝率 (相手の出現率で重み付け)
    field: dict[tuple, tuple[float, int]] = {}
    for k in tops:
        num = den = 0.0
        n_games = 0
        for o in use:
            w, g = head.get((k, o), [0, 0])
            if g == 0:
                continue
            share = use[o] / total
            num += (w / g) * share
            den += share
            n_games += g
        field[k] = (num / den if den else 0.0, n_games)

    print(f"\n{'使用率':>7} {'対環境':>7} {'試合':>6}  アーキタイプ")
    for k in tops:
        rate, n = field[k]
        print(f"{use[k]/total:>7.1%} {rate:>7.1%} {n:>6}  {label(k, ptcg)[:64]}")

    print("\n=== 相性行列 (行が自分、値は行側の勝率、括弧は試合数) ===")
    width = 13
    print(" " * 20 + "".join(f"{'D'+str(j):>{width}}" for j in range(len(tops))))
    for i, k in enumerate(tops):
        cells = []
        for o in tops:
            w, g = head.get((k, o), [0, 0])
            cells.append(f"{w/g:>7.0%}({g:>3})" if g else f"{'-':>{width}}")
        print(f"D{i:<2} {label(k, ptcg)[:15]:<15}" + "".join(f"{c:>{width}}" for c in cells))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "share", "vs_field", "games", "archetype", "deck"])
        for i, k in enumerate(tops):
            rate, n = field[k]
            deck = best_list[k][1]
            w.writerow([f"D{i}", round(use[k] / total, 4), round(rate, 4), n,
                        label(k, ptcg), ",".join(map(str, sorted(deck)))])
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
