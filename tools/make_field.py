"""公開ログの集計から、環境に居るデッキをそのまま対戦相手として書き出す。

  .venv/bin/python tools/make_field.py scratch/decks_0730.csv --top 15

agents/field/<番号>_<名前>/ に deck.csv と main.py を置き、採用率を weights.json に
まとめる。中身の方策は自分と同じものが動くので、本物のチームほど上手くは回らない。
それでも、アラカザムやイワパレスを一度も見ずに学習するよりはましである。

自分と同型のデッキも重みごと残す。7/30 の環境では過半がオーロンゲ系で、ミラーを
落とすと分布のほうが歪む。
"""

import argparse
import csv
import json
import os
import re
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "shared"))


def slug(pokemon: str, i: int) -> str:
    """先頭 2 種類のポケモン名からディレクトリ名を作る。"""
    names = []
    for part in pokemon.split(" / ")[:2]:
        n = re.sub(r"^\d+x", "", part).strip()
        n = re.sub(r"[^A-Za-z0-9]+", "", n).lower()
        if n:
            names.append(n[:14])
    return f"{i:02d}_" + ("_".join(names) or "deck")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("census", help="tools/deck_census.py の出力 CSV")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--out", default="agents/field")
    ap.add_argument("--template", default="agents/d3_candy/main.py")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(os.path.join(ROOT, args.census))))
    rows.sort(key=lambda r: -int(r["games"]))
    top = rows[: args.top]
    total = sum(int(r["games"]) for r in top)

    out_root = os.path.join(ROOT, args.out)
    os.makedirs(out_root, exist_ok=True)
    weights = []

    for i, r in enumerate(top, 1):
        deck = [int(x) for x in r["deck"].split(",")]
        if len(deck) != 60:
            print(f"! {i} 番は {len(deck)} 枚。飛ばす")
            continue
        name = slug(r["pokemon"], i)
        d = os.path.join(out_root, name)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "deck.csv"), "w") as f:
            f.write("\n".join(str(c) for c in deck) + "\n")
        shutil.copy(os.path.join(ROOT, args.template), os.path.join(d, "main.py"))
        weights.append({
            "dir": os.path.join(args.out, name),
            "games": int(r["games"]),
            "share": int(r["games"]) / total,
            "winrate": float(r["winrate"]),
            "pokemon": r["pokemon"],
        })

    path = os.path.join(out_root, "weights.json")
    with open(path, "w") as f:
        json.dump(weights, f, ensure_ascii=False, indent=1)

    print(f"{len(weights)} デッキを {out_root} に書いた (上位 {args.top} で "
          f"公開ログの {total} 戦ぶん)")
    for w in weights:
        print(f"  {w['share']:5.1%}  {w['dir']:<34} {w['pokemon'][:56]}")
    print(f"\n重み: {path}")


if __name__ == "__main__":
    main()
