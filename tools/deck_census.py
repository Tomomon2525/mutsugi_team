"""公開エピソードからデッキを集計する。

Kaggle が日別に公開している対戦ログ (zip、1 日あたり展開後 20GB 超) を、展開せずに
読みながら、両プレイヤーのデッキリストと勝敗を数える。cabt ではデッキは各エージェントの
最初のアクションとして記録されるため、そこから取り出せる。

  .venv/bin/python tools/deck_census.py data/pokemon-tcg-ai-battle-episodes-2026-07-30.zip
  .venv/bin/python tools/deck_census.py <zip> --limit 500 --out scratch/decks.csv

公式パッケージと同様、ログの中身はリポジトリに置かない。data/ と scratch/ は .gitignore 済み。
"""

import argparse
import collections
import csv
import hashlib
import json
import os
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "shared"))


def deck_key(deck: list[int]) -> str:
    """並び順に依存しないデッキの識別子。"""
    return hashlib.sha1(",".join(map(str, sorted(deck))).encode()).hexdigest()[:12]


def extract(episode: dict) -> tuple[list[list[int]], list[float], list[str]] | None:
    """1 エピソードから (両者のデッキ, 報酬, チーム名) を取り出す。

    デッキは steps[1][i]["action"] に 60 枚の card ID として入る。cabt では初手で
    デッキを返す仕様なので、そこが最初のアクションになる。
    報酬とチーム名はトップレベルにある。エピソードにレーティングは含まれない。
    """
    steps = episode.get("steps")
    if not steps or len(steps) < 2:
        return None

    decks: list[list[int]] = []
    for agent in steps[1]:
        a = agent.get("action")
        if isinstance(a, list) and len(a) == 60 and all(isinstance(x, int) for x in a):
            decks.append(a)
    if len(decks) != 2:
        return None

    rewards = episode.get("rewards") or [None, None]
    names = (episode.get("info") or {}).get("TeamNames") or ["", ""]
    return decks, rewards, names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("zip_path")
    ap.add_argument("--limit", type=int, default=0, help="読むエピソード数の上限 (0 で全部)")
    ap.add_argument("--out", default="scratch/decks.csv")
    ap.add_argument("--inspect", action="store_true", help="最初の 1 件の構造だけ表示して終了")
    args = ap.parse_args()

    import ptcg

    stats: dict[str, dict] = {}
    read = skipped = 0

    with zipfile.ZipFile(args.zip_path) as zf:
        entries = [n for n in zf.namelist() if n.endswith(".json")]
        print(f"{len(entries)} 件のエピソード")

        if args.inspect:
            with zf.open(entries[0]) as f:
                ep = json.load(f)
            print("トップレベルのキー:", list(ep.keys()))
            for k, v in ep.items():
                if k == "steps":
                    print(f"  steps: {len(v)} 段  先頭要素の型 {type(v[0])}  長さ {len(v[0])}")
                    print(f"  steps[0][0] キー: {list(v[0][0].keys())}")
                    if len(v) > 1:
                        a = v[1][0].get("action")
                        print(f"  steps[1][0].action: 型 {type(a)} 長さ {len(a) if isinstance(a, list) else '-'}")
                else:
                    print(f"  {k}: {json.dumps(v, ensure_ascii=False)[:200]}")
            return

        for name in entries:
            if args.limit and read >= args.limit:
                break
            try:
                with zf.open(name) as f:
                    ep = json.load(f)
            except Exception:
                skipped += 1
                continue
            got = extract(ep)
            if got is None:
                skipped += 1
                continue
            decks, rewards, team_names = got
            read += 1
            for side in (0, 1):
                key = deck_key(decks[side])
                e = stats.setdefault(key, {"deck": decks[side], "n": 0, "win": 0, "teams": collections.Counter()})
                e["n"] += 1
                if (rewards[side] or 0) > 0:
                    e["win"] += 1
                if side < len(team_names) and team_names[side]:
                    e["teams"][team_names[side]] += 1

    print(f"読めた {read} 件 / 読み飛ばし {skipped} 件 / ユニークデッキ {len(stats)} 種")
    if not stats:
        print("デッキを取り出せなかった。--inspect で構造を確認する。")
        return

    rows = []
    for key, e in stats.items():
        counts = collections.Counter(e["deck"])
        top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        pokemon = [f"{n}x{ptcg.name(c)}" for c, n in top if (ptcg.card(c) or {}).get("cardType") == 0]
        rows.append({
            "deck_id": key,
            "games": e["n"],
            "winrate": round(e["win"] / e["n"], 4) if e["n"] else 0,
            "teams": len(e["teams"]),
            "top_team": e["teams"].most_common(1)[0][0] if e["teams"] else "",
            "pokemon": " / ".join(pokemon[:8]),
            "deck": ",".join(map(str, sorted(e["deck"]))),
        })
    rows.sort(key=lambda r: -r["games"])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")

    print(f"\n{'使用数':>6} {'勝率':>6} {'チーム':>5}  主なポケモン")
    for r in rows[:20]:
        print(f"{r['games']:>6} {r['winrate']:>6.1%} {r['teams']:>5}  {r['pokemon'][:66]}")


if __name__ == "__main__":
    main()
