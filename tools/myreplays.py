"""公開エピソードから自分の対戦だけを抜き出す。

  .venv/bin/python tools/myreplays.py \
      data/pokemon-tcg-ai-battle-episodes-2026-08-03.zip --team mutsugi \
      --out scratch/replays

Kaggle が日別に配る全対戦ログ (zip のまま読む) を走査し、TeamNames に自分の
チーム名が入っているものを取り出す。自己対戦や手元のリーグと違い、実際に
提出したエージェントが本番の相手と戦った記録なので、負け方の傾向はここにしか出ない。

取り出したエピソードは 1 件 1 ファイルで保存する。相手のデッキ、勝敗、決着ターン、
自分が取ったサイドの数を一覧で出す。
"""

import argparse
import collections
import json
import os
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "shared"))


def archetype(deck: list[int], ptcg) -> str:
    """デッキを人間が読める名前にする。進化の終点だけ並べれば区別が付く。"""
    counts = collections.Counter(deck)
    names = []
    for cid, n in counts.most_common():
        c = ptcg.card(cid) or {}
        if c.get("cardType") != 0:  # Pokemon 以外は無視
            continue
        if (c.get("hp") or 0) >= 200 or c.get("stage2"):
            names.append(f"{n}x{c.get('name')}")
    return " / ".join(names[:3]) or "?"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("zip_path")
    ap.add_argument("--team", required=True)
    ap.add_argument("--out", default="scratch/replays")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save", action="store_true", default=True)
    args = ap.parse_args()

    import ptcg

    os.makedirs(args.out, exist_ok=True)
    found = 0
    rows = []

    with zipfile.ZipFile(args.zip_path) as zf:
        entries = [n for n in zf.namelist() if n.endswith(".json")]
        print(f"{len(entries)} 件を走査する")
        for k, name in enumerate(entries, 1):
            if args.limit and found >= args.limit:
                break
            if k % 20000 == 0:
                print(f"  {k}/{len(entries)}  該当 {found}")
            try:
                with zf.open(name) as f:
                    raw = f.read()
            except Exception:
                continue
            # 全件を json.loads すると遅い。チーム名は生のバイト列で先に弾く
            if args.team.encode() not in raw:
                continue
            try:
                ep = json.loads(raw)
            except Exception:
                continue
            names = (ep.get("info") or {}).get("TeamNames") or []
            if args.team not in names:
                continue
            mine = names.index(args.team)

            steps = ep.get("steps") or []
            decks = []
            if len(steps) > 1:
                for agent in steps[1]:
                    a = agent.get("action")
                    if isinstance(a, list) and len(a) == 60:
                        decks.append(a)
            rewards = ep.get("rewards") or [0, 0]
            r = rewards[mine] or 0

            # 最終局面から残りサイドを読む。observation は各段の 0 番に入る
            prizes = None
            for st in reversed(steps):
                obs = (st[0] or {}).get("observation") or {}
                cur = obs.get("current") or {}
                ps = cur.get("players")
                if ps and len(ps) == 2:
                    prizes = [len(p.get("prize") or []) for p in ps]
                    break

            found += 1
            rows.append({
                "file": name,
                "mine": mine,
                "result": "勝ち" if r > 0 else ("負け" if r < 0 else "分け"),
                "steps": len(steps),
                "prize_mine": prizes[mine] if prizes else None,
                "prize_op": prizes[1 - mine] if prizes else None,
                "opponent": names[1 - mine],
                "op_deck": archetype(decks[1 - mine], ptcg) if len(decks) == 2 else "?",
            })
            if args.save:
                out = os.path.join(args.out, f"{found:04d}_{rows[-1]['result']}.json")
                with open(out, "wb") as f:
                    f.write(raw)

    if not rows:
        sys.exit(f"'{args.team}' の対戦が見つからない")

    w = sum(1 for r in rows if r["result"] == "勝ち")
    l = sum(1 for r in rows if r["result"] == "負け")
    d = len(rows) - w - l
    print(f"\n{len(rows)} 戦  {w}勝 {l}敗 {d}分  勝率 {w / len(rows):.1%}")
    print(f"{args.out} に保存した")

    by_deck: dict = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        e = by_deck[r["op_deck"]]
        e[0] += 1
        e[1] += int(r["result"] == "勝ち")
    print("\n[相手デッキ別]")
    for deck, (n, win) in sorted(by_deck.items(), key=lambda t: -t[1][0]):
        print(f"  {n:>4} 戦  勝率 {win / n:5.1%}  {deck}")

    print("\n[サイドの残り] 負けた試合で相手に何枚残していたか")
    left = collections.Counter(r["prize_op"] for r in rows if r["result"] == "負け")
    for k in sorted(x for x in left if x is not None):
        print(f"  残り {k} 枚  {left[k]} 戦")

    with open(os.path.join(args.out, "index.json"), "w") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
