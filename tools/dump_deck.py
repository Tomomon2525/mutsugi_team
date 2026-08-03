"""デッキの全カードを、エンジンが持っているテキストごと吐く。

紙のカードの文面ではなく libcg が実装している内容を見たいときに使う。

  .venv/bin/python tools/dump_deck.py agents/d0_grimmsnarl/deck.csv
"""

import collections
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "shared"))

import ptcg  # noqa: E402
from enums import CardType, EnergyIndex  # noqa: E402


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "agents/d0_grimmsnarl/deck.csv"
    deck = ptcg.load_deck(path)
    counts = collections.Counter(deck)
    print(f"{path}  {len(deck)} 枚 / {len(counts)} 種")

    for cid, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        c = ptcg.card(cid)
        if c is None:
            print(f"\n{n}x 不明なカード id={cid}")
            continue
        head = f"\n{n}x [{ptcg.enum_name(CardType, c['cardType'])}] {c['name']} (id={cid})"
        if c["cardType"] == 0:
            head += (
                f"  HP{c['hp']} {ptcg.enum_name(EnergyIndex, c['energyType'])}"
                f" にげ{c['retreatCost']}"
            )
            if c.get("ex"):
                head += " ex"
            if c.get("evolvesFrom"):
                head += f"  <- {c['evolvesFrom']}"
        print(head)
        for s in c.get("skills") or []:
            print(f"    [特性] {s.get('name', '')}: {s.get('text', '')}")
        for aid in c.get("attacks") or []:
            a = ptcg.attack(aid)
            if not a:
                continue
            cost = "".join(ptcg.enum_name(EnergyIndex, e)[:1] for e in a["energies"]) or "-"
            text = f"  :: {a['text']}" if a.get("text") else ""
            print(f"    [技] {a['name']} dmg={a['damage']} cost={cost}{text}")


if __name__ == "__main__":
    main()
