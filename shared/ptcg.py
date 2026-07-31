"""カード情報の参照レイヤ。

エンジン (`libcg`) が `AllCard` / `AllAttack` を公開しているので、カードデータは実行時に
取り出せる。CSV や PDF をリポジトリに置く必要はない。Kaggle 上でも同じエンジンが動くため、
提出物に同梱するのはこのファイルだけでよい。

  from ptcg import card, attack, describe_option

`enums.py` は tools/gen_enums.py が生成する。無くても動作するが、数値が名前で出なくなる。
"""

import csv
import ctypes
import json
import os

try:
    from enums import AreaType, CardType, EnergyIndex, LogType, OptionType, SelectContext, SelectType
except ImportError:  # 生成前でも import 自体は通す
    AreaType = CardType = EnergyIndex = LogType = OptionType = SelectContext = SelectType = None


_cards: dict[int, dict] | None = None
_attacks: dict[int, dict] | None = None


def _load() -> None:
    global _cards, _attacks
    if _cards is not None:
        return
    from kaggle_environments.envs.cabt.cg.sim import lib

    lib.AllCard.restype = ctypes.c_char_p
    lib.AllAttack.restype = ctypes.c_char_p
    _cards = {c["cardId"]: c for c in json.loads(lib.AllCard().decode())}
    _attacks = {a["attackId"]: a for a in json.loads(lib.AllAttack().decode())}


def cards() -> dict[int, dict]:
    _load()
    return _cards


def attacks() -> dict[int, dict]:
    _load()
    return _attacks


def card(card_id: int | None) -> dict | None:
    return cards().get(card_id) if card_id is not None else None


def attack(attack_id: int | None) -> dict | None:
    return attacks().get(attack_id) if attack_id is not None else None


def name(card_id: int | None) -> str:
    c = card(card_id)
    return c["name"] if c else f"card#{card_id}"


def enum_name(enum_cls, value) -> str:
    """Enum が未生成でも落ちないようにする。"""
    if enum_cls is None or value is None:
        return str(value)
    try:
        return enum_cls(value).name
    except ValueError:
        return str(value)


# AreaType の値 -> observation の player 辞書のキー
_AREA_FIELD = {1: "deck", 2: "hand", 3: "discard", 4: "active", 5: "bench", 6: "prize"}


def resolve(option: dict, obs: dict, area_key: str = "area", index_key: str = "index") -> dict | None:
    """option が指している盤面上のカードを引く。引けなければ None。

    相手の手札やサイドは observation 上 null なので、その場合も None が返る。
    """
    cur = obs.get("current")
    if not cur:
        return None
    area, index = option.get(area_key), option.get(index_key)
    if area is None or index is None:
        return None
    if area == 7:  # Stadium はプレイヤーではなく盤面直下
        stadium = cur.get("stadium") or []
        return stadium[index] if 0 <= index < len(stadium) else None
    field = _AREA_FIELD.get(area)
    if field is None:
        return None
    pi = option.get("playerIndex")
    pi = cur["yourIndex"] if pi is None else pi
    lst = (cur["players"][pi] or {}).get(field)
    if not isinstance(lst, list) or not (0 <= index < len(lst)):
        return None
    return lst[index]


def hand_card(obs: dict, index: int | None) -> dict | None:
    """自分の手札 index 番目のカード。"""
    cur = obs.get("current")
    if not cur or index is None:
        return None
    hand = (cur["players"][cur["yourIndex"]] or {}).get("hand") or []
    return hand[index] if 0 <= index < len(hand) else None


def describe_option(option: dict, obs: dict) -> str:
    """選択肢を人間が読める1行にする。ロジックを書く前の観察用。"""
    t = option.get("type")
    label = enum_name(OptionType, t)

    if t == 13:  # Attack
        a = attack(option.get("attackId"))
        if a:
            cost = "".join(enum_name(EnergyIndex, e)[:1] for e in a["energies"]) or "-"
            text = f" | {a['text'][:50]}" if a.get("text") else ""
            return f"{label}: {a['name']} dmg={a['damage']} cost={cost}{text}"
        return f"{label}: attackId={option.get('attackId')}"

    if t == 0:  # Number
        return f"{label}: {option.get('number')}"

    if t == 7:  # Play は area を持たず、index は自分の手札を指す
        c = hand_card(obs, option.get("index"))
        return f"{label}: {name(c.get('id')) if c else option.get('index')}"

    src = resolve(option, obs)
    parts = [label]
    if src:
        parts.append(name(src.get("id")))
        if src.get("maxHp"):
            parts.append(f"hp={src.get('hp')}/{src['maxHp']}")
    elif option.get("area") is not None:
        parts.append(f"{enum_name(AreaType, option['area'])}[{option.get('index')}]")

    if option.get("inPlayArea") is not None:
        dst = resolve(option, obs, "inPlayArea", "inPlayIndex")
        dst_label = name(dst.get("id")) if dst else enum_name(AreaType, option["inPlayArea"])
        parts.append(f"-> {dst_label}")
    return " ".join(parts)


def describe_select(sel: dict, obs: dict) -> str:
    head = (
        f"[{enum_name(SelectType, sel.get('type'))}/{enum_name(SelectContext, sel.get('context'))}] "
        f"{sel.get('minCount')}-{sel.get('maxCount')} から選ぶ"
    )
    lines = [head]
    for i, o in enumerate(sel.get("option") or []):
        lines.append(f"  {i:2}: {describe_option(o, obs)}")
    return "\n".join(lines)


def load_deck(path: str) -> list[int]:
    ids: list[int] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            for cell in row:
                cell = cell.strip()
                if cell and not cell.startswith("#"):
                    ids.append(int(cell))
    return ids


def deck_summary(deck: list[int]) -> str:
    """デッキを枚数つきで種別ごとに並べる。"""
    import collections

    counts = collections.Counter(deck)
    buckets: dict[str, list[tuple[int, str, int]]] = {}
    for cid, n in counts.items():
        c = card(cid)
        kind = enum_name(CardType, c["cardType"]) if c else "Unknown"
        detail = ""
        if c and c["cardType"] == 0:
            detail = f"HP{c['hp']} {enum_name(EnergyIndex, c['energyType'])}"
            if c.get("ex"):
                detail += " ex"
            if c.get("evolvesFrom"):
                detail += f" <- {c['evolvesFrom']}"
        buckets.setdefault(kind, []).append((n, name(cid), detail))

    out = [f"合計 {len(deck)} 枚 / ユニーク {len(counts)} 種"]
    for kind in sorted(buckets):
        rows = sorted(buckets[kind], key=lambda r: (-r[0], r[1]))
        out.append(f"\n[{kind}] {sum(r[0] for r in rows)} 枚")
        for n, nm, detail in rows:
            out.append(f"  {n} x {nm}" + (f"  ({detail})" if detail else ""))
    return "\n".join(out)


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agents", "baseline", "deck.csv"
    )
    print(deck_summary(load_deck(target)))
