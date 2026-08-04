"""本番のリプレイ (エピソードの JSON) を読んで、何が起きたかを出す。

  .venv/bin/python tools/replay_read.py ~/Downloads/89692194.json
  .venv/bin/python tools/replay_read.py ~/Downloads/89692194.json --log
  .venv/bin/python tools/replay_read.py ~/Downloads/89692194.json --html scratch/r.html

手元のリーグと違い、本番の相手と本番の計算資源で戦った記録である。ここでしか
分からないものが 2 つある。持ち時間の使い方 (remainingOverageTime の推移) と、
本番の相手が実際に何をしてきたか。

エピソードの JSON は対戦ページから取得する。日次で配られるダンプは上位チームに
偏った抽出で、順位が下のチームの試合は 1 件も入っていなかった。

自分で回した試合を見るには tools/replay.py のほう (その場で 1 戦して HTML を書く)。
"""

import argparse
import collections
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "shared"))

GRIMMSNARL = 648


def fmt_log(ev: dict, ptcg, LogType) -> str | None:
    """エンジンのイベント 1 件を 1 行にする。盤面が動くものだけ拾う。"""
    t = ev.get("type")
    who = ev.get("playerIndex")
    name = ptcg.name(ev.get("cardId")) if ev.get("cardId") is not None else ""
    if t == LogType.Attack:
        a = ptcg.attack(ev.get("attackId")) or {}
        return f"P{who} 攻撃 {name} の {a.get('name', ev.get('attackId'))}"
    if t == LogType.HpChange:
        v = ev.get("value") or 0
        return None if v >= 0 else f"P{who} {name} に {-v} ダメージ"
    if t == LogType.Evolve:
        return f"P{who} 進化 → {name}"
    if t == LogType.Play:
        return f"P{who} 使用 {name}"
    if t == LogType.Attach:
        return f"P{who} エネ付与 {name}"
    if t == LogType.Switch:
        return (f"P{who} 入れ替え {ptcg.name(ev.get('cardIdActive'))} ⇄ "
                f"{ptcg.name(ev.get('cardIdBench'))}")
    if t == LogType.Coin:
        return f"P{who} コイン {ev.get('value')}"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--log", action="store_true", help="盤面が動いた手を全部出す")
    ap.add_argument("--side", type=int, default=None, help="この番号の視点だけ集計")
    ap.add_argument("--html", default=None, help="公式ビジュアライザで HTML に書き出す")
    args = ap.parse_args()

    import policy
    import ptcg
    from enums import LogType, SelectContext

    with open(args.path) as f:
        ep = json.load(f)

    steps = ep["steps"]
    names = (ep.get("info") or {}).get("TeamNames") or ["P0", "P1"]
    rewards = ep.get("rewards") or [0, 0]
    win = "P0 の勝ち" if (rewards[0] or 0) > 0 else (
        "P1 の勝ち" if (rewards[1] or 0) > 0 else "引き分け")
    print(f"エピソード {(ep.get('info') or {}).get('EpisodeId')}  {len(steps)} 段")
    print(f"  P0 {names[0]}  vs  P1 {names[1] if len(names) > 1 else '?'}")
    print(f"  結果 {rewards}  ({win})")

    turns: dict = {}
    first_grim: dict = {}
    remaining: dict = {0: [], 1: []}
    ctx_count: collections.Counter = collections.Counter()
    lines: list[str] = []
    last_prize = None

    for i, step in enumerate(steps):
        for side, ag in enumerate(step):
            if ag.get("status") != "ACTIVE":
                continue
            obs = ag.get("observation") or {}
            sel = obs.get("select")
            # action は 1 段あとに入る。kaggle_environments は「行動を受け取って
            # 次の状態を作る」順で記録するので、steps[i] の action は steps[i-1] の
            # observation に対する応答である。ここを取り違えると、攻撃した手を
            # 「攻撃しなかった」と数えてしまう (実際に一度そうなった)。
            nxt = steps[i + 1] if i + 1 < len(steps) else None
            act = (nxt[side].get("action") if nxt else None)
            cur = obs.get("current") or {}
            turn = cur.get("turn")
            rem = obs.get("remainingOverageTime")
            if rem is not None:
                remaining[side].append((i, turn, rem))
            if not sel or not isinstance(act, list) or len(act) == 60:
                continue

            ctx_count[sel.get("context")] += 1

            # 攻撃するとターンが終わるので、1 手ごとではなくターン単位で数える
            # (定義は tools/metrics.py と同じ)
            try:
                atk, ko = policy.attack_options(obs)
            except Exception:
                atk, ko = set(), set()
            t = turns.setdefault((side, turn),
                                 {"atk": 0, "atk_t": 0, "ko": 0, "ko_t": 0})
            got = set(act)
            t["atk"] |= bool(atk)
            t["ko"] |= bool(ko)
            t["atk_t"] |= bool(got & atk)
            t["ko_t"] |= bool(got & ko)

            try:
                if GRIMMSNARL in policy.in_play_ids(obs) and side not in first_grim:
                    first_grim[side] = turn
            except Exception:
                pass

            if args.log:
                opts = sel.get("option") or []
                for j in act:
                    if 0 <= j < len(opts):
                        try:
                            desc = ptcg.describe_option(opts[j], obs)
                        except Exception:
                            desc = str(opts[j])
                        lines.append(f"  T{turn:>2} P{side}  {desc}")

        obs = (step[0] or {}).get("observation") or {}
        for ev in obs.get("logs") or []:
            s = fmt_log(ev, ptcg, LogType)
            if s:
                lines.append(f"       {s}")
        ps = (obs.get("current") or {}).get("players")
        if ps and len(ps) == 2:
            pz = tuple(len(p.get("prize") or []) for p in ps)
            if last_prize is not None and pz != last_prize:
                lines.append(f"  === サイド {last_prize} → {pz}")
            last_prize = pz

    for side in (0, 1):
        if args.side is not None and side != args.side:
            continue
        sub = {k: v for k, v in turns.items() if k[0] == side}
        atk_c = [v for v in sub.values() if v["atk"]]
        ko_c = [v for v in sub.values() if v["ko"]]
        print(f"\n[P{side} {names[side] if side < len(names) else ''}]")
        print(f"  手番 {len(sub)} 回")
        if atk_c:
            n = sum(1 for v in atk_c if v["atk_t"])
            print(f"  攻撃できた手番 {len(atk_c)}  うち攻撃した {n} ({n / len(atk_c):.0%})")
        if ko_c:
            n = sum(1 for v in ko_c if v["ko_t"])
            print(f"  きぜつを取れた手番 {len(ko_c)}  うち取った {n} ({n / len(ko_c):.0%})")
        g = first_grim.get(side)
        print(f"  オーロンゲ ex 成立  {('ターン ' + str(g)) if g is not None else '立たず'}")
        r = remaining[side]
        if r:
            print(f"  持ち時間  {r[0][2]:.0f}s → {r[-1][2]:.0f}s "
                  f"(消費 {r[0][2] - r[-1][2]:.0f}s / {len(r)} 手)")
            if r[-1][2] < 30:
                print("  ! 持ち時間をほぼ使い切っている")

    print("\n[選択文脈の内訳]")
    for c, k in ctx_count.most_common(8):
        print(f"  {ptcg.enum_name(SelectContext, c):<24} {k:>5}")

    if args.log:
        print("\n[経過]")
        print("\n".join(lines))

    if args.html:
        from kaggle_environments import make

        env = make("cabt", steps=steps)
        out = os.path.abspath(args.html)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write(env.render(mode="html", width=1000, height=700))
        print(f"\n{out} に書き出した")


if __name__ == "__main__":
    main()
