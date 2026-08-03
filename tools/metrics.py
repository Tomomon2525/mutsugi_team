"""基本動作がどれだけできているかを測る。勝率だけでは改善点が特定できないため。

PTCG_TRACE が吐いた 1 手 1 行の JSON を集計する。対戦を回してから実行する。

  PTCG_TRACE=$PWD/scratch/tr.jsonl PTCG_TIME_POOL=70 PTCG_MAX_SLICE=0.58 \
  PTCG_RESERVE=5.2 PTCG_MIN_SLICE=0.03 \
  .venv/bin/python tools/league.py agents/d0_grimmsnarl agents/d0_grimmsnarl \
      -n 20 -j 6 -o scratch/league/m.jsonl
  .venv/bin/python tools/metrics.py scratch/tr.jsonl

両者が同じ設定なら、出てくる数字は両側を混ぜたものになる。片側だけ見たい場合は
相手を別のエージェントにして、トレースの出力先を分けること。

指標の意味 (docs/design.md 9 節):
  攻撃を逃した率   攻撃できる選択肢があったのに他を選んだ割合
  きぜつを逃した率 相手を倒せる攻撃があったのに取らなかった割合。ここが高いと
                   探索が終盤を読めていない
  成立ターン       オーロンゲ ex が場に出た最初のターン。遅いほど事故っている
"""

import argparse
import collections
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "shared"))

GRIMMSNARL = 648


def load(prefix: str) -> list[dict]:
    paths = sorted(glob.glob(prefix + ".*")) or ([prefix] if os.path.isfile(prefix) else [])
    rows: list[dict] = []
    for p in paths:
        # ファイル名の末尾が PID。試合を (PID, 試合番号) で区別する
        pid = p.rsplit(".", 1)[-1]
        with open(p) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                r["_pid"] = pid
                rows.append(r)
    return rows


def pct(a: int, b: int) -> str:
    return f"{a / b:.1%}" if b else "-"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", help="PTCG_TRACE に渡したパス (末尾の .PID は不要)")
    args = ap.parse_args()

    rows = load(args.trace)
    if not rows:
        sys.exit(f"トレースが見つからない: {args.trace}(.PID)")

    games = {(r.get("_pid"), r.get("g")) for r in rows}
    print(f"{len(rows)} 手 / 試合数の目安 {len(games)}\n")

    # ---- 攻撃ときぜつの取りこぼし
    #
    # 1 手ごとに数えてはいけない。攻撃するとターンが終わるので、先にグッズや
    # サポートを使ってから最後に攻撃するのが正しい打ち方であり、その途中の手を
    # 「攻撃しなかった」と数えると全部が取りこぼしになる。ターン単位で見る。
    turns: dict = {}
    for r in rows:
        if r.get("turn") is None:
            continue
        key = (r["_pid"], r.get("g"), r.get("side"), r["turn"])
        t = turns.setdefault(key, {"atk": 0, "atk_t": 0, "ko": 0, "ko_t": 0})
        t["atk"] |= (r.get("atk_avail") or 0) > 0
        t["ko"] |= (r.get("ko_avail") or 0) > 0
        t["atk_t"] |= bool(r.get("atk_taken"))
        t["ko_t"] |= bool(r.get("ko_taken"))

    atk_chance = [t for t in turns.values() if t["atk"]]
    atk_taken = sum(1 for t in atk_chance if t["atk_t"])
    ko_chance = [t for t in turns.values() if t["ko"]]
    ko_taken = sum(1 for t in ko_chance if t["ko_t"])

    print(f"[攻撃] 手番 {len(turns)} 回ぶん (攻撃するとターンが終わるのでターン単位で数える)")
    print(f"  攻撃できた手番      {len(atk_chance):>6}")
    print(f"    うち攻撃した      {atk_taken:>6}  ({pct(atk_taken, len(atk_chance))})")
    print(f"    しなかった        {len(atk_chance) - atk_taken:>6}  "
          f"({pct(len(atk_chance) - atk_taken, len(atk_chance))})")
    print(f"  きぜつを取れた手番  {len(ko_chance):>6}")
    print(f"    うち取った        {ko_taken:>6}  ({pct(ko_taken, len(ko_chance))})")
    print(f"    逃した            {len(ko_chance) - ko_taken:>6}  "
          f"({pct(len(ko_chance) - ko_taken, len(ko_chance))})  ← 高いと終盤が読めていない")

    # ---- オーロンゲ ex の成立
    first: dict = {}
    last_turn: dict = {}
    for r in rows:
        key = (r.get("_pid"), r.get("g"))
        t = r.get("turn")
        if t is None:
            continue
        last_turn[key] = max(last_turn.get(key, 0), t)
        if GRIMMSNARL in (r.get("in_play") or []):
            if key not in first or t < first[key]:
                first[key] = t
    if last_turn:
        made = len(first)
        turns = sorted(first.values())
        print("\n[オーロンゲ ex]")
        print(f"  成立した試合        {made}/{len(last_turn)}  ({pct(made, len(last_turn))})")
        if turns:
            mid = turns[len(turns) // 2]
            print(f"  成立ターン          中央値 {mid}  最短 {turns[0]}  最長 {turns[-1]}")
        never = [k for k in last_turn if k not in first]
        if never:
            print(f"  一度も立たなかった  {len(never)} 試合 "
                  f"(平均 {sum(last_turn[k] for k in never) / len(never):.0f} ターン)")

    # ---- 探索の状態
    searched = [r for r in rows if r.get("searched")]
    if searched:
        ro = sum(r.get("rollouts") or 0 for r in searched) / len(searched)
        t = sorted(r.get("t") or 0 for r in rows)
        print("\n[探索]")
        print(f"  探索した手          {len(searched)} / {len(rows)}  "
              f"({pct(len(searched), len(rows))})")
        print(f"  1 手あたり          {ro:.0f} ロールアウト")
        print(f"  思考時間            中央値 {t[len(t) // 2]:.3f}s  最大 {t[-1]:.3f}s")
        bad = sum(1 for r in rows if r.get("error"))
        fails = sum((r.get("step_none") or 0) + (r.get("begin_none") or 0)
                    + (r.get("playout_none") or 0) for r in rows)
        print(f"  例外 {bad} 件 / 探索失敗 {fails} 件")

    # ---- 選択文脈の内訳
    ctx = collections.Counter(r.get("sel_ctx") for r in rows)
    try:
        import ptcg
        from enums import SelectContext
        name = lambda c: ptcg.enum_name(SelectContext, c)  # noqa: E731
    except Exception:
        name = str
    print("\n[選択文脈の内訳]")
    for c, k in ctx.most_common(8):
        print(f"  {name(c):<24} {k:>6}  {pct(k, len(rows))}")


if __name__ == "__main__":
    main()
