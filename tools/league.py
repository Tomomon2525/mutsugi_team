"""対戦を小分けに回し、1 戦ごとに結果を追記する。中断したら続きから再開する。

Colab は資源が保証されず、長時間ジョブは途中で切れる前提で組む必要がある。
そのため、対戦番号を固定し、済んだものを結果ファイルから読み取って飛ばす。
出力先を Google Drive に置けば、ランタイムが落ちても失われない。

時間予算は環境変数で渡す。指定しないとローカルの kaggle_environments が
remainingOverageTime=2000 を渡してくるため、1 手 5 秒の上限いっぱいで思考し、
1 戦に 500 秒前後かかる。Kaggle 相当に縮めるには次を付ける。

  PTCG_TIME_POOL=70 PTCG_MAX_SLICE=0.58 PTCG_RESERVE=5.2 PTCG_MIN_SLICE=0.03 \
  .venv/bin/python tools/league.py agents/d0_grimmsnarl champions/572_6/agent \
      -n 200 -j 6 -o scratch/league/ctx_vs_champ.jsonl

同じコマンドをもう一度流すと、未完了の対戦番号だけを実行する。

対戦番号 i の先後は i%2 で決める。乱数シードはエンジン側が握っていて指定できない
ため、再現性はこのログ自体で担保する (docs/design.md 0 節)。
"""

import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "shared"))


def resolve(spec: str) -> str:
    if spec in ("random", "first"):
        return spec
    p = os.path.abspath(spec)
    if os.path.isdir(p):
        p = os.path.join(p, "main.py")
    if not os.path.isfile(p):
        sys.exit(f"agent not found: {spec}")
    return p


def commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def play(job) -> dict:
    i, a0, a1, swap = job
    from kaggle_environments import make

    t0 = time.time()
    env = make("cabt")
    env.run([a0, a1])
    last = env.steps[-1]
    r = last[0].get("reward") or 0
    return {
        "i": i,
        "reward_p0": r,
        # agent0 視点に直した結果。先後を入れ替えた分をここで戻す
        "result": (-r if swap else r),
        "swap": swap,
        "steps": len(env.steps),
        "seconds": round(time.time() - t0, 1),
        "statuses": [s["status"] for s in last],
    }


def done_indices(path: str) -> set:
    out = set()
    if not os.path.isfile(path):
        return out
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if "i" in rec:
                out.add(rec["i"])
    return out


def summarize(path: str, a0: str, a1: str) -> None:
    win = lose = draw = 0
    bad: list = []
    secs = 0.0
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "result" not in r:
                continue
            secs += r.get("seconds") or 0
            if any(s not in ("DONE", "ACTIVE", "INACTIVE") for s in r.get("statuses") or []):
                bad.append(r["i"])
            if r["result"] > 0:
                win += 1
            elif r["result"] < 0:
                lose += 1
            else:
                draw += 1
    n = win + lose + draw
    if not n:
        print("結果なし")
        return
    rate = win / n
    # 帰無仮説 50% に対する z 値。引き分けも試合数に数える
    z = (rate - 0.5) / ((0.25 / n) ** 0.5) if n else 0.0
    print(f"\n{a0}\n  vs {a1}")
    print(f"  {n} 戦  {win}勝 {lose}敗 {draw}分  勝率 {rate:.1%}  z={z:+.2f}")
    print(f"  延べ実行時間 {secs / 3600:.2f} 時間 (1 戦あたり {secs / n:.1f} 秒)")
    if bad:
        print(f"  ! 異常終了 {len(bad)} 件: {bad[:10]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("agent0")
    ap.add_argument("agent1")
    ap.add_argument("-n", "--games", type=int, default=100)
    ap.add_argument("-j", "--jobs", type=int, default=1)
    ap.add_argument("-b", "--batch", type=int, default=50, help="この数ごとに集計を出す")
    ap.add_argument("-o", "--out", required=True, help="結果の追記先 (.jsonl)")
    args = ap.parse_args()

    a0, a1 = resolve(args.agent0), resolve(args.agent1)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    done = done_indices(args.out)
    todo = [i for i in range(args.games) if i not in done]
    print(f"{args.games} 戦中 {len(done)} 戦が済み。残り {len(todo)} 戦を実行する。")
    if not todo:
        summarize(args.out, args.agent0, args.agent1)
        return

    head = {"meta": True, "commit": commit(), "agent0": args.agent0,
            "agent1": args.agent1, "games": args.games,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with open(args.out, "a") as f:
        f.write(json.dumps(head, ensure_ascii=False) + "\n")

    for start in range(0, len(todo), args.batch):
        chunk = todo[start:start + args.batch]
        jobs = [(i, a1 if i % 2 else a0, a0 if i % 2 else a1, bool(i % 2)) for i in chunk]
        t0 = time.time()
        with open(args.out, "a") as f:
            if args.jobs > 1:
                with cf.ProcessPoolExecutor(max_workers=args.jobs) as ex:
                    for fut in cf.as_completed([ex.submit(play, j) for j in jobs]):
                        f.write(json.dumps(fut.result(), ensure_ascii=False) + "\n")
                        f.flush()
            else:
                for j in jobs:
                    f.write(json.dumps(play(j), ensure_ascii=False) + "\n")
                    f.flush()
        dt = time.time() - t0
        print(f"  バッチ {start // args.batch + 1}: {len(chunk)} 戦 / {dt:.0f} 秒 "
              f"({len(chunk) / dt * 3600:.0f} 戦/時)")
        summarize(args.out, args.agent0, args.agent1)


if __name__ == "__main__":
    main()
