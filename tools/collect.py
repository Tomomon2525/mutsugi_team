"""自己対戦を回して、局面の特徴と最終結果の組を集める。評価関数の学習データ。

  PTCG_SEARCH=0 .venv/bin/python tools/collect.py agents/d0_grimmsnarl \
      -n 4000 -j 6 -o data/value/run1.jsonl

探索を止めると 1 戦 0.4 秒で終わる (Mac j=6 で 35,000 戦/時)。探索ありでは
数百戦/時なので、学習に要る量は探索なしでしか集まらない。ただし局面の分布は
本番と少しずれる。ずれの影響は、学習した重みを探索ありの A/B にかけて確かめる。

エージェント側にログを仕込まず、ここで observation を見る。kaggle_environments は
呼び出し可能オブジェクトも受けるので、エージェント本体を包んで両側に渡せば、
試合の勝敗と局面が同じプロセスに揃う。試合番号の対応付けが要らなくなる。

出力は 1 行 1 局面。

  {"g": 試合番号, "turn": ターン, "y": 1 or 0, "x": [特徴...]}
"""

import argparse
import concurrent.futures as cf
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "shared"))


def load_agent(agent_dir: str, tag: str):
    """エージェントの main.py を専用の名前で読み込む。

    両側に別々のエージェントを置く場合、素の `main` で読むと片方しか残らない。
    """
    import importlib.util

    path = os.path.join(os.path.abspath(agent_dir), "main.py")
    if not os.path.isfile(path):
        sys.exit(f"main.py がない: {path}")
    sys.path.insert(0, os.path.dirname(path))
    try:
        spec = importlib.util.spec_from_file_location(f"agentmain__{tag}", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path.pop(0)


_STATE: dict = {}


def setup(a0: str, a1: str) -> dict:
    """ワーカープロセスで 1 回だけ初期化する。エンジンの読み込みが重いため。"""
    if not _STATE:
        import features

        _STATE["features"] = features
        _STATE["m0"] = load_agent(a0, "p0")
        _STATE["m1"] = load_agent(a1, "p1") if a1 != a0 else _STATE["m0"]
    return _STATE


def play(job) -> tuple[int, list, dict]:
    i, a0, a1, swap, stride = job
    st = setup(a0, a1)
    features = st["features"]
    from kaggle_environments import make

    rows: list[tuple] = []

    def wrap(mod):
        def fn(obs):
            if obs.get("select") is not None:
                cur = obs.get("current") or {}
                mi = cur.get("yourIndex", 0)
                # 両方の視点で記録する。手番側だけを残すと to_move が常に +1 の
                # 定数になり、学習できない。ロールアウトの打ち切りは相手の手番でも
                # 起きるので、そこで手番の有利不利を評価できなくなる。
                rows.append((cur.get("turn"),
                             (0, features.vector(obs, 0)),
                             (1, features.vector(obs, 1))))
            return mod.agent(obs)
        return fn

    m0, m1 = (st["m1"], st["m0"]) if swap else (st["m0"], st["m1"])
    env = make("cabt")
    env.run([wrap(m0), wrap(m1)])
    last = env.steps[-1]
    r = last[0].get("reward") or 0
    ok = all(s["status"] in ("DONE", "ACTIVE", "INACTIVE") for s in last)

    out = []
    if ok and r:
        winner = 0 if r > 0 else 1
        # 同じ試合の隣り合う局面はほとんど同じで、ラベルも共有している。
        # 全部残しても情報は増えず、ファイルだけ膨らむ。間引く。
        for turn, *sides in rows[:: max(1, stride)]:
            for mi, x in sides:
                out.append({"g": i, "turn": turn,
                            "y": 1 if mi == winner else 0,
                            "x": [round(v, 4) for v in x]})
    info = {"i": i, "reward_p0": r, "rows": len(out), "ok": ok,
            "steps": len(env.steps)}
    return i, out, info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("agent0")
    ap.add_argument("agent1", nargs="?", default=None,
                    help="省略すると自己対戦")
    ap.add_argument("-n", "--games", type=int, default=1000)
    ap.add_argument("-j", "--jobs", type=int, default=1)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--stride", type=int, default=1,
                    help="局面を何手おきに残すか")
    args = ap.parse_args()

    a0 = os.path.abspath(args.agent0)
    a1 = os.path.abspath(args.agent1) if args.agent1 else a0
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    # 先後の偏りをそのまま学習させないよう、試合ごとに入れ替える
    jobs = [(i, a0, a1, bool(i % 2), args.stride) for i in range(args.games)]
    t0 = time.time()
    n_rows = draws = bad = 0

    with open(args.out, "w") as f:
        if args.jobs > 1:
            with cf.ProcessPoolExecutor(max_workers=args.jobs) as ex:
                futs = [ex.submit(play, j) for j in jobs]
                for k, fut in enumerate(cf.as_completed(futs), 1):
                    _, out, info = fut.result()
                    for rec in out:
                        f.write(json.dumps(rec) + "\n")
                    n_rows += len(out)
                    draws += int(info["ok"] and not info["reward_p0"])
                    bad += int(not info["ok"])
                    if k % 200 == 0:
                        dt = time.time() - t0
                        print(f"  {k}/{args.games} 戦  {n_rows} 局面  "
                              f"{k / dt * 3600:.0f} 戦/時")
        else:
            for k, j in enumerate(jobs, 1):
                _, out, info = play(j)
                for rec in out:
                    f.write(json.dumps(rec) + "\n")
                n_rows += len(out)
                draws += int(info["ok"] and not info["reward_p0"])
                bad += int(not info["ok"])

    dt = time.time() - t0
    print(f"\n{args.games} 戦 / {dt:.0f} 秒 ({args.games / dt * 3600:.0f} 戦/時)")
    print(f"{n_rows} 局面を {args.out} に書いた "
          f"(引き分け {draws} 戦は捨てた, 異常終了 {bad} 戦)")


if __name__ == "__main__":
    main()
