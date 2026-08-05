"""対戦を回して、局面の特徴と最終結果の組を集める。評価関数の学習データ。

  PTCG_SEARCH=0 .venv/bin/python tools/collect.py agents/d3_candy \
      -n 4000 -j 6 -o data/value/run1.jsonl

  PTCG_SEARCH=0 .venv/bin/python tools/collect.py agents/d3_candy \
      --opponents agents/field/weights.json -n 20000 -j 3 -o data/value/field.jsonl

探索を止めると 1 戦 0.4 秒で終わる (Mac j=6 で 35,000 戦/時)。探索ありでは
数百戦/時なので、学習に要る量は探索なしでしか集まらない。ただし局面の分布は
本番と少しずれる。ずれの影響は、学習した重みを探索ありの A/B にかけて確かめる。

エージェント側にログを仕込まず、ここで observation を見る。kaggle_environments は
呼び出し可能オブジェクトも受けるので、エージェント本体を包んで両側に渡せば、
試合の勝敗と局面が同じプロセスに揃う。試合番号の対応付けが要らなくなる。

--opponents を渡すと、試合ごとに相手デッキを採用率で抽選する。ミラーだけで学習
すると、環境の半分を占める他のデッキを一度も見ないまま重みが決まる。

記録するのは手番側の視点だけである。相手の手札は None で来るので、手番でない側の
視点を作ると evolve_path と punk_up_ready が必ず 0 になる。以前は両側を記録して
いて、学習データの半分がこの状態だった。to_move は常に +1 になるが、探索側も
手番側の視点で評価して符号を返すようにしたので、推論と揃っている。

出力は 1 行 1 局面。

  {"g": 試合番号, "turn": ターン, "y": 1 or 0, "x": [特徴...], "opp": 相手デッキ}
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


def setup(dirs: tuple) -> dict:
    """ワーカープロセスで 1 回だけ初期化する。エンジンの読み込みが重いため。

    相手を抽選する場合、1 プロセスが複数のエージェントを抱える。読み込み済みの
    ものは使い回す。エージェントごとに専用の名前で読むので、共有モジュールが
    互いを潰すことはない。
    """
    if "features" not in _STATE:
        import features

        _STATE["features"] = features
        _STATE["mods"] = {}
    mods = _STATE["mods"]
    for d in dirs:
        if d not in mods:
            mods[d] = load_agent(d, "a%04x" % (abs(hash(d)) & 0xFFFF))
    return _STATE


def play(job) -> tuple[int, list, dict]:
    i, ours, opp, swap, stride = job
    st = setup((ours, opp))
    features = st["features"]
    mods = st["mods"]
    from kaggle_environments import make

    rows: list[tuple] = []
    # swap が真なら自分は 1 番。以降 mi はこの固定の index を指す
    my_index = 1 if swap else 0

    def wrap(mod):
        def fn(obs):
            if obs.get("select") is not None:
                cur = obs.get("current") or {}
                # 手番側の視点だけを記録する。相手の手札は None で来るので、
                # 手番でない側のベクトルは手札由来の特徴が必ず 0 になる。
                mi = cur.get("yourIndex", 0)
                rows.append((cur.get("turn"), mi, features.vector(obs, mi)))
            return mod.agent(obs)
        return fn

    m0 = mods[opp] if swap else mods[ours]
    m1 = mods[ours] if swap else mods[opp]
    env = make("cabt")
    env.run([wrap(m0), wrap(m1)])
    last = env.steps[-1]
    r = last[0].get("reward") or 0
    ok = all(s["status"] in ("DONE", "ACTIVE", "INACTIVE") for s in last)

    out = []
    if ok and r:
        winner = 0 if r > 0 else 1
        tag = os.path.basename(opp)
        # 同じ試合の隣り合う局面はほとんど同じで、ラベルも共有している。
        # 全部残しても情報は増えず、ファイルだけ膨らむ。間引く。
        for turn, mi, x in rows[:: max(1, stride)]:
            # 自分の側の視点だけを学習に使う。相手デッキが違う試合では、
            # 相手視点のベクトルはこちらの勝ち筋を表す特徴が全部 0 になる
            if mi != my_index:
                continue
            out.append({"g": i, "turn": turn,
                        "y": 1 if mi == winner else 0,
                        "x": [round(v, 4) for v in x], "opp": tag})
    info = {"i": i, "reward_p0": r, "rows": len(out), "ok": ok,
            "win": int(ok and r and (0 if r > 0 else 1) == my_index),
            "opp": os.path.basename(opp), "steps": len(env.steps)}
    return i, out, info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("agent0")
    ap.add_argument("agent1", nargs="?", default=None,
                    help="省略すると自己対戦")
    ap.add_argument("--opponents", default=None,
                    help="make_field.py の weights.json。採用率で相手を抽選する")
    ap.add_argument("-n", "--games", type=int, default=1000)
    ap.add_argument("-j", "--jobs", type=int, default=1)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--stride", type=int, default=1,
                    help="局面を何手おきに残すか")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    a0 = os.path.abspath(args.agent0)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    if args.opponents:
        import random as _random

        with open(args.opponents) as f:
            field = json.load(f)
        base = os.path.dirname(os.path.dirname(os.path.abspath(args.opponents)))
        pool = [os.path.join(ROOT, w["dir"]) for w in field]
        share = [w["share"] for w in field]
        rng = _random.Random(args.seed)
        # 抽選は親プロセスで済ませる。ワーカーごとに引くと再現できなくなる
        opps = rng.choices(pool, weights=share, k=args.games)
        print(f"相手を {len(pool)} 種類から採用率で抽選する ({base})")
    else:
        a1 = os.path.abspath(args.agent1) if args.agent1 else a0
        opps = [a1] * args.games

    # 先後の偏りをそのまま学習させないよう、試合ごとに入れ替える
    jobs = [(i, a0, opps[i], bool(i % 2), args.stride)
            for i in range(args.games)]
    t0 = time.time()
    n_rows = draws = bad = 0
    tally: dict = {}

    def record(out, info):
        nonlocal n_rows, draws, bad
        n_rows += len(out)
        draws += int(info["ok"] and not info["reward_p0"])
        bad += int(not info["ok"])
        if info["ok"] and info["reward_p0"]:
            t = tally.setdefault(info["opp"], [0, 0])
            t[0] += info["win"]
            t[1] += 1

    with open(args.out, "w") as f:
        if args.jobs > 1:
            with cf.ProcessPoolExecutor(max_workers=args.jobs) as ex:
                futs = [ex.submit(play, j) for j in jobs]
                for k, fut in enumerate(cf.as_completed(futs), 1):
                    _, out, info = fut.result()
                    for rec in out:
                        f.write(json.dumps(rec) + "\n")
                    record(out, info)
                    if k % 200 == 0:
                        dt = time.time() - t0
                        print(f"  {k}/{args.games} 戦  {n_rows} 局面  "
                              f"{k / dt * 3600:.0f} 戦/時")
        else:
            for k, j in enumerate(jobs, 1):
                _, out, info = play(j)
                for rec in out:
                    f.write(json.dumps(rec) + "\n")
                record(out, info)

    dt = time.time() - t0
    print(f"\n{args.games} 戦 / {dt:.0f} 秒 ({args.games / dt * 3600:.0f} 戦/時)")
    print(f"{n_rows} 局面を {args.out} に書いた "
          f"(引き分け {draws} 戦は捨てた, 異常終了 {bad} 戦)")
    if len(tally) > 1:
        print("\n相手デッキごとの勝率")
        for k, (w, n) in sorted(tally.items(), key=lambda t: -t[1][1]):
            print(f"  {k:<34} {w:>5}/{n:<5} {w / n:6.1%}")


if __name__ == "__main__":
    main()
