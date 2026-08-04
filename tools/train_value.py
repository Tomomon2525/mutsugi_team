"""集めた局面から評価関数を学習する。ロジスティック回帰。

  .venv/bin/python tools/train_value.py data/value/run1.jsonl \
      --agent agents/d0_value

特徴が 17 個しかないので、勾配法ではなくニュートン法 (IRLS) で解く。数十回の
反復で厳密解に落ちるうえ、学習率のような調整項目が無い。

汎化の確認は試合単位で切る。同じ試合の局面は結果ラベルを共有していて、行単位で
分けると訓練と検証に同じ試合が混ざり、検証の値が実力より良く出る。

出力は 17 個の重み。エージェントの config.json に "value_weights" として書き込むと、
policy.evaluate が手書きの式の代わりにこれを使う。
"""

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "shared"))

import features  # noqa: E402


def load(paths: list[str]) -> tuple:
    xs, ys, gs, ts = [], [], [], []
    for p in paths:
        with open(p) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                x = r.get("x")
                if not x or len(x) != features.N:
                    continue
                xs.append(x)
                ys.append(r["y"])
                gs.append((p, r.get("g")))
                ts.append(r.get("turn") or 0)
    if not xs:
        sys.exit("学習データが空である")
    seen: dict = {}
    gid = np.array([seen.setdefault(g, len(seen)) for g in gs])
    return (np.asarray(xs, float), np.asarray(ys, float), gid,
            np.asarray(ts, int))


def fit(X: np.ndarray, y: np.ndarray, l2: float, iters: int = 60) -> np.ndarray:
    n, d = X.shape
    w = np.zeros(d)
    # 定数項 (先頭) は罰則から外す。ここを縮めると全体の基準がずれる
    pen = np.full(d, l2)
    pen[0] = 0.0
    for _ in range(iters):
        z = np.clip(X @ w, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        g = X.T @ (p - y) + pen * w
        s = np.clip(p * (1 - p), 1e-6, None)
        H = (X * s[:, None]).T @ X + np.diag(pen)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w


def report(name: str, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    z = np.clip(X @ w, -30, 30)
    p = 1.0 / (1.0 + np.exp(-z))
    ll = -np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))
    acc = np.mean((p >= 0.5) == (y >= 0.5))
    print(f"  {name:<6} 対数損失 {ll:.4f}  的中 {acc:.1%}  ({len(y)} 局面)")
    return ll


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="+")
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--tol", type=float, default=0.002,
                    help="この対数損失の悪化までは、強い罰則のほうを選ぶ")
    ap.add_argument("--agent", default=None,
                    help="この config.json に value_weights を書き込む")
    ap.add_argument("--min-turn", type=int, default=0,
                    help="このターン未満の局面を捨てる。序盤は勝敗をほぼ決めない")
    args = ap.parse_args()

    X, y, gid, turn = load(args.data)
    if args.min_turn:
        keep = turn >= args.min_turn
        X, y, gid = X[keep], y[keep], gid[keep]
        print(f"ターン {args.min_turn} 未満を捨てた ({(~keep).sum()} 局面)")
    print(f"{len(y)} 局面 / {len(set(gid.tolist()))} 試合  勝ちラベル {y.mean():.1%}")

    # 試合単位で 5 分割。1 つを検証に回す
    val = (gid % 5) == 0
    Xt, yt = X[~val], y[~val]
    Xv, yv = X[val], y[val]

    # 罰則の強さを検証データで選ぶ。ただし損失が最小のものは採らない。
    #
    # 特徴どうしの相関が強く (サイドの残り枚数とトラッシュの枚数で 0.80)、
    # 弱い罰則では係数が相方に食われて符号が反転する。実測では、サイドを
    # 取っているほど不利という重みが付いた。予測値としては問題なくても、
    # 探索の指針としては危ない。損失がほぼ並ぶ範囲では強い側を選ぶ。
    curve = []
    for l2 in (0.1, 1.0, 10.0, 100.0, 1000.0, 3000.0, 10000.0):
        w = fit(Xt, yt, l2)
        z = np.clip(Xv @ w, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        ll = -np.mean(yv * np.log(p + 1e-12) + (1 - yv) * np.log(1 - p + 1e-12))
        print(f"  l2={l2:<8} 検証 対数損失 {ll:.4f}")
        curve.append((l2, ll))
    floor = min(ll for _, ll in curve)
    l2 = max(v for v, ll in curve if ll <= floor + args.tol)
    print(f"\n選んだ罰則 l2={l2}  (最小損失 {floor:.4f} + {args.tol} 以内で最も強いもの)")

    w = fit(Xt, yt, l2)
    print("\n[分割して評価]")
    report("訓練", Xt, yt, w)
    report("検証", Xv, yv, w)

    # 常に 50% と答えるだけのモデル。これを下回らなければ学習していない
    base = -np.mean(yv * np.log(yv.mean()) + (1 - yv) * np.log(1 - yv.mean()))
    print(f"  {'基準':<6} 対数損失 {base:.4f}  (常に平均を返す)")

    w = fit(X, y, l2)  # 提出用は全データで引き直す
    print("\n[重み] 全データで再推定")
    for nm, v in sorted(zip(features.NAMES, w), key=lambda t: -abs(t[1])):
        print(f"  {nm:<20} {v:+.4f}")

    out = [round(float(v), 6) for v in w]
    print("\n" + json.dumps({"value_weights": out}))

    if args.agent:
        path = os.path.join(args.agent, "config.json")
        cfg = {}
        if os.path.isfile(path):
            with open(path) as f:
                cfg = json.load(f)
        cfg["value_weights"] = out
        os.makedirs(args.agent, exist_ok=True)
        with open(path, "w") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        print(f"{path} に書き込んだ")


if __name__ == "__main__":
    main()
