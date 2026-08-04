"""集めた局面から評価関数を学習する。線形と小さな多層パーセプトロンの両方。

  .venv/bin/python tools/train_value.py data/value/d3_run1.jsonl
  .venv/bin/python tools/train_value.py data/value/d3_run1.jsonl --model mlp --hidden 16
  .venv/bin/python tools/train_value.py data/value/*.jsonl --model both --agent agents/d3_value

線形はニュートン法 (IRLS) で解く。特徴が数十個しかないので、数十回の反復で
厳密解に落ちるうえ、学習率のような調整項目が無い。

汎化の確認は試合単位で切る。同じ試合の局面は結果ラベルを共有していて、行単位で
分けると訓練と検証に同じ試合が混ざり、検証の値が実力より良く出る。

推論はロールアウトの打ち切りごとに呼ばれる。隠れ層を大きくすると探索の回転数が
落ちるので、既定は 16 に抑えてある。
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "shared"))

import features  # noqa: E402


def load(paths: list[str]) -> tuple:
    files: list[str] = []
    for p in paths:
        files.extend(sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p])
    xs, ys, gs, ts = [], [], [], []
    bad = 0
    for p in files:
        with open(p) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                x = r.get("x")
                if not x or len(x) != features.N:
                    bad += 1
                    continue
                xs.append(x)
                ys.append(r["y"])
                gs.append((p, r.get("g")))
                ts.append(r.get("turn") or 0)
    if bad:
        print(f"! 特徴の数が合わない行を {bad} 件捨てた (期待 {features.N})")
    if not xs:
        sys.exit("学習データが空である。特徴を変えたなら集め直しが要る")
    seen: dict = {}
    gid = np.array([seen.setdefault(g, len(seen)) for g in gs])
    return (np.asarray(xs, float), np.asarray(ys, float), gid,
            np.asarray(ts, int))


def logloss(p: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


# ------------------------------------------------------------------ 線形

def fit_linear(X: np.ndarray, y: np.ndarray, l2: float, iters: int = 60) -> np.ndarray:
    d = X.shape[1]
    w = np.zeros(d)
    pen = np.full(d, l2)
    pen[0] = 0.0  # 定数項は罰則から外す。ここを縮めると全体の基準がずれる
    for _ in range(iters):
        p = sigmoid(X @ w)
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


def pick_l2(Xt, yt, Xv, yv, tol: float) -> float:
    """罰則の強さを検証データで選ぶ。ただし損失が最小のものは採らない。

    特徴どうしの相関が強く、弱い罰則では係数が相方に食われて符号が反転する。
    予測値としては問題なくても、探索の指針としては危ない。損失がほぼ並ぶ
    範囲では強い側を選ぶ。
    """
    curve = []
    for l2 in (0.1, 1.0, 10.0, 100.0, 1000.0, 3000.0, 10000.0):
        p = sigmoid(Xv @ fit_linear(Xt, yt, l2))
        ll = logloss(p, yv)
        print(f"  l2={l2:<8} 検証 対数損失 {ll:.4f}")
        curve.append((l2, ll))
    floor = min(ll for _, ll in curve)
    l2 = max(v for v, ll in curve if ll <= floor + tol)
    print(f"  → l2={l2} (最小 {floor:.4f} + {tol} 以内で最も強いもの)")
    return l2


# ------------------------------------------------------------------ MLP

def fit_mlp(Xt, yt, Xv, yv, hidden: int, l2: float, epochs: int, seed: int) -> dict:
    """1 隠れ層。Adam で回し、検証損失が最良だった時点の重みを返す。

    定数項の列 (先頭) は入力から外す。バイアス項が別にあるので二重になる。
    """
    rng = np.random.default_rng(seed)
    Xt, Xv = Xt[:, 1:], Xv[:, 1:]
    n, d = Xt.shape
    # He 初期化。tanh ではなく relu を使う。推論を Python で書くとき分岐 1 つで済む
    w1 = rng.normal(0, np.sqrt(2.0 / d), (d, hidden))
    b1 = np.zeros(hidden)
    w2 = rng.normal(0, np.sqrt(2.0 / hidden), hidden)
    b2 = 0.0

    params = [w1, b1, w2, np.array([b2])]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    lr, b1a, b2a, eps = 0.01, 0.9, 0.999, 1e-8
    batch = 4096
    best = None
    t = 0

    for ep in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            xb, yb = Xt[idx], yt[idx]
            z1 = xb @ params[0] + params[1]
            h = np.maximum(z1, 0.0)
            p = sigmoid(h @ params[2] + params[3][0])
            dz = (p - yb) / len(idx)
            gw2 = h.T @ dz + l2 * params[2] / n
            gb2 = np.array([dz.sum()])
            dh = np.outer(dz, params[2]) * (z1 > 0)
            gw1 = xb.T @ dh + l2 * params[0] / n
            gb1 = dh.sum(axis=0)
            t += 1
            for k, g in enumerate((gw1, gb1, gw2, gb2)):
                m[k] = b1a * m[k] + (1 - b1a) * g
                v[k] = b2a * v[k] + (1 - b2a) * g * g
                mh = m[k] / (1 - b1a ** t)
                vh = v[k] / (1 - b2a ** t)
                params[k] -= lr * mh / (np.sqrt(vh) + eps)

        hv = np.maximum(Xv @ params[0] + params[1], 0.0)
        ll = logloss(sigmoid(hv @ params[2] + params[3][0]), yv)
        if best is None or ll < best[0]:
            best = (ll, [p.copy() for p in params])
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"  epoch {ep:>3}  検証 対数損失 {ll:.4f}")
    print(f"  → 最良 {best[0]:.4f}")
    w1, b1, w2, b2 = best[1]
    return {"w1": w1, "b1": b1, "w2": w2, "b2": float(b2[0]), "val": best[0]}


# ------------------------------------------------------------------

def report(name: str, p: np.ndarray, y: np.ndarray) -> None:
    acc = float(np.mean((p >= 0.5) == (y >= 0.5)))
    print(f"  {name:<8} 対数損失 {logloss(p, y):.4f}  的中 {acc:.1%}  ({len(y)} 局面)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="+")
    ap.add_argument("--model", choices=("linear", "mlp", "both"), default="both")
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--mlp-l2", type=float, default=1.0)
    ap.add_argument("--tol", type=float, default=0.002,
                    help="この対数損失の悪化までは、強い罰則のほうを選ぶ")
    ap.add_argument("--agent", default=None,
                    help="この config.json に学習結果を書き込む")
    ap.add_argument("--min-turn", type=int, default=0)
    args = ap.parse_args()

    X, y, gid, turn = load(args.data)
    if args.min_turn:
        keep = turn >= args.min_turn
        X, y, gid = X[keep], y[keep], gid[keep]
        print(f"ターン {args.min_turn} 未満を捨てた ({int((~keep).sum())} 局面)")
    print(f"{len(y)} 局面 / {len(set(gid.tolist()))} 試合 / 特徴 {X.shape[1]} 個  "
          f"勝ちラベル {y.mean():.1%}")

    val = (gid % 5) == 0
    Xt, yt, Xv, yv = X[~val], y[~val], X[val], y[val]
    base = logloss(np.full(len(yv), yv.mean()), yv)
    print(f"基準 (常に平均) 対数損失 {base:.4f}\n")

    out: dict = {}

    if args.model in ("linear", "both"):
        print("[線形]")
        l2 = pick_l2(Xt, yt, Xv, yv, args.tol)
        w = fit_linear(Xt, yt, l2)
        report("訓練", sigmoid(Xt @ w), yt)
        report("検証", sigmoid(Xv @ w), yv)
        w_full = fit_linear(X, y, l2)
        print("\n[重み] 全データで再推定")
        for nm, v in sorted(zip(features.NAMES, w_full), key=lambda t: -abs(t[1])):
            print(f"  {nm:<20} {v:+.4f}")
        out["value_weights"] = [round(float(v), 6) for v in w_full]

    if args.model in ("mlp", "both"):
        print(f"\n[MLP {X.shape[1] - 1}-{args.hidden}-1]")
        mlp = fit_mlp(Xt, yt, Xv, yv, args.hidden, args.mlp_l2, args.epochs, 0)
        hv = np.maximum(Xv[:, 1:] @ mlp["w1"] + mlp["b1"], 0.0)
        report("検証", sigmoid(hv @ mlp["w2"] + mlp["b2"]), yv)
        out["value_mlp"] = {
            "w1": [[round(float(x), 5) for x in row] for row in mlp["w1"]],
            "b1": [round(float(x), 5) for x in mlp["b1"]],
            "w2": [round(float(x), 5) for x in mlp["w2"]],
            "b2": round(float(mlp["b2"]), 5),
        }

    if args.agent:
        path = os.path.join(args.agent, "config.json")
        cfg = {}
        if os.path.isfile(path):
            with open(path) as f:
                cfg = json.load(f)
        cfg.update(out)
        os.makedirs(args.agent, exist_ok=True)
        with open(path, "w") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=1)
        print(f"\n{path} に書き込んだ")


if __name__ == "__main__":
    main()
