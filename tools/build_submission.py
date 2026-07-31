"""提出用 submission.tar.gz を作る。

  .venv/bin/python tools/build_submission.py agents/baseline

トップレベルに main.py と deck.csv が並ぶ形で固める。__pycache__ 等は除外。
"""

import argparse
import os
import sys
import tarfile

EXCLUDE_DIRS = {"__pycache__", ".git", ".venv", ".ipynb_checkpoints"}
EXCLUDE_SUFFIX = {".pyc", ".pyo", ".tar.gz", ".zip"}

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHARED = os.path.join(ROOT, "shared")


def check(agent_dir: str) -> None:
    main_py = os.path.join(agent_dir, "main.py")
    deck = os.path.join(agent_dir, "deck.csv")
    if not os.path.isfile(main_py):
        sys.exit(f"main.py がない: {main_py}")
    if not os.path.isfile(deck):
        sys.exit(f"deck.csv がない: {deck}")

    import py_compile

    py_compile.compile(main_py, doraise=True)

    n = 0
    with open(deck, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                n += len(line.split(","))
    if n != 60:
        sys.exit(f"deck.csv は 60 枚である必要がある (現在 {n} 枚)")


def validate(tar_path: str) -> None:
    """Kaggle 側の検証と同じく自分自身と 1 戦させる。

    エージェントディレクトリではなく tar を展開して実行する。shared/ の同梱漏れなど、
    提出物の形でしか出ない不具合をここで捕まえるためである。
    """
    import tempfile

    from kaggle_environments import make

    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(tar_path) as tar:
            tar.extractall(tmp)
        main_py = os.path.join(tmp, "main.py")
        env = make("cabt")
        env.run([main_py, main_py])
        statuses = [a.status for a in env.steps[-1]]
        if any(s not in ("DONE", "ACTIVE", "INACTIVE") for s in statuses):
            err = env.steps[0][0].get("error")
            sys.exit(f"検証対戦に失敗: statuses={statuses} error={err}")
        print(f"validation game: OK ({len(env.steps)} steps, statuses={statuses})")


def build(agent_dir: str, out: str) -> str:
    def filt(ti: tarfile.TarInfo):
        base = os.path.basename(ti.name)
        if base in EXCLUDE_DIRS:
            return None
        if any(base.endswith(s) for s in EXCLUDE_SUFFIX):
            return None
        return ti

    with tarfile.open(out, "w:gz") as tar:
        for name in sorted(os.listdir(agent_dir)):
            tar.add(os.path.join(agent_dir, name), arcname=name, filter=filt)
        # shared/ の中身は tar のトップレベルに置く。kaggle_environments は main.py のある
        # ディレクトリを sys.path に足すので、`import ptcg` がそのまま通る。
        for name in sorted(os.listdir(SHARED)) if os.path.isdir(SHARED) else []:
            if name.endswith(".py") and not os.path.exists(os.path.join(agent_dir, name)):
                tar.add(os.path.join(SHARED, name), arcname=name, filter=filt)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("agent_dir")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--skip-validate", action="store_true", help="自己対戦による検証を省略")
    args = ap.parse_args()

    agent_dir = os.path.abspath(args.agent_dir)
    out = args.out or os.path.join(agent_dir, "submission.tar.gz")

    check(agent_dir)
    build(agent_dir, out)
    if not args.skip_validate:
        validate(out)

    size = os.path.getsize(out)
    print(f"built: {out} ({size / 1024:.1f} KiB)")
    with tarfile.open(out) as tar:
        for m in tar.getmembers():
            print(f"  {m.name}")
    print("\n提出:")
    print(f"  .venv/bin/kaggle competitions submit pokemon-tcg-ai-battle -f {out} -m 'message'")


if __name__ == "__main__":
    main()
