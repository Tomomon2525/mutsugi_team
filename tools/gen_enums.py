"""公式エンジンの C++ ヘッダから Enum 定義を抜き出し、shared/enums.py を生成する。

公式パッケージは再配布が禁じられているため、リポジトリには含めない。各自が Kaggle から
ダウンロードした手元のコピーを読ませる。生成物 shared/enums.py は機構名の定数だけで
カード名・カードテキストを含まないため、こちらは追跡対象としている。

  .venv/bin/python tools/gen_enums.py
  .venv/bin/python tools/gen_enums.py --src ~/Downloads/pokemon-tcg-ai-battle
"""

import argparse
import glob
import keyword
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SRC = os.path.expanduser("~/Downloads/pokemon-tcg-ai-battle")
OUT = os.path.join(ROOT, "shared", "enums.py")

# (Python 側の名前, C++ の enum 名, ヘッダ, 先頭の None を落とすか)
#
# ToJson.h の SelectJson は select.type / select.context を `enum - 1` で書き出している。
# 先頭の None が JSON に現れないため、その 2 つだけ 1 個ずらす。option.type と logs[].type は
# ApiJson.h が生の (int) を書いているのでずらさない。
WANTED = [
    ("SelectType", "SelectType", "ApiType.h", True),
    ("SelectContext", "SelectContext", "ApiType.h", True),
    ("OptionType", "SelectOptionType", "ApiType.h", False),
    ("LogType", "LogType", "ApiType.h", False),
    ("AreaType", "AreaType", "Types.h", False),
    ("CardType", "CardType", "Types.h", False),
    ("PokemonType", "PokemonType", "Types.h", False),
    ("EvolutionType", "EvolutionType", "Types.h", False),
]


def find_headers(src: str) -> dict[str, str]:
    """公式パッケージ内から *.h を集める。ディレクトリ名に空白や版番号が入るため glob で探す。"""
    found: dict[str, str] = {}
    for path in glob.glob(os.path.join(src, "**", "*.h"), recursive=True):
        found.setdefault(os.path.basename(path), path)
    return found


def parse_enum(text: str, name: str) -> list[str]:
    """`enum class <name> : <type> { A, B, // comment\n C, };` から識別子を順に取る。"""
    m = re.search(r"enum\s+class\s+%s\s*(?::\s*\w[\w\s]*)?\s*\{(.*?)\}\s*;" % re.escape(name), text, re.S)
    if not m:
        raise SystemExit(f"enum が見つからない: {name}")
    body = re.sub(r"//[^\n]*", "", m.group(1))
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
    members = []
    for part in body.split(","):
        part = part.strip()
        if not part:
            continue
        ident = part.split("=")[0].strip()
        if re.fullmatch(r"[A-Za-z_]\w*", ident):
            members.append(ident)
    return members


def parse_energy_order(text: str) -> list[str]:
    """EnergyTypes 配列の並びが、JSON に出てくる energy のインデックスと一致する。"""
    m = re.search(r"EnergyTypes\s*=\s*\{(.*?)\}\s*;", text, re.S)
    if not m:
        raise SystemExit("EnergyTypes 配列が見つからない")
    names = re.findall(r"EnergyType::(\w+)", m.group(1))
    out: list[str] = []
    for n in names:
        if n == "All" or n in out:
            break
        out.append(n)
    return out


def safe(name: str) -> str:
    """Python の予約語 (None など) はそのままだと構文エラーになるため後置アンダースコアを付ける。"""
    return name + "_" if keyword.iskeyword(name) else name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC, help="公式パッケージを展開したディレクトリ")
    args = ap.parse_args()

    src = os.path.expanduser(args.src)
    if not os.path.isdir(src):
        sys.exit(f"見つからない: {src}\nKaggle のコンペページから取得したディレクトリを --src で指定する。")

    headers = find_headers(src)
    cache: dict[str, str] = {}

    def read(fname: str) -> str:
        if fname not in cache:
            if fname not in headers:
                sys.exit(f"{fname} が {src} 以下に見つからない")
            with open(headers[fname], encoding="utf-8-sig") as f:
                cache[fname] = f.read()
        return cache[fname]

    blocks = []
    for py_name, cpp_name, fname, drop_first in WANTED:
        members = parse_enum(read(fname), cpp_name)
        if drop_first:
            if members[0] != "None":
                sys.exit(f"{cpp_name} の先頭が None ではない: {members[0]}。ToJson.h の採番を再確認する。")
            members = members[1:]
        lines = [f"class {py_name}(IntEnum):"]
        lines += [f"    {safe(m)} = {i}" for i, m in enumerate(members)]
        blocks.append("\n".join(lines))

    energy = parse_energy_order(read("Types.h"))
    lines = ["class EnergyIndex(IntEnum):"]
    lines += [f"    {safe(m)} = {i}" for i, m in enumerate(energy)]
    blocks.append("\n".join(lines))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write('"""自動生成。直接編集しない。tools/gen_enums.py で再生成する。\n\n')
        f.write("公式エンジンのヘッダから抽出した機構名の定数である。カード名・カードテキストは含まない。\n")
        f.write('"""\n\nfrom enum import IntEnum\n\n\n')
        f.write("\n\n\n".join(blocks))
        f.write("\n")

    print(f"wrote {OUT}")
    for py_name, _, _, _ in WANTED:
        print(f"  {py_name}")
    print(f"  EnergyIndex ({', '.join(energy)})")


if __name__ == "__main__":
    main()
