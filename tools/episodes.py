"""提出した submission が本番で戦った対戦記録を Kaggle から引く。

  .venv/bin/python tools/episodes.py 55228295 55207880 \
      --leaderboard scratch/pokemon-tcg-ai-battle-publicleaderboard-*.csv

日次で配られるエピソードのダンプは上位チームに偏った抽出で、順位が下のチームの
対戦はまず入っていない (4720 件中に自分の試合は 0 件だった)。自分の記録は
Kaggle の内部 API から submissionId 指定で引く。

  POST /api/i/competitions.EpisodeService/ListEpisodes  {"submissionId": N}

返るのは対戦ごとの相手 (teamId / submissionId)、勝敗、レートの前後である。
盤面のログ (steps) を返す口は見つかっていないので、何をして負けたかはここでは分からない。
分かるのは誰に負けたかと、レートがどう動いたか。
"""

import argparse
import csv
import glob
import json
import os
import sys
import time
import urllib.error
import urllib.request

API = "https://www.kaggle.com/api/i/competitions.EpisodeService"


def post(svc: str, payload: dict) -> dict:
    req = urllib.request.Request(
        f"{API}/{svc}", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def team_names(patterns: list[str]) -> dict:
    """リーダーボードの CSV から teamId → (名前, スコア) を作る。"""
    out: dict = {}
    for pat in patterns:
        for path in glob.glob(pat):
            with open(path, encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    try:
                        out[int(row["TeamId"])] = (row["TeamName"], float(row["Score"]))
                    except Exception:
                        continue
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("submissions", nargs="+", type=int)
    ap.add_argument("--team", type=int, default=None, help="自分の teamId (省略時は自動判定)")
    ap.add_argument("--leaderboard", nargs="*", default=[],
                    help="teamId を名前に直すための CSV")
    ap.add_argument("--out", default=None, help="生の JSON を書き出す先")
    args = ap.parse_args()

    names = team_names(args.leaderboard)
    allrows = []

    for sid in args.submissions:
        try:
            data = post("ListEpisodes", {"submissionId": sid})
        except urllib.error.HTTPError as e:
            print(f"submission {sid}: 取得に失敗 (HTTP {e.code})")
            continue
        eps = data.get("episodes") or []
        if args.out:
            os.makedirs(args.out, exist_ok=True)
            with open(os.path.join(args.out, f"{sid}.json"), "w") as f:
                json.dump(data, f, ensure_ascii=False, indent=1)

        me_team = args.team
        if me_team is None:
            for ep in eps:
                for a in ep.get("agents") or []:
                    if a.get("submissionId") == sid:
                        me_team = a.get("teamId")
                        break
                if me_team:
                    break

        rows = []
        for ep in eps:
            mine = op = None
            for a in ep.get("agents") or []:
                if a.get("submissionId") == sid:
                    mine = a
                else:
                    op = a
            if not mine or not op:
                continue
            rows.append({
                "episode": ep.get("id"),
                "end": (ep.get("endTime") or "")[:19].replace("T", " "),
                "reward": mine.get("reward"),
                "score_before": mine.get("initialScore"),
                "score_after": mine.get("updatedScore"),
                "op_team": op.get("teamId"),
                "op_score": op.get("initialScore"),
            })
        rows.sort(key=lambda r: r["end"])
        allrows.append((sid, me_team, rows))

    for sid, me_team, rows in allrows:
        if not rows:
            print(f"\n== submission {sid}: エピソードなし")
            continue
        w = sum(1 for r in rows if (r["reward"] or 0) > 0)
        l = sum(1 for r in rows if (r["reward"] or 0) < 0)
        d = len(rows) - w - l
        first = rows[0]["score_before"] or 0
        last = rows[-1]["score_after"] or 0
        print(f"\n== submission {sid} (team {me_team})")
        print(f"  {len(rows)} 戦  {w}勝 {l}敗 {d}分  勝率 {w / len(rows):.1%}")
        print(f"  レート {first:.1f} → {last:.1f}")
        print(f"  {'終了':<19} {'結果':<4} {'自分':>7} {'相手':>7}  相手チーム")
        for r in rows:
            res = "勝ち" if (r["reward"] or 0) > 0 else ("負け" if (r["reward"] or 0) < 0 else "分け")
            nm, sc = names.get(r["op_team"], ("?", 0.0))
            print(f"  {r['end']:<19} {res:<4} {r['score_before'] or 0:7.1f} "
                  f"{r['op_score'] or 0:7.1f}  {nm} ({r['op_team']})")

        # 相手のレート帯ごとの成績。負けている相手の水準を見る
        band: dict = {}
        for r in rows:
            k = int((r["op_score"] or 0) // 100 * 100)
            e = band.setdefault(k, [0, 0])
            e[0] += 1
            e[1] += int((r["reward"] or 0) > 0)
        print("  [相手のレート帯]")
        for k in sorted(band):
            n, win = band[k]
            print(f"    {k:>5}台  {n:>3} 戦  {win} 勝")


if __name__ == "__main__":
    main()
