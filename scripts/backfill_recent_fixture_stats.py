#!/usr/bin/env python3
import os, time, math
import pandas as pd
import numpy as np
import requests
from argparse import ArgumentParser
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "data" / "features"
API_BASE = "https://v3.football.api-sports.io"

def _hdr():
    k = os.getenv("APIFOOTBALL_KEY")
    if not k:
        raise RuntimeError("APIFOOTBALL_KEY not set")
    return {"x-apisports-key": k}

def _get(endpoint, params, timeout=25, sleep=0.0):
    r = requests.get(f"{API_BASE}/{endpoint}", headers=_hdr(), params=params, timeout=timeout)
    if r.status_code == 429:
        # blid backoff
        time.sleep(max(1.0, sleep))
        return {"response": []}
    r.raise_for_status()
    return r.json()

def _normkey(s: str) -> str:
    return str(s).strip().lower().replace(" ", "_").replace("%","pct").replace("-", "_")

# Hent statistik for EN færdig fixture (home/away i samme kald uden team-param – API returnerer 2 entries)
def fetch_fixture_statistics_both(fixture_id: int, timeout=20, sleep=0.1) -> dict:
    js = _get("fixtures/statistics", {"fixture": int(fixture_id)}, timeout=timeout, sleep=sleep)
    resp = js.get("response", [])
    sides = {"home": {}, "away": {}}
    for item in resp:
        team = (item.get("team") or {}).get("name","")
        # API returnerer typisk [home, away] i korrekt rækkefølge; team-name kan bruges hvis man vil være mere robust
        side = "home" if not sides["home"] else "away"
        for st in (item.get("statistics") or []):
            typ = _normkey(st.get("type",""))
            val = st.get("value")
            if isinstance(val, str) and val.endswith("%"):
                try:
                    val = float(val.replace("%","").strip())
                except Exception:
                    val = None
            try:
                if isinstance(val, str):
                    val = float(val)
            except Exception:
                pass
            if val is not None:
                sides[side][typ] = val
    # Prefix
    out = {}
    for k,v in sides["home"].items():
        out[f"home_{k}"] = v
    for k,v in sides["away"].items():
        out[f"away_{k}"] = v
    return out

def safe_mean(s):
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() == 0:
        return np.nan
    return s.mean()

def aggregate_form(rows: list[dict]) -> dict:
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    # Vælg relevante nøgler (tilpas efter behov)
    keys = [
        "shots_on_goal","total_shots","shots_off_goal",
        "shots_inside_box","shots_outside_box",
        "ball_possession","passes_pct","passes_accurate",
        "corners","yellow_cards","red_cards","fouls","offsides",
        "expected_goals","xg"  # nogle vendors bruger xg i stedet for expected_goals
    ]
    out = {}
    # Aggreger separat for home_*/away_* for at lave pr-hold form
    for side in ("home","away"):
        for k in keys:
            col = f"{side}_{k}"
            if col in df.columns:
                out[f"{side}_form_{k}_avg{len(df)}"] = safe_mean(df[col])
    return out

def main():
    ap = ArgumentParser()
    ap.add_argument("--season", type=int, default=datetime.now().year, help="Season year (for fixtures?team=.. filter)")
    ap.add_argument("--last", type=int, default=8, help="Antal seneste færdige fixtures pr. hold at bruge")
    ap.add_argument("--limit-fixtures", type=int, default=80, help="Max antal kommende fixtures at behandle i ét run")
    ap.add_argument("--sleep", type=float, default=0.12, help="Pause mellem API-kald")
    args = ap.parse_args()

    up_path = FEAT / "upcoming_set.parquet"
    if not up_path.exists():
        print(f"⛔ {up_path} findes ikke. Kør build_features først.")
        return
    up = pd.read_parquet(up_path)
    up = up.dropna(subset=["fixture_id","home_id","away_id","date"]).copy()
    up["date"] = pd.to_datetime(up["date"], utc=True, errors="coerce")
    up = up.sort_values("date")

    if args.limit_fixtures and len(up) > args.limit_fixtures:
        up = up.head(args.limit_fixtures).copy()
        print(f"… capping to first {len(up)} upcoming fixtures")

    # find sidste K færdigspillede fixtures pr. hold og hent statistics for dem
    rows = []
    for i, r in enumerate(up.itertuples(index=False), 1):
        hid = int(getattr(r, "home_id"))
        aid = int(getattr(r, "away_id"))
        # hent sidste K færdige fixtures for home
        def last_fix_ids(team_id):
            js = _get("fixtures", {"team": team_id, "season": args.season, "last": args.last})
            resp = js.get("response", [])
            # filtrér til færdige
            out = []
            for it in resp:
                status = ((it.get("fixture") or {}).get("status") or {}).get("short")
                if status in ("FT","AET","PEN"):
                    out.append((it.get("fixture") or {}).get("id"))
            return [fid for fid in out if fid]

        h_last = last_fix_ids(hid) or []
        a_last = last_fix_ids(aid) or []

        sample_rows = []
        for fid in h_last[:args.last] + a_last[:args.last]:
            try:
                stats = fetch_fixture_statistics_both(int(fid), sleep=args.sleep)
                if stats:
                    sample_rows.append(stats)
                time.sleep(args.sleep)
            except Exception:
                time.sleep(args.sleep)

        agg = aggregate_form(sample_rows)
        agg.update({
            "fixture_id": int(getattr(r,"fixture_id")),
            "home_id": hid,
            "away_id": aid,
        })
        rows.append(agg)
        if i % 10 == 0 or i == 1:
            print(f"   progress: {i}/{len(up)}")

    if not rows:
        print("⚠️ Ingen form-stats aggregeret")
        return

    form_df = pd.DataFrame(rows)
    outp = FEAT / "upcoming_form_stats.parquet"
    form_df.to_parquet(outp, index=False)
    print(f"✅ saved → {outp} (rows={len(form_df)}, cols={form_df.shape[1]})")

    # merge ind i upcoming_set
    base = pd.read_parquet(up_path)
    merged = base.merge(form_df, on=["fixture_id"], how="left")
    merged.to_parquet(up_path, index=False)
    print(f"🔁 merged form-stats into → {up_path} (shape={merged.shape})")

if __name__ == "__main__":
    main()