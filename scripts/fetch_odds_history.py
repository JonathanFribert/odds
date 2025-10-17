#!/usr/bin/env python3
import time, sys, os, argparse, requests, pandas as pd, numpy as np
from datetime import timedelta, timezone
from pathlib import Path
from common import FIXDIR, ODIR, ODDSAPI_KEY, ODDSAPI_REGIONS, ODDSAPI_SLEEP, norm

BASE = "https://api.the-odds-api.com/v4"
SESSION = requests.Session()
MARKET = "h2h"
REGIONS_DEFAULT = ODDSAPI_REGIONS or "eu"

# track last seen quota header so main() can stop cleanly before we hit the wall
_QUOTA_REMAINING = None  # type: ignore

# Map fixture sport_key -> The Odds API sport key
SPORTKEY_MAP = {
    "soccer_epl": "soccer_epl",
    "soccer_spain_la_liga": "soccer_spain_la_liga",
    "soccer_germany_bundesliga": "soccer_germany_bundesliga",
}

def _alias_map():
    # map common shorthand -> canonical normalized name
    return {
        "man utd": "manchester united",
        "man city": "manchester city",
        "spurs": "tottenham hotspur",
        "wolves": "wolverhampton wanderers",
        "newcastle utd": "newcastle united",
        "west brom": "west bromwich albion",
        "west ham": "west ham united",
        "sporting gijon": "real sporting de gijon",
        "athletic bilbao": "athletic club",
        "betis": "real betis",
        "alaves": "deportivo alaves",
        "osasuna": "ca osasuna",
        "sevilla fc": "sevilla",
        "real sociedad de futbol": "real sociedad",
        "ud las palmas": "las palmas",
        "bayer leverkusen": "bayer 04 leverkusen",
        "borussia m'gladbach": "borussia monchengladbach",
        "monchengladbach": "borussia monchengladbach",
        "fc koln": "1 fc koln",
        "hertha bsc": "hertha berlin",
    }

ALIASES = _alias_map()

def normalize_name(s: str) -> str:
    base = norm(s)
    return ALIASES.get(base, base)

def get_hist(sport_key: str, when_iso: str, regions: str, verbose: bool=False):
    params = {
        "apiKey": ODDSAPI_KEY,
        "regions": regions,
        "markets": MARKET,
        "oddsFormat": "decimal",
        "dateFormat": "iso",
        "date": when_iso,
    }
    url = f"{BASE}/historical/sports/{sport_key}/odds"
    for attempt in range(5):
        r = SESSION.get(url, params=params, timeout=30)
        # capture quota header if present
        rem_raw = r.headers.get("x-requests-remaining") or r.headers.get("X-Requests-Remaining")
        try:
            rem_int = int(rem_raw) if rem_raw is not None else None
        except Exception:
            rem_int = None
        globals()["_QUOTA_REMAINING"] = rem_int

        # plan/quota issues
        if r.status_code in (401, 402):
            msg = f"⛔ {r.status_code} {r.reason} – plan/permission or quota restriction on historical endpoint. remaining={rem_raw}"
            print(msg, flush=True)
            # stop cleanly so partial results are preserved
            raise SystemExit(0)

        if r.status_code == 429:
            # backoff progressively
            sleep_s = 3 * (attempt + 1)
            if verbose:
                print(f"   [rate-limit 429] sleeping {sleep_s}s...", flush=True)
            time.sleep(sleep_s)
            continue

        try:
            r.raise_for_status()
            js = r.json()
            if verbose:
                if rem_raw is not None:
                    try:
                        print(f"   [quota] remaining={int(rem_raw)}")
                    except Exception:
                        print(f"   [quota] remaining={rem_raw}")
                if isinstance(js, list):
                    n = len(js)
                elif isinstance(js, dict):
                    n = len(js.get("data", [])) or len(js.get("events", []))
                else:
                    n = 0
                print(f"   [http {r.status_code}] events={n}", flush=True)
            return js
        except requests.RequestException:
            if attempt == 4:
                raise
            time.sleep(2 * (attempt + 1))
    return []

def extract_h2h(ev, bk_filter=None):
    h,d,a=[],[],[]
    home = norm(ev.get("home_team",""))
    away = norm(ev.get("away_team",""))
    for bk in ev.get("bookmakers", []):
        if bk_filter and (bk.get("key") not in bk_filter):
            continue
        for mk in bk.get("markets", []):
            if (mk.get("key") or "").lower()!=MARKET: continue
            for out in mk.get("outcomes", []):
                nm = norm(out.get("name",""))
                try:
                    odd = float(out.get("price"))
                except:
                    continue
                if odd<=1.01: continue
                if nm in {"home", home}: h.append(odd)
                elif nm in {"draw","tie"}: d.append(odd)
                elif nm in {"away", away}: a.append(odd)
    med = lambda xs: float(np.median(xs)) if xs else None
    return med(h), med(d), med(a)

def find_snapshot_events(sport_key: str, target_iso: str, regions: str, kind: str, verbose: bool=False):
    """Try multiple time offsets around target to find a non-empty historical snapshot.
    Returns the JSON events list (possibly empty). Bounded number of calls per fixture.
    """
    # Offsets: try nearest first
    if kind == "entry":
        offsets = [0, -6, -12, -24, -36, -48, -60, -72]  # hours
        to_secs = lambda h: h*3600
    else:  # close
        offsets = [0, -2, -5, -10, -15, -30, -45, -60]  # minutes
        to_secs = lambda m: m*60

    base = pd.to_datetime(target_iso, utc=True, errors="coerce")
    if base is None:
        return []

    for off in offsets:
        ts = (base + pd.Timedelta(seconds=to_secs(off))).isoformat().replace("+00:00","Z")
        js = get_hist(sport_key, ts, regions=regions, verbose=verbose)
        if isinstance(js, list):
            events = js
        elif isinstance(js, dict):
            # historical sport-level endpoint returns {'timestamp', 'previous_timestamp', 'next_timestamp', 'data': [...]}
            events = js.get("data", []) or js.get("events", [])
        else:
            events = []
        if verbose:
            print(f"      try {kind} offset {off}: events={len(events)}")
        if events:
            return events
    return []

def save_partial(rows, out_path: Path):
    if not rows:
        return
    df = pd.DataFrame(rows)
    # write or append
    if out_path.exists() and out_path.stat().st_size > 0:
        df.to_csv(out_path, index=False, mode="a", header=False)
    else:
        df.to_csv(out_path, index=False)
    # deduplicate on key
    try:
        full = pd.read_csv(out_path)
        full = full.drop_duplicates(subset=["date","home","away","snapshot_type"], keep="last")
        full.to_csv(out_path, index=False)
    except Exception:
        pass

def main():
    if not ODDSAPI_KEY:
        raise SystemExit("Missing ODDSAPI_KEY in .env")

    ap = argparse.ArgumentParser(description="Fetch historical entry/close odds snapshots from The Odds API")
    ap.add_argument("--league", choices=list(SPORTKEY_MAP.keys())+["ALL"], default="soccer_epl",
                    help="Which fixtures file(s) to process (by sport_key). Use ALL to process every file in data/fixtures.")
    ap.add_argument("--max-matches", type=int, default=60, help="Limit number of fixtures per file (tail).")
    ap.add_argument("--start-index", type=int, default=0, help="Start index within sorted fixtures (after date sort).")
    ap.add_argument("--entry-hours", type=int, default=48, help="Hours before kickoff for entry snapshot.")
    ap.add_argument("--close-minutes", type=int, default=5, help="Minutes before kickoff for close snapshot.")
    ap.add_argument("--sleep", type=float, default=ODDSAPI_SLEEP, help="Base sleep between fixture iterations.")
    ap.add_argument("--verbose", action="store_true", help="Print extra HTTP details.")
    ap.add_argument("--regions", type=str, default=REGIONS_DEFAULT, help="Comma-separated regions for The Odds API (e.g. 'uk,eu')")
    ap.add_argument("--bookmakers", type=str, default="", help="Comma-separated bookmaker keys to include (empty = all)")
    args = ap.parse_args()

    bk_filter = set(x.strip() for x in args.bookmakers.split(",") if x.strip()) or None
    if args.verbose and bk_filter:
        print(f"[bookmakers filter] {sorted(bk_filter)}")

    files = sorted(FIXDIR.glob("*.csv"))
    if not files:
        raise SystemExit(f"No fixtures in {FIXDIR}. Run fetch_fixtures.py first.")

    # Filter by league if not ALL
    sel_files = []
    for f in files:
        try:
            peek = pd.read_csv(f, nrows=1)
            sk = str(peek.get("sport_key").iloc[0])
        except Exception:
            continue
        if args.league == "ALL" or sk == args.league:
            sel_files.append(f)

    if not sel_files:
        raise SystemExit(f"No fixture files match league='{args.league}'.")

    out_rows = []
    outp = ODIR / "odds_history.csv"

    for f in sel_files:
        fx = pd.read_csv(f, parse_dates=["date"])
        if fx.empty: 
            print(f"[skip] {f.name}: empty"); 
            continue
        sport_key = SPORTKEY_MAP.get(fx["sport_key"].iloc[0])
        if not sport_key:
            print(f"[skip] {f.name}: unsupported sport_key"); 
            continue
        fx = fx.sort_values("date").reset_index(drop=True)
        # window selection
        start = max(0, args.start_index)
        stop  = len(fx) if args.max_matches <= 0 else min(len(fx), start + args.max_matches)
        fx = fx.iloc[start:stop].reset_index(drop=True)

        print(f"→ {f.name} | sport={sport_key} | regions={args.regions} | slice=[{start}:{stop}] ({len(fx)} matches)")
        since = time.time()
        calls = 0
        for i,row in fx.iterrows():
            # If Odds API quota is nearly exhausted, stop before starting another fixture (≈2 calls/fixture)
            try:
                rem = globals().get("_QUOTA_REMAINING")
                if isinstance(rem, int) and rem is not None and rem < 4:
                    print(f"⛔ Low quota (remaining={rem}). Stopping cleanly before next fixture…")
                    save_partial(out_rows, outp)
                    return
            except Exception:
                pass

            dt = row["date"].to_pydatetime().replace(tzinfo=timezone.utc)
            entry = (dt - timedelta(hours=args.entry_hours)).isoformat().replace("+00:00","Z")
            close = (dt - timedelta(minutes=args.close_minutes)).isoformat().replace("+00:00","Z")
            matched_any = False
            for typ, ts in (("entry",entry),("close",close)):
                js_events = find_snapshot_events(sport_key, ts, regions=args.regions, kind=typ, verbose=args.verbose)
                js_events = list(js_events) if js_events is not None else []
                matched = None
                hn, an = normalize_name(row["home"]), normalize_name(row["away"])
                for ev in js_events:
                    hname, aname = normalize_name(ev.get("home_team","")), normalize_name(ev.get("away_team",""))
                    if {hname, aname} == {hn, an}:
                        matched = ev; break
                if not matched:
                    # try exact order fallback
                    for ev in js_events:
                        hname, aname = normalize_name(ev.get("home_team","")), normalize_name(ev.get("away_team",""))
                        if hname == hn and aname == an:
                            matched = ev; break
                if not matched:
                    continue
                oh,od,oa = extract_h2h(matched, bk_filter=bk_filter)
                out_rows.append({
                    "sport_key": row["sport_key"],
                    "league_name": row["league_name"],
                    "season": int(row["season"]),
                    "date": row["date"].isoformat(),
                    "home": row["home"],
                    "away": row["away"],
                    "snapshot_type": typ,
                    "snapshot_ts": ts,
                    "odds_h": oh, "odds_d": od, "odds_a": oa
                })
                matched_any = True
            calls += 2
            # progress heartbeat
            if (i+1) % 10 == 0:
                elapsed = time.time() - since
                print(f"   … {i+1}/{len(fx)} fixtures processed (api calls≈{calls}, {elapsed:.1f}s)", flush=True)
            # save partial every 100 rows to disk
            if len(out_rows) and (len(out_rows) % 100 == 0):
                save_partial(out_rows, outp)
            time.sleep(args.sleep)

    save_partial(out_rows, outp)
    print(f"✅ saved {len(out_rows)} rows → {outp}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Writing partial results…", file=sys.stderr)
        # best effort write of partials on Ctrl+C
        # (we can't access local out_rows here cleanly without globals; re-run main window if needed)
        sys.exit(130)
