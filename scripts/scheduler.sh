#!/usr/bin/env bash
set -euo pipefail
# Basic logging + ensure logs dir exists when called manually
mkdir -p logs
start_ts="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "== $(date) :: scheduler start :: job=${1:-(none)} =="

mask(){ local v="${1:-}"; [ -n "$v" ] && echo "${v:0:6}***" || echo "(unset)"; }
echo "APIFOOTBALL_KEY=$(mask "$APIFOOTBALL_KEY")  ODDSAPI_KEY=$(mask "$ODDSAPI_KEY")"

# End-of-run status line
trap 'rc=$?; echo "== $(date) :: scheduler end :: job=${1:-(none)} :: rc=$rc =="' EXIT
# 🔐 Sørg for at cron altid har nøglerne
export APIFOOTBALL_KEY=42a6c99a90e23f294a580f08bf1d8415
export ODDSAPI_KEY=7a9e62c6e8dcb5803e25c10d6ed47353
cd "$(dirname "$0")/.."
export PROJ_ROOT="$(pwd)"
echo "PROJ_ROOT=$PROJ_ROOT"
source .venv/bin/activate
LEAGUES="39,140,78"
SEASON="2025"
# --- Dynamic windows from upcoming_set.parquet (no extra files needed) ---
py_eval_windows() {
python - <<'PY'
import pandas as pd, os
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pytz
root = Path(os.environ.get("PROJ_ROOT", ".")).resolve()
feat = root/"data"/"features"/"upcoming_set.parquet"
now = datetime.now(pytz.timezone("Europe/Copenhagen"))
NEAR_H = 6     # 0–6 timer til KO ⇒ odds + picks
MID_H  = 18    # 6–18 timer ⇒ let odds-refresh
DAY_H  = 36    # 18–36 timer ⇒ bred refresh
out = {"NEAR_COUNT":0,"MID_COUNT":0,"DAY_COUNT":0,"NEXT_KO":""}
if feat.exists():
    df = pd.read_parquet(feat)
    if not df.empty and 'date' in df.columns:
        s = pd.to_datetime(df['date'], utc=True).dt.tz_convert('Europe/Copenhagen')
        deltas = (s - now).dt.total_seconds()/3600.0
        near = ((deltas >= 0) & (deltas <= NEAR_H))
        mid  = ((deltas > NEAR_H) & (deltas <= MID_H))
        day  = ((deltas > MID_H) & (deltas <= DAY_H))
        out['NEAR_COUNT'] = int(near.sum())
        out['MID_COUNT']  = int(mid.sum())
        out['DAY_COUNT']  = int(day.sum())
        if len(s)>0:
            out['NEXT_KO'] = s.min().strftime('%Y-%m-%d %H:%M %Z')
print(f"NEAR_COUNT={out['NEAR_COUNT']}")
print(f"MID_COUNT={out['MID_COUNT']}")
print(f"DAY_COUNT={out['DAY_COUNT']}")
print(f"NEXT_KO=\"{out['NEXT_KO']}\"")
PY
}
case "${1:-}" in
  morning)
    python scripts/fetch_apifootball_odds.py --days 14 --use-mapping --sleep 0.35 --write-parquet
    python scripts/build_features.py --season "$SEASON" --leagues "$LEAGUES" --sleep-between 0.3
    python scripts/train_models.py
    python scripts/predict_upcoming.py --days 21 --require-real-odds --top-k 25 --export-md --no-oddsapi
    ;;
  live_odds)
    python scripts/fetch_apifootball_odds.py --days 2 --use-mapping --sleep 0.4 --write-parquet
    ;;
  live_picks)
    python scripts/predict_upcoming.py --days 3 --require-real-odds --top-k 25 --export-md --no-oddsapi
    ;;
  night_backfill)
    python scripts/fetch_apifootball_odds.py --days 1 --include-ou25 --sleep 0.4 --write-parquet
    python scripts/build_features.py --season "$SEASON" --leagues "$LEAGUES" --sleep-between 0.35
    python scripts/train_models.py
    ;;
  schedule)
    # Print/save a simple upcoming schedule CSV for visibility
    python - "$PROJ_ROOT" <<'PY'
import sys, pandas as pd, os
from pathlib import Path
root = Path(sys.argv[1]).resolve()
feat = root/"data"/"features"/"upcoming_set.parquet"
out = root/"data"/"reports"/"upcoming_schedule.csv"
out.parent.mkdir(parents=True, exist_ok=True)
if not feat.exists():
    raise SystemExit(f"upcoming_set not found at {feat}. Run build_features.py first.")
df = pd.read_parquet(feat)
df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_convert('Europe/Copenhagen')
df[['date','league_name','home','away']].sort_values('date').to_csv(out, index=False)
print(f"Saved → {out}")
PY
    ;;
  smart)
    # Run often via cron (e.g., */15 * * * *) — decide what to do based on KO windows
    eval "$(py_eval_windows)"
    echo "windows: NEAR=$NEAR_COUNT MID=$MID_COUNT DAY=$DAY_COUNT next_ko=$NEXT_KO"
    echo "smart: decision=start"
    # If we are within 0–6h of any kickoff → refresh odds tightly + picks
    if [ "${NEAR_COUNT:-0}" -gt 0 ]; then
      echo "smart: action=live_picks reason=NEAR>0"
      python scripts/fetch_apifootball_odds.py --days 1 --use-mapping --sleep 0.35 --write-parquet
      python scripts/predict_upcoming.py --days 1 --require-real-odds --top-k 25 --export-md --no-oddsapi
      exit 0
    fi
    # If 6–18h window has games → light refresh odds (no picks to save credits)
    if [ "${MID_COUNT:-0}" -gt 0 ]; then
      echo "smart: action=odds_refresh reason=MID>0"
      python scripts/fetch_apifootball_odds.py --days 2 --use-mapping --sleep 0.4 --write-parquet
      exit 0
    fi
    # If 18–36h window has games → broader refresh
    if [ "${DAY_COUNT:-0}" -gt 0 ]; then
      echo "smart: action=odds_refresh_broad reason=DAY>0"
      python scripts/fetch_apifootball_odds.py --days 3 --use-mapping --sleep 0.45 --write-parquet
      exit 0
    fi
    echo "smart: action=none reason=no_window"
    ;;
  *)
    echo "Usage: $0 {morning|live_odds|live_picks|night_backfill|smart|schedule}"
    exit 1
    ;;
esac
