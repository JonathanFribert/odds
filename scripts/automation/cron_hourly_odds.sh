#!/usr/bin/env bash
set -euo pipefail

# Project root
PROJ="$HOME/Desktop/odds_value_ai"
cd "$PROJ"

# Activate venv python path explicitly (more reliable under cron)
PY="$PROJ/.venv/bin/python"

# Load API keys if a local .env exists (exports ODDSAPI_KEY/APIFOOTBALL_KEY)
if [ -f "$PROJ/.env" ]; then
  # shellcheck disable=SC1090
  . "$PROJ/.env"
fi

export TZ=Europe/Copenhagen
mkdir -p "$PROJ/logs"
ts="$(date +%Y%m%d_%H%M)"

# Only odds + picks (no feature building / training) to save credits
"$PY" scripts/predict_upcoming.py \
  --days 14 --require-real-odds \
  --min-ev 0.02 --min-kelly 0.01 --top-k 20 \
  --export-md --log-bets \
  | tee -a "$PROJ/logs/hourly_${ts}.log"