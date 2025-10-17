#!/usr/bin/env bash
set -euo pipefail

# Project root and explicit python path (cron-safe)
PROJ="$HOME/Desktop/odds_value_ai"
PY="$PROJ/.venv/bin/python"
ENV_FILE="$PROJ/.env"
LOG_DIR="$PROJ/logs"

# Move to project root
cd "$PROJ"

# Load API keys if present
if [ -f "$ENV_FILE" ]; then
  # shellcheck disable=SC1090
  . "$ENV_FILE"
fi

# Ensure logs dir
mkdir -p "$LOG_DIR"
export TZ=Europe/Copenhagen

# Timestamp (portable on macOS)
ts="$(date -u +%Y%m%d_%H%M)"
LOGFILE="$LOG_DIR/morning_${ts}.log"

# Portable lock (works without flock)
LOCKDIR="$LOG_DIR/morning.lock"
if mkdir "$LOCKDIR" 2>/dev/null; then
  trap 'rmdir "$LOCKDIR"' EXIT
else
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) another morning job is running, exiting" | tee -a "$LOGFILE"
  exit 0
fi

echo "== MORNING RUN $ts ==" | tee -a "$LOGFILE"

# 1) Fixtures + standings + teamstats (+ optional per-fixture stats with cap)
"$PY" scripts/build_features.py \
  --season 2025 --leagues 39,140,78 \
  --with-stats --with-stats-limit 40 --with-stats-sleep 0.25 \
  2>&1 | tee -a "$LOGFILE"

# 2) Train models
"$PY" scripts/train_models.py 2>&1 | tee -a "$LOGFILE"

# 3) Generate picks and append to ledger (keeps credits modest)
ARGS=(
  scripts/predict_upcoming.py
  --days 21 --require-real-odds
  --min-ev 0.02 --min-kelly 0.01 --top-k 25
  --export-md --log-bets
)

# Add Postgres logging if POSTGRES_URL is available
if [ -n "${POSTGRES_URL:-}" ]; then
  ARGS+=( --postgres-url "$POSTGRES_URL" )
fi

"$PY" "${ARGS[@]}" 2>&1 | tee -a "$LOGFILE"

echo "== DONE MORNING $ts ==" | tee -a "$LOGFILE"