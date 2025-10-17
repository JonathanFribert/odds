#!/usr/bin/env bash
set -euo pipefail

# Portable settle runner for macOS (no flock, no GNU date flags)
# - Uses a mkdir-based lock instead of flock
# - Uses POSIX-compatible date formatting

PROJ="$HOME/Desktop/odds_value_ai"
PY="$PROJ/.venv/bin/python"
ENV_FILE="$PROJ/.env"
LOG_DIR="$PROJ/logs"
LOCK_DIR="$LOG_DIR/settle.lockdir"

mkdir -p "$LOG_DIR"
cd "$PROJ"

# Load env if present (API keys, POSTGRES_URL, etc.)
if [ -f "$ENV_FILE" ]; then
  # shellcheck disable=SC1090
  . "$ENV_FILE"
fi

# Acquire lock using mkdir (portable). If already exists, exit quietly.
if mkdir "$LOCK_DIR" 2>/dev/null; then
  # Ensure the lock is released on exit
  cleanup() { rmdir "$LOCK_DIR" 2>/dev/null || true; }
  trap cleanup EXIT INT TERM
else
  echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') another settle job is running, exiting"
  exit 0
fi

# Timestamp for filenames (UTC)
TS="$(date -u '+%Y%m%d_%H%M')"
LOGFILE="$LOG_DIR/settle_${TS}.log"

{
  echo "== SETTLE RUN $TS =="

  ARGS=(
    scripts/predict_upcoming.py
    --settle
    --settle-lookback-days 90
  )

  # Optional Postgres logging
  if [ -n "${POSTGRES_URL:-}" ]; then
    ARGS+=( --postgres-url "$POSTGRES_URL" )
  fi

  # Execute settle
  "$PY" "${ARGS[@]}"

  echo "== DONE SETTLE $TS =="
} | tee -a "$LOGFILE"