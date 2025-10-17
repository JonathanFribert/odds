
set -euo pipefail

PROJ="$HOME/Desktop/odds_value_ai"
ENV="$PROJ/.env"
PY="$PROJ/.venv/bin/python"
LOGS="$PROJ/logs"

mkdir -p "$LOGS"
if [ -f "$ENV" ]; then . "$ENV"; fi

cd "$PROJ"
$PY scripts/predict_upcoming.py --settle --settle-lookback-days 90
