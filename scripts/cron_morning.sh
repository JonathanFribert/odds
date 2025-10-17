
set -euo pipefail

PROJ="$HOME/Desktop/odds_value_ai"
ENV="$PROJ/.env"
PY="$PROJ/.venv/bin/python"
LOGS="$PROJ/logs"

mkdir -p "$LOGS"

if [ -f "$ENV" ]; then . "$ENV"; fi

cd "$PROJ"

$PY scripts/fetch_fixtures.py --season 2025

$PY scripts/build_features.py

$PY scripts/predict_upcoming.py \
  --days 21 --require-real-odds --min-ev 0.02 --min-kelly 0.01 \
  --top-k 25 --export-md --log-bets
