#!/usr/bin/env bash
set -euxo pipefail

cd "/Users/mycomputer/Desktop/odds_value_ai"

if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
fi

export DB_HOST="gondola.proxy.rlwy.net"
export DB_PORT="32418"
export DB_USER="postgres"
export DB_NAME="railway"
export DB_PASS="uHNwGYrTlCRSioIQyNkixASvLhjIhnCZ"
export POSTGRES_URL="postgresql+psycopg://${DB_USER}:${DB_PASS}@${DB_HOST}:${DB_PORT}/${DB_NAME}?sslmode=require"

# Brug din rigtige nøgle her (eller behold den vi har, hvis den virker i dit setup)
export APIFOOTBALL_KEY="${APIFOOTBALL_KEY:-42a6c99a90e23f294a580f08bf1d8415}"
# export ODDSAPI_KEY="PUT_VALID_KEY_HERE"  # valgfrit men anbefalet

python3 -m pip install -q --upgrade pip
python3 -m pip install -q "sqlalchemy>=2" "psycopg[binary]" pandas numpy pyarrow joblib scikit-learn requests

python3 - <<'PY'
import os
from sqlalchemy import create_engine
e=create_engine(os.environ["POSTGRES_URL"], pool_pre_ping=True)
with e.connect() as c:
    print("DB OK:", c.exec_driver_sql("select now()").scalar())
PY

# KØR predict (uden --verbose, det flag findes ikke i din parser)
python3 scripts/predict_upcoming.py --postgres-url "$POSTGRES_URL"
