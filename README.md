
# Odds Value AI (clean start)

## Setup
```bash
cd odds_value_ai
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Check `.env` has your keys (already filled).

## Pipeline
```bash
# 1) Fixtures/results (API-Football)
python scripts/fetch_fixtures.py

# 2) Historical odds (The Odds API) — entry (T-48h) & close (T-5m)
python scripts/fetch_odds_history.py

# 3) Build features
python scripts/build_features.py

# 4) Train models
python scripts/train_models.py

# 5) Dashboard
streamlit run app/app.py
```
