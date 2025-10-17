#!/usr/bin/env python3
from pathlib import Path
import re
import pandas as pd

# Project paths
ROOT = Path(__file__).resolve().parents[1]
DATADIR = ROOT / "data"
FEAT = DATADIR / "features"
ODIR = DATADIR / "odds_history"
FIXDIR = DATADIR / "fixtures"
PICKDIR = DATADIR / "picks"

# League mapping: API-Football league_id -> OddsAPI sport_key
LEAGUE_MAP = {
    39: "soccer_epl",                  # Premier League
    140: "soccer_spain_la_liga",       # La Liga
    78: "soccer_germany_bundesliga",   # Bundesliga
}

SPORT_WHITELIST = set(LEAGUE_MAP.values())

# --- Team name normalization ---
_norm_re = re.compile(r"\b(fc|cf|afc|club|deportivo|ud|cd|sc)\b|[\.-]", re.I)

def normalize_team(name: str) -> str:
    if not isinstance(name, str):
        return ""
    s = name
    s = _norm_re.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    # simple canonicalisations for common differences
    s = s.replace("manchester utd", "manchester united")
    s = s.replace("athletic club", "athletic bilbao")
    s = s.replace("real betis balompie", "real betis")
    return s


def add_normalized_teams(df: pd.DataFrame, home_col="home", away_col="away") -> pd.DataFrame:
    df = df.copy()
    df["home_norm"] = df[home_col].map(normalize_team)
    df["away_norm"] = df[away_col].map(normalize_team)
    return df


def map_league_id_to_sport_key(league_id):
    try:
        return LEAGUE_MAP.get(int(league_id))
    except Exception:
        return None


def ensure_dirs():
    for d in (DATADIR, FEAT, ODIR, FIXDIR, PICKDIR):
        d.mkdir(parents=True, exist_ok=True)


def load_upcoming():
    f = FEAT / "upcoming_set.parquet"
    if not f.exists():
        return pd.DataFrame(), []
    up = pd.read_parquet(f)
    need = ["p_hat_h","p_hat_d","p_hat_a","odds_h","odds_d","odds_a"]
    return up, need

# Default focus sport for UI
SPORT = "soccer_epl"