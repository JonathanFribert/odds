#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import os
import re
import datetime as dt
from typing import Iterable, List, Tuple
import pandas as pd

# ------------------------------------------------------------
# Project root & well-known folders (absolute, cron-safe)
# ------------------------------------------------------------

# Resolve repo root relative to this file (…/odds_value_ai)
ROOT: Path = Path(__file__).resolve().parents[1]
DATADIR: Path = ROOT / "data"
FEAT: Path = DATADIR / "features"
ODIR: Path = DATADIR / "odds_history"
FIXDIR: Path = DATADIR / "fixtures"
PICKDIR: Path = DATADIR / "picks"
LOGDIR: Path = ROOT / "logs"
MODELDIR: Path = ROOT / "models"
DBDIR: Path = DATADIR / "db"
SQLITE_PATH: Path = DBDIR / "picks.sqlite"

# Ensure expected dirs exist (safe to call often)
def ensure_dirs() -> None:
    for d in (DATADIR, FEAT, ODIR, FIXDIR, PICKDIR, LOGDIR, MODELDIR, DBDIR):
        d.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Environment helpers
# ------------------------------------------------------------

def _getenv(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name, default)
    if v is None or str(v).strip() == "":
        return None
    return v

ODDSAPI_KEY: str | None = _getenv("ODDSAPI_KEY")
APIFOOTBALL_KEY: str | None = _getenv("APIFOOTBALL_KEY")
POSTGRES_URL: str | None = _getenv("POSTGRES_URL")

# ------------------------------------------------------------
# Leagues & sport keys
# ------------------------------------------------------------
# API-Football league_id -> OddsAPI sport_key
LEAGUE_MAP: dict[int, str] = {
    39:  "soccer_epl",                 # Premier League
    140: "soccer_spain_la_liga",      # La Liga
    78:  "soccer_germany_bundesliga", # Bundesliga
}
SPORT_WHITELIST = set(LEAGUE_MAP.values())

LEAGUE_NAME: dict[int, str] = {
    39:  "Premier League",
    140: "La Liga",
    78:  "Bundesliga",
}

# ------------------------------------------------------------
# Team-name normalisation (used across fetch/merge/predict)
# ------------------------------------------------------------
_norm_re = re.compile(r"\b(fc|cf|afc|club|deportivo|ud|cd|sc)\b|[\.-]", re.I)

_DEF_REPLACEMENTS = {
    "manchester utd": "manchester united",
    "athletic club": "athletic bilbao",
    "real betis balompie": "real betis",
}

def normalize_team(name: str) -> str:
    if not isinstance(name, str):
        return ""
    s = _norm_re.sub(" ", name)
    s = re.sub(r"\s+", " ", s).strip().lower()
    for a, b in _DEF_REPLACEMENTS.items():
        if a in s:
            s = s.replace(a, b)
    return s


def add_normalized_teams(df: pd.DataFrame, home_col: str = "home", away_col: str = "away") -> pd.DataFrame:
    out = df.copy()
    out["home_norm"] = out[home_col].map(normalize_team)
    out["away_norm"] = out[away_col].map(normalize_team)
    return out


def map_league_id_to_sport_key(league_id) -> str | None:
    try:
        return LEAGUE_MAP.get(int(league_id))
    except Exception:
        return None

# ------------------------------------------------------------
# Time helpers
# ------------------------------------------------------------

def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def ts_compact(d: dt.datetime | None = None) -> str:
    d = d or now_utc()
    return d.strftime("%Y%m%d_%H%M")

# ------------------------------------------------------------
# IO helpers
# ------------------------------------------------------------

def coerce_float_cols(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def load_parquet_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)

# ------------------------------------------------------------
# Upcoming loader used by app/predict
# ------------------------------------------------------------

NEEDED_PROBA_COLS: Tuple[str, ...] = ("p_hat_h", "p_hat_d", "p_hat_a")
NEEDED_ODDS_COLS: Tuple[str, ...] = ("odds_h", "odds_d", "odds_a")


def load_upcoming() -> tuple[pd.DataFrame, List[str]]:
    """Load upcoming_set.parquet and return (df, missing_cols).
    Ensures odds/proba columns exist and are numeric; returns a list of
    still-missing columns the UI can warn about.
    """
    f = FEAT / "upcoming_set.parquet"
    if not f.exists():
        return pd.DataFrame(), list(NEEDED_PROBA_COLS + NEEDED_ODDS_COLS)

    up = pd.read_parquet(f)

    # Ensure required columns exist
    for c in (*NEEDED_PROBA_COLS, *NEEDED_ODDS_COLS):
        if c not in up.columns:
            up[c] = pd.NA

    up = coerce_float_cols(up, [*NEEDED_PROBA_COLS, *NEEDED_ODDS_COLS])

    # Missing that the app/predict should surface
    missing = [c for c in (*NEEDED_PROBA_COLS, *NEEDED_ODDS_COLS) if c not in up.columns]
    return up, missing

# Default focus sport for UI (can be changed by app)
SPORT = "soccer_epl"

# Make sure directories exist if module imported from scripts
ensure_dirs()