#!/usr/bin/env python3
import os
import pandas as pd, numpy as np
import requests
from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
import json
import joblib
from typing import Optional

# Optional Postgres logging via SQLAlchemy
try:
    import sqlalchemy as _sa
    _HAS_SA = True
except Exception:
    _HAS_SA = False
from common import FEAT, PICKDIR, SPORT_WHITELIST, add_normalized_teams
from common import ODIR

ODDSAPI_KEY = os.getenv("ODDSAPI_KEY")
ODDS_BASE = "https://api.the-odds-api.com/v4"

# -------------------- Helpers --------------------
def _kelly_fraction(p: float, o: float) -> float:
    b = o - 1.0
    if b <= 0 or not np.isfinite(b) or not np.isfinite(p):
        return 0.0
    q = 1.0 - p
    f = (b * p - q) / b
    return max(0.0, float(f))

def _ev(p: float, o: float) -> float:
    if not np.isfinite(p) or not np.isfinite(o):
        return np.nan
    return p * o - 1.0

# -------------------- Model scoring helpers --------------------
def _load_model_safe(path: Path):
    """Load a sklearn model saved with joblib, return (estimator, feature_names) or (None, None)."""
    try:
        est = joblib.load(path)
        # sklearn estimators often expose feature_names_in_
        feat = getattr(est, "feature_names_in_", None)
        if isinstance(feat, np.ndarray):
            feat = list(feat)
        return est, feat
    except Exception as e:
        print(f"[proba] could not load {path.name}: {e}")
        return None, None

def _align_X(df: pd.DataFrame, feature_names) -> pd.DataFrame:
    """Return X with the estimator's expected feature columns, filling missing with 0.0."""
    if not feature_names:
        return pd.DataFrame(index=df.index)
    X = df.reindex(columns=feature_names)
    # Avoid FutureWarning on fillna(dtype object)
    X = X.apply(pd.to_numeric, errors="ignore")
    X = X.fillna(0.0)
    return X

def _score_1x2(up: pd.DataFrame, model_dir: Path = Path("models")) -> pd.DataFrame:
    """Attach p_hat_h/d/a using model_1x2.pkl if present."""
    mpath = model_dir / "model_1x2.pkl"
    est, feat = _load_model_safe(mpath)
    if est is None:
        return up.assign(p_hat_h=np.nan, p_hat_d=np.nan, p_hat_a=np.nan)
    X = _align_X(up, feat)
    try:
        proba = est.predict_proba(X)
        # map columns using classes_
        cls = list(getattr(est, "classes_", []))
        # classes could be ['H','D','A'] or similar
        ph = pd.Series(0.0, index=up.index, dtype="float64")
        pdw = pd.Series(0.0, index=up.index, dtype="float64")
        pa = pd.Series(0.0, index=up.index, dtype="float64")
        for i, c in enumerate(cls):
            k = str(c).upper()
            if k in ("H","HOME","1"):
                ph = proba[:, i]
            elif k in ("D","DRAW","X"):
                pdw = proba[:, i]
            elif k in ("A","AWAY","2"):
                pa = proba[:, i]
        out = up.copy()
        out["p_hat_h"] = ph
        out["p_hat_d"] = pdw
        out["p_hat_a"] = pa
        return out
    except Exception as e:
        print(f"[proba] 1x2 scoring failed: {e}")
        return up.assign(p_hat_h=np.nan, p_hat_d=np.nan, p_hat_a=np.nan)

def _score_ou25(up: pd.DataFrame, model_dir: Path = Path("models")) -> pd.DataFrame:
    """Attach p_over_25 and p_under_25 using model_ou25.pkl if present."""
    mpath = model_dir / "model_ou25.pkl"
    est, feat = _load_model_safe(mpath)
    if est is None:
        return up.assign(p_over_25=np.nan, p_under_25=np.nan)
    X = _align_X(up, feat)
    try:
        proba = est.predict_proba(X)
        cls = list(getattr(est, "classes_", []))
        p_over = None
        p_under = None
        for i, c in enumerate(cls):
            k = str(c).lower()
            if "over" in k or k in ("o","1", "over2.5"):
                p_over = proba[:, i]
            if "under" in k or k in ("u","0", "under2.5"):
                p_under = proba[:, i]
        if p_over is None and proba.shape[1] == 2:
            p_over = proba[:, 1]
        if p_under is None and proba.shape[1] == 2:
            p_under = proba[:, 0]
        out = up.copy()
        out["p_over_25"] = p_over if p_over is not None else np.nan
        out["p_under_25"] = p_under if p_under is not None else np.nan
        return out
    except Exception as e:
        print(f"[proba] ou25 scoring failed: {e}")
        return up.assign(p_over_25=np.nan, p_under_25=np.nan)

def _score_ahm05(up: pd.DataFrame, model_dir: Path = Path("models")) -> pd.DataFrame:
    """Attach p_home_cover for AH Home -0.5 using model_ah_home_m0_5.pkl if present."""
    mpath = model_dir / "model_ah_home_m0_5.pkl"
    est, feat = _load_model_safe(mpath)
    if est is None:
        return up.assign(p_home_cover=np.nan)
    X = _align_X(up, feat)
    try:
        proba = est.predict_proba(X)
        cls = list(getattr(est, "classes_", []))
        p_home = None
        for i, c in enumerate(cls):
            k = str(c).lower()
            if k in ("home","h","1","cover","win"):
                p_home = proba[:, i]
        if p_home is None and proba.shape[1] == 2:
            p_home = proba[:, 1]
        out = up.copy()
        out["p_home_cover"] = p_home if p_home is not None else np.nan
        return out
    except Exception as e:
        print(f"[proba] ahm05 scoring failed: {e}")
        return up.assign(p_home_cover=np.nan)

def _select_picks(scored: pd.DataFrame, args) -> pd.DataFrame:
    """Build a picks DataFrame across requested markets with EV/Kelly filters and top-k selection."""
    rows = []

    # 1X2
    if "1x2" in args.markets.lower():
        df = scored.copy()
        # EVs
        df["EV_H"] = _ev(pd.to_numeric(df.get("p_hat_h"), errors="coerce"),
                         pd.to_numeric(df.get("odds_h"), errors="coerce"))
        df["EV_D"] = _ev(pd.to_numeric(df.get("p_hat_d"), errors="coerce"),
                         pd.to_numeric(df.get("odds_d"), errors="coerce"))
        df["EV_A"] = _ev(pd.to_numeric(df.get("p_hat_a"), errors="coerce"),
                         pd.to_numeric(df.get("odds_a"), errors="coerce"))
        # choose best side
        ev_cols = ["EV_H","EV_D","EV_A"]
        best = df[ev_cols].astype(float).idxmax(axis=1)
        side_map = {"EV_H":("Home","H"), "EV_D":("Draw","D"), "EV_A":("Away","A")}
        df["recommended_market"] = "1X2"
        df["recommended_side_text"] = best.map(lambda x: side_map.get(x, ("?", None))[0])
        df["recommended_side"] = best.map(lambda x: side_map.get(x, ("?", None))[1])
        # pick price/prob
        df["recommended_price"] = np.where(df["recommended_side"]=="H", df.get("odds_h"),
                                    np.where(df["recommended_side"]=="D", df.get("odds_d"),
                                             df.get("odds_a")))
        df["model_prob"] = np.where(df["recommended_side"]=="H", df.get("p_hat_h"),
                             np.where(df["recommended_side"]=="D", df.get("p_hat_d"),
                                      df.get("p_hat_a")))
        df["best_EV"] = np.where(df["recommended_side"]=="H", df["EV_H"],
                          np.where(df["recommended_side"]=="D", df["EV_D"], df["EV_A"]))
        # Kelly/stake
        df["kelly_best"] = df.apply(lambda r: _kelly_fraction(float(r["model_prob"]) if pd.notna(r["model_prob"]) else np.nan,
                                                              float(r["recommended_price"]) if pd.notna(r["recommended_price"]) else np.nan), axis=1)
        if args.bankroll is not None:
            df["stake"] = np.clip(df["kelly_best"], 0.0, 1.0) * float(args.bankroll) * float(args.stake_kelly_frac)
        # columns for output
        rows.append(df)

    # OU 2.5
    if "ou25" in args.markets.lower():
        df = scored.copy()
        # odds columns expected: ou25_over_odds, ou25_under_odds (ensured in loader)
        df["EV_OVER"] = _ev(pd.to_numeric(df.get("p_over_25"), errors="coerce"),
                            pd.to_numeric(df.get("ou25_over_odds"), errors="coerce"))
        df["EV_UNDER"] = _ev(pd.to_numeric(df.get("p_under_25"), errors="coerce"),
                             pd.to_numeric(df.get("ou25_under_odds"), errors="coerce"))
        best = df[["EV_OVER","EV_UNDER"]].astype(float).idxmax(axis=1)
        side_map = {"EV_OVER":("Over","O"), "EV_UNDER":("Under","U")}
        df["recommended_market"] = "OU25"
        df["recommended_side_text"] = best.map(lambda x: side_map.get(x, ("?", None))[0])
        df["recommended_side"] = best.map(lambda x: side_map.get(x, ("?", None))[1])
        df["recommended_price"] = np.where(df["recommended_side"]=="O", df.get("ou25_over_odds"), df.get("ou25_under_odds"))
        df["model_prob"] = np.where(df["recommended_side"]=="O", df.get("p_over_25"), df.get("p_under_25"))
        df["best_EV"] = np.where(df["recommended_side"]=="O", df["EV_OVER"], df["EV_UNDER"])
        df["kelly_best"] = df.apply(lambda r: _kelly_fraction(float(r["model_prob"]) if pd.notna(r["model_prob"]) else np.nan,
                                                              float(r["recommended_price"]) if pd.notna(r["recommended_price"]) else np.nan), axis=1)
        if args.bankroll is not None:
            df["stake"] = np.clip(df["kelly_best"], 0.0, 1.0) * float(args.bankroll) * float(args.stake_kelly_frac)
        rows.append(df)

    # AH Home -0.5
    if "ahm05" in args.markets.lower():
        df = scored.copy()
        # expected: ah_home_m0_5_odds / ah_away_p0_5_odds
        df["EV_HOME"] = _ev(pd.to_numeric(df.get("p_home_cover"), errors="coerce"),
                            pd.to_numeric(df.get("ah_home_m0_5_odds"), errors="coerce"))
        # For away +0.5, probability is (1 - p_home_cover)
        df["EV_AWAY"] = _ev(1.0 - pd.to_numeric(df.get("p_home_cover"), errors="coerce"),
                            pd.to_numeric(df.get("ah_away_p0_5_odds"), errors="coerce"))
        best = df[["EV_HOME","EV_AWAY"]].astype(float).idxmax(axis=1)
        side_map = {"EV_HOME":("AH Home -0.5","H"), "EV_AWAY":("AH Away +0.5","A")}
        df["recommended_market"] = "AHM0.5"
        df["recommended_side_text"] = best.map(lambda x: side_map.get(x, ("?", None))[0])
        df["recommended_side"] = best.map(lambda x: side_map.get(x, ("?", None))[1])
        df["recommended_price"] = np.where(df["recommended_side"]=="H", df.get("ah_home_m0_5_odds"), df.get("ah_away_p0_5_odds"))
        df["model_prob"] = np.where(df["recommended_side"]=="H", df.get("p_home_cover"), 1.0 - df.get("p_home_cover"))
        df["best_EV"] = np.where(df["recommended_side"]=="H", df["EV_HOME"], df["EV_AWAY"])
        df["kelly_best"] = df.apply(lambda r: _kelly_fraction(float(r["model_prob"]) if pd.notna(r["model_prob"]) else np.nan,
                                                              float(r["recommended_price"]) if pd.notna(r["recommended_price"]) else np.nan), axis=1)
        if args.bankroll is not None:
            df["stake"] = np.clip(df["kelly_best"], 0.0, 1.0) * float(args.bankroll) * float(args.stake_kelly_frac)
        rows.append(df)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True, sort=False)

    # Optional filters
    if args.min_ev is not None:
        out = out[pd.to_numeric(out["best_EV"], errors="coerce") >= float(args.min_ev)]
    if args.min_kelly is not None and "kelly_best" in out:
        out = out[pd.to_numeric(out["kelly_best"], errors="coerce") >= float(args.min_kelly)]

    # Keep only rows that have both odds & probs
    has_odds = pd.to_numeric(out["recommended_price"], errors="coerce").notna()
    has_prob = pd.to_numeric(out["model_prob"], errors="coerce").notna()
    out = out[has_odds & has_prob].copy()

    # Top-K by EV
    if args.top_k and len(out) > int(args.top_k):
        out = out.sort_values("best_EV", ascending=False).head(int(args.top_k)).copy()

    # Standardize a few identifiers for logging
    for c in ["fixture_id","league_id","season"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")

    return out

# -------------------- Odds fetch --------------------

# --- OddsAPI helpers ---
def get_odds_for_sport(sport_key: str, regions: str = "uk,eu", market: str = "h2h") -> pd.DataFrame:
    """Fetch current market odds (default h2h) for a given OddsAPI sport key and return best price per side."""
    import pandas as pd
    import numpy as np

    if not ODDSAPI_KEY:
        print("⛔ Set ODDSAPI_KEY in your environment.")
        return pd.DataFrame()

    url = f"{ODDS_BASE}/sports/{sport_key}/odds"
    params = {
        "apiKey": ODDSAPI_KEY,
        "regions": regions,
        "markets": market,
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }
    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        key_hint = (ODDSAPI_KEY[:6] + "…") if ODDSAPI_KEY else "<missing>"
        print(f"⚠️ OddsAPI {sport_key} HTTP {r.status_code}: {r.text[:160]} (ODDSAPI_KEY={key_hint})")
        return pd.DataFrame()

    evs = r.json()
    rows = []
    for ev in evs:
        home = ev.get("home_team")
        away = ev.get("away_team")
        best_h = best_d = best_a = np.nan
        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                if mk.get("key") != "h2h":
                    continue
                for out in mk.get("outcomes", []):
                    nm = (out.get("name") or "").strip().lower()
                    pr = out.get("price")
                    if nm == "draw":
                        best_d = np.nanmax([best_d, pr])
                    elif home and nm == home.strip().lower():
                        best_h = np.nanmax([best_h, pr])
                    elif away and nm == away.strip().lower():
                        best_a = np.nanmax([best_a, pr])
        rows.append({
            "sport_key": ev.get("sport_key"),
            "date": pd.to_datetime(ev.get("commence_time"), utc=True, errors="coerce"),
            "home": home,
            "away": away,
            "odds_h": best_h,
            "odds_d": best_d,
            "odds_a": best_a,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = add_normalized_teams(df, "home", "away")
    return df


def get_totals_for_sport(sport_key: str, regions: str = "uk,eu", total_point: float = 2.5) -> pd.DataFrame:
    """
    Safe stub: (Optional) Fetch Over/Under market odds for a given sport_key and total_point (e.g., 2.5 goals).
    For now we return an empty, well-typed DataFrame so the script doesn't crash when this path is used.
    """
    import pandas as pd
    cols = [
        "fixture_id", "sport_key", "bookmaker", "bet_type",
        "total_point", "odds_over", "odds_under",
        "fetched_at_utc", "home", "away", "home_norm", "away_norm",
    ]
    return pd.DataFrame(columns=cols)


def get_spreads_for_sport(sport_key: str, regions: str = "uk,eu", target_line: float = -0.5) -> pd.DataFrame:
    """Fetch spreads (AH) from OddsAPI and return best odds at Home -0.5 / Away +0.5 (closest line)."""
    import pandas as pd
    import numpy as np

    if not ODDSAPI_KEY:
        return pd.DataFrame()

    url = f"{ODDS_BASE}/sports/{sport_key}/odds"
    params = {
        "apiKey": ODDSAPI_KEY,
        "regions": regions,
        "markets": "spreads",
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }
    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        key_hint = (ODDSAPI_KEY[:6] + "…") if ODDSAPI_KEY else "<missing>"
        print(f"⚠️ OddsAPI spreads {sport_key} HTTP {r.status_code}: {r.text[:160]} (ODDSAPI_KEY={key_hint})")
        return pd.DataFrame()

    evs = r.json()
    rows = []
    for ev in evs:
        home = ev.get("home_team")
        away = ev.get("away_team")
        best_home = best_away = np.nan
        best_line_home = best_line_away = None
        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                if mk.get("key") != "spreads":
                    continue
                for out in mk.get("outcomes", []):
                    nm = (out.get("name") or "").strip()
                    pr = out.get("price")
                    pt = out.get("point")
                    if not isinstance(pr, (int, float)) or pt is None:
                        continue
                    if nm == home:
                        if best_line_home is None or abs(float(pt) - target_line) < abs(float(best_line_home) - target_line):
                            best_line_home, best_home = pt, pr
                        elif float(pt) == float(best_line_home):
                            best_home = np.nanmax([best_home, pr])
                    elif nm == away:
                        target_away = -target_line
                        if best_line_away is None or abs(float(pt) - target_away) < abs(float(best_line_away) - target_away):
                            best_line_away, best_away = pt, pr
                        elif float(pt) == float(best_line_away):
                            best_away = np.nanmax([best_away, pr])
        rows.append({
            "sport_key": ev.get("sport_key"),
            "date": pd.to_datetime(ev.get("commence_time"), utc=True, errors="coerce"),
            "home": home,
            "away": away,
            "ah_home_line": best_line_home,
            "ah_away_line": best_line_away,
            "ah_home_m0_5_odds": best_home if best_line_home is not None else np.nan,
            "ah_away_p0_5_odds": best_away if best_line_away is not None else np.nan,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = add_normalized_teams(df, "home", "away")
    return df

# -------------------- Main --------------------
# -------------------- Main --------------------

LEDGER_COLS = [
    "pick_ts_utc","fixture_id","date","league_id","season","sport_key",
    "home","away","recommended_market","recommended_side","recommended_book","recommended_price",
    "model_prob","implied_prob","fair_odds","edge_pp","edge_pct","best_EV","kelly_best","stake",
    "odds_source","status","result","pnl_units","closed_at_utc"
]

def _ensure_ledger(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        import pandas as _pd
        _pd.DataFrame(columns=LEDGER_COLS).to_csv(path, index=False)
    return path

def _append_ledger(path: Path, df: pd.DataFrame, pick_ts_utc: str) -> None:
    # Map the picks frame to ledger columns and append
    use = df.copy()
    use = use.assign(
        pick_ts_utc=pick_ts_utc,
        status="open",
        result=pd.NA,
        pnl_units=np.nan,
        closed_at_utc=pd.NA,
    )
    # keep only known columns if present
    keep = [c for c in LEDGER_COLS if c in use.columns or c in ["pick_ts_utc","status","result","pnl_units","closed_at_utc"]]
    # Ensure all required exist
    for c in LEDGER_COLS:
        if c not in use.columns:
            use[c] = pd.NA
    use = use[LEDGER_COLS]
    # append
    if path.exists() and path.stat().st_size > 0:
        base = pd.read_csv(path)
        out = pd.concat([base, use], ignore_index=True)
    else:
        out = use
    out.to_csv(path, index=False)


# -------------------- Postgres logging (optional) --------------------


# --- Helper: check if a table has required columns in Postgres ---
def _pg_table_has_columns(engine_or_conn, table: str, required_cols) -> bool:
    """
    Returns True if all required_cols exist in given table (case-insensitive).
    engine_or_conn: SQLAlchemy Connection or Engine
    """
    try:
        # Accept both engine and connection
        conn = engine_or_conn
        if hasattr(engine_or_conn, "connect"):
            conn = engine_or_conn.connect()
        q = _sa.text("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = :table
        """)
        res = conn.execute(q, {"table": table})
        cols = {r[0].lower() for r in res}
        return all(str(c).lower() in cols for c in required_cols)
    except Exception:
        return False

# --- Postgres picks append compatibility wrapper ---
def _append_postgres_picks(db_url: str, df: pd.DataFrame, pick_ts_utc: str) -> bool:
    """
    Compatibility wrapper: detects legacy picks table (id but not pick_id), and routes to legacy or v2 logic.
    """
    if not db_url:
        return False
    if not _HAS_SA:
        print("⚠️ SQLAlchemy not available; skipping Postgres logging.")
        return False
    eng = _sa.create_engine(db_url, pool_pre_ping=True)
    try:
        with eng.connect() as conn:
            res = conn.execute(_sa.text("SELECT to_regclass('public.picks')"))
            exists = bool(res.scalar())
            if exists:
                has_id = _pg_table_has_columns(conn, 'picks', ['id'])
                has_pick_uuid = _pg_table_has_columns(conn, 'picks', ['pick_id'])
                if has_id and not has_pick_uuid:
                    return _append_postgres_picks_legacy(db_url, df, pick_ts_utc)
    except Exception:
        # if detection fails, fall through to v2 which is safe/no-op if table is new
        pass
    # default to v2 normalized upsert
    return _append_postgres_picks_v2(db_url, df, pick_ts_utc)

# --- Internal: legacy append (wide table, append-only, no schema change) ---
def _append_postgres_picks_legacy(db_url: str, df: pd.DataFrame, pick_ts_utc: str) -> bool:
    """
    Appends picks to legacy wide picks table (append-only, no schema changes).
    """
    try:
        use = df.copy()
        legacy_cols = [
            "date", "sport_key", "league_id", "season", "fixture_id", "home", "away",
            "recommended_market", "recommended_side", "recommended_book", "recommended_price",
            "model_prob", "implied_prob", "fair_odds", "edge_pp", "edge_pct", "best_EV",
            "kelly_best", "stake", "odds_h", "odds_d", "odds_a", "p_hat_h", "p_hat_d", "p_hat_a",
            "odds_source", "status", "result", "pnl_units", "closed_at_utc"
        ]
        # Ensure all legacy columns plus pick_ts_utc
        for c in legacy_cols + ["pick_ts_utc"]:
            if c not in use.columns:
                use[c] = pd.NA
        # Defensive: ensure columns order
        out_cols = ["pick_ts_utc"] + legacy_cols
        use = use.assign(pick_ts_utc=pick_ts_utc)
        use = use[out_cols]
        # Convert date columns to timezone-aware timestamps where possible
        for dtcol in ["pick_ts_utc", "date", "closed_at_utc"]:
            if dtcol in use.columns:
                use[dtcol] = pd.to_datetime(use[dtcol], utc=True, errors="coerce")
        eng = _sa.create_engine(db_url, pool_pre_ping=True)
        # Use pandas to_sql to append (no schema changes)
        use.to_sql("picks", eng, if_exists="append", index=False)
        print(f"[pg] ↩︎ legacy picks append: {len(use)} rows")
        return True
    except Exception as e:
        print(f"Postgres legacy logging failed: {e}")
        return False

# --- Internal: v2 normalized upsert (moved from previous implementation) ---
def _append_postgres_picks_v2(db_url: str, df: pd.DataFrame, pick_ts_utc: str) -> bool:
    """Map picks DataFrame to normalized Postgres schema and upsert.

    Target table schema (already created via migrations):
        picks(
          pick_id uuid primary key default gen_random_uuid(),
          created_at timestamptz not null default now(),
          fixture_id bigint not null,
          league_id int,
          season int,
          kick_off timestamptz,
          home text,
          away text,
          market text not null default '1x2',
          selection text not null check (selection in ('H','D','A')),
          model_prob double precision,
          best_odds double precision,
          ev double precision,
          kelly double precision,
          stake_units double precision,
          source text,
          model_tag text,
          unique (fixture_id, market, selection, model_tag)
        )
    """
    # Defensive copy
    use = df.copy()

    # Normalize required columns from the script's internal naming
    # Expected input columns (best-effort):
    #   fixture_id, league_id, season, date, home, away,
    #   recommended_market, recommended_side, recommended_price,
    #   model_prob, best_EV, kelly_best, stake, odds_source, model_tag
    # We will gracefully coerce alternatives where possible.

    # Ensure datetime is timezone-aware for kick_off
    if "date" in use.columns:
        use["date"] = pd.to_datetime(use["date"], utc=True, errors="coerce")

    # Map side → selection ('H','D','A')
    def _side_to_sel(x: str) -> Optional[str]:
        if x is None:
            return None
        s = str(x).strip().lower()
        if s in ("h", "home", "1"): return "H"
        if s in ("d", "draw", "x"): return "D"
        if s in ("a", "away", "2"): return "A"
        return None

    # Fallback: if recommended_* are missing, try to build them from EV columns
    if "recommended_market" not in use.columns and {"EV_H","EV_D","EV_A"}.issubset(set(use.columns)):
        use["recommended_market"] = "1x2"
        # pick the best EV side
        ev_cols = ["EV_H","EV_D","EV_A"]
        best_idx = use[ev_cols].astype(float).idxmax(axis=1)
        use["recommended_side"] = best_idx.str[-1].map({"H":"H","D":"D","A":"A"})
        # price from odds_* for that side
        def best_price_row(r):
            m = r.get("recommended_side")
            return {"H": r.get("odds_h"), "D": r.get("odds_d"), "A": r.get("odds_a")}.get(m)
        use["recommended_price"] = use.apply(best_price_row, axis=1)
        # model_prob from p_hat_*
        def best_prob_row(r):
            m = r.get("recommended_side")
            return {"H": r.get("p_hat_h"), "D": r.get("p_hat_d"), "A": r.get("p_hat_a")}.get(m)
        use["model_prob"] = use.apply(best_prob_row, axis=1)
        # kelly (if not already computed)
        if "kelly_best" not in use.columns:
            def kelly_row(r):
                p = r.get("model_prob"); o = r.get("recommended_price")
                try:
                    return _kelly_fraction(float(p), float(o))
                except Exception:
                    return None
            use["kelly_best"] = use.apply(kelly_row, axis=1)

    # Safe defaults
    for col in [
        "fixture_id","league_id","season","date","home","away",
        "recommended_market","recommended_side","recommended_price",
        "model_prob","best_EV","kelly_best","stake","odds_source","model_tag"
    ]:
        if col not in use.columns:
            use[col] = pd.NA

    # Normalize selection & market
    sel = use["recommended_side"].apply(_side_to_sel)
    use["selection"] = sel
    # default market
    use["market"] = use.get("recommended_market").fillna("1x2").astype(str)

    # Compute EV if missing
    if "best_EV" not in use or use["best_EV"].isna().all():
        try:
            p = pd.to_numeric(use["model_prob"], errors="coerce")
            o = pd.to_numeric(use["recommended_price"], errors="coerce")
            use["best_EV"] = p * o - 1.0
        except Exception:
            use["best_EV"] = pd.NA

    # Kelly/stake normalization
    use["kelly"] = pd.to_numeric(use.get("kelly_best"), errors="coerce")
    use["stake_units"] = pd.to_numeric(use.get("stake"), errors="coerce")

    # kick_off
    use["kick_off"] = use.get("date")

    # Source / tag
    use["source"] = use.get("odds_source").fillna("mixed")
    if "model_tag" not in use.columns or use["model_tag"].isna().all():
        # simple default tag by UTC date
        ts = (pd.to_datetime(pick_ts_utc, utc=True, errors="coerce") or pd.Timestamp.utcnow()).strftime("%Y-%m-%d")
        use["model_tag"] = f"logreg_1x2_{ts}"

    # Final column mapping for INSERT
    mapped = use[[
        "fixture_id","league_id","season","kick_off","home","away",
        "market","selection","model_prob","recommended_price","best_EV",
        "kelly","stake_units","source","model_tag"
    ]].copy()

    mapped = mapped.rename(columns={
        "recommended_price": "best_odds",
        "best_EV": "ev",
    })

    # Drop rows without essential fields
    essential = ["fixture_id","market","selection","model_tag"]
    for c in essential:
        mapped = mapped[mapped[c].notna()]
    if mapped.empty:
        print("⚠️ No picks to log (after mapping).")
        return False

    try:
        eng = _sa.create_engine(db_url, pool_pre_ping=True)
        upsert_sql = _sa.text(
            """
            INSERT INTO picks (
                fixture_id, league_id, season, kick_off, home, away,
                market, selection, model_prob, best_odds, ev, kelly, stake_units, source, model_tag
            ) VALUES (
                :fixture_id, :league_id, :season, :kick_off, :home, :away,
                :market, :selection, :model_prob, :best_odds, :ev, :kelly, :stake_units, :source, :model_tag
            )
            ON CONFLICT (fixture_id, market, selection, model_tag) DO UPDATE SET
                model_prob = EXCLUDED.model_prob,
                best_odds  = EXCLUDED.best_odds,
                ev         = EXCLUDED.ev,
                kelly      = EXCLUDED.kelly,
                stake_units= EXCLUDED.stake_units,
                source     = EXCLUDED.source
            """
        )
        create_sql = _sa.text(
            """
            CREATE TABLE IF NOT EXISTS picks (
              pick_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
              created_at timestamptz NOT NULL DEFAULT now(),
              fixture_id bigint NOT NULL,
              league_id int,
              season int,
              kick_off timestamptz,
              home text,
              away text,
              market text NOT NULL DEFAULT '1x2',
              selection text NOT NULL CHECK (selection IN ('H','D','A')),
              model_prob double precision,
              best_odds double precision,
              ev double precision,
              kelly double precision,
              stake_units double precision,
              source text,
              model_tag text,
              UNIQUE (fixture_id, market, selection, model_tag)
            );
            """
        )

        with eng.begin() as conn:
            conn.execute(create_sql)
            # type conversions for safe execute
            mapped["fixture_id"] = pd.to_numeric(mapped["fixture_id"], errors="coerce").astype("Int64")
            # Execute in chunks to be safe
            records = mapped.to_dict(orient="records")
            for chunk_start in range(0, len(records), 500):
                chunk = records[chunk_start:chunk_start+500]
                conn.execute(upsert_sql, chunk)
        print(f"[pg] ✅ logged {len(mapped)} picks → picks")
        return True
    except Exception as e:
        print(f"⚠️ Postgres logging failed: {e}")
        return False


# -------------------- SQLite logging (optional) --------------------
try:
    import sqlite3
    _HAS_SQLITE = True
except Exception:
    _HAS_SQLITE = False

def _ensure_sqlite(db_path: Path) -> Optional['sqlite3.Connection']:
    if not _HAS_SQLITE:
        print("⚠️ sqlite3 not available; skipping SQLite logging.")
        return None
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    # minimal schema for picks; flexible TEXT/REAL types for portability
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS picks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pick_ts_utc TEXT,
            date TEXT,
            sport_key TEXT,
            league_id INTEGER,
            season INTEGER,
            fixture_id INTEGER,
            home TEXT,
            away TEXT,
            recommended_market TEXT,
            recommended_side TEXT,
            recommended_book TEXT,
            recommended_price REAL,
            model_prob REAL,
            implied_prob REAL,
            fair_odds REAL,
            edge_pp REAL,
            edge_pct REAL,
            best_EV REAL,
            kelly_best REAL,
            stake REAL,
            odds_h REAL,
            odds_d REAL,
            odds_a REAL,
            p_hat_h REAL,
            p_hat_d REAL,
            p_hat_a REAL,
            odds_source TEXT,
            status TEXT,
            result TEXT,
            pnl_units REAL,
            closed_at_utc TEXT
        )
        """
    )
    conn.commit()
    return conn

def _append_sqlite_picks(db_path: Path, df: pd.DataFrame, pick_ts_utc: str) -> None:
    conn = _ensure_sqlite(db_path)
    if conn is None:
        return
    try:
        use = df.copy()
        use = use.assign(pick_ts_utc=pick_ts_utc)
        # ensure all expected columns exist
        cols = [
            "pick_ts_utc","date","sport_key","league_id","season","fixture_id",
            "home","away","recommended_market","recommended_side","recommended_book","recommended_price",
            "model_prob","implied_prob","fair_odds","edge_pp","edge_pct","best_EV","kelly_best","stake",
            "odds_h","odds_d","odds_a","p_hat_h","p_hat_d","p_hat_a","odds_source",
            "status","result","pnl_units","closed_at_utc"
        ]
        for c in cols:
            if c not in use.columns:
                use[c] = pd.NA
        # cast dates to iso strings for sqlite
        for c in ("date","closed_at_utc"):
            if c in use.columns:
                use[c] = pd.to_datetime(use[c], errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        use = use[cols]
        use.to_sql("picks", conn, if_exists="append", index=False)
    finally:
        conn.close()


def _load_all_fixtures(fixtures_dir: Path, lookback_days: int = 35) -> pd.DataFrame:
    # Load all league fixtures CSVs and return a combined DataFrame with useful columns
    frames = []
    now = pd.Timestamp.now(tz="UTC")
    cutoff = now - pd.Timedelta(days=lookback_days)
    for p in fixtures_dir.glob("*.csv"):
        try:
            df = pd.read_csv(p)
            # best-effort parse
            for col in ["date","fixture_date","kickoff"]:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
            # unify common columns
            if "date" not in df.columns:
                if "fixture_date" in df.columns:
                    df["date"] = df["fixture_date"]
            # keep reasonably recent only to speed up
            if "date" in df.columns:
                df = df[df["date"] >= cutoff]
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    allx = pd.concat(frames, ignore_index=True)
    # Normalize fixture_id dtype if present
    if "fixture_id" in allx.columns:
        allx["fixture_id"] = pd.to_numeric(allx["fixture_id"], errors="coerce").astype("Int64")
    return allx


def _extract_score(row: pd.Series) -> Optional[tuple]:
    # Try multiple common score column names
    for h,a in [("home_goals","away_goals"),("goals_home","goals_away"),("home_score","away_score"),("goals.home","goals.away")]:
        if h in row and a in row and pd.notna(row[h]) and pd.notna(row[a]):
            try:
                return int(row[h]), int(row[a])
            except Exception:
                pass
    return None


def _settle_1x2(side: str, score: tuple) -> str:
    hg, ag = score
    if hg == ag:
        return "win" if side.lower().startswith("draw") else "lose"
    if hg > ag:
        return "win" if side.lower().startswith("home") else "lose"
    return "win" if side.lower().startswith("away") else "lose"


def _settle_ledger(ledger_path: Path, fixtures_dir: Path, lookback_days: int = 35) -> tuple:
    if not ledger_path.exists():
        return (0,0,0.0)
    led = pd.read_csv(ledger_path)
    if led.empty:
        return (0,0,0.0)
    # Only open 1X2 bets for now
    open_mask = (led["status"] == "open") & (led["recommended_market"].fillna("").str.upper().isin(["1X2"]))
    if "fixture_id" in led.columns:
        led["fixture_id"] = pd.to_numeric(led["fixture_id"], errors="coerce").astype("Int64")
    open_bets = led[open_mask].copy()
    if open_bets.empty:
        return (0,0,0.0)
    allx = _load_all_fixtures(fixtures_dir, lookback_days)
    if allx.empty or "fixture_id" not in allx.columns:
        return (0,0,0.0)
    merged = open_bets.merge(allx, on="fixture_id", how="left", suffixes=("","_fx"))
    settled = 0
    pnl_sum = 0.0
    for idx, r in merged.iterrows():
        # Determine finished
        status = str(r.get("status_short", r.get("status", ""))).upper()
        if status not in ("FT","AET","PEN","WO"):  # finished codes
            continue
        sc = _extract_score(r)
        if not sc:
            continue
        res = _settle_1x2(str(r.get("recommended_side","")), sc)
        stake = float(r.get("stake") or 0.0)
        price = float(r.get("recommended_price") or 0.0)
        pnl = (stake * (price - 1.0)) if res == "win" else (-stake)
        pnl_sum += pnl
        settled += 1
        # write back into main ledger frame by index match on original led
        lid = r.name
    # Update led based on merged (need to re-iterate using fixture_id)
    for i, r in merged.iterrows():
        status = str(r.get("status_short", r.get("status", ""))).upper()
        sc = _extract_score(r)
        if status in ("FT","AET","PEN","WO") and sc:
            res = _settle_1x2(str(r.get("recommended_side","")), sc)
            stake = float(r.get("stake") or 0.0)
            price = float(r.get("recommended_price") or 0.0)
            pnl = (stake * (price - 1.0)) if res == "win" else (-stake)
            mask = (led["fixture_id"].astype("Int64") == r.get("fixture_id")) & (led["status"] == "open")
            led.loc[mask, ["status","result","pnl_units","closed_at_utc"]] = [
                "closed", res, pnl, datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            ]
    led.to_csv(ledger_path, index=False)
    closed_n = (led["status"] == "closed").sum()
    return (int(open_bets.shape[0]), int(closed_n), float(led.get("pnl_units", pd.Series()).fillna(0).sum()))


# -------------------- Load latest API-Football OU/AH snapshots --------------------

def _latest_file(dirpath: Path, prefix: str) -> Optional[Path]:
    try:
        files = sorted(dirpath.glob(f"{prefix}_*.parquet"))
        if not files:
            files = sorted(dirpath.glob(f"{prefix}_*.csv"))
        return files[-1] if files else None
    except Exception:
        return None


def _load_apifootball_ou25() -> pd.DataFrame:
    """Load the latest OU(2.5) snapshot saved by fetch_apifootball_odds.py and normalize columns."""
    p = _latest_file(ODIR, "apifootball_ou25")
    if not p:
        return pd.DataFrame()
    try:
        if p.suffix == ".parquet":
            df = pd.read_parquet(p)
        else:
            df = pd.read_csv(p)
        # expected columns from fetch_apifootball_odds: fixture_id, home, away, home_norm, away_norm,
        # bookmaker_count, ou_line, ou_over, ou_under, date, fetched_at_utc, odds_source
        # Create standard names used in this script
        if "ou25_over_odds" not in df.columns and "ou_over" in df.columns:
            df = df.rename(columns={"ou_over": "ou25_over_odds"})
        if "ou25_under_odds" not in df.columns and "ou_under" in df.columns:
            df = df.rename(columns={"ou_under": "ou25_under_odds"})
        # Keep only needed columns
        keep = [
            c for c in [
                "fixture_id","sport_key","date","home","away","home_norm","away_norm",
                "bookmaker_count","ou25_over_odds","ou25_under_odds"
            ] if c in df.columns
        ]
        df = df[keep].copy()
        # Types
        if "fixture_id" in df.columns:
            df["fixture_id"] = pd.to_numeric(df["fixture_id"], errors="coerce").astype("Int64")
        return df
    except Exception as e:
        print(f"[warn] Could not load OU(2.5) from {p.name}: {e}")
        return pd.DataFrame()


def _load_apifootball_ah(target_line: float = -0.5) -> pd.DataFrame:
    """Load latest AH snapshot and compute best odds at the closest line to target_line (home) and -target_line (away)."""
    p = _latest_file(ODIR, "apifootball_ah")
    if not p:
        return pd.DataFrame()
    try:
        if p.suffix == ".parquet":
            df = pd.read_parquet(p)
        else:
            df = pd.read_csv(p)
        # expected columns: fixture_id, bookmaker_id/name, ah_line, ah_home, ah_away, home_norm, away_norm, date
        for c in ["ah_line","ah_home","ah_away"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        if "fixture_id" in df.columns:
            df["fixture_id"] = pd.to_numeric(df["fixture_id"], errors="coerce").astype("Int64")
        # choose best price at closest line to target
        if df.empty or "ah_line" not in df.columns:
            return pd.DataFrame()
        # compute closest line per fixture for home/away
        # home target = target_line, away target = -target_line
        df["line_diff_home"] = (df["ah_line"] - float(target_line)).abs()
        df["line_diff_away"] = (df["ah_line"] - float(-target_line)).abs()
        # best line per fixture for home side
        idx_home = df.sort_values(["fixture_id","line_diff_home"]).groupby("fixture_id", as_index=False).head(1).index
        # best line per fixture for away side
        idx_away = df.sort_values(["fixture_id","line_diff_away"]).groupby("fixture_id", as_index=False).head(1).index
        best_home = df.loc[idx_home, ["fixture_id","home_norm","away_norm","ah_line","ah_home","date"]].copy()
        best_away = df.loc[idx_away, ["fixture_id","home_norm","away_norm","ah_line","ah_away","date"]].copy()
        best_home = best_home.rename(columns={"ah_line":"ah_home_line","ah_home":"ah_home_m0_5_odds"})
        best_away = best_away.rename(columns={"ah_line":"ah_away_line","ah_away":"ah_away_p0_5_odds"})
        # merge the two so each fixture_id has both columns
        out = pd.merge(
            best_home,
            best_away[["fixture_id","ah_away_line","ah_away_p0_5_odds"]],
            on="fixture_id",
            how="outer"
        )
        # keep identifiers
        id_cols = [c for c in ["fixture_id","home_norm","away_norm","date"] if c in out.columns]
        price_cols = [c for c in ["ah_home_line","ah_away_line","ah_home_m0_5_odds","ah_away_p0_5_odds"] if c in out.columns]
        out = out[id_cols + price_cols]
        return out
    except Exception as e:
        print(f"[warn] Could not load AH from {p.name}: {e}")
        return pd.DataFrame()

def main():
    ap = ArgumentParser()
    ap.add_argument("--days", type=int, default=14, help="Limit to fixtures within the next N days")
    ap.add_argument("--require-real-odds", action="store_true", help="Only keep rows with odds from API-Football or The Odds API")
    ap.add_argument("--min-ev", type=float, default=None, help="Min best_EV to keep (e.g., 0.02)")
    ap.add_argument("--min-kelly", type=float, default=None, help="Min kelly_best to keep (e.g., 0.01)")
    ap.add_argument("--top-k", type=int, default=25, help="Only keep top-K picks by best_EV for the final output")
    ap.add_argument("--bankroll", type=float, default=None, help="If provided, compute suggested stake = bankroll * kelly_best (capped)")
    ap.add_argument("--export-md", action="store_true", help="Also export a human-readable Markdown summary of top picks")
    ap.add_argument("--offline-only", action="store_true", help="Use cached/API-Football data only; skip The Odds API calls")
    ap.add_argument("--no-oddsapi", action="store_true", help="Skip The Odds API even if a key is set")
    ap.add_argument("--safe-test", action="store_true", help="Limit scope per league for fast, cheap iteration")
    ap.add_argument(
        "--stake-kelly-frac",
        type=float,
        default=0.25,
        help="Fractional Kelly til stakes (0.25 = quarter-Kelly). Brug 0 for at deaktivere stakes."
    )
    ap.add_argument("--markets", type=str, default="1x2,ou25,ahm05",
                    help="Comma-separated markets to consider in picks: 1x2, ou25 (Over/Under 2.5), ahm05 (AH Home -0.5)")
    ap.add_argument("--log-bets", action="store_true", help="Append today's picks to a persistent bets ledger (status=open)")
    ap.add_argument("--ledger-path", type=str, default=str(Path("data/ledger/bets_ledger.csv")), help="Path to the bets ledger CSV")
    ap.add_argument("--postgres-url", type=str, default=os.getenv("POSTGRES_URL", ""), help="SQLAlchemy URL til Postgres (fx Supabase). Hvis sat, logger vi picks til tabellen 'picks'.")
    ap.add_argument("--sqlite-db", type=str, default=str(Path("data/db/picks.sqlite")), help="Path til SQLite database for picks (valgfrit)")
    ap.add_argument("--settle", action="store_true", help="Settle open bets in the ledger using finished fixtures from data/fixtures")
    ap.add_argument("--settle-lookback-days", type=int, default=35, help="Look back this many days for finished fixtures when settling")
    args = ap.parse_args()

    # Settlement-only mode
    if getattr(args, "settle", False):
        ledger_path = Path(args.ledger_path)
        fixtures_dir = Path("data/fixtures")
        _ensure_ledger(ledger_path)
        opened, closed_total, pnl_sum = _settle_ledger(ledger_path, fixtures_dir, args.settle_lookback_days)
        print(f"✅ settle: examined open bets={opened}, total closed now={closed_total}, cumulative PnL(units)={pnl_sum:.2f}")
        return

    up_path = FEAT / "upcoming_set.parquet"
    if not up_path.exists():
        print("No upcoming_set.parquet. Run: python scripts/fetch_fixtures.py --season 2025 && python scripts/build_features.py")
        return

    up = pd.read_parquet(up_path)
    # ---- Harden upcoming filter: ensure dates are UTC and rows are truly upcoming ----
    if "date" in up.columns:
        up["date"] = pd.to_datetime(up["date"], utc=True, errors="coerce")
    # Drop rows without a fixture_id (those are likely legacy/train rows)
    if "fixture_id" in up.columns:
        up = up[up["fixture_id"].notna()].copy()
    # Exclude finished/historical rows if such flags exist
    if "finished" in up.columns:
        try:
            up = up[~up["finished"].astype(bool)].copy()
        except Exception:
            pass
    # If API-Football status codes are present, keep only not-started/upcoming ones
    if "status_short" in up.columns:
        upcoming_codes = {"NS","TBD","PST","SUSP","INT"}
        up = up[up["status_short"].isin(upcoming_codes) | up["status_short"].isna()].copy()
    # Align dtype for fixture_id early to ensure merges work
    if "fixture_id" in up.columns:
        up["fixture_id"] = pd.to_numeric(up["fixture_id"], errors="coerce").astype("Int64")
    # Filter to supported leagues only & next N days
    if "sport_key" in up.columns:
        up = up[up["sport_key"].isin(SPORT_WHITELIST)].copy()
    now = pd.Timestamp.now(tz="UTC")
    cutoff = now + pd.Timedelta(days=args.days)

    # Optional: trim by season if column exists (prevents mixing older seasons)
    if "season" in up.columns:
        try:
            this_year = int(now.year)
            up = up[up["season"].isin([this_year, this_year + 1])].copy()
        except Exception:
            pass

    if "date" in up.columns:
        # Keep only fixtures in the horizon and not obviously in the past
        up = up[(up["date"] >= now - pd.Timedelta(hours=1)) & (up["date"] <= cutoff)].copy()

    # Debug counters
    n_after_date = len(up)
    print(f"[debug] rows after date filter: {n_after_date}")
    # extra debug to understand scale
    if "sport_key" in up.columns:
        try:
            print("[debug] sport_key counts:", up["sport_key"].value_counts().to_dict())
        except Exception:
            pass
    if "date" in up.columns and len(up) > 0:
        print(f"[debug] date range in up: {up['date'].min()} → {up['date'].max()}")

    # Ensure one row per fixture to avoid duplicate picks later
    if "fixture_id" in up.columns:
        up = (up.sort_values(["fixture_id", "date"]) if "date" in up.columns else up)
        up = up.drop_duplicates(subset=["fixture_id"], keep="first").copy()
    else:
        # fallback uniqueness by teams+date if fixture_id missing
        uniq_keys = [c for c in ["sport_key", "home_norm", "away_norm", "date"] if c in up.columns]
        if uniq_keys:
            up = up.sort_values(uniq_keys).drop_duplicates(subset=uniq_keys, keep="first").copy()

    if up.empty:
        print("No upcoming fixtures for supported leagues in the selected horizon.")
        return

    # Ensure odds columns exist before any merges (prevents KeyError on combine_first)
    for col in ["odds_h","odds_d","odds_a"]:
        if col not in up.columns:
            up[col] = np.nan

    # Track where odds came from
    if "odds_source" not in up.columns:
        up["odds_source"] = pd.Series([None] * len(up), dtype="object")

    # Prefer API-Football odds if we have a recent snapshot (joins by fixture_id)
    try:
        apifo_files = sorted(ODIR.glob("apifootball_live_odds_*.csv"))
    except Exception:
        apifo_files = []
    if apifo_files:
        apifo_path = apifo_files[-1]
        # Parse 'date' only if present to avoid ValueError on files without a date column
        with open(apifo_path, "r") as _fh:
            first_line = _fh.readline()
        has_date = ("date," in first_line) or first_line.strip().startswith("date") or (",date," in first_line)
        apifo = pd.read_csv(apifo_path, dtype={"fixture_id": "Int64"}, parse_dates=["date"] if has_date else None)
        if "fixture_id" in apifo.columns:
            apifo["fixture_id"] = pd.to_numeric(apifo["fixture_id"], errors="coerce").astype("Int64")
        for col in ("odds_h","odds_d","odds_a"):
            if col in apifo.columns:
                apifo[col] = pd.to_numeric(apifo[col], errors="coerce")
        # merge by fixture_id (most reliable)
        if "fixture_id" in up.columns and "fixture_id" in apifo.columns:
            for col in ["odds_h","odds_d","odds_a"]:
                if col in apifo.columns:
                    aux = col + "_apifb"
                    old_nan = up[col].isna()
                    up = up.merge(
                        apifo[["fixture_id", col]].rename(columns={col: aux}),
                        on="fixture_id", how="left"
                    )
                    if aux in up.columns:
                        filled = old_nan & up[aux].notna()
                        # fill and tag source
                        up[col] = up[col].where(~filled, up[aux])
                        up.loc[filled & up["odds_source"].isna(), "odds_source"] = "apifootball"
                        up.drop(columns=[aux], inplace=True)
            # Ensure destination columns exist before we try to overwrite/where
            for _col in ["src_book_h", "src_book_d", "src_book_a", "bookmaker"]:
                if _col not in up.columns:
                    up[_col] = pd.NA

    # === BEGIN: model scoring & pick selection ===
    # Attach model probabilities
    scored = _score_1x2(up)
    scored = _score_ou25(scored)
    scored = _score_ahm05(scored)

    # If there are no odds rows at all, let the user know
    any_odds_cols = ["odds_h","odds_d","odds_a","ou25_over_odds","ou25_under_odds","ah_home_m0_5_odds","ah_away_p0_5_odds"]
    any_odds = scored[ [c for c in any_odds_cols if c in scored.columns] ]
    n_any_odds = any_odds.notna().any(axis=1).sum() if not any_odds.empty else 0
    print(f"[debug] rows with any_odds: {n_any_odds}, with full_p: {(scored[['p_hat_h','p_hat_d','p_hat_a']].notna().all(axis=1)).sum() if set(['p_hat_h','p_hat_d','p_hat_a']).issubset(scored.columns) else 0}")

    picks = _select_picks(scored, args)
    if picks.empty:
        print("No picks selected with current filters.")
        return

    # Fill in some missing identifiers for logging
    if "odds_source" in scored.columns and "odds_source" in picks.columns:
        picks["odds_source"] = picks["odds_source"].fillna(scored["odds_source"])
    # prefer normalized names if present
    if "home_norm" in scored.columns:
        picks["home"] = picks.get("home_norm", picks.get("home", pd.NA))
    if "away_norm" in scored.columns:
        picks["away"] = picks.get("away_norm", picks.get("away", pd.NA))
    # carry date
    if "date" in scored.columns and "date" in picks.columns:
        picks["date"] = pd.to_datetime(picks["date"], utc=True, errors="coerce").fillna(scored["date"])

    # Record timestamp
    pick_ts_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Ledger append if requested
    if args.log_bets:
        led_path = _ensure_ledger(Path(args.ledger_path))
        _append_ledger(led_path, picks, pick_ts_utc)
        print(f"📒 appended {len(picks)} picks to ledger → {led_path}")

    # Postgres logging if URL is provided
    if getattr(args, "postgres_url", ""):
        ok = _append_postgres_picks(args.postgres_url, picks, pick_ts_utc)
        if ok:
            print("🏦 also logged picks to Postgres (table=picks)")

    # SQLite logging (optional)
    if getattr(args, "sqlite_db", ""):
        _append_sqlite_picks(Path(args.sqlite_db), picks, pick_ts_utc)
        print(f"🗄️  also logged {len(picks)} picks to SQLite → {args.sqlite_db}")

    # Save picks CSV + optional MD
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    PICKDIR.mkdir(parents=True, exist_ok=True)
    out_csv = PICKDIR / f"picks_{ts}.csv"
    picks.to_csv(out_csv, index=False)
    print(f"✅ saved picks → {out_csv} (rows={len(picks)})")
    if args.export_md:
        md = PICKDIR / f"picks_{ts}.md"
        with open(md, "w") as fh:
            fh.write("# Top picks\n\n")
            for _, r in picks.sort_values("best_EV", ascending=False).iterrows():
                line = f"- {r.get('date')} — {r.get('home')} vs {r.get('away')} — **{r.get('recommended_market')} {r.get('recommended_side_text')}** @ {r.get('recommended_price')} (EV={r.get('best_EV'):.3f})\n"
                fh.write(line)
        print(f"📝 also wrote {md}")
    # === END: model scoring & pick selection ===

    # Return to avoid running any legacy code paths below (if any)
    return