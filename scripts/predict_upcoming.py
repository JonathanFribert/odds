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

def _append_postgres_picks(db_url: str, df: pd.DataFrame, pick_ts_utc: str) -> bool:
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
    if not db_url:
        return False
    if not _HAS_SA:
        print("⚠️ SQLAlchemy not available; skipping Postgres logging.")
        return False

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
            for _col in ["src_book_h", "src_book_d", "src_book_a", "bookmaker_count"]:
                if _col not in up.columns:
                    up[_col] = pd.NA
            # also merge the source-bookmaker columns if present (to explain where price came from)
            apifb_cols = [c for c in ["src_book_h","src_book_d","src_book_a","bookmaker_count"] if c in apifo.columns]
            if apifb_cols:
                aux_ren = {c: f"{c}_apifb" for c in apifb_cols}
                up = up.merge(apifo[["fixture_id"] + apifb_cols].rename(columns=aux_ren), on="fixture_id", how="left")
                # Prefer APIFootball source labels where our chosen odds came from APIFootball (or where our label is empty)
                apifb_mask = (up.get("odds_source").fillna("") == "apifootball")
                for c in apifb_cols:
                    aux = f"{c}_apifb"
                    if aux in up.columns:
                        if c not in up.columns:
                            up[c] = pd.NA
                        if c.startswith("src_book_"):
                            up[c] = up[c].where(~apifb_mask, up[aux])
                        elif c == "bookmaker_count":
                            up[c] = up[c].fillna(up[aux])
                        up.drop(columns=[aux], inplace=True)

    # Debug: how many have any odds and source after API-Football merge
    has_odds_apifb = up[["odds_h","odds_d","odds_a"]].notna().any(axis=1).sum()
    print(f"[debug] rows with any odds after API-Football merge: {has_odds_apifb}")

    # Reduce workload when testing: cap ~30 fixtures per league
    if args.safe_test and not up.empty and "league_id" in up.columns:
        up = (
            up.sort_values(["league_id", "date", "fixture_id"])\
              .groupby("league_id", group_keys=False).head(30)
        )

    # Ensure model probability columns exist
    for p in ("p_hat_h","p_hat_d","p_hat_a"):
        if p not in up.columns:
            up[p] = np.nan

    # If probabilities are missing, score upcoming using the trained outcome model
    need_proba = up[["p_hat_h","p_hat_d","p_hat_a"]].isna().all(axis=1).any()
    if need_proba:
        # Prefer our new artifact saved by train_models.py
        model_dict_path = Path("models/model_1x2.pkl")
        used_new_artifact = False
        if model_dict_path.exists():
            try:
                obj = joblib.load(model_dict_path)
                if isinstance(obj, dict) and "model" in obj and "features" in obj:
                    clf = obj["model"]
                    x_cols = obj["features"]
                    used_new_artifact = True
                    print("[proba] using models/model_1x2.pkl (features from artifact)")
                    missing = [c for c in x_cols if c not in up.columns]
                    if missing:
                        print(f"⚠️ Upcoming missing {len(missing)} training features (e.g., {missing[:5]}). Filling 0.")
                    X_up = up.reindex(columns=x_cols).fillna(0.0)
                    X_up = X_up.replace([np.inf, -np.inf], 0.0).clip(-10, 10)
                    proba = clf.predict_proba(X_up)
                    if proba.ndim == 2 and proba.shape[1] == 3:
                        up.loc[:, "p_hat_h"] = proba[:, 0]
                        up.loc[:, "p_hat_d"] = proba[:, 1]
                        up.loc[:, "p_hat_a"] = proba[:, 2]
                    else:
                        print(f"⚠️ Unexpected proba shape {proba.shape}; expected (n,3). Skipping proba assignment.")
            except Exception as e:
                print(f"⚠️ Could not use model_1x2.pkl: {e}")

        if not used_new_artifact:
            # Backwards-compatible: support legacy outcome_model + x_cols.json
            model_candidates = [
                Path("models/outcome_logreg.pkl"),
                Path("models/outcome_model.pkl"),
                Path("data/models/outcome_model.pkl"),
                Path("models/outcome.pkl"),
            ]
            xcols_candidates = [
                Path("models/x_cols.json"),
                Path("data/models/x_cols.json"),
            ]
            model_path = next((p for p in model_candidates if p.exists()), None)
            xcols_path = next((p for p in xcols_candidates if p.exists()), None)
            if model_path is None or xcols_path is None:
                print("⚠️ Could not locate model and/or x_cols.json to score upcoming. Skipping proba scoring.")
            else:
                try:
                    clf = joblib.load(model_path)
                    x_cols = json.loads(xcols_path.read_text())
                    missing = [c for c in x_cols if c not in up.columns]
                    if missing:
                        print(f"⚠️ Upcoming missing {len(missing)} feature columns used in training (e.g., {missing[:5]}...). Filling with 0.")
                    X_up = up.reindex(columns=x_cols).fillna(0.0)
                    X_up = X_up.replace([np.inf, -np.inf], 0.0).clip(-10, 10)
                    proba = clf.predict_proba(X_up)
                    if proba.shape[1] == 3:
                        up.loc[:, "p_hat_h"] = proba[:, 0]
                        up.loc[:, "p_hat_d"] = proba[:, 1]
                        up.loc[:, "p_hat_a"] = proba[:, 2]
                    else:
                        print(f"⚠️ Unexpected proba shape {proba.shape}; expected (n,3). Skipping proba assignment.")
                except Exception as e:
                    print(f"⚠️ Could not compute predict_proba on upcoming: {e}")

    # Debug: how many have all three probabilities now
    has_proba = up[["p_hat_h","p_hat_d","p_hat_a"]].notna().all(axis=1).sum()
    print(f"[debug] rows with full p_hat after scoring: {has_proba}")

    # Normalize names for joining with Odds API
    up = add_normalized_teams(up, "home", "away")

    # Fetch live odds for each supported league and merge
    if args.offline_only or args.no_oddsapi:
        print("⚠️ Skipping The Odds API (offline/no-oddsapi flag). Using API-Football/implied entry odds only.")
        odds_all = pd.DataFrame()
    else:
        odds_frames = []
        for sk in sorted(SPORT_WHITELIST):
            live = get_odds_for_sport(sk)
            if not live.empty:
                odds_frames.append(live)
        odds_all = pd.concat(odds_frames, ignore_index=True) if odds_frames else pd.DataFrame()

    if odds_all.empty:
        print("⚠️ No live odds returned from OddsAPI. Falling back to any existing entry odds as implied.")
    else:
        key = ["sport_key","home_norm","away_norm"]
        for _, col in [("h","odds_h"),("d","odds_d"),("a","odds_a")]:
            live_col = col + "_live"
            old_nan = up[col].isna()
            up = up.merge(odds_all[key + [col]].rename(columns={col: live_col}), on=key, how="left")
            if live_col in up.columns:
                filled = old_nan & up[live_col].notna()
                up[col] = up[col].where(~filled, up[live_col])
                up.loc[filled & up["odds_source"].isna(), "odds_source"] = "oddsapi"
                up.drop(columns=[live_col], inplace=True)

    # Fallback: implied from entry if live missing
    for _, pcol, ocol, ecol in [
        ("h","p_entry_h","odds_h","entry_h"),
        ("d","p_entry_d","odds_d","entry_d"),
        ("a","p_entry_a","odds_a","entry_a"),
    ]:
        if ecol in up.columns:
            up[ocol] = up[ocol].fillna(pd.to_numeric(up[ecol], errors="coerce"))
        if pcol in up.columns:
            miss = up[ocol].isna()
            up.loc[miss, ocol] = (1.0 / up.loc[miss, pcol].clip(lower=1e-6))

    # Any odds now present without a source → came from implied/entry
    got_odds = up[["odds_h","odds_d","odds_a"]].notna().any(axis=1)
    up.loc[got_odds & up["odds_source"].isna(), "odds_source"] = "entry_implied"

    # Clamp odds
    for oc in ("odds_h","odds_d","odds_a"):
        up[oc] = pd.to_numeric(up[oc], errors="coerce").clip(1.01, 200.0)

    # EV + Kelly (fractional Kelly, dynamic)
    for side, pcol, ocol in [("H","p_hat_h","odds_h"),("D","p_hat_d","odds_d"),("A","p_hat_a","odds_a")]:
        up[pcol] = pd.to_numeric(up[pcol], errors="coerce")
        up[f"EV_{side}"] = up[pcol]*up[ocol] - 1.0
        b = (up[ocol] - 1.0).replace(0, np.nan)
        q = 1.0 - up[pcol]
        f = (b*up[pcol] - q) / b
        frac = float(getattr(args, "stake_kelly_frac", 0.25) or 0.0)
        up[f"kelly_{side}"] = (frac * f).clip(lower=0.0).fillna(0.0)

    # Best Kelly across outcomes
    up["kelly_best"] = up[["kelly_H","kelly_D","kelly_A"]].max(axis=1)

    # Compute best_EV/best_side BEFORE filtering so CLI filters can use it (NA-safe)
    up[["EV_H","EV_D","EV_A"]] = up[["EV_H","EV_D","EV_A"]].replace([np.inf, -np.inf], np.nan)
    ev_cols = ["EV_H","EV_D","EV_A"]
    valid_ev = up[ev_cols].notna().any(axis=1)
    # default
    up["best_side"] = pd.Series([None] * len(up), dtype="object")
    up["best_EV"] = np.nan
    # compute only on valid rows
    if valid_ev.any():
        idxmax = up.loc[valid_ev, ev_cols].idxmax(axis=1, skipna=True)
        up.loc[valid_ev, "best_side"] = idxmax.str[-1]
        up.loc[valid_ev, "best_EV"] = up.loc[valid_ev, ev_cols].max(axis=1, skipna=True)

    # Implied probabilities from current odds
    up["imp_h"] = (1.0 / up["odds_h"]).clip(upper=1.0)
    up["imp_d"] = (1.0 / up["odds_d"]).clip(upper=1.0)
    up["imp_a"] = (1.0 / up["odds_a"]).clip(upper=1.0)

    # Fair odds from model probs
    up["fair_h"] = (1.0 / up["p_hat_h"]).replace([np.inf, -np.inf], np.nan)
    up["fair_d"] = (1.0 / up["p_hat_d"]).replace([np.inf, -np.inf], np.nan)
    up["fair_a"] = (1.0 / up["p_hat_a"]).replace([np.inf, -np.inf], np.nan)

    # Probability deltas (percentage points)
    up["edge_pp_h"] = 100.0 * (up["p_hat_h"] - up["imp_h"])
    up["edge_pp_d"] = 100.0 * (up["p_hat_d"] - up["imp_d"])
    up["edge_pp_a"] = 100.0 * (up["p_hat_a"] - up["imp_a"])

    # Price edge (market vs fair). Positive means market price > fair price
    up["edge_price_h"] = (up["odds_h"] / up["fair_h"]) - 1.0
    up["edge_price_d"] = (up["odds_d"] / up["fair_d"]) - 1.0
    up["edge_price_a"] = (up["odds_a"] / up["fair_a"]) - 1.0

    # Build human-friendly recommendation fields
    side_map = {"H": ("Home", "odds_h", "p_hat_h", "imp_h", "fair_h", "edge_pp_h", "edge_price_h", "src_book_h"),
                "D": ("Draw", "odds_d", "p_hat_d", "imp_d", "fair_d", "edge_pp_d", "edge_price_d", "src_book_d"),
                "A": ("Away", "odds_a", "p_hat_a", "imp_a", "fair_a", "edge_pp_a", "edge_price_a", "src_book_a")}
    up["recommended_market"] = "1X2"
    up["recommended_side"] = up["best_side"].map({"H":"Home","D":"Draw","A":"Away"})
    up["recommended_price"] = np.nan
    if "recommended_book" not in up.columns:
        up["recommended_book"] = pd.Series([None]*len(up), dtype="object")
    up["model_prob"] = np.nan
    up["implied_prob"] = np.nan
    up["fair_odds"] = np.nan
    up["edge_pp"] = np.nan
    up["edge_pct"] = np.nan

    for code, (label, oc, pc, ic, fc, epc, eprc, sbc) in side_map.items():
        mask = up["best_side"].eq(code)
        if not mask.any():
            continue
        up.loc[mask, "recommended_price"] = up.loc[mask, oc]
        if sbc in up.columns:
            up.loc[mask, "recommended_book"] = up.loc[mask, sbc]
        up.loc[mask, "model_prob"] = up.loc[mask, pc]
        up.loc[mask, "implied_prob"] = up.loc[mask, ic]
        up.loc[mask, "fair_odds"] = up.loc[mask, fc]
        up.loc[mask, "edge_pp"] = up.loc[mask, epc]
        up.loc[mask, "edge_pct"] = up.loc[mask, eprc]

    # stake sizing applied later on the final `good` frame using chosen Kelly fraction

    # Debug: rows with both any odds and full p_hat BEFORE dropna
    any_odds = up[["odds_h","odds_d","odds_a"]].notna().any(axis=1)
    full_p = up[["p_hat_h","p_hat_d","p_hat_a"]].notna().all(axis=1)
    print(f"[debug] rows with any_odds: {int(any_odds.sum())}, with full_p: {int(full_p.sum())}, with both: {int((any_odds & full_p).sum())}")

    # Keep only rows with both probs & odds
    need = ["p_hat_h","p_hat_d","p_hat_a","odds_h","odds_d","odds_a"]
    for c in need:
        up[c] = pd.to_numeric(up[c], errors="coerce")

    # ========== Extra market logic (OU 2.5, AH -0.5) ==========
    markets = set([m.strip().lower() for m in str(args.markets).split(",") if m.strip()])

    # Container for multi-market picks; start with the existing 1X2 rows (computed below)
    # We'll build 1X2 as 'base' and optionally extend with OU/AH.

    # === Extra markets: try to load models if available ===
    def _load_model(path: Path):
        try:
            if path.exists():
                obj = joblib.load(path)
                if isinstance(obj, dict) and "model" in obj and "features" in obj:
                    return obj["model"], obj["features"]
                return obj, None
        except Exception as e:
            print(f"⚠️ Could not load model {path.name}: {e}")
        return None, None

    model_ou, feat_ou = _load_model(Path("models/model_ou25.pkl"))
    model_ah, feat_ah = _load_model(Path("models/model_ah_home_m0_5.pkl"))

    # Fetch OU / AH odds: prefer API-Football saved snapshots, optionally extend with OddsAPI if allowed
    ou_df = _load_apifootball_ou25()
    ah_df = _load_apifootball_ah(target_line=-0.5)

    if (not args.offline_only) and (not args.no_oddsapi) and ODDSAPI_KEY:
        frames_tot = []
        frames_spread = []
        for sk in sorted(SPORT_WHITELIST):
            if "ou25" in markets:
                tot = get_totals_for_sport(sk)
                if not tot.empty:
                    frames_tot.append(tot)
            if "ahm05" in markets:
                spr = get_spreads_for_sport(sk)
                if not spr.empty:
                    frames_spread.append(spr)
        ou_api = pd.concat(frames_tot, ignore_index=True) if frames_tot else pd.DataFrame()
        ah_api = pd.concat(frames_spread, ignore_index=True) if frames_spread else pd.DataFrame()
        # Harmonize and merge OddsAPI frames (use normalized team keys)
        if not ou_api.empty:
            ou_api = add_normalized_teams(ou_api, "home", "away")
            need_cols = ["sport_key","home_norm","away_norm","ou25_over_odds","ou25_under_odds"]
            if set(need_cols).issubset(ou_api.columns):
                if ou_df.empty:
                    ou_df = ou_api[need_cols]
                else:
                    ou_df = pd.concat([ou_df, ou_api[need_cols]], ignore_index=True)
        if not ah_api.empty:
            ah_api = add_normalized_teams(ah_api, "home", "away")
            need_cols = ["sport_key","home_norm","away_norm","ah_home_m0_5_odds","ah_away_p0_5_odds","ah_home_line","ah_away_line"]
            have_cols = [c for c in need_cols if c in ah_api.columns]
            if have_cols:
                if ah_df.empty:
                    ah_df = ah_api[have_cols]
                else:
                    ah_df = pd.concat([ah_df, ah_api[have_cols]], ignore_index=True)

    # Merge OU odds and score if model exists
    if ("ou25" in markets) and model_ou is not None:
        key = ["sport_key","home_norm","away_norm"]
        if not ou_df.empty:
            if "fixture_id" in ou_df.columns and "fixture_id" in up.columns:
                up = up.merge(ou_df[["fixture_id","ou25_over_odds","ou25_under_odds"]], on="fixture_id", how="left")
            else:
                up = up.merge(ou_df[key + ["ou25_over_odds","ou25_under_odds"]], on=key, how="left")
        if feat_ou is not None and set(feat_ou).issubset(up.columns):
            Xou = up.reindex(columns=feat_ou).fillna(0.0).replace([np.inf,-np.inf],0.0).clip(-10,10)
            try:
                p_over = model_ou.predict_proba(Xou)
                if p_over.ndim == 2 and p_over.shape[1] == 2:
                    p_over = p_over[:,1]
                up["p_over25"] = pd.to_numeric(p_over, errors="coerce")
                up["p_under25"] = 1.0 - up["p_over25"]
            except Exception as e:
                print(f"⚠️ Could not score OU model: {e}")
        # EV/Kelly if we have both probs and odds
        if {"p_over25","ou25_over_odds"}.issubset(up.columns):
            up["ev_over25"] = _ev(up["p_over25"], up["ou25_over_odds"]) 
            up["kelly_over25"] = up.apply(lambda r: _kelly_fraction(r.get("p_over25"), r.get("ou25_over_odds")), axis=1)
        if {"p_under25","ou25_under_odds"}.issubset(up.columns):
            up["ev_under25"] = _ev(up["p_under25"], up["ou25_under_odds"]) 
            up["kelly_under25"] = up.apply(lambda r: _kelly_fraction(r.get("p_under25"), r.get("ou25_under_odds")), axis=1)

    # Merge AH -0.5 odds and score if model exists
    if ("ahm05" in markets) and model_ah is not None:
        key = ["sport_key","home_norm","away_norm"]
        if not ah_df.empty:
            if "fixture_id" in ah_df.columns and "fixture_id" in up.columns:
                cols = [c for c in ["ah_home_m0_5_odds","ah_away_p0_5_odds","ah_home_line","ah_away_line"] if c in ah_df.columns]
                up = up.merge(ah_df[["fixture_id"] + cols], on="fixture_id", how="left")
            else:
                cols = [c for c in ["ah_home_m0_5_odds","ah_away_p0_5_odds","ah_home_line","ah_away_line"] if c in ah_df.columns]
                up = up.merge(ah_df[key + cols], on=key, how="left")
        if feat_ah is not None and set(feat_ah).issubset(up.columns):
            Xah = up.reindex(columns=feat_ah).fillna(0.0).replace([np.inf,-np.inf],0.0).clip(-10,10)
            try:
                p_home = model_ah.predict_proba(Xah)
                if p_home.ndim == 2 and p_home.shape[1] == 2:
                    p_home = p_home[:,1]
                up["p_home_m0_5"] = pd.to_numeric(p_home, errors="coerce")
                up["p_away_p0_5"] = 1.0 - up["p_home_m0_5"]
            except Exception as e:
                print(f"⚠️ Could not score AH model: {e}")
        # EV/Kelly if we have both probs and odds
        if {"p_home_m0_5","ah_home_m0_5_odds"}.issubset(up.columns):
            up["ev_ah_home_m0_5"] = _ev(up["p_home_m0_5"], up["ah_home_m0_5_odds"]) 
            up["kelly_ah_home_m0_5"] = up.apply(lambda r: _kelly_fraction(r.get("p_home_m0_5"), r.get("ah_home_m0_5_odds")), axis=1)

    good = up.dropna(subset=need).copy()

    # ========== Add extra market picks for OU/AH ==========
    extra_frames = []
    # OU 2.5 picks
    if ("ou25" in markets) and {"p_over25","ou25_over_odds","ev_over25","kelly_over25"}.issubset(up.columns):
        ou_pick = up.dropna(subset=["p_over25","ou25_over_odds"]).copy()
        ou_pick["recommended_market"] = "OU 2.5"
        # choose Over vs Under by EV
        ou_pick["ev_over"] = ou_pick.get("ev_over25")
        ou_pick["ev_under"] = ou_pick.get("ev_under25")
        sel_over = ou_pick[["ev_over","ev_under"]].idxmax(axis=1) == "ev_over"
        ou_pick["recommended_side"] = np.where(sel_over, "Over 2.5", "Under 2.5")
        ou_pick["recommended_price"] = np.where(sel_over, ou_pick["ou25_over_odds"], ou_pick["ou25_under_odds"]) 
        ou_pick["model_prob"] = np.where(sel_over, ou_pick["p_over25"], 1.0 - ou_pick["p_over25"]) 
        ou_pick["implied_prob"] = 1.0 / ou_pick["recommended_price"].clip(lower=1e-9)
        ou_pick["fair_odds"] = 1.0 / ou_pick["model_prob"].clip(lower=1e-9)
        ou_pick["edge_pct"] = ou_pick["recommended_price"] / ou_pick["fair_odds"] - 1.0
        ou_pick["edge_pp"] = 100.0 * (ou_pick["model_prob"] - ou_pick["implied_prob"]) 
        # Kelly of the chosen side
        ou_pick["kelly_best"] = np.where(sel_over, ou_pick.get("kelly_over25", 0.0), ou_pick.get("kelly_under25", 0.0))
        # Set best_EV for OU rows so downstream filters/sorting/dedup work correctly
        ou_pick["best_EV"] = np.where(sel_over, ou_pick["ev_over"], ou_pick["ev_under"]) 
        # mark source when coming from OddsAPI totals
        if "odds_source" in ou_pick.columns:
            ou_pick.loc[ou_pick["odds_source"].isna(), "odds_source"] = "oddsapi"
        else:
            ou_pick["odds_source"] = "oddsapi"
        # If OU odds came via API-Football loader, mark them
        if "ou25_over_odds" in ou_df.columns and ou_df.shape[0] > 0:
            # naive flag: if we had no OddsAPI totals merged for a given row but have OU odds, treat as apifootball
            from_apifootball_mask = ou_pick["ou25_over_odds"].notna() & ou_pick["odds_source"].isna()
            ou_pick.loc[from_apifootball_mask, "odds_source"] = "apifootball"
        extra_frames.append(ou_pick)

    # AH -0.5 picks
    if ("ahm05" in markets) and {"p_home_m0_5","ah_home_m0_5_odds","ev_ah_home_m0_5","kelly_ah_home_m0_5"}.issubset(up.columns):
        ah_pick = up.dropna(subset=["p_home_m0_5","ah_home_m0_5_odds"]).copy()
        ah_pick["recommended_market"] = "AH Home -0.5"
        ah_pick["recommended_side"] = "Home -0.5"
        ah_pick["recommended_price"] = ah_pick["ah_home_m0_5_odds"]
        ah_pick["model_prob"] = ah_pick["p_home_m0_5"]
        ah_pick["implied_prob"] = 1.0 / ah_pick["recommended_price"].clip(lower=1e-9)
        ah_pick["fair_odds"] = 1.0 / ah_pick["model_prob"].clip(lower=1e-9)
        ah_pick["edge_pct"] = ah_pick["recommended_price"] / ah_pick["fair_odds"] - 1.0
        ah_pick["edge_pp"] = 100.0 * (ah_pick["model_prob"] - ah_pick["implied_prob"]) 
        ah_pick["kelly_best"] = ah_pick.get("kelly_ah_home_m0_5", 0.0)
        # Ensure best_EV is set for proper ranking/dedup
        ah_pick["best_EV"] = ah_pick["ev_ah_home_m0_5"]
        if "odds_source" in ah_pick.columns:
            ah_pick.loc[ah_pick["odds_source"].isna(), "odds_source"] = "oddsapi"
        else:
            ah_pick["odds_source"] = "oddsapi"
        # If AH odds came via API-Football loader, mark them
        if ("ah_home_m0_5_odds" in ah_df.columns) and (ah_df.shape[0] > 0):
            from_apifootball_mask = ah_pick["ah_home_m0_5_odds"].notna() & ah_pick["odds_source"].isna()
            ah_pick.loc[from_apifootball_mask, "odds_source"] = "apifootball"
        extra_frames.append(ah_pick)

    # If we prepared extra market frames, combine them with the base 1X2 rows later

    if good.empty:
        print("No upcoming fixtures with both model probabilities and odds. Nothing to save.")
        return

    # optional filters
    if args.require_real_odds:
        good = good[good["odds_source"].isin(["apifootball", "oddsapi"])].copy()
    if args.min_ev is not None:
        good = good[good["best_EV"] >= args.min_ev].copy()
    if args.min_kelly is not None:
        good = good[good["kelly_best"] >= args.min_kelly].copy()

    if extra_frames:
        # align columns and stack
        extra = pd.concat(extra_frames, ignore_index=True, sort=False)
        # For multi-market concat we don't want to lose the base 1X2 picks already in `good`.
        good = pd.concat([good, extra], ignore_index=True, sort=False)

    if good.empty:
        print("No picks after filters.")
        return

    # Final de-duplication of picks: keep the row with highest best_EV per fixture/matchup
    if "fixture_id" in good.columns:
        idx = good.groupby("fixture_id")["best_EV"].idxmax()
        good = good.loc[idx].copy()
    else:
        dedup_keys = [k for k in ["sport_key", "home_norm", "away_norm", "date"] if k in good.columns]
        if dedup_keys:
            good = (
                good.sort_values(dedup_keys + ["best_EV"], ascending=[True, True, True, True, False])
                .drop_duplicates(subset=dedup_keys, keep="first")
                .copy()
            )

    # Order by value and trim to top-k picks if requested
    order_cols = ["best_EV","kelly_best","edge_pct","edge_pp"]
    keep_cols = [c for c in [
        "date","sport_key","league_id","season","fixture_id",
        "home","away","odds_source","recommended_market","recommended_side","recommended_book","recommended_price",
        "model_prob","implied_prob","fair_odds","edge_pp","edge_pct","best_EV","kelly_best","stake",
        "odds_h","odds_d","odds_a","p_hat_h","p_hat_d","p_hat_a",
    ] if c in good.columns]

    good = good.sort_values(order_cols, ascending=[False, False, False, False])
    if args.top_k is not None and args.top_k > 0 and len(good) > args.top_k:
        good = good.head(args.top_k).copy()

    # Reorder columns for readability
    good = good[[c for c in keep_cols if c in good.columns] + [c for c in good.columns if c not in keep_cols]]

    # Final stake sizing using Kelly fraction
    frac = float(getattr(args, "stake_kelly_frac", 0.25) or 0.0)
    if args.bankroll is not None and frac > 0:
        good["stake"] = (args.bankroll * good["kelly_best"]).clip(lower=0.0, upper=0.05 * float(args.bankroll))
    elif frac > 0:
        good["stake"] = (100.0 * good["kelly_best"]).round(2)

    # Append to persistent ledger if requested
    ledger_path = Path(args.ledger_path)
    if args.log_bets:
        _ensure_ledger(ledger_path)
        _append_ledger(ledger_path, good.copy(), datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
        print(f"📒 appended {len(good)} picks to ledger → {ledger_path}")

    # Optional: log picks to Postgres
    try:
        pg_url = getattr(args, "postgres_url", "").strip()
        if pg_url:
            pg_ok = _append_postgres_picks(pg_url, good.copy(), datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
            if pg_ok:
                print("🏦 also logged picks to Postgres (table=picks)")
    except Exception as e:
        print(f"⚠️ Postgres logging error: {e}")

    # summary by source
    try:
        src_counts = good["odds_source"].value_counts(dropna=False).to_dict()
        print(f"ℹ️ picks by source: {src_counts}")
    except Exception:
        pass

    # Helper for concise explanation string for each pick
    def _mk_reason(row):
        parts = []
        def add(label, val, fmt="{:+.2f}"):
            try:
                if pd.notna(val):
                    parts.append(f"{label}: " + fmt.format(float(val)))
            except Exception:
                pass
        # recent form (if available)
        if "home_form5_pts" in row and "away_form5_pts" in row:
            add("form5_pts(H-A)", (row.get("home_form5_pts") or 0) - (row.get("away_form5_pts") or 0))
        # table rank diff
        if "rank_diff" in row:
            add("rank_diff(H-A)", row.get("rank_diff"))
        # shots on target per game
        if "home_shots_on_for_pg" in row and "away_shots_on_for_pg" in row:
            add("shots_on_pg(H-A)", (row.get("home_shots_on_for_pg") or 0) - (row.get("away_shots_on_for_pg") or 0))
        # injuries last 30d (fewer home better → A-H)
        if "home_injuries_30d" in row and "away_injuries_30d" in row:
            add("injuries30d(A-H)", (row.get("away_injuries_30d") or 0) - (row.get("home_injuries_30d") or 0))
        # market overround (lower better; show as negative)
        if "overround" in row:
            add("overround", -(row.get("overround") or 0))
        # EV / Kelly
        if "best_EV" in row:
            add("EV", row.get("best_EV"), fmt="{:+.1%}")
        if "kelly_best" in row:
            add("Kelly", row.get("kelly_best"), fmt="{:.1%}")
        return " | ".join(parts[:6])

    try:
        good["why"] = good.apply(_mk_reason, axis=1)
    except Exception:
        pass

    md = None
    if args.export_md:
        def _pct(x):
            try:
                return f"{100*x:.1f}%"
            except Exception:
                return "-"
        # build a short table-like markdown of top picks
        rows = []
        for _, r in good.iterrows():
            dt = pd.to_datetime(r.get("date"))
            dt_str = dt.strftime("%Y-%m-%d %H:%M") if pd.notna(dt) else ""
            why_txt = f" — {r.get('why')}" if pd.notna(r.get("why")) and str(r.get("why")).strip() else ""
            rows.append(
                f"- **{r.get('home')} vs {r.get('away')}** ({dt_str}) — "
                f"[{r.get('recommended_market')}] Play **{r.get('recommended_side')}** @ {r.get('recommended_price')}"
                f" (book: {r.get('recommended_book') or 'best'}); "
                f"model {_pct(r.get('model_prob'))} vs implied {_pct(r.get('implied_prob'))}; "
                f"edge {_pct(r.get('edge_pct'))}, Kelly {r.get('kelly_best'):.3f}{why_txt}"
            )
        md = "\n".join(rows)

    PICKDIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    out = PICKDIR / f"picks_{ts}.csv"
    good.to_csv(out, index=False)
    print(f"✅ saved picks → {out} (rows={len(good)})")
    if args.export_md and md:
        md_path = PICKDIR / f"picks_{ts}.md"
        md_path.write_text(md)
        print(f"📝 also wrote summary → {md_path}")

    # Optional: log picks to SQLite
    try:
        db_path = Path(getattr(args, "sqlite_db", "").strip())
        if db_path:
            _append_sqlite_picks(db_path, good.copy(), datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
            print(f"🗄️  also logged {len(good)} picks to SQLite → {db_path}")
    except Exception as e:
        print(f"⚠️ SQLite logging failed: {e}")


if __name__ == "__main__":
    main()
    
# --- DB logging helpers ---
import os
from sqlalchemy import create_engine

def _pg_url():
    url = os.getenv("POSTGRES_URL")
    if not url:
        return None
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg://", 1)
    elif url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url

def log_picks_to_pg(picks_df, model_tag: str = "logreg_1x2"):
    url = _pg_url()
    if not url or picks_df is None or picks_df.empty:
        return False
    eng = create_engine(url, pool_pre_ping=True)
    cols = ["fixture_id","league_id","season","kick_off","home","away",
            "market","selection","model_prob","best_odds","ev","kelly",
            "stake_units","source","model_tag"]
    df = picks_df.copy()
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df["model_tag"] = model_tag
    with eng.begin() as conn:
        tmp = "_picks_tmp"
        df[cols].to_sql(tmp, conn, if_exists="replace", index=False)
        conn.exec_driver_sql(f"""
        insert into picks ({",".join(cols)})
        select {",".join(cols)} from {tmp}
        on conflict (fixture_id, market, selection, model_tag) do update
        set model_prob = excluded.model_prob,
            best_odds  = excluded.best_odds,
            ev         = excluded.ev,
            kelly      = excluded.kelly,
            stake_units= excluded.stake_units,
            source     = excluded.source,
            created_at = now();
        drop table if exists {tmp};
        """)
    return True
