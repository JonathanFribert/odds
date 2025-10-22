#!/usr/bin/env python3
import os, sys, time
import argparse
import json, requests
import pandas as pd, numpy as np
from datetime import datetime, timedelta, timezone as _TZ
from pathlib import Path
from typing import List, Optional

from common import ROOT, FIXDIR, FEAT, ODIR, map_league_id_to_sport_key, add_normalized_teams

API_FOOTBALL_BASE = "https://v3.football.api-sports.io"
FEAT_DIR = Path("data/features")

def _write_train_set(df):
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    out = FEAT_DIR / "train_set.parquet"
    # Sikre deterministiske kolonner (valgfrit)
    try:
        cols = sorted(df.columns)
        df = df[cols]
    except Exception:
        pass
    df.to_parquet(out, index=False)
    print(f"✅ wrote {out} rows={len(df)}")

def _apif_headers():
    k = os.getenv("APIFOOTBALL_KEY")
    if not k:
        raise RuntimeError("APIFOOTBALL_KEY not set")
    return {"x-apisports-key": k}

def _apif_get(endpoint: str, **params):
    url = f"{API_FOOTBALL_BASE}/{endpoint.lstrip('/')}"
    last = None
    for i in range(5):
        try:
            r = requests.get(url, headers=_apif_headers(), params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(0.7 * (i + 1))
                continue
            if r.status_code in (401, 403):
                print(f"⛔ {endpoint} auth {r.status_code}: {r.text[:160]}")
                return []
            r.raise_for_status()
            js = r.json()
            return js.get("response", [])
        except Exception as e:
            last = e
            time.sleep(0.7 * (i + 1))
    if last:
        print(f"⚠️ _apif_get {endpoint} failed: {last}")
    return []

_STAT_KEYMAP = {
    "Shots on Goal": "shots_on_goal",
    "Shots off Goal": "shots_off_goal",
    "Total Shots": "shots_total",
    "Shots insidebox": "shots_inside_box",
    "Shots outsidebox": "shots_outside_box",
    "Fouls": "fouls",
    "Corner Kicks": "corners",
    "Offsides": "offsides",
    "Ball Possession": "ball_possession",
    "Yellow Cards": "yellow_cards",
    "Red Cards": "red_cards",
    "Goalkeeper Saves": "saves",
    "Passes %": "passes_pct",
    "Passes accurate": "passes_accurate",
    "expected_goals": "expected_goals",
    "xG": "expected_goals",
}

def _to_num(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if s.endswith("%"):
        try:
            return float(s[:-1])
        except Exception:
            return None
    try:
        return float(s)
    except Exception:
        return None

def flatten_fixture_statistics(stats_response: list) -> dict:
    out = {}
    if not stats_response:
        return out
    home_team_id = (stats_response[0].get("team") or {}).get("id") if stats_response else None
    for side in stats_response or []:
        team = side.get("team") or {}
        team_id = team.get("id")
        prefix = "home_" if home_team_id and team_id == home_team_id else "away_"
        for it in side.get("statistics") or []:
            name = (it.get("type") or "").strip()
            val = it.get("value")
            key = _STAT_KEYMAP.get(name)
            if key:
                out[f"{prefix}{key}"] = _to_num(val)
            else:
                nn = name.lower().replace(" ", "_")
                if "possession" in nn:
                    out[f"{prefix}ball_possession"] = _to_num(val)
                elif "shots_on_target" in nn or "shots_on_goal" in nn:
                    out[f"{prefix}shots_on_goal"] = _to_num(val)
                elif "total_shots" in nn or "shots_total" in nn:
                    out[f"{prefix}shots_total"] = _to_num(val)
    return out

def fetch_fixture_closing_odds(fid: int, tz: str = "Europe/Copenhagen") -> dict:
    rows = _apif_get("odds", fixture=int(fid), timezone=tz)
    best_h = best_d = best_a = None
    bk_count = 0
    for it in rows or []:
        for bk in it.get("bookmakers", []) or []:
            bk_count += 1
            for bet in bk.get("bets", []) or []:
                nm = (bet.get("name") or "").lower()
                if bet.get("id") == 1 or any(k in nm for k in ("1x2","match winner","win-draw-win","fulltime result")):
                    for v in bet.get("values", []) or []:
                        lbl = (v.get("value") or v.get("label") or "").strip().lower()
                        try:
                            price = float(v.get("odd"))
                        except Exception:
                            price = None
                        if price is None:
                            continue
                        if lbl in ("home","1","team 1","1 (home)"):
                            best_h = max(best_h, price) if best_h is not None else price
                        elif "draw" in lbl or lbl == "x" or "tie" in lbl:
                            best_d = max(best_d, price) if best_d is not None else price
                        elif lbl in ("away","2","team 2","2 (away)"):
                            best_a = max(best_a, price) if best_a is not None else price
    return {"closing_odds_h": best_h, "closing_odds_d": best_d, "closing_odds_a": best_a, "closing_bk_count": bk_count}

def harvest_postmatch(leagues: list[int], season: int, days_back: int = 1, days_fwd: int = 0,
                      timezone_str: str = "Europe/Copenhagen", sleep: float = 0.2):
    """
    Finder fixtures med status=FT i [today-days_back, today+days_fwd] for valgte ligaer,
    henter /fixtures/statistics og 'closing' odds, fladgør og gemmer:
      - data/postmatch/postmatch_raw_YYYYMMDD.csv
      - data/features/postmatch_flat.parquet (upsert på fixture_id)
    """
    outdir = ROOT / "data" / "postmatch"
    outdir.mkdir(parents=True, exist_ok=True)
    flat_rows, raw_rows = [], []
    today = datetime.now(_TZ.utc).date()

    for delta in range(-abs(days_back), days_fwd + 1):
        date = today + timedelta(days=delta)
        for lg in leagues:
            fixtures = _apif_get("fixtures", league=int(lg), season=int(season), date=str(date))
            fixtures = [f for f in fixtures if (f.get("fixture") or {}).get("status", {}).get("short") == "FT"]
            for f in fixtures:
                fx = f.get("fixture") or {}
                fid = fx.get("id")
                if not fid:
                    continue
                home = (f.get("teams") or {}).get("home", {}).get("name")
                away = (f.get("teams") or {}).get("away", {}).get("name")
                dt = fx.get("date")
                goals_h = (f.get("goals") or {}).get("home")
                goals_a = (f.get("goals") or {}).get("away")

                stats = _apif_get("fixtures/statistics", fixture=int(fid))
                closing = fetch_fixture_closing_odds(int(fid), tz=timezone_str)

                raw_rows.append({
                    "fixture_id": fid, "league_id": lg, "season": season, "date": dt,
                    "home": home, "away": away, "goals_h": goals_h, "goals_a": goals_a,
                    "raw_stats": json.dumps(stats), "raw_closing": json.dumps(closing)
                })

                flat = {
                    "fixture_id": fid, "league_id": lg, "season": season, "date": dt,
                    "home": home, "away": away, "goals_h": goals_h, "goals_a": goals_a,
                }
                fstats = flatten_fixture_statistics(stats)
                flat.update(fstats)
                flat.update(closing)
                flat_rows.append(flat)
                time.sleep(sleep)

    ts = datetime.now(_TZ.utc).strftime("%Y%m%d_%H%M")
    if raw_rows:
        pd.DataFrame(raw_rows).to_csv(outdir / f"postmatch_raw_{ts}.csv", index=False)

    FEAT.mkdir(parents=True, exist_ok=True)
    flat_path = FEAT / "postmatch_flat.parquet"
    if flat_rows:
        new_df = pd.DataFrame(flat_rows)
        if flat_path.exists():
            old = pd.read_parquet(flat_path)
            combined = pd.concat(
                [old[~old["fixture_id"].isin(new_df["fixture_id"])], new_df],
                ignore_index=True,
            )
        else:
            combined = new_df
        combined.to_parquet(flat_path, index=False)
        print(f"✅ postmatch harvested: {len(new_df)} new rows; total={len(combined)} → {flat_path}")
    else:
        print("ℹ️ No FT fixtures found in chosen window.")

# === Simple filesystem cache for API-Football ===
from hashlib import sha1

CACHEDIR = ROOT / "data" / ".cache" / "apifootball"
CACHEDIR.mkdir(parents=True, exist_ok=True)

def _cache_key(endpoint: str, params: dict) -> str:
    try:
        items = sorted((params or {}).items())
    except Exception:
        items = []
    payload = json.dumps([endpoint, items], ensure_ascii=True, separators=(",", ":"))
    return sha1(payload.encode("utf-8")).hexdigest()

def _cache_path(endpoint: str, params: dict):
    safe = endpoint.replace("/", "_")
    return CACHEDIR / f"{safe}_{_cache_key(endpoint, params)}.json"

def _cache_read(path, ttl_s: float):
    try:
        if path.exists():
            age = time.time() - path.stat().st_mtime
            if ttl_s <= 0 or age <= ttl_s:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
    except Exception:
        return None
    return None

def _cache_write(path, obj: dict):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f)
    except Exception:
        pass

# Cached GET with retry + optional stale fallback
def _get_cached(endpoint: str, params: dict, ttl_s: float = 600, allow_stale: bool = True):
    path = _cache_path(endpoint, params)
    # Try fresh cache first
    js = _cache_read(path, ttl_s)
    if js is not None:
        resp = (js or {}).get("response", [])
        if resp:
            return resp
    # Otherwise hit network (with the same backoff as _get)
    backoff = 1.2
    for attempt in range(6):
        try:
            r = requests.get(f"{API_BASE}/{endpoint}", headers=HEADERS, params=params, timeout=30)
        except Exception:
            r = None
        if r is not None and r.status_code == 429:
            wait = min(6.0, backoff * (attempt + 1))
            print(f"   ↪︎ 429 rate-limited on /{endpoint} (attempt {attempt+1}), sleeping {wait:.1f}s …")
            time.sleep(wait)
            continue
        try:
            if r is None:
                raise RuntimeError("no response")
            r.raise_for_status()
            js = r.json()
            _cache_write(path, js)
            return (js or {}).get("response", [])
        except Exception as e:
            code = getattr(r, "status_code", "?")
            print(f"   ⚠️ {endpoint} HTTP {code}: {e}")
            break
    # If we reach here, try stale cache as a fallback
    if allow_stale:
        js = _cache_read(path, ttl_s=10**12)  # effectively allow any age
        if js is not None:
            print(f"   ℹ️ using STALE cache for /{endpoint} params={params}")
            return (js or {}).get("response", [])
    print(f"   ❌ cache+network failed for /{endpoint}; params={params}")
    return []

# --- robust nested getter (API may vary keys slightly) ---
def _dig(d: dict, *keys):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur

def _safediv(a, b):
    """Safe division for scalars and pandas Series.
    - Returns NaN on 0-division or invalid inputs.
    - Preserves Series shape/index when any input is a Series.
    """
    import pandas as pd
    import numpy as np

    # Hvis en af dem er Series → vectoriseret vej
    if isinstance(a, pd.Series) or isinstance(b, pd.Series):
        if isinstance(a, pd.Series) and isinstance(b, pd.Series):
            A = pd.to_numeric(a, errors="coerce")
            B = pd.to_numeric(b, errors="coerce")
        elif isinstance(a, pd.Series):
            A = pd.to_numeric(a, errors="coerce")
            try:
                b_val = float(b)
            except Exception:
                b_val = np.nan
            B = pd.Series(b_val, index=A.index, dtype="float64")
        else:  # b er Series
            B = pd.to_numeric(b, errors="coerce")
            try:
                a_val = float(a)
            except Exception:
                a_val = np.nan
            A = pd.Series(a_val, index=B.index, dtype="float64")
        with np.errstate(divide='ignore', invalid='ignore'):
            res = A / B
        return res.replace([np.inf, -np.inf], np.nan)

    # Scalar vej
    try:
        a_val = float(a)
    except Exception:
        a_val = np.nan
    try:
        b_val = float(b)
    except Exception:
        b_val = np.nan
    if not np.isfinite(a_val) or not np.isfinite(b_val) or b_val == 0.0:
        return np.nan
    return a_val / b_val

API_BASE = "https://v3.football.api-sports.io"
API_KEY = os.getenv("APIFOOTBALL_KEY")

HEADERS = {"x-apisports-key": API_KEY}

# ---- Extra API helpers: fetch per-fixture statistics ----
def _apif_headers():
    key = os.getenv("APIFOOTBALL_KEY")
    if not key:
        raise RuntimeError("APIFOOTBALL_KEY not set")
    return {
        "x-apisports-key": key,
        # Some users use RapidAPI headers; we stay with native header already used elsewhere in this file.
    }

def _norm_stat_key(s: str) -> str:
    return (
        str(s).strip().lower().replace(" ", "_").replace("%", "pct").replace("-", "_")
    )

def _extract_stats_map(js_resp) -> dict:
    out = {}
    stats = (js_resp or {}).get("statistics", [])
    for st in stats:
        k = _norm_stat_key(st.get("type", ""))
        v = st.get("value")
        if v is None:
            continue
        try:
            if isinstance(v, str) and v.endswith("%"):
                v = float(v.strip().replace('%','')) / 100.0
            else:
                v = float(v)
            out[k] = v
        except Exception:
            # non-numeric values are ignored
            pass
    return out

def fetch_fixture_statistics(fixture_id: int, team_id: int, sleep: float = 0.25) -> dict:
    """Fetch per-team statistics for a fixture from /fixtures/statistics.
    Returns a dict of normalized numeric stats (keys like 'shots_on_goal', 'possession', 'expected_goals', ...).
    """
    params = {"fixture": int(fixture_id), "team": int(team_id)}
    resp = _get_cached("fixtures/statistics", params, ttl_s=2*60*60, allow_stale=True)
    if not resp:
        if sleep:
            time.sleep(sleep)
        return {}
    m = _extract_stats_map(resp[0])
    if sleep:
        time.sleep(sleep)
    return m

def _attach_upcoming_fixture_stats(up_df: pd.DataFrame, limit: int = 60, per_req_sleep: float = 0.15, timeout: float = 10.0) -> pd.DataFrame:
    """
    Hent /fixtures/statistics for kommende fixtures (home & away) og attach som home_*/away_* kolonner.
    Caper automatisk antal kald med 'limit' for at skåne kvoter.
    """
    if up_df.empty:
        return up_df
    need_cols = {"fixture_id", "home_id", "away_id"}
    if not need_cols.issubset(up_df.columns):
        return up_df

    mask = up_df["fixture_id"].notna()
    pending = up_df.loc[mask, ["fixture_id", "home_id", "away_id"]].drop_duplicates().copy()
    if limit and limit > 0:
        pending = pending.head(limit)

    out_rows = {}
    total = len(pending)
    print(f"   → fetching /fixtures/statistics for upcoming fixtures (home/away)…")
    print(f"   … capping stats fetch to first {len(pending)} fixtures (of {len(up_df)})")

    for idx, row in enumerate(pending.itertuples(index=False), start=1):
        fx = int(row.fixture_id) if pd.notna(row.fixture_id) else None
        hid = int(row.home_id) if pd.notna(row.home_id) else None
        aid = int(row.away_id) if pd.notna(row.away_id) else None
        if fx is None or hid is None or aid is None:
            continue
        if idx % 5 == 0 or idx == 1:
            print(f"      progress: {idx}/{total} (fixture={fx})")

        # Home
        try:
            h_stats = fetch_fixture_statistics(fx, hid, sleep=per_req_sleep) or {}
        except Exception:
            h_stats = {}
        # Away
        try:
            a_stats = fetch_fixture_statistics(fx, aid, sleep=per_req_sleep) or {}
        except Exception:
            a_stats = {}

        prefixed = {}
        for k, v in (h_stats or {}).items():
            prefixed[f"home_{k}"] = v
        for k, v in (a_stats or {}).items():
            prefixed[f"away_{k}"] = v
        out_rows[fx] = prefixed

    if not out_rows:
        return up_df

    fx_list = []
    for fx, kv in out_rows.items():
        d = {"fixture_id": fx}
        d.update(kv)
        fx_list.append(d)
    stats_df = pd.DataFrame(fx_list)

    return up_df.merge(stats_df, on="fixture_id", how="left")

def _attach_historic_fixture_stats(hist_df: pd.DataFrame, limit: int = 400, per_req_sleep: float = 0.15) -> pd.DataFrame:
    """Fetch /fixtures/statistics for FINISHED fixtures in hist_df and return a tidy dataframe
    with columns: fixture_id + prefixed home_*/away_* stats. Caller merges by fixture_id.
    """
    if hist_df is None or hist_df.empty or "fixture_id" not in hist_df.columns:
        return pd.DataFrame(columns=["fixture_id"])

    # kun færdige kampe (filtrér status hvis tilgængelig)
    done = hist_df.copy()
    if "status_short" in done.columns:
        done = done[~done["status_short"].isin({"NS","TBD","PST","SUSP","INT"})]
    done = done.dropna(subset=["fixture_id","home_id","away_id"]).drop_duplicates("fixture_id")
    if done.empty:
        return pd.DataFrame(columns=["fixture_id"])

    # cap for at spare kreditter
    if limit and limit > 0:
        done = done.sort_values("date", ascending=False).head(limit)

    rows = []
    print(f"   → fetching /fixtures/statistics for FINISHED fixtures (cap={len(done)}) …")
    for i, r in enumerate(done.itertuples(index=False), start=1):
        fx = int(getattr(r, "fixture_id"))
        hid = int(getattr(r, "home_id"))
        aid = int(getattr(r, "away_id"))
        if i % 25 == 1 or i == len(done):
            print(f"      progress: {i}/{len(done)} (fixture={fx})")
        try:
            h_stats = fetch_fixture_statistics(fx, hid, sleep=per_req_sleep) or {}
        except Exception:
            h_stats = {}
        try:
            a_stats = fetch_fixture_statistics(fx, aid, sleep=per_req_sleep) or {}
        except Exception:
            a_stats = {}

        row = {"fixture_id": fx}
        for k, v in (h_stats or {}).items():
            row[f"home_{k}"] = v
        for k, v in (a_stats or {}).items():
            row[f"away_{k}"] = v
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["fixture_id"])
    return pd.DataFrame(rows)


def _build_upcoming_set(leagues: list[int], season: int, days_fwd: int = 2,
                        timezone_str: str = "Europe/Copenhagen",
                        stats_limit: int = 40, stats_sleep: float = 0.15) -> pd.DataFrame:
    """Fetch upcoming fixtures within [today, today+days_fwd] for the given leagues/season,
    attach limited /fixtures/statistics (per-team), and write FEAT_DIR/upcoming_set.parquet.
    Returns the dataframe (possibly empty).
    """
    today = datetime.now(_TZ.utc).date()
    rows = []
    for delta in range(0, max(0, int(days_fwd)) + 1):
        date = today + timedelta(days=delta)
        for lg in leagues:
            fixtures = _get("fixtures", {
                "league": int(lg),
                "season": int(season),
                "date": str(date),
                "timezone": "UTC",
            })
            for f in fixtures or []:
                fx = (f or {}).get("fixture", {})
                st = (fx.get("status") or {}).get("short")
                # Keep not-started/scheduled; allow early in-play windows for hourly overlap
                if st not in {"NS", "TBD", "PST", "1H", "HT", "2H"}:
                    continue
                fid = fx.get("id")
                if not fid:
                    continue
                teams = (f.get("teams") or {})
                home = (teams.get("home") or {})
                away = (teams.get("away") or {})
                rows.append({
                    "fixture_id": fid,
                    "league_id": int(lg),
                    "season": int(season),
                    "date": fx.get("date"),
                    "status_short": st,
                    "home": home.get("name"),
                    "away": away.get("name"),
                    "home_id": home.get("id"),
                    "away_id": away.get("id"),
                })

    df = pd.DataFrame(rows)
    FEAT_DIR.mkdir(parents=True, exist_ok=True)

    if df.empty:
        # Skriv en tom fil, så downstream steps kan se at "upcoming_set.parquet" findes (selv uden kampe)
        outp = FEAT_DIR / "upcoming_set.parquet"
        try:
            df.to_parquet(outp, index=False)
            print(f"✅ wrote empty upcoming_set → {outp}")
        except Exception as e:
            print(f"⚠️ could not write empty upcoming_set: {e}")
        return df

    # Prøv at attach'e per-team upcoming statistics i et rimeligt cap
    try:
        df = _attach_upcoming_fixture_stats(df, limit=int(stats_limit), per_req_sleep=float(stats_sleep))
    except Exception as e:
        print(f"⚠️ attach upcoming statistics failed: {e}")

    # Normaliser dato + navne
    try:
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    except Exception:
        pass
    try:
        df = add_normalized_teams(df.copy(), "home", "away")
    except Exception:
        pass

    outp = FEAT_DIR / "upcoming_set.parquet"
    try:
        df.to_parquet(outp, index=False)
        print(f"✅ wrote upcoming_set: rows={len(df)} cols={df.shape[1]} → {outp}")
    except Exception as e:
        print(f"⚠️ could not write upcoming_set: {e}")
    return df

# ---- DB merge helpers: enrich train_set.parquet with outcomes stats from Postgres ----
def _pg_engine():
    """Create a SQLAlchemy engine from POSTGRES_URL if available, else return None."""
    try:
        from sqlalchemy import create_engine  # lazy import so script works without SA
    except Exception:
        return None
    url = os.getenv("POSTGRES_URL")
    if not url:
        return None
    try:
        return create_engine(url, pool_pre_ping=True)
    except Exception:
        return None

def _coalesce_variants(df: pd.DataFrame, bases: list[str]) -> pd.DataFrame:
    """For each base name, coalesce any variant columns (e.g., _x/_y/_left/_right/_train/_db)
    into a single `base` column, preferring the first non-null values in order of appearance.
    Drops the variant columns after coalescing. Returns the modified df.
    """
    if df is None or df.empty:
        return df
    for base in bases:
        cand = []
        for suf in ["", "_x", "_y", "_left", "_right", "_train", "_db"]:
            col = f"{base}{suf}" if suf else base
            if col in df.columns:
                cand.append(col)
        if len(cand) <= 1:
            continue
        s = None
        for cn in cand:
            v = pd.to_numeric(df[cn], errors="coerce") if df[cn].dtype != "O" else df[cn]
            s = v if s is None else s.combine_first(v)
        df[base] = s
        for cn in cand:
            if cn != base and cn in df.columns:
                df.drop(columns=[cn], inplace=True, errors="ignore")
    return df

def _repair_train_set(train_path: Path):
    """
    Ensure train_set has a non-null 'result' label and no duplicate suffix columns.
    - Backfills 'result' from Postgres outcomes for all fixture_ids in train_set.
    - Falls back to reconstructing 'result' from goals_h/goals_a or goals_home/goals_away.
    - Coalesces duplicate variants (_x/_y/_left/_right/_train/_db) and folds goalkeeper_saves -> saves.
    """
    try:
        df = pd.read_parquet(train_path)
    except Exception as e:
        print(f"ℹ️ could not read {train_path}: {e}")
        return

    # Backfill labels from DB
    eng = _pg_engine()
    if eng is not None and "fixture_id" in df.columns and not df.empty:
        fids = df["fixture_id"].dropna().astype("int64").unique().tolist()
        if fids:
            from sqlalchemy import text
            with eng.connect() as c:
                lab = pd.DataFrame(
                    c.execute(text("SELECT fixture_id, result, goals_h, goals_a FROM outcomes WHERE fixture_id = ANY(:ids)"),
                              {"ids": fids}).mappings().all()
                )
            if not lab.empty:
                lab["fixture_id"] = pd.to_numeric(lab["fixture_id"], errors="coerce").astype("Int64")
                df = df.merge(lab, on="fixture_id", how="left", suffixes=("", "_db"))

    # Coalesce result from any variants into a single Series
    def _ser(col):
        return df[col] if (col in df.columns and not isinstance(df[col], pd.DataFrame)) else pd.Series(index=df.index, dtype="object")
    res = _ser("result").copy() if "result" in df.columns else pd.Series(index=df.index, dtype="object")
    for cand in ["result_db","result_x","result_y","result_left","result_right","result_train"]:
        if cand in df.columns and not isinstance(df[cand], pd.DataFrame):
            res = res.where(res.notna(), df[cand])
    # Fallback from goals
    gh = df.get("goals_h", df.get("goals_home", df.get("goals_h_db")))
    ga = df.get("goals_a", df.get("goals_away", df.get("goals_a_db")))
    if gh is not None and ga is not None and res.isna().any():
        def _rf(h,a):
            try:
                h,a = int(h), int(a)
            except Exception:
                return np.nan
            return "H" if h>a else ("A" if a>h else "D")
        need = res.isna()
        ghv = pd.to_numeric(gh, errors="coerce")
        gav = pd.to_numeric(ga, errors="coerce")
        res.loc[need] = [ _rf(h,a) for h,a in zip(ghv[need], gav[need]) ]
    df["result"] = res

    # --- derive training labels ---
    # Goals sources (prefer canonical, then variants from DB/train)
    gh_src = df.get("goals_h", df.get("goals_home", df.get("goals_h_db")))
    ga_src = df.get("goals_a", df.get("goals_away", df.get("goals_a_db")))
    ghv = pd.to_numeric(gh_src, errors="coerce") if gh_src is not None else pd.Series(index=df.index, dtype="float64")
    gav = pd.to_numeric(ga_src, errors="coerce") if ga_src is not None else pd.Series(index=df.index, dtype="float64")

    # OU 2.5: 1 if total goals > 2.5 else 0
    tot = ghv.add(gav, fill_value=np.nan)
    ou = np.where(tot > 2.5, 1.0, 0.0)
    ou[~np.isfinite(tot.values)] = np.nan  # keep NaN where goals missing
    df["label_ou25"] = ou

    # AH home -0.5: 1 if home wins by any margin else 0
    diff = ghv - gav
    ah = np.where(diff > 0, 1.0, 0.0)
    ah[~np.isfinite(diff.values)] = np.nan
    df["label_ah_home_m0_5"] = ah

    # Fold goalkeeper_saves -> saves
    for side in ("home","away"):
        gk = f"{side}_goalkeeper_saves"
        sv = f"{side}_saves"
        if gk in df.columns:
            if sv in df.columns:
                df[sv] = pd.to_numeric(df[sv], errors="coerce").combine_first(pd.to_numeric(df[gk], errors="coerce"))
            else:
                df.rename(columns={gk: sv}, inplace=True)

    # Drop all suffix variants
    suffixes = ("_x","_y","_left","_right","_train","_db")
    dup_cols = [c for c in df.columns if c.endswith(suffixes)]
    if dup_cols:
        df.drop(columns=dup_cols, inplace=True, errors="ignore")

    try:
        df.to_parquet(train_path, index=False)
    except Exception as e:
        print(f"ℹ️ could not write repaired train_set: {e}")
    else:
        nn = int(df["result"].notna().sum()) if "result" in df.columns else 0
        ou_ok = int(pd.notna(df.get("label_ou25")).sum()) if "label_ou25" in df.columns else 0
        ah_ok = int(pd.notna(df.get("label_ah_home_m0_5")).sum()) if "label_ah_home_m0_5" in df.columns else 0
        print(f"✅ repaired train_set: result non-null rows={nn}; ou25 labels={ou_ok}; ah(-0.5) labels={ah_ok}; cols={df.shape[1]}; rows={df.shape[0]}")

def _merge_train_stats_from_db(train_path: Path, batch: int = 2000) -> int:
    """
    Load data/features/train_set.parquet, look up matching fixture_ids in outcomes (Postgres),
    and merge numeric stats like shots_on_goal, possession, expected_goals as
    home_*/away_* columns. Saves back to the same parquet.
    Returns number of rows enriched.
    """
    if not train_path.exists():
        print(f"ℹ️ train_set missing → {train_path}; nothing to merge.")
        return 0
    eng = _pg_engine()
    if eng is None:
        print("ℹ️ POSTGRES_URL or SQLAlchemy missing — skipping DB merge of train stats.")
        return 0

    df = pd.read_parquet(train_path)
    if "fixture_id" not in df.columns or df.empty:
        print("ℹ️ train_set has no fixture_id or is empty — skipping DB merge.")
        return 0

    # FULL desired outcomes schema for training merge (will be filtered to existing cols at runtime)
    wanted = [
        "fixture_id", "result", "goals_h", "goals_a",
        "home_shots_on_goal","away_shots_on_goal",
        "home_ball_possession","away_ball_possession",
        "home_expected_goals","away_expected_goals",
        # extended granular stats (shots/pass/corners/cards/fouls/offsides/saves)
        "home_shots_total","away_shots_total",
        "home_shots_off_goal","away_shots_off_goal",
        "home_shots_inside_box","away_shots_inside_box",
        "home_shots_outside_box","away_shots_outside_box",
        "home_passes_pct","away_passes_pct",
        "home_passes_accurate","away_passes_accurate",
        "home_corners","away_corners",
        "home_yellow_cards","away_yellow_cards",
        "home_red_cards","away_red_cards",
        "home_fouls","away_fouls",
        "home_offsides","away_offsides",
        "home_saves","away_saves",
    ]
    from sqlalchemy import text  # late import to avoid hard dependency

    # Discover which of the wanted columns actually exist in outcomes
    with eng.connect() as c:
        rows = c.execute(text("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name='outcomes' AND column_name = ANY(:cols)
        """), {"cols": [w for w in wanted if w != "fixture_id"]}).fetchall()
    existing = {"fixture_id"} | {r[0] for r in rows}
    cols = [c for c in wanted if c in existing]
    if len(cols) <= 1:
        print("ℹ️ outcomes has none of the requested stat columns — skipping DB merge.")
        return 0
    cols_sql = ", ".join(cols)
    # Pull only fixture_ids we actually have in train_set
    fids = df["fixture_id"].dropna().astype("int64").unique().tolist()
    if not fids:
        print("ℹ️ no fixture_ids in train_set — skipping DB merge.")
        return 0

    merged = df.copy()
    # Clean up any legacy duplicates that may already exist in train_set
    merged = _coalesce_variants(merged, [
        # core
        "home_shots_on_goal","away_shots_on_goal",
        "home_ball_possession","away_ball_possession",
        "home_expected_goals","away_expected_goals",
        # extended
        "home_shots_total","away_shots_total",
        "home_shots_off_goal","away_shots_off_goal",
        "home_shots_inside_box","away_shots_inside_box",
        "home_shots_outside_box","away_shots_outside_box",
        "home_passes_pct","away_passes_pct",
        "home_passes_accurate","away_passes_accurate",
        "home_corners","away_corners",
        "home_yellow_cards","away_yellow_cards",
        "home_red_cards","away_red_cards",
        "home_fouls","away_fouls",
        "home_offsides","away_offsides",
        "home_saves","away_saves",
    ])
    enriched = 0
    # Chunk to keep IN () manageable
    for i in range(0, len(fids), batch):
        chunk = fids[i:i+batch]
        with eng.connect() as c:
            rows = pd.DataFrame(
                c.execute(
                    text(f"SELECT {cols_sql} FROM outcomes WHERE fixture_id = ANY(:ids)"),
                    {"ids": chunk},
                ).mappings().all()
            )
        if rows.empty:
            continue
        # Coerce numeric columns
        for col in rows.columns:
            if col == "fixture_id":
                continue
            rows[col] = pd.to_numeric(rows[col], errors="coerce")
        # Merge
        before_na = merged.filter(regex=r'^(home|away)_(shots|ball|expected|passes|corners|yellow|red|fouls|offsides)').isna().sum().sum() if any(
            c.startswith(("home_","away_")) for c in merged.columns) else None
        # Avoid overlapping 'result' column during merge (train_set already has it)
        cols = [c for c in cols if c != "result"]
        merged = merged.merge(rows, on="fixture_id", how="left", suffixes=("_train", "_db"))

        # Coalesce potential duplicate suffix variants into a single canonical column
        merged = _coalesce_variants(merged, [
            # core
            "home_shots_on_goal","away_shots_on_goal",
            "home_ball_possession","away_ball_possession",
            "home_expected_goals","away_expected_goals",
            # extended
            "home_shots_total","away_shots_total",
            "home_shots_off_goal","away_shots_off_goal",
            "home_shots_inside_box","away_shots_inside_box",
            "home_shots_outside_box","away_shots_outside_box",
            "home_passes_pct","away_passes_pct",
            "home_passes_accurate","away_passes_accurate",
            "home_corners","away_corners",
            "home_yellow_cards","away_yellow_cards",
            "home_red_cards","away_red_cards",
            "home_fouls","away_fouls",
            "home_offsides","away_offsides",
            "home_saves","away_saves",
        ])
        enriched += len(rows)

    # Save back if anything merged
    if enriched:
        merged.to_parquet(train_path, index=False)
        print(f"✅ train_set enriched with DB stats for {enriched} rows → {train_path}")
        # Auto-repair labels and suffixes so training never skips due to missing 'result'
        _repair_train_set(train_path)
    else:
        print("ℹ️ no matching outcomes rows merged into train_set.")
    return int(enriched)

def _build_train_from_db(train_path: Path) -> Optional[pd.DataFrame]:
    """Run DB merge for train_set and return the resulting dataframe if available."""
    enriched = _merge_train_stats_from_db(train_path)
    if not train_path.exists():
        print(f"⚠️ train_set missing → {train_path}; returning None.")
        return None
    try:
        df = pd.read_parquet(train_path)
    except Exception as exc:
        print(f"⚠️ could not load train_set after DB merge: {exc}")
        return None
    if enriched:
        print(f"ℹ️ loaded train_set after DB merge: rows={len(df)} cols={df.shape[1]}")
    return df

# ------------- New helper: ensure train_set.parquet exists by seeding from available sources -------------
def _ensure_train_set(train_path: Path, days_back: int = 365) -> Optional[pd.DataFrame]:
    """Ensure data/features/train_set.parquet exists by seeding it from either
    - postmatch_flat.parquet, if available, or
    - a minimal outcomes export from Postgres (last `days_back` days), if possible.
    Returns the loaded dataframe or None if unavailable.
    """
    # Case 1: already exists
    if train_path.exists():
        try:
            return pd.read_parquet(train_path)
        except Exception:
            pass

    # Case 2: seed from postmatch_flat
    flat_path = FEAT / "postmatch_flat.parquet"
    if flat_path.exists():
        try:
            df = pd.read_parquet(flat_path)
            if not df.empty:
                _write_train_set(df)
                return df
        except Exception as e:
            print(f"ℹ️ could not read {flat_path}: {e}")

    # Case 3: seed from DB outcomes (if available)
    eng = _pg_engine()
    if eng is not None:
        try:
            sql = """
                SELECT fixture_id, league_id, season, kick_off as date,
                       home as home, away as away,
                       goals_h, goals_a, result,
                       home_shots_on_goal, away_shots_on_goal,
                       home_ball_possession, away_ball_possession,
                       home_expected_goals, away_expected_goals
                FROM outcomes
                WHERE updated_at > (now() - %(days)s::interval)
            """
            with eng.connect() as c:
                outc = pd.read_sql(sql, c, params={"days": f"{int(days_back)} days"})
                print(f"[merge-db] outcomes fetched from PG: {len(outc)} rows")
            df = outc
            if not df.empty:
                # normalize datetime
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
                _write_train_set(df)
                return df
        except Exception as e:
            print(f"ℹ️ could not seed from DB outcomes: {e}")

    print("ℹ️ unable to ensure train_set.parquet — no sources available.")
    return None

# default leagues we use in this project (EPL, La Liga, Bundesliga)
LEAGUES = [39, 140, 78]


def _check_key():
    if not API_KEY:
        print("⛔ APIFOOTBALL_KEY not set in environment.")
        sys.exit(1)


def _get(endpoint: str, params: dict):
    backoff = 1.2
    for attempt in range(6):
        r = requests.get(f"{API_BASE}/{endpoint}", headers=HEADERS, params=params, timeout=30)
        if r.status_code == 429:
            wait = min(6.0, backoff * (attempt + 1))
            print(f"   ↪︎ 429 rate-limited on /{endpoint} (attempt {attempt+1}), sleeping {wait:.1f}s …")
            time.sleep(wait)
            continue
        try:
            r.raise_for_status()
        except Exception as e:
            print(f"   ⚠️ {endpoint} HTTP {r.status_code}: {e}")
            return []
        js = r.json()
        resp = js.get("response", [])
        if not resp:
            # helpful breadcrumb for debugging empty payloads per league/season
            pg = js.get("paging", {})
            print(f"   ⚠️ empty response from /{endpoint} params={params} paging={pg}")
        return resp
    print(f"   ❌ giving up on /{endpoint} after retries; params={params}")
    return []



def _result_from_goals(home_goals, away_goals):
    try:
        h = int(home_goals); a = int(away_goals)
    except Exception:
        return None
    return "H" if h > a else ("A" if a > h else "D")

# -------------------- H2H helpers (leakage-safe via our hist frame) --------------------

def _pts_from_result_for_team(res: str, is_team_home: bool) -> float:
    if res == "D":
        return 1.0
    if res == "H":
        return 3.0 if is_team_home else 0.0
    if res == "A":
        return 0.0 if is_team_home else 3.0
    return float("nan")


def _compute_h2h_for_pair(hist_df: pd.DataFrame, home_team: str, away_team: str, cutoff: pd.Timestamp, last_n: int = 10):
    """Compute H2H aggregates for (home_team vs away_team) using only matches < cutoff.
    Returns (games, pts_for_home, gd_for_home, win_rate_home).
    Teams are compared on normalized names (home_norm/away_norm already in hist_df).
    """
    if hist_df is None or hist_df.empty or not isinstance(cutoff, pd.Timestamp):
        return 0, np.nan, np.nan, np.nan

    # Filter to this pair only and keep only rows strictly before cutoff to avoid leakage
    mask_pair = (
        ((hist_df["home_norm"] == home_team) & (hist_df["away_norm"] == away_team)) |
        ((hist_df["home_norm"] == away_team) & (hist_df["away_norm"] == home_team))
    )
    dfp = hist_df.loc[mask_pair].copy()
    if dfp.empty:
        return 0, np.nan, np.nan, np.nan

    dfp = dfp[dfp["date"] < cutoff].sort_values("date", ascending=False)
    if dfp.empty:
        return 0, np.nan, np.nan, np.nan
    dfp = dfp.head(last_n).copy()

    # Ensure numeric goals
    for c in ("goals_home", "goals_away"):
        if c in dfp.columns:
            dfp[c] = pd.to_numeric(dfp[c], errors="coerce")

    pts_list = []
    gd_list = []
    win_flag = []  # 1 if current-home-team won that historic match, else 0

    for _, r in dfp.iterrows():
        if pd.isna(r.get("result")):
            continue
        this_is_home = bool(r.get("home_norm") == home_team)
        pts = _pts_from_result_for_team(r.get("result"), this_is_home)
        if np.isnan(pts):
            continue
        if this_is_home:
            gd = (r.get("goals_home") or 0) - (r.get("goals_away") or 0)
            won = 1.0 if r.get("result") == "H" else 0.0
        else:
            gd = (r.get("goals_away") or 0) - (r.get("goals_home") or 0)
            won = 1.0 if r.get("result") == "A" else 0.0
        pts_list.append(pts)
        gd_list.append(gd if pd.notna(gd) else 0.0)
        win_flag.append(won)

    games = int(len(pts_list))
    if games == 0:
        return 0, np.nan, np.nan, np.nan
    pts_sum = float(np.nansum(pts_list))
    gd_sum = float(np.nansum(gd_list))
    wr = float(np.nanmean(win_flag)) if len(win_flag) else np.nan
    return games, pts_sum, gd_sum, wr


#
# -------------------- Extra feature helpers --------------------

def _mk_outcome_flags(result: str, is_home: bool):
    # returns (win, draw, loss) flags from a team's perspective
    if result == "D":
        return 0, 1, 0
    if result == "H":
        return (1, 0, 0) if is_home else (0, 0, 1)
    if result == "A":
        return (0, 0, 1) if is_home else (1, 0, 0)
    return np.nan, np.nan, np.nan


def _compute_team_timeseries(hist_matches: pd.DataFrame) -> pd.DataFrame:
    """Build long per-team time series with date, win/draw/loss flags and rest days between appearances.
    hist_matches must contain: home_norm, away_norm, date, result.
    """
    if hist_matches.empty:
        return pd.DataFrame(columns=["team","date","win","draw","loss","rest_days"])  # empty shape

    # Ensure normalized names & datetime
    hf = add_normalized_teams(hist_matches.copy(), "home", "away")
    hf = hf.dropna(subset=["date"]).copy()
    hf["date"] = pd.to_datetime(hf["date"], utc=True, errors="coerce")

    # Home perspective rows
    wdl_home = [ _mk_outcome_flags(r, True) for r in hf["result"].tolist() ]
    win_h, draw_h, loss_h = zip(*wdl_home) if wdl_home else ([],[],[])
    long_home = pd.DataFrame({
        "team": hf["home_norm"],
        "date": hf["date"],
        "win": pd.to_numeric(win_h, errors="coerce"),
        "draw": pd.to_numeric(draw_h, errors="coerce"),
        "loss": pd.to_numeric(loss_h, errors="coerce"),
    })
    # Away perspective rows
    wdl_away = [ _mk_outcome_flags(r, False) for r in hf["result"].tolist() ]
    win_a, draw_a, loss_a = zip(*wdl_away) if wdl_away else ([],[],[])
    long_away = pd.DataFrame({
        "team": hf["away_norm"],
        "date": hf["date"],
        "win": pd.to_numeric(win_a, errors="coerce"),
        "draw": pd.to_numeric(draw_a, errors="coerce"),
        "loss": pd.to_numeric(loss_a, errors="coerce"),
    })

    perf = pd.concat([long_home, long_away], ignore_index=True)
    perf = perf.dropna(subset=["team","date"]).sort_values(["team","date"]).reset_index(drop=True)

    # Rest days
    perf["prev_date"] = perf.groupby("team")["date"].shift(1)
    perf["rest_days"] = (perf["date"] - perf["prev_date"]).dt.total_seconds() / (3600*24)

    return perf[["team","date","win","draw","loss","rest_days"]]


def _attach_rolling_wdl_and_rest(target_df: pd.DataFrame, hist_matches: pd.DataFrame, side_prefix: str, team_col: str, window: int = 5) -> pd.DataFrame:
    """For each row in target_df (must have columns [fixture_id, date, team_col]),
    attach prior rolling sums of win/draw/loss and last rest_days from hist_matches.
    """
    ts = _compute_team_timeseries(hist_matches)
    if ts.empty:
        # create empty shell to merge
        empty = pd.DataFrame({
            "fixture_id": pd.Series(dtype="Int64"),
            f"{side_prefix}_wins{window}": pd.Series(dtype="float64"),
            f"{side_prefix}_draws{window}": pd.Series(dtype="float64"),
            f"{side_prefix}_losses{window}": pd.Series(dtype="float64"),
            f"{side_prefix}_rest_days": pd.Series(dtype="float64"),
        })
        return target_df.merge(empty, on="fixture_id", how="left")

    # Build rolling sums (shifted so we only use prior matches)
    ts = ts.sort_values(["team","date"]).reset_index(drop=True)
    ts["win_prev"] = ts.groupby("team")["win"].shift(1)
    ts["draw_prev"] = ts.groupby("team")["draw"].shift(1)
    ts["loss_prev"] = ts.groupby("team")["loss"].shift(1)

    roll_w = (
        ts.groupby("team")["win_prev"].rolling(window, min_periods=1).sum().reset_index(level=0, drop=True)
    )
    roll_d = (
        ts.groupby("team")["draw_prev"].rolling(window, min_periods=1).sum().reset_index(level=0, drop=True)
    )
    roll_l = (
        ts.groupby("team")["loss_prev"].rolling(window, min_periods=1).sum().reset_index(level=0, drop=True)
    )
    ts[f"wins{window}"] = roll_w
    ts[f"draws{window}"] = roll_d
    ts[f"losses{window}"] = roll_l

    # For asof join, prepare left (target) and right (ts)
    left = target_df[["fixture_id","date", team_col]].rename(columns={team_col: "team"}).copy()
    left["date"] = pd.to_datetime(left["date"], utc=True, errors="coerce")

    right = ts[["team","date", f"wins{window}", f"draws{window}", f"losses{window}", "rest_days"]].copy()

    out_parts = []
    for t in sorted(set(left["team"]).intersection(set(right["team"]))):
        l = left[left["team"] == t].sort_values("date").reset_index(drop=True)
        r = right[right["team"] == t].sort_values("date").reset_index(drop=True)
        merged = pd.merge_asof(l, r.drop(columns=["team"]), on="date", direction="backward", allow_exact_matches=True)
        out_parts.append(merged[["fixture_id", f"wins{window}", f"draws{window}", f"losses{window}", "rest_days"]])

    if out_parts:
        m_final = pd.concat(out_parts, ignore_index=True)
    else:
        m_final = pd.DataFrame({
            "fixture_id": pd.Series(dtype="Int64"),
            f"wins{window}": pd.Series(dtype="float64"),
            f"draws{window}": pd.Series(dtype="float64"),
            f"losses{window}": pd.Series(dtype="float64"),
            "rest_days": pd.Series(dtype="float64"),
        })

    m_final = m_final.rename(columns={
        f"wins{window}": f"{side_prefix}_wins{window}",
        f"draws{window}": f"{side_prefix}_draws{window}",
        f"losses{window}": f"{side_prefix}_losses{window}",
        "rest_days": f"{side_prefix}_rest_days",
    })

    return target_df.merge(m_final, on="fixture_id", how="left")


# ... rest of file unchanged ...

# -------------------- CLI entrypoint --------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Build/harvest features and write train_set.parquet")
    p.add_argument("--harvest-postmatch", action="store_true",
                   help="Harvest FT fixtures (outcomes + stats + closing odds) into postmatch_flat.parquet")
    p.add_argument("--merge-train-stats-from-db", action="store_true",
                   help="Merge DB outcomes/stats into train_set.parquet (creates it if missing)")
    p.add_argument("--write-train-set", action="store_true",
                   help="Write data/features/train_set.parquet from the in-memory dataframe (or seed sources)")
    p.add_argument("--write-upcoming", action="store_true",
                   help="Write data/features/upcoming_set.parquet for upcoming fixtures (uses --days-fwd, --leagues, --season; attaches limited /fixtures/statistics)")
    p.add_argument("--leagues", type=str, default="39,140,78", help="Comma-separated league ids for harvest")
    p.add_argument("--season", type=int, default=2025, help="Season year for harvest")
    p.add_argument("--days-back", type=int, default=1, help="Days back for harvest window")
    p.add_argument("--days-fwd", type=int, default=0, help="Days forward for harvest window")
    p.add_argument("--timezone", type=str, default="Europe/Copenhagen", help="Timezone for closing odds fetch")
    p.add_argument("--with-stats-limit", type=int, default=40, help="Cap statistics requests per run (upcoming)")
    p.add_argument("--with-stats-sleep", type=float, default=0.25, help="Sleep between statistics calls")
    return p.parse_args()


def main():
    args = _parse_args()
    train_path = FEAT_DIR / "train_set.parquet"
    ran = False

    # 1) Harvest
    if args.harvest_postmatch:
        ran = True
        try:
            leagues = [int(x) for x in str(args.leagues).split(',') if str(x).strip()]
        except Exception:
            leagues = LEAGUES
        print(f"[build_features] harvest_postmatch leagues={leagues} season={args.season} window=[-{args.days_back}, +{args.days_fwd}] …")
        harvest_postmatch(
            leagues=leagues,
            season=int(args.season),
            days_back=int(args.days_back),
            days_fwd=int(args.days_fwd),
            timezone_str=args.timezone,
            sleep=float(args.with_stats_sleep),
        )

    # 2) Merge from DB into train_set (ensure seed exists)
    df_mem = None
    if args.merge_train_stats_from_db:
        ran = True
        df_mem = _ensure_train_set(train_path)
        _merge_train_stats_from_db(train_path)
        try:
            df_mem = pd.read_parquet(train_path)
        except Exception:
            df_mem = None

    # 3) Write train_set if requested
    if args.write_train_set:
        ran = True
        if df_mem is None:
            df_mem = _ensure_train_set(train_path)
        if df_mem is not None:
            _write_train_set(df_mem)
        else:
            print("❌ could not write train_set.parquet — no dataframe available")

    # 4) Write upcoming_set if requested
    if getattr(args, "write_upcoming", False):
        ran = True
        try:
            leagues = [int(x) for x in str(args.leagues).split(',') if str(x).strip()]
        except Exception:
            leagues = LEAGUES
        print(f"[build_features] write_upcoming leagues={leagues} season={args.season} days_fwd={args.days_fwd} …")
        _build_upcoming_set(
            leagues=leagues,
            season=int(args.season),
            days_fwd=int(args.days_fwd),
            timezone_str=args.timezone,
            stats_limit=int(args.with_stats_limit),
            stats_sleep=float(args.with_stats_sleep),
        )

    if not ran:
        # Default behavior if no flags: do nothing but print help to guide users in CI
        print("ℹ️ No action flags provided. Try --harvest-postmatch or --merge-train-stats-from-db --write-train-set")


if __name__ == "__main__":
    main()
