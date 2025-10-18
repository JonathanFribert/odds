#!/usr/bin/env python3
import os, sys, time
import json, requests
import pandas as pd, numpy as np
from datetime import datetime, timedelta, timezone as _TZ
from pathlib import Path
from typing import List

from common import ROOT, FIXDIR, FEAT, ODIR, map_league_id_to_sport_key, add_normalized_teams

API_FOOTBALL_BASE = "https://v3.football.api-sports.io"

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
        print(f"✅ repaired train_set: result non-null rows={nn}; cols={df.shape[1]}; rows={df.shape[0]}")

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


def _compute_league_home_adv(hist_matches: pd.DataFrame) -> pd.DataFrame:
    if hist_matches.empty:
        return pd.DataFrame(columns=["league_id","home_win_rate"])  # shell
    df = hist_matches.dropna(subset=["result","league_id"]).copy()
    df["home_win"] = (df["result"] == "H").astype(float)
    agg = df.groupby("league_id")["home_win"].mean().reset_index().rename(columns={"home_win":"league_home_win_rate"})
    return agg


def fetch_fixtures_for_league(league_id: int, season: int) -> pd.DataFrame:
    league_resp = _get_cached("leagues", {"id": league_id}, ttl_s=12*60*60)
    league_name = league_resp[0]["league"]["name"] if league_resp else str(league_id)
    print(f"→ Fetching fixtures: {league_name} ({league_id}) season={season}")

    resp = _get_cached("fixtures", {"league": league_id, "season": season}, ttl_s=6*60*60)
    print(f"   fixtures received: {len(resp)}")
    rows = []
    for it in resp:
        fix = it.get("fixture", {})
        teams = it.get("teams", {})
        goals = it.get("goals", {})
        home = teams.get("home", {})
        away = teams.get("away", {})
        date_iso = fix.get("date")
        try:
            date = pd.to_datetime(date_iso, utc=True)
        except Exception:
            date = pd.NaT
        res = _result_from_goals(goals.get("home"), goals.get("away"))
        status = fix.get("status", {})
        rows.append({
            "fixture_id": fix.get("id"),
            "league_id": league_id,
            "league_name": league_name,
            "season": season,
            "date": date,
            "status_short": status.get("short"),
            "status_long": status.get("long"),
            "home_id": home.get("id"),
            "home": home.get("name"),
            "away_id": away.get("id"),
            "away": away.get("name"),
            "goals_home": goals.get("home"),
            "goals_away": goals.get("away"),
            "result": res,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        print(f"   ⚠️ no fixtures parsed for league={league_id} season={season}")

    if "fixture_id" in df.columns:
        df["fixture_id"] = pd.to_numeric(df["fixture_id"], errors="coerce").astype("Int64")

    # sport_key + normalized team names for joins
    df["sport_key"] = df["league_id"].map(map_league_id_to_sport_key)
    df = add_normalized_teams(df, "home", "away")

    # save main fixtures
    sk = df["sport_key"].dropna().unique()
    if len(sk) == 1:
        out = FIXDIR / f"{sk[0]}_{season}.csv"
    else:
        out = FIXDIR / f"league_{league_id}_{season}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"✅ saved {len(df)} rows → {out}")

    # standings
    std = _get_cached("standings", {"league": league_id, "season": season}, ttl_s=12*60*60)
    print(f"   standings payload groups: {len(std)}")
    std_rows = []
    for item in std:
        for table in item.get("league", {}).get("standings", []):
            for t in table:
                std_rows.append({
                    "team_id": t.get("team", {}).get("id"),
                    "team_name": t.get("team", {}).get("name"),
                    "rank": t.get("rank"),
                    "points": t.get("points"),
                    "goalsDiff": t.get("goalsDiff"),
                })
    if not std_rows:
        print("   ⚠️ no standings rows returned")
    if std_rows:
        p = out.with_name(out.stem + "_standings.csv")
        pd.DataFrame(std_rows).to_csv(p, index=False)
        print(f"   ➕ standings → {p} ({len(std_rows)})")

    # team statistics (overall)
    ts_rows = []
    teams_df = df[["home_id","home"]].drop_duplicates().rename(columns={"home_id":"team_id","home":"team_name"})
    teams_df = pd.concat([
        teams_df,
        df[["away_id","away"]].drop_duplicates().rename(columns={"away_id":"team_id","away":"team_name"})
    ], ignore_index=True).drop_duplicates("team_id")

    for tid, tname in teams_df.itertuples(index=False):
        if len(ts_rows) % 5 == 0 and len(ts_rows) > 0:
            print(f"   … teamstats rows so far: {len(ts_rows)}")
        ts = _get_cached("teams/statistics", {"league": league_id, "season": season, "team": tid}, ttl_s=12*60*60)
        if not ts:
            continue
        st = ts[0] if isinstance(ts, list) and ts else ts

        # Defensive parsing (API can change shape). Use _dig() to probe multiple likely paths.
        played = _dig(st, "fixtures", "played", "total")
        wins   = _dig(st, "fixtures", "wins", "total")
        draws  = _dig(st, "fixtures", "draws", "total")
        losses = _dig(st, "fixtures", "losses", "total")

        g_for_avg     = _dig(st, "goals", "for", "average", "total") or _dig(st, "goals", "for", "average")
        g_against_avg = _dig(st, "goals", "against", "average", "total") or _dig(st, "goals", "against", "average")

        clean = _dig(st, "clean_sheet", "total") or _dig(st, "clean_sheet")
        fts   = _dig(st, "failed_to_score", "total") or _dig(st, "failed_to_score")

        # Extra features (best-effort, may be None if not covered for a league)
        shots_for_tot      = _dig(st, "shots", "for", "total")
        shots_on_for_tot   = _dig(st, "shots", "for", "on")
        shots_against_tot  = _dig(st, "shots", "against", "total")
        shots_on_against_tot = _dig(st, "shots", "against", "on")
        corners_for_tot    = _dig(st, "corners", "total") or _dig(st, "corners")
        corners_against_tot = _dig(st, "corners", "against", "total")  # may be missing
        yc_tot             = _dig(st, "cards", "yellow", "total") or _dig(st, "cards", "yellow")
        rc_tot             = _dig(st, "cards", "red", "total") or _dig(st, "cards", "red")
        possession_avg     = _dig(st, "ball_possession", "average") or _dig(st, "ball_possession") or _dig(st, "possession", "average")

        # Coerce to numeric where appropriate
        def _num(x):
            return pd.to_numeric(x, errors="coerce")

        played = _num(played)
        wins   = _num(wins)
        draws  = _num(draws)
        losses = _num(losses)
        g_for_avg     = _num(g_for_avg)
        g_against_avg = _num(g_against_avg)
        clean = _num(clean)
        fts   = _num(fts)

        shots_for_tot       = _num(shots_for_tot)
        shots_on_for_tot    = _num(shots_on_for_tot)
        shots_against_tot   = _num(shots_against_tot)
        shots_on_against_tot= _num(shots_on_against_tot)
        corners_for_tot     = _num(corners_for_tot)
        corners_against_tot = _num(corners_against_tot)
        yc_tot              = _num(yc_tot)
        rc_tot              = _num(rc_tot)
        possession_avg      = _num(str(possession_avg).replace('%','')) if possession_avg is not None else np.nan

        # Per-game rates (safe divide)
        def _pg(x):
            return _safediv(x, played)

        ts_rows.append({
            "team_id": tid,
            "played_total": played,
            "wins_total": wins,
            "draws_total": draws,
            "losses_total": losses,
            "goals_for_avg": g_for_avg,
            "goals_against_avg": g_against_avg,
            "clean_sheet": clean,
            "failed_to_score": fts,
            # volumes
            "shots_for": shots_for_tot,
            "shots_on_for": shots_on_for_tot,
            "shots_against": shots_against_tot,
            "shots_on_against": shots_on_against_tot,
            "corners_for": corners_for_tot,
            "corners_against": corners_against_tot,
            "yc_total": yc_tot,
            "rc_total": rc_tot,
            "possession_avg": possession_avg,
            # per-game
            "shots_for_pg": _pg(shots_for_tot),
            "shots_on_for_pg": _pg(shots_on_for_tot),
            "shots_against_pg": _pg(shots_against_tot),
            "shots_on_against_pg": _pg(shots_on_against_tot),
            "corners_for_pg": _pg(corners_for_tot),
            "corners_against_pg": _pg(corners_against_tot),
            "yc_pg": _pg(yc_tot),
            "rc_pg": _pg(rc_tot),
        })
        time.sleep(0.15)

    if not ts_rows:
        print("   ⚠️ no teamstats rows returned (endpoint teams/statistics)")
    if ts_rows:
        p = out.with_name(out.stem + "_teamstats.csv")
        pd.DataFrame(ts_rows).to_csv(p, index=False)
        print(f"   ➕ teamstats → {p} ({len(ts_rows)})")

    # injuries (last 30 days)
    inj_rows = []
    since = (datetime.now(_TZ.utc) - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    injuries = _get_cached("injuries", {"league": league_id, "season": season, "date": since}, ttl_s=6*60*60)
    print(f"   injuries payload items: {len(injuries)} (since last 30d)")
    for it in injuries:
        team = it.get("team", {})
        player = it.get("player", {})
        fixture = it.get("fixture", {})
        inj_rows.append({
            "team_id": team.get("id"),
            "team_name": team.get("name"),
            "player_id": player.get("id"),
            "player_name": player.get("name"),
            "type": it.get("type"),
            "date": pd.to_datetime(fixture.get("date"), utc=True, errors="coerce"),
        })
    if not inj_rows:
        print("   ⚠️ no injuries in last 30 days for this league/season (common in breaks)")
    if inj_rows:
        p = out.with_name(out.stem + "_injuries_last30.csv")
        pd.DataFrame(inj_rows).to_csv(p, index=False)
        print(f"   ➕ injuries(30d) → {p} ({len(inj_rows)})")

    return df


def main():
    _check_key()
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument("--season", type=int, default=datetime.now(_TZ.utc).year,
                help="Primary season (defaults to current UTC year).")
    ap.add_argument("--leagues", type=str, default="39,140,78", help="Comma-separated league IDs to process")
    ap.add_argument("--sleep-between", type=float, default=0.3, help="Seconds to sleep between leagues")
    ap.add_argument("--harvest-postmatch", action="store_true",
                help="Harvest FT fixtures (stats + closing odds) til postmatch_flat.parquet og exit")
    ap.add_argument("--pm-leagues", type=str, default="39,140,78",
                help="Comma-separerede league IDs (default EPL, LaLiga, Bundesliga)")
    ap.add_argument("--pm-season", type=int, default=datetime.now(_TZ.utc).year,
                help="Season for post-match harvest (defaults to current UTC year).")
    ap.add_argument("--pm-days-back", type=int, default=1,
                help="Dage bagud (default 1)")
    ap.add_argument("--pm-days-fwd", type=int, default=0,
                help="Dage frem (default 0)")
    ap.add_argument("--pm-timezone", type=str, default="Europe/Copenhagen",
                help="Timezone for odds-kald")
    ap.add_argument("--pm-sleep", type=float, default=0.2,
                help="Sleep mellem fixture-kald")
    ap.add_argument("--with-stats", action="store_true",
                help="Fetch /fixtures/statistics for upcoming fixtures (home/away) and merge as features (shots, possession, xG, etc.)")
    ap.add_argument("--with-stats-limit", type=int, default=60, help="Max fixtures to fetch per run for /fixtures/statistics (0 = no cap)")
    ap.add_argument("--with-stats-timeout", type=float, default=10.0, help="Per-request timeout seconds for /fixtures/statistics")
    ap.add_argument("--with-stats-sleep", type=float, default=0.15, help="Sleep seconds between stats requests to respect rate limits")
    ap.add_argument("--with-stats-train", action="store_true",
                help="Fetch /fixtures/statistics for FINISHED fixtures (per-team) and merge into TRAIN set features.")
    ap.add_argument("--with-stats-train-limit", type=int, default=400,
                help="Max finished fixtures to fetch stats for (0 = no cap).")
    ap.add_argument("--merge-train-stats-from-db", action="store_true",
                help="Merge outcomes stats from Postgres into data/features/train_set.parquet (home/away SOG, possession, xG, etc.) and exit.")
    args = ap.parse_args()
    # Fast path: only merge DB stats into existing train_set and exit
    if getattr(args, "merge_train_stats_from_db", False):
        FEAT.mkdir(parents=True, exist_ok=True)
        train_path = FEAT / "train_set.parquet"
        _merge_train_stats_from_db(train_path)
        return
    if args.harvest_postmatch:
        leagues = [int(x) for x in args.pm_leagues.replace(";", ",").split(",") if x.strip()]
        harvest_postmatch(
            leagues=leagues,
            season=args.pm_season,
            days_back=args.pm_days_back,
            days_fwd=args.pm_days_fwd,
            timezone_str=args.pm_timezone,
            sleep=args.pm_sleep,
        )
        return
    
    leagues: List[int]
    if args.leagues:
        leagues = [int(x.strip()) for x in args.leagues.split(",") if x.strip()]
    else:
        leagues = LEAGUES
    all_league_frames = []
    for idx, lg in enumerate(leagues, start=1):
        print(f"\n=== League {idx}/{len(leagues)} → {lg} (season {args.season}) ===")
        df = fetch_fixtures_for_league(lg, args.season)
        all_league_frames.append(df)
        time.sleep(max(0.0, args.sleep_between))

    # ---- Build upcoming_set.parquet from freshly fetched league fixtures ----
    if all_league_frames:
        up = pd.concat(all_league_frames, ignore_index=True)
        # normalize dates to UTC
        if "date" in up.columns:
            up["date"] = pd.to_datetime(up["date"], utc=True, errors="coerce")
        # ensure normalized team names and sport_key exist
        up = add_normalized_teams(up, "home", "away")
        up["sport_key"] = up["league_id"].map(map_league_id_to_sport_key)
        # keep only one row per fixture
        if "fixture_id" in up.columns:
            up = up.dropna(subset=["fixture_id"]).drop_duplicates(subset=["fixture_id"]).copy()
            up["fixture_id"] = pd.to_numeric(up["fixture_id"], errors="coerce").astype("Int64")
        # status-based upcoming filter if present
        if "status_short" in up.columns:
            upcoming_codes = {"NS","TBD","PST","SUSP","INT"}
            up = up[up["status_short"].isin(upcoming_codes) | up["status_short"].isna()].copy()
        # time window: now → +60 days (predict_upcoming narrows further via --days)
        now = pd.Timestamp.now(tz="UTC")
        up = up[(up["date"] >= now - pd.Timedelta(hours=1)) & (up["date"] <= now + pd.Timedelta(days=60))].copy()
        # Relative time to kickoff in days (useful feature)
        up["days_to_match"] = (up["date"] - now).dt.total_seconds() / 86400.0

        # Optionally enrich with per-fixture statistics (home/away) for upcoming fixtures
        if args.with_stats:
            up = _attach_upcoming_fixture_stats(
                up,
                limit=args.with_stats_limit,
                per_req_sleep=args.with_stats_sleep,
                timeout=args.with_stats_timeout,
            )
            # Ensure a stable schema for stats columns across runs
            expected_stats_cols = [
                "home_expected_goals", "away_expected_goals",
                "home_ball_possession", "away_ball_possession",
                "home_shots_on_goal", "away_shots_on_goal",
                "home_total_shots", "away_total_shots",
                "home_shots_off_goal", "away_shots_off_goal",
                "home_shots_inside_box", "away_shots_inside_box",
                "home_shots_outside_box", "away_shots_outside_box",
                "home_passes_pct", "away_passes_pct",
                "home_passes_accurate", "away_passes_accurate",
                "home_corners", "away_corners",
                "home_yellow_cards", "away_yellow_cards",
                "home_red_cards", "away_red_cards",
                "home_fouls", "away_fouls",
                "home_offsides", "away_offsides",
            ]
            present = [c for c in expected_stats_cols if c in up.columns]
            missing = [c for c in expected_stats_cols if c not in up.columns]
            for c in missing:
                up[c] = np.nan
            print(
                f"   ensured schema: added {len(missing)} empty stats columns (as NA)\n"
                f"   stats columns present: {len(present)}; missing (ignored): {len(missing)}"
            )
        # ---- Load and prepare teamstats for this season, then merge into upcoming ----
        stats_frames = []
        for lg in leagues:
            sk = map_league_id_to_sport_key(lg)
            p = FIXDIR / f"{sk}_{args.season}_teamstats.csv"
            if p.exists():
                tmp = pd.read_csv(p)
                tmp["league_id"] = lg
                stats_frames.append(tmp)
        if stats_frames:
            stats = pd.concat(stats_frames, ignore_index=True)
            # normalize types
            for c in ["team_id", "played_total", "wins_total", "draws_total", "losses_total", "clean_sheet", "failed_to_score"]:
                if c in stats.columns:
                    stats[c] = pd.to_numeric(stats[c], errors="coerce")
            for c in ["goals_for_avg", "goals_against_avg"]:
                if c in stats.columns:
                    stats[c] = pd.to_numeric(stats[c], errors="coerce")
            # derived rates
            stats["win_rate"]  = _safediv(stats.get("wins_total"),  stats.get("played_total"))
            stats["draw_rate"] = _safediv(stats.get("draws_total"), stats.get("played_total"))
            stats["loss_rate"] = _safediv(stats.get("losses_total"),stats.get("played_total"))
            stats["cs_rate"]   = _safediv(stats.get("clean_sheet"), stats.get("played_total"))
            stats["fts_rate"]  = _safediv(stats.get("failed_to_score"), stats.get("played_total"))
            # clip improbable values
            for c in ["win_rate","draw_rate","loss_rate","cs_rate","fts_rate"]:
                if c in stats.columns:
                    stats[c] = stats[c].clip(0, 1)
            # select columns to merge
            base_cols = [
                "team_id",
                "goals_for_avg", "goals_against_avg",
                "win_rate", "draw_rate", "loss_rate", "cs_rate", "fts_rate",
                # volumes and per-game extras (keep both; PG are more comparable across leagues)
                "shots_for", "shots_on_for", "shots_against", "shots_on_against",
                "corners_for", "corners_against", "yc_total", "rc_total",
                "shots_for_pg", "shots_on_for_pg", "shots_against_pg", "shots_on_against_pg",
                "corners_for_pg", "corners_against_pg", "yc_pg", "rc_pg",
                "possession_avg",
            ]
            stats = stats[[c for c in base_cols if c in stats.columns]].dropna(subset=["team_id"]).drop_duplicates("team_id")
            # merge into upcoming on team_id (home/away)
            if "home_id" in up.columns:
                up = up.merge(stats.add_prefix("home_"), left_on="home_id", right_on="home_team_id", how="left")
                up = up.drop(columns=[c for c in ["home_team_id"] if c in up.columns])
            if "away_id" in up.columns:
                up = up.merge(stats.add_prefix("away_"), left_on="away_id", right_on="away_team_id", how="left")
                up = up.drop(columns=[c for c in ["away_team_id"] if c in up.columns])
        # ---- Merge standings (rank/points/goalsDiff) into upcoming ----
        std_frames = []
        for lg in leagues:
            sk = map_league_id_to_sport_key(lg)
            pstd = FIXDIR / f"{sk}_{args.season}_standings.csv"
            if pstd.exists():
                tmp = pd.read_csv(pstd)
                tmp["league_id"] = lg
                std_frames.append(tmp)
        if std_frames:
            std_all = pd.concat(std_frames, ignore_index=True)
            for c in ["team_id","rank","points","goalsDiff"]:
                if c in std_all.columns:
                    std_all[c] = pd.to_numeric(std_all[c], errors="coerce")
            # home side
            if "home_id" in up.columns:
                up = up.merge(std_all.add_prefix("home_"), left_on=["home_id","league_id"], right_on=["home_team_id","home_league_id"], how="left")
                up = up.drop(columns=[c for c in ["home_team_id","home_league_id"] if c in up.columns])
            # away side
            if "away_id" in up.columns:
                up = up.merge(std_all.add_prefix("away_"), left_on=["away_id","league_id"], right_on=["away_team_id","away_league_id"], how="left")
                up = up.drop(columns=[c for c in ["away_team_id","away_league_id"] if c in up.columns])
            # simple diffs
            for a,b,name in [("home_rank","away_rank","rank_diff"),("home_points","away_points","points_diff"),("home_goalsDiff","away_goalsDiff","goalsdiff_diff")]:
                if a in up.columns and b in up.columns:
                    up[name] = pd.to_numeric(up[a], errors="coerce") - pd.to_numeric(up[b], errors="coerce")

        # ---- Injuries count (last 30d) per team ----
        inj_frames = []
        for lg in leagues:
            sk = map_league_id_to_sport_key(lg)
            pinj = FIXDIR / f"{sk}_{args.season}_injuries_last30.csv"
            if pinj.exists():
                tmp = pd.read_csv(pinj)
                tmp["league_id"] = lg
                inj_frames.append(tmp)
        if inj_frames:
            inj_all = pd.concat(inj_frames, ignore_index=True)
            inj_all["team_id"] = pd.to_numeric(inj_all.get("team_id"), errors="coerce")
            inj_ct = inj_all.groupby(["league_id","team_id"]).size().reset_index(name="injuries_30d")
            if "home_id" in up.columns:
                up = up.merge(inj_ct.add_prefix("home_"), left_on=["league_id","home_id"], right_on=["home_league_id","home_team_id"], how="left")
                up = up.drop(columns=[c for c in ["home_league_id","home_team_id"] if c in up.columns])
            if "away_id" in up.columns:
                up = up.merge(inj_ct.add_prefix("away_"), left_on=["league_id","away_id"], right_on=["away_league_id","away_team_id"], how="left")
                up = up.drop(columns=[c for c in ["away_league_id","away_team_id"] if c in up.columns])

        # ---- Add form5 features from historical fixtures ----
        hist = pd.concat(all_league_frames, ignore_index=True)
        # Only finished matches with valid dates and goals
        hist = hist[(hist["date"].notna()) & hist["result"].notna()].copy()
        # Coerce numeric goals (some rows may be None)
        for c in ("goals_home","goals_away"):
            if c in hist.columns:
                hist[c] = pd.to_numeric(hist[c], errors="coerce")
        # Build per-team match rows (home and away views)
        def _mk_points(res, is_home):
            if res == "D":
                return 1
            if res == "H":
                return 3 if is_home else 0
            if res == "A":
                return 0 if is_home else 3
            return np.nan
        # ensure normalized names exist in hist
        hist = add_normalized_teams(hist, "home", "away")
        # Home perspective
        home_rows = pd.DataFrame({
            "team": hist["home_norm"],
            "opp": hist["away_norm"],
            "date": hist["date"],
            "points": hist["result"].map(lambda r: _mk_points(r, True)),
            "gd": (hist.get("goals_home") - hist.get("goals_away")) if "goals_home" in hist.columns and "goals_away" in hist.columns else np.nan,
        })
        # Away perspective
        away_rows = pd.DataFrame({
            "team": hist["away_norm"],
            "opp": hist["home_norm"],
            "date": hist["date"],
            "points": hist["result"].map(lambda r: _mk_points(r, False)),
            "gd": (hist.get("goals_away") - hist.get("goals_home")) if "goals_home" in hist.columns and "goals_away" in hist.columns else np.nan,
        })
        perf = pd.concat([home_rows, away_rows], ignore_index=True).dropna(subset=["team","date"]).sort_values("date")
        # Rolling last 5 per team
        perf["points"] = pd.to_numeric(perf["points"], errors="coerce")
        perf["gd"] = pd.to_numeric(perf["gd"], errors="coerce")
        form = (
            perf.groupby("team")[ ["points","gd"] ]
                .rolling(5, min_periods=1)
                .sum()
                .reset_index()
                .rename(columns={"points":"form5_pts","gd":"form5_gd"})
        )
        # Keep the last known form per team (as of now)
        last_idx = form.groupby("team")["level_1"].idxmax()
        form_now = form.loc[last_idx, ["team","form5_pts","form5_gd"]].copy().reset_index(drop=True)
        # Merge into upcoming for both home/away
        up = up.merge(form_now.rename(columns={"team":"home_norm","form5_pts":"home_form5_pts","form5_gd":"home_form5_gd"}), on="home_norm", how="left")
        up = up.merge(form_now.rename(columns={"team":"away_norm","form5_pts":"away_form5_pts","form5_gd":"away_form5_gd"}), on="away_norm", how="left")

        # ---- Add entry/implied probabilities from latest API-Football odds snapshot ----
        try:
            apifo_files = sorted(ODIR.glob("apifootball_live_odds_*.csv"))
        except Exception:
            apifo_files = []
        if apifo_files:
            apifo = pd.read_csv(apifo_files[-1], parse_dates=["date"], dtype={"fixture_id":"Int64"})
            apifo["fixture_id"] = pd.to_numeric(apifo["fixture_id"], errors="coerce").astype("Int64")
            for col in ["odds_h","odds_d","odds_a"]:
                if col in apifo.columns:
                    apifo[col] = pd.to_numeric(apifo[col], errors="coerce")
            # implied and normalized (remove overround)
            probs = pd.DataFrame(index=apifo.index)
            for s,col in [("h","odds_h"),("d","odds_d"),("a","odds_a")]:
                if col in apifo.columns:
                    probs[f"p_entry_{s}"] = 1.0 / apifo[col]
            if not probs.empty and {"p_entry_h","p_entry_d","p_entry_a"}.issubset(probs.columns):
                total = probs[["p_entry_h","p_entry_d","p_entry_a"]].sum(axis=1)
                for s in ("h","d","a"):
                    probs[f"p_entry_{s}"] = probs[f"p_entry_{s}"] / total
                apifo = pd.concat([apifo, probs], axis=1)
            # merge into upcoming by fixture_id
            cols = [c for c in ["fixture_id","p_entry_h","p_entry_d","p_entry_a","odds_h","odds_d","odds_a"] if c in apifo.columns]
            up = up.merge(apifo[cols], on="fixture_id", how="left", suffixes=("", "_apifb"))
            # do not overwrite odds if already present later; here we only supply p_entry_* primarily

        # market overround (how juiced the book is) – robust to NaN/inf
        if {"odds_h","odds_d","odds_a"}.issubset(up.columns):
            inv_h = _safediv(1.0, up["odds_h"]) if "odds_h" in up.columns else np.nan
            inv_d = _safediv(1.0, up["odds_d"]) if "odds_d" in up.columns else np.nan
            inv_a = _safediv(1.0, up["odds_a"]) if "odds_a" in up.columns else np.nan
            invsum = pd.to_numeric(inv_h, errors="coerce") + pd.to_numeric(inv_d, errors="coerce") + pd.to_numeric(inv_a, errors="coerce")
            up["overround"] = pd.to_numeric(invsum, errors="coerce") - 1.0
        
        # fill NaNs with neutral defaults where sensible
        for c in ["home_form5_pts","home_form5_gd","away_form5_pts","away_form5_gd"]:
            if c in up.columns:
                up[c] = pd.to_numeric(up[c], errors="coerce")
        
        # ---- Rolling W/D/L and rest-days from history (window=5) ----
        hist_finished = pd.concat(all_league_frames, ignore_index=True)
        hist_finished = hist_finished[(hist_finished["date"].notna()) & hist_finished["result"].notna()].copy()
        if not hist_finished.empty:
            up = _attach_rolling_wdl_and_rest(up, hist_finished, "home", "home_norm", window=5)
            up = _attach_rolling_wdl_and_rest(up, hist_finished, "away", "away_norm", window=5)

        lha = _compute_league_home_adv(hist_finished)
        if not lha.empty:
            up = up.merge(lha, on="league_id", how="left")

        # ---- Add H2H features for upcoming (using only past matches to avoid leakage) ----
        try:
            hist_for_h2h = pd.concat(all_league_frames, ignore_index=True)
            hist_for_h2h = add_normalized_teams(hist_for_h2h, "home", "away")
            hist_for_h2h = hist_for_h2h.dropna(subset=["date"]).copy()
            hist_for_h2h["date"] = pd.to_datetime(hist_for_h2h["date"], utc=True, errors="coerce")
        except Exception:
            hist_for_h2h = pd.DataFrame()

        def _row_h2h_up(row):
            cutoff = pd.Timestamp.utcnow().tz_localize("UTC") if pd.isna(row.get("date")) else pd.to_datetime(row.get("date"), utc=True, errors="coerce")
            g, pts, gd, wr = _compute_h2h_for_pair(hist_for_h2h, row.get("home_norm"), row.get("away_norm"), cutoff)
            return pd.Series({
                "h2h_games": g,
                "h2h_pts_home": pts,
                "h2h_gd_home": gd,
                "h2h_win_rate_home": wr,
            })

        if not up.empty and {"home_norm","away_norm","date"}.issubset(up.columns):
            h2h_up = up.apply(_row_h2h_up, axis=1)
            up = pd.concat([up, h2h_up], axis=1)


        # --- Ensure stable schema for per-fixture stats: create columns as NA if missing ---
        desired_stats_cols = [
            "home_expected_goals","away_expected_goals",
            "home_ball_possession","away_ball_possession",
            "home_shots_on_goal","away_shots_on_goal",
            "home_total_shots","away_total_shots",
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
            "home_xg","away_xg",
        ]
        created_cols = 0
        for col in desired_stats_cols:
            if col not in up.columns:
                up[col] = pd.NA
                created_cols += 1
        if created_cols:
            print(f"   ensured schema: added {created_cols} empty stats columns (as NA)")
        # mirror expected_goals into xg aliases if provided by API
        if "home_expected_goals" in up.columns and "home_xg" in up.columns:
            up["home_xg"] = up["home_xg"].fillna(up["home_expected_goals"])
        if "away_expected_goals" in up.columns and "away_xg" in up.columns:
            up["away_xg"] = up["away_xg"].fillna(up["away_expected_goals"])

        # simple non-null summary for per-fixture stats
        _stats_cols = [c for c in desired_stats_cols if c in up.columns]
        try:
            _nn = int(up[_stats_cols].notna().sum().sum())
            print(f"   per-fixture stats non-nulls: {_nn} cells across {len(_stats_cols)} columns")
        except Exception:
            pass
        # ---- Decide which columns to keep in upcoming_set ----
        base_keep = [
            "fixture_id","league_id","league_name","season","date","days_to_match","sport_key",
            "home","away","home_norm","away_norm","status_short","status_long",
            "home_id","away_id",
            # standings-derived
            "home_rank","away_rank","rank_diff","home_points","away_points","points_diff","home_goalsDiff","away_goalsDiff","goalsdiff_diff",
            # recent form (rolling)
            "home_form5_pts","home_form5_gd","away_form5_pts","away_form5_gd",
            # rolling WDL + rest-days
            "home_wins5","home_draws5","home_losses5","home_rest_days",
            "away_wins5","away_draws5","away_losses5","away_rest_days",
            # entry probs + overround
            "p_entry_h","p_entry_d","p_entry_a","overround",
            # teamstats-derived (hold stats — always present if teams/statistics succeeded)
            "home_goals_for_avg","home_goals_against_avg","home_win_rate","home_draw_rate","home_loss_rate","home_cs_rate","home_fts_rate",
            "away_goals_for_avg","away_goals_against_avg","away_win_rate","away_draw_rate","away_loss_rate","away_cs_rate","away_fts_rate",
            "home_shots_for_pg","home_shots_on_for_pg","home_shots_against_pg","home_shots_on_against_pg",
            "away_shots_for_pg","away_shots_on_for_pg","away_shots_against_pg","away_shots_on_against_pg",
            "home_corners_for_pg","home_corners_against_pg","away_corners_for_pg","away_corners_against_pg",
            "home_yc_pg","home_rc_pg","away_yc_pg","away_rc_pg",
            "home_possession_avg","away_possession_avg",
            # injuries (last 30d)
            "home_injuries_30d","away_injuries_30d",
            # league-level priors
            "league_home_win_rate",
            # h2h aggregates
            "h2h_games","h2h_pts_home","h2h_gd_home","h2h_win_rate_home",
        ]
        # Optional per-fixture stats (only if actually present; e.g., when --with-stats was used and API had data)
        per_fix_candidates = [
            "home_expected_goals","away_expected_goals",
            "home_ball_possession","away_ball_possession",
            "home_shots_on_goal","away_shots_on_goal",
            "home_total_shots","away_total_shots",
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
            "home_xg","away_xg",
        ]
        dyn_cols = [c for c in per_fix_candidates if c in up.columns]
        keep_cols = [c for c in base_keep if c in up.columns] + dyn_cols

        # Small summary for debug
        present_dyn = [c for c in dyn_cols]
        missing_dyn = [c for c in per_fix_candidates if c not in up.columns]
        print(f"   stats columns present: {len(present_dyn)}; missing (ignored): {len(missing_dyn)}")
        up = up[keep_cols].sort_values(["date","league_id","fixture_id"]).reset_index(drop=True)
        FEAT.mkdir(parents=True, exist_ok=True)
        outp = FEAT / "upcoming_set.parquet"
        up.to_parquet(outp, index=False)
        print(f"   ➕ upcoming_set → {outp} ({len(up)})")

        # ---- Build train_set.parquet (finished matches with leakage-safe form5) ----
        hist_all = pd.concat(all_league_frames, ignore_index=True)
        # ---- Historic per-fixture stats for TRAIN (finished fixtures only) ----
        if args.with_stats_train:
            hist_stats_df = _attach_historic_fixture_stats(
                hist_all,
                limit=args.with_stats_train_limit,
                per_req_sleep=args.with_stats_sleep
            )
            if not hist_stats_df.empty:
                # sikr stabilt schema
                expected_cols = [
                    "home_expected_goals","away_expected_goals",
                    "home_ball_possession","away_ball_possession",
                    "home_shots_on_goal","away_shots_on_goal",
                    "home_total_shots","away_total_shots",
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
                ]
                for c in expected_cols:
                    if c not in hist_stats_df.columns:
                        hist_stats_df[c] = np.nan
                # gem rå (valgfrit til debug)
                (FEAT / "historic_fixture_stats.parquet").parent.mkdir(parents=True, exist_ok=True)
                hist_stats_df.to_parquet(FEAT / "historic_fixture_stats.parquet", index=False)
            else:
                print("   ⚠️ no historic per-fixture stats returned; TRAIN set will not include per-fixture stats.")
        else:
            hist_stats_df = pd.DataFrame(columns=["fixture_id"])
        hist = hist_all[(hist_all["date"].notna()) & hist_all["result"].notna()].copy()
        # keep finished matches with valid dates and labeled results
        hist = add_normalized_teams(hist, "home", "away")
        # ensure chronological order
        hist = hist.sort_values("date").reset_index(drop=True)

        # per-team long table for rolling (points, goal diff), using shift(1) to avoid target leakage
        def _mk_points(res, is_home):
            if res == "D":
                return 1
            if res == "H":
                return 3 if is_home else 0
            if res == "A":
                return 0 if is_home else 3
            return np.nan

        # coerce goals just in case (not strictly needed for gd if missing)
        for c in ("goals_home","goals_away"):
            if c in hist.columns:
                hist[c] = pd.to_numeric(hist[c], errors="coerce")

        home_long = pd.DataFrame({
            "team": hist["home_norm"],
            "date": hist["date"],
            "points": hist["result"].map(lambda r: _mk_points(r, True)),
            "gd": (hist.get("goals_home") - hist.get("goals_away")) if "goals_home" in hist.columns and "goals_away" in hist.columns else np.nan,
        })
        away_long = pd.DataFrame({
            "team": hist["away_norm"],
            "date": hist["date"],
            "points": hist["result"].map(lambda r: _mk_points(r, False)),
            "gd": (hist.get("goals_away") - hist.get("goals_home")) if "goals_home" in hist.columns and "goals_away" in hist.columns else np.nan,
        })
        perf = pd.concat([home_long, away_long], ignore_index=True).dropna(subset=["team","date"]).sort_values(["team","date"])
        # shift(1) so current match isn't included in its own form window
        perf["points"] = pd.to_numeric(perf["points"], errors="coerce")
        perf["gd"] = pd.to_numeric(perf["gd"], errors="coerce")
        # compute rolling sums directly on shifted columns (leakage-safe)
        perf["points_prev"] = perf.groupby("team")["points"].shift(1)
        perf["gd_prev"] = perf.groupby("team")["gd"].shift(1)
        perf["form5_pts"] = (
            perf.groupby("team")["points_prev"]
                .rolling(5, min_periods=1)
                .sum()
                .reset_index(level=0, drop=True)
        )
        perf["form5_gd"] = (
            perf.groupby("team")["gd_prev"]
                .rolling(5, min_periods=1)
                .sum()
                .reset_index(level=0, drop=True)
        )
        # build per-team time series for asof (no overlapping columns)
        form_ts = (
            perf[["team","date","form5_pts","form5_gd"]]
                .dropna(subset=["team","date"]) 
                .sort_values(["team","date"])
                .reset_index(drop=True)
        )

        # helper to merge form into side (home/away) without leakage
        def merge_form(df_matches, side_col_prefix, team_col):
            # create left frame with standardized schema
            m = df_matches[["fixture_id", "date", team_col]].rename(columns={team_col: "team"}).copy()
            m["date"] = pd.to_datetime(m["date"], utc=True, errors="coerce")
            m = m.dropna(subset=["team", "date"]).copy()

            # make a clean, typed copy of the form time series
            ts = form_ts.dropna(subset=["team", "date"]).copy()
            ts["date"] = pd.to_datetime(ts["date"], utc=True, errors="coerce")

            out_parts = []
            for t in sorted(set(m["team"]).intersection(set(ts["team"]))):
                left = m.loc[m["team"] == t, ["fixture_id", "date", "team"]].sort_values("date").reset_index(drop=True)
                right = ts.loc[ts["team"] == t, ["date", "team", "form5_pts", "form5_gd"]].sort_values("date").reset_index(drop=True)
                if left.empty:
                    continue
                if right.empty:
                    # no history for this team yet; fill NaNs
                    tmp = left.copy()
                    tmp["form5_pts"] = np.nan
                    tmp["form5_gd"] = np.nan
                    out_parts.append(tmp[["fixture_id", "form5_pts", "form5_gd"]])
                    continue
                merged = pd.merge_asof(
                    left,
                    right.drop(columns=["team"]),
                    on="date",
                    direction="backward",
                    allow_exact_matches=True,
                )
                out_parts.append(merged[["fixture_id", "form5_pts", "form5_gd"]])

            if out_parts:
                m_final = pd.concat(out_parts, ignore_index=True)
            else:
                # no teams matched; return empty frame with expected columns
                m_final = pd.DataFrame({
                    "fixture_id": pd.Series(dtype="Int64"),
                    "form5_pts": pd.Series(dtype="float64"),
                    "form5_gd": pd.Series(dtype="float64"),
                })

            m_final = m_final.rename(columns={
                "form5_pts": f"{side_col_prefix}_form5_pts",
                "form5_gd": f"{side_col_prefix}_form5_gd",
            })
            return m_final

        # start from finished match rows to avoid creating extras
        train = hist[[
            "fixture_id","league_id","league_name","season","date","sport_key",
            "home","away","home_norm","away_norm","home_id","away_id","result"
        ]].dropna(subset=["fixture_id","date"]).copy()
        # cast fixture_id for stability
        train["fixture_id"] = pd.to_numeric(train["fixture_id"], errors="coerce").astype("Int64")

        # merge home/away form5 via asof
        home_form = merge_form(train, "home", "home_norm")
        away_form = merge_form(train, "away", "away_norm")
        train = train.merge(home_form, on="fixture_id", how="left")
        train = train.merge(away_form, on="fixture_id", how="left")

        # merge teamstats into train on team_id (home/away)
        if stats_frames:
            stats_tr = stats.copy()
            if "home_id" in train.columns:
                train = train.merge(stats_tr.add_prefix("home_"), left_on="home_id", right_on="home_team_id", how="left")
                train = train.drop(columns=[c for c in ["home_team_id"] if c in train.columns])
            if "away_id" in train.columns:
                train = train.merge(stats_tr.add_prefix("away_"), left_on="away_id", right_on="away_team_id", how="left")
                train = train.drop(columns=[c for c in ["away_team_id"] if c in train.columns])

        # optional: attach entry probs if latest snapshot includes same fixtures (rare for historical)
        # we leave p_entry_* as NaN here; model will still train using form-features.

        # ---- Add H2H to train (cutoff = match date) ----
        if not train.empty and {"home_norm","away_norm","date"}.issubset(train.columns):
            def _row_h2h_tr(row):
                cutoff = pd.to_datetime(row.get("date"), utc=True, errors="coerce")
                g, pts, gd, wr = _compute_h2h_for_pair(hist, row.get("home_norm"), row.get("away_norm"), cutoff)
                return pd.Series({
                    "h2h_games": g,
                    "h2h_pts_home": pts,
                    "h2h_gd_home": gd,
                    "h2h_win_rate_home": wr,
                })
            h2h_tr = train.apply(_row_h2h_tr, axis=1)
            train = pd.concat([train, h2h_tr], axis=1)

        # standings into train
        if std_frames:
            std_all_tr = pd.concat(std_frames, ignore_index=True)
            for c in ["team_id","rank","points","goalsDiff"]:
                if c in std_all_tr.columns:
                    std_all_tr[c] = pd.to_numeric(std_all_tr[c], errors="coerce")
            if "home_id" in train.columns:
                train = train.merge(std_all_tr.add_prefix("home_"), left_on=["home_id","league_id"], right_on=["home_team_id","home_league_id"], how="left")
                train = train.drop(columns=[c for c in ["home_team_id","home_league_id"] if c in train.columns])
            if "away_id" in train.columns:
                train = train.merge(std_all_tr.add_prefix("away_"), left_on=["away_id","league_id"], right_on=["away_team_id","away_league_id"], how="left")
                train = train.drop(columns=[c for c in ["away_team_id","away_league_id"] if c in train.columns])
            for a,b,name in [("home_rank","away_rank","rank_diff"),("home_points","away_points","points_diff"),("home_goalsDiff","away_goalsDiff","goalsdiff_diff")]:
                if a in train.columns and b in train.columns:
                    train[name] = pd.to_numeric(train[a], errors="coerce") - pd.to_numeric(train[b], errors="coerce")

        # rolling wdl + rest-days into train (window=5)
        if not hist.empty:
            train = _attach_rolling_wdl_and_rest(train, hist, "home", "home_norm", window=5)
            train = _attach_rolling_wdl_and_rest(train, hist, "away", "away_norm", window=5)

        # league home-adv
        lha_tr = _compute_league_home_adv(hist)
        if not lha_tr.empty:
            train = train.merge(lha_tr, on="league_id", how="left")

        # injuries last30 — **note**: these are anchored to NOW, so only meaningful for upcoming; we skip for train to avoid leakage

        # save training set
        # --- attach final scores for finished fixtures into train ---
        # Antag at du allerede har en DataFrame 'train' med 'fixture_id', 'league_id', 'season'
        # og at dine rå fixtures er gemt pr. liga i data/fixtures/soccer_*_{season}.csv

        import glob, json

        def _load_fixtures_with_goals(season: int, league_ids: list[int]) -> pd.DataFrame:
            # læs alle fixtures csv for pågældende sæson/liger
            paths = glob.glob(str(ROOT / "data" / "fixtures" / f"*.csv"))
            frames = []
            for p in paths:
                try:
                    df_fx = pd.read_csv(p)
                except Exception:
                    continue
                # heuristik: kræv kolonner der typisk findes i dine fixtures-CSV
                needed = {"fixture_id","league_id","season","status_short","goals_home","goals_away"}
                if not needed.issubset(set(c.lower() for c in df_fx.columns)):
                    # prøv at normalisere kolonnenavne
                    df_fx.columns = [c.lower() for c in df_fx.columns]
                    if not needed.issubset(df_fx.columns):
                        continue
                # filter sæson/liger
                df_fx = df_fx[(df_fx["season"] == season) & (df_fx["league_id"].isin(league_ids))].copy()
                # behold kun færdigspillede
                df_fx = df_fx[df_fx["status_short"].isin(["FT","AET","PEN"])]
                # vælg relevante kolonner og rename til konsistente navne
                df_fx = df_fx[["fixture_id","league_id","season","goals_home","goals_away"]].copy()
                df_fx = df_fx.rename(columns={"goals_home":"home_goals","goals_away":"away_goals"})
                frames.append(df_fx)
            if not frames:
                return pd.DataFrame(columns=["fixture_id","home_goals","away_goals"])
            out = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["fixture_id"], keep="last")
            return out

        # hent mål og merge
        _league_ids = sorted(train["league_id"].dropna().astype(int).unique().tolist())
        _season = int(pd.to_numeric(train["season"], errors="coerce").dropna().iloc[0])
        print(f"   → attaching goals snapshot for season={_season} leagues={json.dumps(_league_ids)}")
        _goals = _load_fixtures_with_goals(_season, _league_ids)
        if not _goals.empty:
            train = train.merge(_goals[["fixture_id","home_goals","away_goals"]], on="fixture_id", how="left")

        # --- labels for extra markets ---
        if {"home_goals","away_goals"}.issubset(train.columns):
            train["home_goals"] = pd.to_numeric(train["home_goals"], errors="coerce")
            train["away_goals"] = pd.to_numeric(train["away_goals"], errors="coerce")
            train["label_ou25"] = ((train["home_goals"] + train["away_goals"]) >= 3).astype("Int64")
            train["label_ah_home_m0_5"] = (train["home_goals"] > train["away_goals"]).astype("Int64")
            print(
                "   ➕ labels:",
                f"OU2.5 positives={int(pd.to_numeric(train['label_ou25'], errors='coerce').sum())},",
                f"AH(Home -0.5) positives={int(pd.to_numeric(train['label_ah_home_m0_5'], errors='coerce').sum())}"
            )
        else:
            print("   ⚠️ missing goals → could not create OU/AH labels")
        keep_cols_tr = [c for c in [
            "fixture_id","league_id","league_name","season","date","sport_key",
            "home","away","home_norm","away_norm","home_id","away_id","result","home_goals","away_goals",
            # standings-derived
            "home_rank","away_rank","rank_diff","home_points","away_points","points_diff","home_goalsDiff","away_goalsDiff","goalsdiff_diff",
            # form features
            "home_form5_pts","home_form5_gd","away_form5_pts","away_form5_gd",
            # rolling WDL + rest-days
            "home_wins5","home_draws5","home_losses5","home_rest_days",
            "away_wins5","away_draws5","away_losses5","away_rest_days",
            # teamstats-derived
            "home_goals_for_avg","home_goals_against_avg","home_win_rate","home_draw_rate","home_loss_rate","home_cs_rate","home_fts_rate",
            "away_goals_for_avg","away_goals_against_avg","away_win_rate","away_draw_rate","away_loss_rate","away_cs_rate","away_fts_rate",
            # league priors
            "league_home_win_rate",
            # H2H
            "h2h_games","h2h_pts_home","h2h_gd_home","h2h_win_rate_home",
        ] if c in train.columns]
        for extra_col in ["label_ou25", "label_ah_home_m0_5"]:
            if extra_col in train.columns and extra_col not in keep_cols_tr:
                keep_cols_tr.append(extra_col)
        train = train[keep_cols_tr].sort_values(["date","league_id","fixture_id"]).reset_index(drop=True)
        train_set = train.copy()

        # Merge historic per-fixture stats into TRAIN set (by fixture_id)
        try:
            if 'train_set' in locals() and not train_set.empty and not hist_stats_df.empty:
                train_set["fixture_id"] = pd.to_numeric(train_set["fixture_id"], errors="coerce").astype("Int64")
                hist_stats_df["fixture_id"] = pd.to_numeric(hist_stats_df["fixture_id"], errors="coerce").astype("Int64")
                train_set = train_set.merge(hist_stats_df, on="fixture_id", how="left")
                print(f"   ➕ merged historic per-fixture stats into TRAIN set (cols_added≈{hist_stats_df.shape[1]-1})")
        except Exception as e:
            print(f"   ⚠️ could not merge historic stats into TRAIN set: {e}")

        train_df = train_set

        pm_path = FEAT / "postmatch_flat.parquet"
        if pm_path.exists() and "fixture_id" in train_df.columns:
            try:
                pm = pd.read_parquet(pm_path)
                keep = [
                    c for c in pm.columns
                    if c in (
                        "fixture_id",
                        "closing_odds_h",
                        "closing_odds_d",
                        "closing_odds_a",
                        "closing_bk_count",
                    )
                    or c.startswith("home_") or c.startswith("away_")
                ]
                pm = pm[keep].copy()
                n_before = train_df.shape[1]
                train_df = train_df.merge(pm, on="fixture_id", how="left")
                print(f"➕ merged postmatch stats → train_set (+{train_df.shape[1]-n_before} kolonner)")
            except Exception as e:
                print(f"⚠️ postmatch merge skipped: {e}")

        train = train_df
        out_tr = FEAT / "train_set.parquet"
        FEAT.mkdir(parents=True, exist_ok=True)
        train.to_parquet(out_tr, index=False)
        print(f"   ➕ train_set → {out_tr} ({len(train)})")


if __name__ == "__main__":
    main()
