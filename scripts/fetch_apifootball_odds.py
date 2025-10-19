#!/usr/bin/env python3
import os, time
import pandas as pd, numpy as np
import requests
from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Iterable
from typing import List, Dict
import re
import hashlib
import json
# Optional: Postgres upsert for snapshots
try:
    from sqlalchemy import create_engine, text  # type: ignore
except Exception:
    create_engine = None  # type: ignore
    text = None  # type: ignore

from common import FEAT, ODIR, add_normalized_teams

API_BASE = "https://v3.football.api-sports.io"

# ---- simple on-disk cache for API-Football odds/bookmakers/mapping ----
CACHE_DIR = Path("data/.cache/apifootball_odds")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_CACHE_DIR = CACHE_DIR

def _cache_path(endpoint: str, params: dict):
    # stable key from endpoint + sorted params
    key_src = endpoint + "?" + "&".join([f"{k}={params[k]}" for k in sorted(params.keys())])
    h = hashlib.sha1(key_src.encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{endpoint.replace('/','_')}__{h}.json"

def _now_ts():
    return int(datetime.now(timezone.utc).timestamp())

def _get_cached(endpoint: str, params: dict, ttl_seconds: int, retries=5, backoff=1.4, allow_stale=True):
    """
    Get JSON 'response' list with TTL caching.
    If network fails and allow_stale=True, returns stale cache if present.
    """
    p = _cache_path(endpoint, params)
    # try fresh cache
    if p.exists():
        try:
            with p.open("r", encoding="utf-8") as f:
                blob = json.load(f)
            if isinstance(blob, dict) and "saved_at" in blob and "response" in blob:
                age = _now_ts() - int(blob.get("saved_at", 0))
                if age <= ttl_seconds:
                    return blob.get("response", [])
        except Exception:
            pass

    # fetch live
    last_exc = None
    for i in range(retries):
        try:
            r = requests.get(f"{API_BASE}/{endpoint}", headers=_headers(), params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(backoff * (i + 1))
                continue
            if r.status_code in (401, 403):
                print(f"⛔ {endpoint} auth error {r.status_code}: {r.text[:140]}")
                break
            r.raise_for_status()
            js = r.json()
            resp = js.get("response", [])
            try:
                with p.open("w", encoding="utf-8") as f:
                    json.dump({"saved_at": _now_ts(), "response": resp}, f)
            except Exception:
                pass
            return resp
        except Exception as e:
            last_exc = e
            time.sleep(backoff * (i + 1))

    # fall back to stale cache
    if allow_stale and p.exists():
        try:
            with p.open("r", encoding="utf-8") as f:
                blob = json.load(f)
            print(f"ℹ️ using STALE cache for {endpoint} params={params}")
            return blob.get("response", [])
        except Exception:
            pass
    if last_exc:
        print(f"⚠️ GET {endpoint} failed after {retries} retries: {last_exc}")
    return []

def _headers():
    k = os.getenv("APIFOOTBALL_KEY")
    if not k:
        raise RuntimeError("APIFOOTBALL_KEY not set")
    return {"x-apisports-key": k}

BET_ID_MATCH_WINNER = 1  # API-Football: bet parameter is an INTEGER id; 1 = Match Winner (1X2)
BET_ID_OU_GOALS = 5  # Goals Over/Under (full time)

# Asian Handicap is not consistent by id across time on some accounts; match by name as well
AH_NAME_TOKENS = ("asian handicap", "handicap")

#
# default shortlist (can be overridden via --bookmakers)
DEFAULT_WANTED_BOOKMAKERS = {"888sport", "nordicbet", "unibet", "betano"}

def _parse_bookmaker_targets(arg: Optional[str]) -> set[str]:
    """Parse a comma/space separated list of bookmaker names from CLI and normalize to lowercase no-space keys."""
    if not arg:
        return set(DEFAULT_WANTED_BOOKMAKERS)
    # split on comma and strip
    toks = [t.strip() for t in arg.replace(";", ",").split(",") if t.strip()]
    return set(toks) if toks else set(DEFAULT_WANTED_BOOKMAKERS)

DEFAULT_TZ = "Europe/Copenhagen"

# ------------------- HTTP util -------------------

def _get(endpoint: str, params: dict, retries=5, backoff=1.4):
    last_exc = None
    for i in range(retries):
        try:
            r = requests.get(f"{API_BASE}/{endpoint}", headers=_headers(), params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(backoff * (i + 1))
                continue
            if r.status_code in (401, 403):
                # surface a helpful message then stop
                print(f"⛔ {endpoint} auth error {r.status_code}: {r.text[:140]}")
                return []
            r.raise_for_status()
            js = r.json()
            return js.get("response", [])
        except Exception as e:
            last_exc = e
            time.sleep(backoff * (i + 1))
    if last_exc:
        print(f"⚠️ GET {endpoint} failed after {retries} retries: {last_exc}")
    return []

# ------------------- Data loaders -------------------

def load_upcoming_fixture_ids(days=14):
    up_path = FEAT / "upcoming_set.parquet"
    if not up_path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(up_path)
    now = pd.Timestamp.now(tz="UTC")
    cutoff = now + pd.Timedelta(days=days)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        df = df[(df["date"] >= now - pd.Timedelta(hours=1)) & (df["date"] <= cutoff)]
    if "fixture_id" not in df.columns:
        return pd.DataFrame()
    keep = [c for c in [
        "fixture_id","league_id","season","sport_key","home","away","home_norm","away_norm","date"
    ] if c in df.columns]
    df = df[keep].copy()
    if "season" not in df.columns or df["season"].isna().all():
        df["season"] = df["date"].dt.year
    df["fixture_id"] = pd.to_numeric(df["fixture_id"], errors="coerce").astype("Int64")
    if "league_id" in df.columns:
        df["league_id"] = pd.to_numeric(df["league_id"], errors="coerce").astype("Int64")
    return df.dropna(subset=["fixture_id"]).drop_duplicates("fixture_id")

# --- helpers: robust set extraction from a column-like ---

def _unique_int_set(col_like) -> set[int]:
    """Return a set of ints from a Series-like; robust against scalars/None."""
    try:
        s = pd.to_numeric(col_like, errors="coerce")
        # if scalar, s won't have dropna
        if not hasattr(s, "dropna"):
            return set()
        s = s.dropna()
        if s.empty:
            return set()
        return set(s.astype(int).tolist())
    except Exception:
        return set()

# ------------------- Bookmakers -------------------

def discover_bookmakers(targets: Optional[Iterable[str]] = None, ttl_seconds: int = 7*24*3600):
    """Return (all_map, wanted_map) where keys are lowercase names; values are (id, display)."""
    resp = _get_cached("odds/bookmakers", {}, ttl_seconds=ttl_seconds)
    name_to_meta = {}
    for row in resp:
        nm = str(row.get("name", "")).strip()
        bid = row.get("id")
        if not nm or bid is None:
            continue
        name_to_meta[nm.lower()] = (int(bid), nm)
    if not targets:
        targets = DEFAULT_WANTED_BOOKMAKERS
    # normalize comparison keys (lower + nospace)
    targets_lc = {t.lower(): t for t in targets}
    targets_flat = {t.replace(" ", ""): t for t in targets_lc}

    wanted = {}
    for nm_l, (bid, disp) in name_to_meta.items():
        nm_flat = nm_l.replace(" ", "")
        for key in list(targets_lc) + list(targets_flat):
            if key in nm_l or key in nm_flat:
                wanted[nm_l] = (bid, disp)
                break
    return name_to_meta, wanted

# ------------------- Mapping helper -------------------

def fetch_mapping_fixture_ids(days: int,
                              leagues: set[int] | None = None,
                              season: int | None = None,
                              timezone_str: str = DEFAULT_TZ,
                              max_pages: int = 5,
                              ttl_seconds: int = 6*3600):
    """Return a set of fixture_ids that API-Football reports as having odds within the next N days.
    Some accounts require league+season filters for mapping to return anything. We therefore iterate per-league, per-date.
    """
    fx_ids: set[int] = set()
    today = pd.Timestamp.now(tz="UTC").date()

    if leagues:
        league_list = sorted({int(x) for x in leagues})
    else:
        league_list = [None]

    for d in range(days + 1):
        day = today + pd.Timedelta(days=d)
        for lg in league_list:
            page = 1
            while page <= max_pages:
                params = {"date": str(day), "timezone": timezone_str, "page": page}
                if lg is not None:
                    params["league"] = int(lg)
                if season is not None:
                    params["season"] = int(season)
                resp = _get_cached("odds/mapping", params, ttl_seconds=ttl_seconds)
                if not resp:
                    break
                for item in resp:
                    lg_obj = item.get("league", {})
                    if leagues and int(lg_obj.get("id", 0)) not in leagues:
                        continue
                    fx = item.get("fixture", {})
                    fid = fx.get("id")
                    if fid:
                        fx_ids.add(int(fid))
                page += 1
                time.sleep(0.05)
    return fx_ids

# ------------------- Odds parsing -------------------

def parse_1x2_rows(resp_item):
    out = []
    fixture = resp_item.get("fixture", {})
    fid = fixture.get("id")
    for bk in resp_item.get("bookmakers", []) or []:
        bid = bk.get("id")
        bname = bk.get("name") or ""
        odds_h = odds_d = odds_a = None
        for bet in bk.get("bets", []) or []:
            bet_id = bet.get("id")
            bet_name = (bet.get("name") or "").lower()
            if not (bet_id == BET_ID_MATCH_WINNER or any(k in bet_name for k in ("match winner","1x2","fulltime result","win-draw-win"))):
                continue
            for v in bet.get("values", []) or []:
                label_raw = (v.get("value") or v.get("label") or '').strip()
                val = label_raw.lower()
                try:
                    price = float(v.get("odd"))
                except Exception:
                    price = None
                if price is None:
                    continue
                if val in ("home", "1", "team 1", "1 (home)"):
                    odds_h = price
                elif val in ("away", "2", "team 2", "2 (away)"):
                    odds_a = price
                else:
                    # be permissive on draw/tie labels
                    if any(k in val for k in ("draw", "x", "tie")):
                        odds_d = price
        if fid is not None and (odds_h or odds_d or odds_a):
            out.append({
                "fixture_id": fid,
                "bookmaker_id": bid,
                "bookmaker_name": bname,
                "odds_h": odds_h,
                "odds_d": odds_d,
                "odds_a": odds_a,
            })
    return out

# ------------------- Over/Under 2.5 parsing -------------------

def parse_ou25_rows(resp_item):
    """Return one row per bookmaker *per goal line* with both over and under odds if present."""
    out = []
    fixture = resp_item.get("fixture", {})
    fid = fixture.get("id")
    for bk in resp_item.get("bookmakers", []) or []:
        bid = bk.get("id")
        bname = bk.get("name") or ""
        # line -> {over: x, under: y}
        by_line = {}
        for bet in bk.get("bets", []) or []:
            if bet.get("id") != BET_ID_OU_GOALS:
                continue
            for v in bet.get("values", []) or []:
                label_raw = (v.get("value") or v.get("label") or "").strip()
                label = label_raw.lower()
                try:
                    price = float(v.get("odd"))
                except Exception:
                    price = None
                # Extract numeric line (e.g., "Over 2.5")
                line_val = None
                for p in label_raw.replace("/", " ").split():
                    try:
                        line_val = float(p)
                        break
                    except Exception:
                        continue
                if line_val is None:
                    # sometimes line sits in separate field
                    try:
                        line_val = float(v.get("handicap"))
                    except Exception:
                        pass
                if line_val is None:
                    continue
                slot = by_line.setdefault(line_val, {"over": None, "under": None})
                if "over" in label and price is not None:
                    slot["over"] = max(filter(lambda x: x is not None, [slot["over"], price])) if slot["over"] else price
                elif "under" in label and price is not None:
                    slot["under"] = max(filter(lambda x: x is not None, [slot["under"], price])) if slot["under"] else price
        for line, kv in by_line.items():
            if kv.get("over") is None and kv.get("under") is None:
                continue
            out.append({
                "fixture_id": fid,
                "bookmaker_id": bid,
                "bookmaker_name": bname,
                "ou_line": float(line),
                "ou_over": kv.get("over"),
                "ou_under": kv.get("under"),
            })
    return out

# ------------------- Asian Handicap parsing -------------------

def parse_ah_rows(resp_item):
    """Parse Asian Handicap market per bookmaker. We collect best prices per line for home/away handicaps.
    The API sometimes exposes handicap info in `value`/`label` or `handicap`. We try both.
    """
    out = []
    fixture = resp_item.get("fixture", {})
    fid = fixture.get("id")
    for bk in resp_item.get("bookmakers", []) or []:
        bid = bk.get("id")
        bname = bk.get("name") or ""
        by_line = {}  # line -> {home: price, away: price}
        for bet in bk.get("bets", []) or []:
            bet_name = (bet.get("name") or "").lower()
            # Asian handicap detection by name tokens (id varies between feeds)
            if not any(tok in bet_name for tok in AH_NAME_TOKENS):
                continue
            for v in bet.get("values", []) or []:
                label_raw = (v.get("value") or v.get("label") or "").strip()
                lbl = label_raw.lower()
                try:
                    price = float(v.get("odd"))
                except Exception:
                    price = None
                # Extract numeric handicap line, prefer `handicap` field if present
                line_val = None
                try:
                    line_val = float(v.get("handicap"))
                except Exception:
                    pass
                if line_val is None:
                    # Try to find last signed float in label like "Home -0.5" or "Away +0.25"
                    m = re.search(r"([+-]?\d+(?:\.\d+)?)", label_raw)
                    if m:
                        try:
                            line_val = float(m.group(1))
                        except Exception:
                            line_val = None
                if line_val is None:
                    continue
                slot = by_line.setdefault(float(line_val), {"home": None, "away": None})
                # Determine side (home/away)
                side = None
                if "home" in lbl or "1 (home)" in lbl or lbl.startswith("1 "):
                    side = "home"
                elif "away" in lbl or "2 (away)" in lbl or lbl.startswith("2 "):
                    side = "away"
                elif "team 1" in lbl:
                    side = "home"
                elif "team 2" in lbl:
                    side = "away"
                # If side not detected, infer from sign convention used by many feeds: negative is favourite (home) sometimes
                if side is None:
                    side = "home" if str(label_raw).strip().lower().startswith("home") else ("away" if str(label_raw).strip().lower().startswith("away") else None)
                if side and price is not None:
                    cur = slot.get(side)
                    slot[side] = price if (cur is None or price > cur) else cur
        for line, kv in by_line.items():
            if kv.get("home") is None and kv.get("away") is None:
                continue
            out.append({
                "fixture_id": fid,
                "bookmaker_id": bid,
                "bookmaker_name": bname,
                "ah_line": float(line),
                "ah_home": kv.get("home"),
                "ah_away": kv.get("away"),
            })
    return out

# ------------------- Fetchers -------------------
# ------------------- Fetcher for Asian Handicap -------------------

def fetch_fixture_ah(fid: int, sleep: float, ttl_seconds: int):
    params = {"fixture": int(fid), "timezone": DEFAULT_TZ}
    resp = _get_cached("odds", params, ttl_seconds=ttl_seconds)
    rows = []
    for it in resp:
        rows.extend(parse_ah_rows(it))
    if rows:
        return pd.DataFrame(rows)
    time.sleep(sleep)
    return pd.DataFrame(columns=["fixture_id","bookmaker_id","bookmaker_name","ah_line","ah_home","ah_away"])

def fetch_fixture_odds(fid: int, sleep: float, ttl_seconds: int):
    params = {"fixture": int(fid), "bet": BET_ID_MATCH_WINNER, "timezone": DEFAULT_TZ}
    resp = _get_cached("odds", params, ttl_seconds=ttl_seconds)
    rows = []
    for it in resp:
        rows.extend(parse_1x2_rows(it))
    if rows:
        return pd.DataFrame(rows)
    time.sleep(sleep)
    return pd.DataFrame(columns=["fixture_id","bookmaker_id","bookmaker_name","odds_h","odds_d","odds_a"])


# ------------------- Fetcher for OU 2.5 -------------------
def fetch_fixture_ou25(fid: int, sleep: float, ttl_seconds: int):
    params = {"fixture": int(fid), "bet": BET_ID_OU_GOALS, "timezone": DEFAULT_TZ}
    resp = _get_cached("odds", params, ttl_seconds=ttl_seconds)
    rows = []
    for it in resp:
        rows.extend(parse_ou25_rows(it))
    if rows:
        return pd.DataFrame(rows)
    time.sleep(sleep)
    return pd.DataFrame(columns=["fixture_id","bookmaker_id","bookmaker_name","ou_line","ou_over","ou_under"])


def fetch_league_odds(league_id: int, season: int, sleep: float, bookmaker_id: int | None = None, max_pages: int = 20, days_ahead: int = 14, ttl_seconds_date: int = 15*60):
    """Fetch pre-match 1x2 odds by iterating date windows (docs: pre-match odds exist 1–14 days before KO).
    We iterate each date from today to today+days_ahead and paginate up to max_pages per date.
    """
    base_params = {"league": int(league_id), "season": int(season), "bet": BET_ID_MATCH_WINNER}
    if bookmaker_id:
        base_params["bookmaker"] = int(bookmaker_id)
    rows = []
    today = pd.Timestamp.now(tz="UTC").date()
    for d in range(days_ahead + 1):
        day = today + pd.Timedelta(days=d)
        params = dict(base_params)
        params["date"] = str(day)
        params["page"] = 1
        while params["page"] <= max_pages:
            params["timezone"] = DEFAULT_TZ
            resp = _get_cached("odds", params, ttl_seconds=ttl_seconds_date)
            if not resp:
                break
            for it in resp:
                rows.extend(parse_1x2_rows(it))
            params["page"] += 1
            time.sleep(sleep)
        # be gentle between dates
        time.sleep(min(0.2, sleep))
    if not rows:
        return pd.DataFrame(columns=["fixture_id","bookmaker_id","bookmaker_name","odds_h","odds_d","odds_a"])
    return pd.DataFrame(rows)

# ------------------- Aggregation -------------------

def best_of(df_rows: pd.DataFrame, wanted_bids: set):
    if df_rows.empty:
        return None
    sub = df_rows[df_rows["bookmaker_id"].isin(list(wanted_bids))] if wanted_bids else df_rows
    if sub.empty:
        sub = df_rows
    row = {"bookmaker_count": int(sub["bookmaker_id"].nunique())}
    for col, suf in (("odds_h","h"),("odds_d","d"),("odds_a","a")):
        cand = sub[[col,"bookmaker_name"]].dropna(subset=[col])
        if cand.empty:
            row[f"odds_{suf}"] = np.nan
            row[f"src_book_{suf}"] = None
        else:
            idx = cand[col].astype(float).idxmax()
            row[f"odds_{suf}"] = float(cand.loc[idx, col])
            row[f"src_book_{suf}"] = str(cand.loc[idx, "bookmaker_name"])[:64]
    return row


# ------------------- Snapshot helpers (optional Postgres) -------------------
def _pg():
    """Return a SQLAlchemy engine if POSTGRES_URL is set and sqlalchemy is available; else None."""
    url = os.environ.get("POSTGRES_URL")
    if not url or create_engine is None:
        return None
    try:
        return create_engine(url, pool_pre_ping=True)  # type: ignore
    except Exception:
        return None

def _snap_rows_from_1x2_df(out_df: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    if out_df is None or out_df.empty:
        return rows
    for _, r in out_df.iterrows():
        for sel, col, src_b in (("H","odds_h","src_book_h"),
                                ("D","odds_d","src_book_d"),
                                ("A","odds_a","src_book_a")):
            price = r.get(col)
            if pd.notna(price):
                rows.append({
                    "fixture_id": int(r["fixture_id"]),
                    "market": "1X2",
                    "selection": sel,
                    "line": None,
                    "price": float(price),
                    "bookmaker": (r.get(src_b) or None),
                    "fetched_at_utc": pd.to_datetime(r.get("fetched_at_utc"), utc=True, errors="coerce"),
                    "home": r.get("home"),
                    "away": r.get("away"),
                    "source": "apifootball",
                })
    return rows


# Emit snapshot rows for all bookmaker quotes present in the raw per-bookmaker 1X2 dataframe
def _snap_rows_from_1x2_book_df(df_rows: pd.DataFrame) -> List[Dict]:
    """
    Build snapshot rows for all bookmaker quotes present in the raw per-bookmaker 1X2 dataframe.
    Expected columns: fixture_id, bookmaker_name, odds_h/odds_d/odds_a, and fetched_at_utc/home/away.
    """
    rows: List[Dict] = []
    if df_rows is None or df_rows.empty:
        return rows
    for _, r in df_rows.iterrows():
        bname = r.get("bookmaker_name") or None
        for sel, col in (("H","odds_h"), ("D","odds_d"), ("A","odds_a")):
            price = r.get(col)
            if pd.notna(price):
                rows.append({
                    "fixture_id": int(r["fixture_id"]),
                    "market": "1X2",
                    "selection": sel,
                    "line": None,
                    "price": float(price),
                    "bookmaker": bname if bname else None,
                    "fetched_at_utc": pd.to_datetime(r.get("fetched_at_utc"), utc=True, errors="coerce"),
                    "home": r.get("home"),
                    "away": r.get("away"),
                    "source": "apifootball",
                })
    return rows

def _snap_rows_from_ou_df(ou_df: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    if ou_df is None or ou_df.empty:
        return rows
    for _, r in ou_df.iterrows():
        line = r.get("ou_line")
        if pd.notna(line):
            if pd.notna(r.get("ou_over")):
                rows.append({
                    "fixture_id": int(r["fixture_id"]),
                    "market": "totals",
                    "selection": "O",
                    "line": float(line),
                    "price": float(r["ou_over"]),
                    "bookmaker": None,
                    "fetched_at_utc": pd.to_datetime(r.get("fetched_at_utc"), utc=True, errors="coerce"),
                    "home": r.get("home"), "away": r.get("away"),
                    "source": "apifootball",
                })
            if pd.notna(r.get("ou_under")):
                rows.append({
                    "fixture_id": int(r["fixture_id"]),
                    "market": "totals",
                    "selection": "U",
                    "line": float(line),
                    "price": float(r["ou_under"]),
                    "bookmaker": None,
                    "fetched_at_utc": pd.to_datetime(r.get("fetched_at_utc"), utc=True, errors="coerce"),
                    "home": r.get("home"), "away": r.get("away"),
                    "source": "apifootball",
                })
    return rows

def _snap_rows_from_ah_df(ah_df: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    if ah_df is None or ah_df.empty:
        return rows
    for _, r in ah_df.iterrows():
        line = r.get("ah_line")
        if pd.notna(line):
            if pd.notna(r.get("ah_home")):
                rows.append({
                    "fixture_id": int(r["fixture_id"]),
                    "market": "ahc",
                    "selection": "H",
                    "line": float(line),
                    "price": float(r["ah_home"]),
                    "bookmaker": None,
                    "fetched_at_utc": pd.to_datetime(r.get("fetched_at_utc"), utc=True, errors="coerce"),
                    "home": r.get("home"), "away": r.get("away"),
                    "source": "apifootball",
                })
            if pd.notna(r.get("ah_away")):
                rows.append({
                    "fixture_id": int(r["fixture_id"]),
                    "market": "ahc",
                    "selection": "A",
                    "line": float(line),
                    "price": float(r["ah_away"]),
                    "bookmaker": None,
                    "fetched_at_utc": pd.to_datetime(r.get("fetched_at_utc"), utc=True, errors="coerce"),
                    "home": r.get("home"), "away": r.get("away"),
                    "source": "apifootball",
                })
    return rows

def _upsert_snapshots(rows: list[dict]):
    eng = _pg()
    if eng is None or not rows:
        return
    # Ensure table exists (idempotent)
    with eng.begin() as c:
        c.execute(text("""
        CREATE TABLE IF NOT EXISTS odds_snapshots (
          id BIGSERIAL PRIMARY KEY,
          fixture_id BIGINT NOT NULL,
          market TEXT NOT NULL,
          selection TEXT,
          line DOUBLE PRECISION,
          price DOUBLE PRECISION,
          bookmaker TEXT,
          fetched_at_utc TIMESTAMPTZ NOT NULL,
          home TEXT,
          away TEXT,
          source TEXT,
          UNIQUE(fixture_id, market, selection, line, bookmaker, fetched_at_utc)
        );
        CREATE INDEX IF NOT EXISTS idx_snap_fx ON odds_snapshots(fixture_id);
        CREATE INDEX IF NOT EXISTS idx_snap_time ON odds_snapshots(fetched_at_utc);
        """))
        stmt = text("""
          INSERT INTO odds_snapshots
            (fixture_id, market, selection, line, price, bookmaker, fetched_at_utc, home, away, source)
          VALUES
            (:fixture_id, :market, :selection, :line, :price, :bookmaker, :fetched_at_utc, :home, :away, :source)
          ON CONFLICT (fixture_id, market, selection, line, bookmaker, fetched_at_utc) DO NOTHING
        """)
        for i in range(0, len(rows), 500):
            c.execute(stmt, rows[i:i+500])

# ------------------- Main -------------------

def main():
    global CACHE_DIR
    ap = ArgumentParser()
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--sleep", type=float, default=0.35)
    ap.add_argument("--use-mapping", action="store_true", help="Filter fixtures to those present in odds/mapping for the date range")
    ap.add_argument("--timezone", type=str, default=DEFAULT_TZ, help="Timezone for date-based odds/mapping calls")
    ap.add_argument("--no-fallback", action="store_true", help="Only call odds by fixture (skip league/season sweeps)")
    ap.add_argument("--include-ou25", action="store_true", help="Also fetch Goals Over/Under (2.5) market and save a separate CSV")
    ap.add_argument("--include-ah", action="store_true", help="Also fetch Asian Handicap market and save a separate CSV")
    ap.add_argument("--ah-line-target", type=float, default=-0.5, help="Preferred AH line for home side; closest line chosen if exact not present")
    ap.add_argument("--bookmakers", type=str, default=",".join(sorted(DEFAULT_WANTED_BOOKMAKERS)),
                    help="Comma-separated shortlist of bookmakers to prioritize (default: 888sport,nordicbet,unibet,betano)")
    ap.add_argument("--ou-line-target", type=float, default=2.5,
                    help="Preferred Over/Under goal line; closest line will be chosen when exact is unavailable.")
    ap.add_argument("--write-parquet", action="store_true", help="Also save output as Parquet alongside CSV")
    ap.add_argument("--force-all-books", action="store_true", help="Ignore shortlist and consider all bookmakers equally")
    ap.add_argument("--all-books-snapshots", action="store_true",
                    help="Upsert snapshots for every bookmaker quote (not just the best price).")
    ap.add_argument("--imminent-mins", type=int, default=0,
                    help="If >0, only process fixtures with kickoff within the next N minutes (filters base dataframe).")
    ap.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR), help="Directory for API cache files")
    ap.add_argument("--ttl-bookmakers", type=int, default=7*24*3600, help="TTL seconds for odds/bookmakers cache (default 7 days)")
    ap.add_argument("--ttl-mapping", type=int, default=6*3600, help="TTL seconds for odds/mapping cache (default 6h)")
    ap.add_argument("--ttl-odds-fixture", type=int, default=15*60, help="TTL seconds for odds?fixture= cache (default 15min)")
    ap.add_argument("--ttl-odds-date", type=int, default=15*60, help="TTL seconds for odds by date cache (default 15min)")
    args = ap.parse_args()

    global CACHE_DIR
    CACHE_DIR = Path(args.cache_dir)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    try:
        _ = _headers()
    except Exception as e:
        print(f"⛔ {e}")
        return

    base = load_upcoming_fixture_ids(args.days)
    if args.imminent_mins and "date" in base.columns:
        now = pd.Timestamp.now(tz="UTC")
        horizon = now + pd.Timedelta(minutes=int(args.imminent_mins))
        before = len(base)
        base = base[(base["date"] >= now - pd.Timedelta(minutes=1)) & (base["date"] <= horizon)].copy()
        print(f"ℹ️ imminent filter: {len(base)}/{before} fixtures within next {args.imminent_mins} mins")
    if (base is not None) and (not base.empty) and ("league_id" in base.columns):
        leagues_in_base = _unique_int_set(base["league_id"])  # robust against scalar/None
    else:
        leagues_in_base = set()
    if args.use_mapping:
        season_infer = None
        if (base is not None) and (not base.empty) and ("season" in base.columns) and base["season"].notna().any():
            try:
                season_infer = int(pd.to_numeric(base["season"], errors="coerce").dropna().mode().iloc[0])
            except Exception:
                season_infer = None
        map_ids = fetch_mapping_fixture_ids(args.days, leagues=leagues_in_base or None, season=season_infer, timezone_str=args.timezone, ttl_seconds=args.ttl_mapping)
        if map_ids:
            before = len(base)
            base = base[ base["fixture_id"].astype(int).isin(map_ids) ].copy()
            print(f"ℹ️ mapping filter active: {len(base)}/{before} fixtures remain (timezone={args.timezone}, season={season_infer})")
        else:
            print("ℹ️ mapping returned 0 fixtures for range — proceeding without mapping filter (league/season/date may not match window)")

    if base.empty:
        print("⚠️ No upcoming_set.parquet or no fixtures found. Run build_features first.")
        return

    # Discover bookmakers
    try:
        targets = _parse_bookmaker_targets(args.bookmakers)
        _, wanted = discover_bookmakers(targets, ttl_seconds=args.ttl_bookmakers)
        wanted_bids = {meta[0] for meta in wanted.values()}
        print("ℹ️ targeting bookmakers:", sorted([v[1].lower() for v in wanted.values()]), "-> IDs", sorted(list(wanted_bids)))
        print(f"ℹ️ timezone: {args.timezone}")
    except Exception as e:
        print(f"⚠️ bookmaker discovery failed: {e}; proceeding with all books")
        wanted_bids = set()
    if args.force_all_books:
        wanted_bids = set()
        print("ℹ️ force_all_books: using all bookmakers (no shortlist filter)")
    prefer_books = bool(wanted_bids)

    base_map = base.set_index("fixture_id")[ [c for c in ("league_id","season") if c in base.columns] ].to_dict(orient="index")

    out_rows = []
    ou_rows = [] if args.include_ou25 else None
    ah_rows = [] if args.include_ah else None
    misses = 0
    for i, (fid, row) in enumerate(base.set_index("fixture_id").iterrows(), start=1):
        try:
            df = fetch_fixture_odds(int(fid), args.sleep, args.ttl_odds_fixture)
            if df.empty:
                if args.no_fallback:
                    misses += 1
                    continue
                meta = base_map.get(fid, {})
                lg = meta.get("league_id")
                ssn = meta.get("season")
                if pd.notna(lg) and pd.notna(ssn):
                    found = False
                    # (a) targeted books, per-date
                    if prefer_books and wanted_bids:
                        for bid in wanted_bids:
                            df_lg = fetch_league_odds(int(lg), int(ssn), args.sleep, bookmaker_id=int(bid), max_pages=10, days_ahead=min(args.days,7), ttl_seconds_date=args.ttl_odds_date)
                            if (not df_lg.empty) and ("fixture_id" in df_lg.columns):
                                df = df_lg[df_lg["fixture_id"].astype("Int64") == int(fid)]
                                if not df.empty:
                                    found = True
                                    break
                        if df_lg.empty:
                            print(f"… no odds for book={bid} league={lg} season={ssn} on date-sweep")
                    # (b) all books, per-date
                    if (not found):
                        df_lg = fetch_league_odds(int(lg), int(ssn), args.sleep, bookmaker_id=None, max_pages=10, days_ahead=min(args.days,7), ttl_seconds_date=args.ttl_odds_date)
                        if (not df_lg.empty) and ("fixture_id" in df_lg.columns):
                            df = df_lg[df_lg["fixture_id"].astype("Int64") == int(fid)]
                            found = (not df.empty)
                        if df_lg.empty:
                            print(f"… no league/date odds for league={lg} season={ssn} date-sweep; will try broader sweep")
                    # (c) all books, season-wide (last resort)
                    if (not found):
                        df_lg = fetch_league_odds(int(lg), int(ssn), args.sleep, bookmaker_id=None, max_pages=10, days_ahead=0, ttl_seconds_date=args.ttl_odds_date)
                        if (not df_lg.empty) and ("fixture_id" in df_lg.columns):
                            df = df_lg[df_lg["fixture_id"].astype("Int64") == int(fid)]
            if not df.empty:
                df["sport_key"] = row.get("sport_key")
                df["date"] = row.get("date")
                df["home"] = row.get("home")
                df["away"] = row.get("away")
                df = add_normalized_teams(df, "home", "away")
                df_book = df.copy()
                df_book["fetched_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                best = best_of(df, wanted_bids)
                if best:
                    best.update({
                        "fixture_id": int(fid),
                        "date": row.get("date"),
                        "home": row.get("home"),
                        "away": row.get("away"),
                        "home_norm": row.get("home_norm"),
                        "away_norm": row.get("away_norm"),
                        "fetched_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                        "odds_source": "apifootball",
                    })
                    out_rows.append(best)
            # ---- fetch OU(2.5) if enabled ----
            if args.include_ou25:
                try:
                    df_ou = fetch_fixture_ou25(int(fid), args.sleep, args.ttl_odds_fixture)
                    if not df_ou.empty:
                        df_ou["sport_key"] = row.get("sport_key")
                        df_ou["date"] = row.get("date")
                        df_ou["home"] = row.get("home")
                        df_ou["away"] = row.get("away")
                        df_ou = add_normalized_teams(df_ou, "home", "away")
                        # filter to preferred books if available
                        sub = df_ou[df_ou["bookmaker_id"].isin(list(wanted_bids))] if wanted_bids else df_ou
                        if sub.empty:
                            sub = df_ou
                        # pick closest line to target
                        sub = sub.dropna(subset=["ou_line"]).copy()
                        if sub.empty:
                            continue
                        sub["line_diff"] = (sub["ou_line"] - float(args.ou_line_target)).abs()
                        sub = sub.sort_values(["line_diff"], ascending=[True])
                        best_line = float(sub.iloc[0]["ou_line"])  # exact or closest
                        at_line = df_ou[np.isclose(df_ou["ou_line"], best_line, rtol=0, atol=1e-6)].copy()
                        if at_line.empty:
                            # fallback if float compare fails
                            at_line = sub[sub["ou_line"] == best_line].copy()
                        # best over/under on that line across candidate bookmakers
                        over_best = at_line["ou_over"].dropna().max() if at_line["ou_over"].notna().any() else None
                        under_best = at_line["ou_under"].dropna().max() if at_line["ou_under"].notna().any() else None
                        ou_rows.append({
                            "fixture_id": int(fid),
                            "date": row.get("date"),
                            "home": row.get("home"),
                            "away": row.get("away"),
                            "home_norm": row.get("home_norm"),
                            "away_norm": row.get("away_norm"),
                            "bookmaker_count": int(df_ou["bookmaker_id"].nunique()),
                            "ou_line": best_line,
                            "ou_over": float(over_best) if over_best is not None else None,
                            "ou_under": float(under_best) if under_best is not None else None,
                            "fetched_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                            "odds_source": "apifootball",
                        })
                except Exception as e:
                    print(f"[warn-ou] fixture={fid}: {e}")
            # ---- fetch AH if enabled ----
            if args.include_ah:
                try:
                    df_ah = fetch_fixture_ah(int(fid), args.sleep, args.ttl_odds_fixture)
                    if not df_ah.empty:
                        df_ah["sport_key"] = row.get("sport_key")
                        df_ah["date"] = row.get("date")
                        df_ah["home"] = row.get("home")
                        df_ah["away"] = row.get("away")
                        df_ah = add_normalized_teams(df_ah, "home", "away")
                        sub = df_ah[df_ah["bookmaker_id"].isin(list(wanted_bids))] if wanted_bids else df_ah
                        if sub.empty:
                            sub = df_ah
                        sub = sub.dropna(subset=["ah_line"]).copy()
                        if sub.empty:
                            pass
                        else:
                            sub["line_diff"] = (sub["ah_line"] - float(args.ah_line_target)).abs()
                            sub = sub.sort_values(["line_diff"], ascending=[True])
                            best_line = float(sub.iloc[0]["ah_line"])  # exact or closest
                            at_line = df_ah[np.isclose(df_ah["ah_line"], best_line, rtol=0, atol=1e-6)].copy()
                            if at_line.empty:
                                at_line = sub[sub["ah_line"] == best_line].copy()
                            home_best = at_line["ah_home"].dropna().max() if at_line["ah_home"].notna().any() else None
                            away_best = at_line["ah_away"].dropna().max() if at_line["ah_away"].notna().any() else None
                            ah_rows.append({
                                "fixture_id": int(fid),
                                "date": row.get("date"),
                                "home": row.get("home"),
                                "away": row.get("away"),
                                "home_norm": row.get("home_norm"),
                                "away_norm": row.get("away_norm"),
                                "bookmaker_count": int(df_ah["bookmaker_id"].nunique()),
                                "ah_line": best_line,
                                "ah_home": float(home_best) if home_best is not None else None,
                                "ah_away": float(away_best) if away_best is not None else None,
                                "fetched_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                                "odds_source": "apifootball",
                            })
                except Exception as e:
                    print(f"[warn-ah] fixture={fid}: {e}")
            else:
                misses += 1
        except Exception as e:
            print(f"[warn] fixture={fid}: {e}")
        time.sleep(args.sleep)
        if i % 25 == 0:
            print(f"… {i}/{len(base)} processed; rows so far: {len(out_rows)}")

    if misses:
        print(f"ℹ️ fixtures with no odds found: {misses}/{len(base)}")

    print(f"ℹ️ processed fixtures: {len(base)}; with odds: {len(out_rows)}")
    if not out_rows:
        print("⚠️ No odds retrieved from API-Football.")
        return

    out = pd.DataFrame(out_rows)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    ODIR.mkdir(parents=True, exist_ok=True)
    # de-duplicate by fixture and price triplet
    out = out.drop_duplicates(subset=["fixture_id","odds_h","odds_d","odds_a"], keep="last")
    outfile = ODIR / f"apifootball_live_odds_{ts}.csv"
    # Ensure ISO UTC string for date
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S%z")
    cols = [
        "fixture_id","date","odds_source","fetched_at_utc",
        "home","away","home_norm","away_norm",
        "bookmaker_count","odds_h","odds_d","odds_a",
        "src_book_h","src_book_d","src_book_a",
    ]
    out = out[[c for c in cols if c in out.columns]]
    out.to_csv(outfile, index=False)
    print(f"✅ saved API-Football odds → {outfile} (rows={len(out)})")
    if args.write_parquet:
        out_parq = ODIR / f"apifootball_live_odds_{ts}.parquet"
        out.to_parquet(out_parq, index=False)
        print(f"📦 also saved Parquet → {out_parq}")

    # Upsert 1X2 snapshots to Postgres (optional)
    try:
        if args.all_books_snapshots:
            try:
                # join minimal context (home/away) from 'out' back into df_book if needed
                if "home" in df_book.columns and "away" in df_book.columns:
                    pass
                else:
                    df_book["home"] = out["home"].iloc[0] if not out.empty else None
                    df_book["away"] = out["away"].iloc[0] if not out.empty else None
                _upsert_snapshots(_snap_rows_from_1x2_book_df(df_book))
                print("🧾 upserted 1X2 snapshots (all bookmakers) → Postgres")
            except Exception as e:
                print(f"[snapshots-1x2-allbooks] skip: {e}")
        else:
            _upsert_snapshots(_snap_rows_from_1x2_df(out))
            print("🧾 upserted 1X2 snapshots → Postgres")
    except Exception as e:
        print(f"[snapshots-1x2] skip: {e}")

    if args.include_ou25:
        print(f"ℹ️ OU2.5 rows collected: {0 if not ou_rows else len(ou_rows)}")
    if args.include_ah:
        print(f"ℹ️ AH rows collected: {0 if not ah_rows else len(ah_rows)}")

    if args.include_ou25 and ou_rows:
        ou = pd.DataFrame(ou_rows)
        if "date" in ou.columns:
            ou["date"] = pd.to_datetime(ou["date"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S%z")
        outfile_ou = ODIR / f"apifootball_ou25_{ts}.csv"
        ou_cols = [
            "fixture_id","date","odds_source","fetched_at_utc",
            "home","away","home_norm","away_norm",
            "bookmaker_count","ou_line","ou_over","ou_under",
        ]
        ou = ou[[c for c in ou_cols if c in ou.columns]]
        ou.to_csv(outfile_ou, index=False)
        print(f"✅ saved API-Football OU(2.5) → {outfile_ou} (rows={len(ou)})")
        if args.write_parquet:
            ou_parq = ODIR / f"apifootball_ou25_{ts}.parquet"
            ou.to_parquet(ou_parq, index=False)
            print(f"📦 also saved Parquet → {ou_parq}")
        try:
            _upsert_snapshots(_snap_rows_from_ou_df(ou))
            print("🧾 upserted OU snapshots → Postgres")
        except Exception as e:
            print(f"[snapshots-ou] skip: {e}")

    if args.include_ah and ah_rows:
        ah = pd.DataFrame(ah_rows)
        if "date" in ah.columns:
            ah["date"] = pd.to_datetime(ah["date"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S%z")
        outfile_ah = ODIR / f"apifootball_ah_{ts}.csv"
        ah_cols = [
            "fixture_id","date","odds_source","fetched_at_utc",
            "home","away","home_norm","away_norm",
            "bookmaker_count","ah_line","ah_home","ah_away",
        ]
        ah = ah[[c for c in ah_cols if c in ah.columns]]
        ah.to_csv(outfile_ah, index=False)
        print(f"✅ saved API-Football AH → {outfile_ah} (rows={len(ah)})")
        if args.write_parquet:
            ah_parq = ODIR / f"apifootball_ah_{ts}.parquet"
            ah.to_parquet(ah_parq, index=False)
            print(f"📦 also saved Parquet → {ah_parq}")
        try:
            _upsert_snapshots(_snap_rows_from_ah_df(ah))
            print("🧾 upserted AH snapshots → Postgres")
        except Exception as e:
            print(f"[snapshots-ah] skip: {e}")


if __name__ == "__main__":
    main()
