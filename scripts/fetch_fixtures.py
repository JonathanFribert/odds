#!/usr/bin/env python3
import os, sys, time
import pandas as pd
import requests
from datetime import datetime, timezone
from pathlib import Path
from common import FIXDIR, map_league_id_to_sport_key, add_normalized_teams

API_BASE = "https://v3.football.api-sports.io"
API_KEY = os.getenv("APIFOOTBALL_KEY")
HEADERS = {"x-apisports-key": API_KEY}

# Vores fokus-ligaer (API-Football IDs): EPL=39, La Liga=140, Bundesliga=78
LEAGUES = [39, 140, 78]


def _check_key():
    if not API_KEY:
        print("⛔ APIFOOTBALL_KEY not set in environment.")
        sys.exit(1)


def _get(endpoint: str, params: dict):
    """Rate-limit tolerant GET med 429 backoff."""
    for _ in range(3):
        r = requests.get(f"{API_BASE}/{endpoint}", headers=HEADERS, params=params, timeout=30)
        if r.status_code == 429:
            time.sleep(1.2)
            continue
        r.raise_for_status()
        js = r.json()
        return js.get("response", [])
    return []


def _result_from_goals(home_goals, away_goals):
    try:
        h = int(home_goals); a = int(away_goals)
    except Exception:
        return None
    return "H" if h > a else ("A" if a > h else "D")


def fetch_fixtures_for_league(league_id: int, season: int) -> pd.DataFrame:
    league_resp = _get("leagues", {"id": league_id})
    league_name = league_resp[0]["league"]["name"] if league_resp else str(league_id)
    print(f"→ Fetching fixtures: {league_name} ({league_id}) season={season}")

    # Fixtures
    resp = _get("fixtures", {"league": league_id, "season": season})
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
        rows.append({
            "fixture_id": fix.get("id"),
            "league_id": league_id,
            "league_name": league_name,
            "season": season,
            "date": date,
            "home_id": home.get("id"),
            "home": home.get("name"),
            "away_id": away.get("id"),
            "away": away.get("name"),
            "result": res,
        })
    df = pd.DataFrame(rows)

    # sport_key + normaliserede navne til join mod OddsAPI
    df["sport_key"] = df["league_id"].map(map_league_id_to_sport_key)
    df = add_normalized_teams(df, "home", "away")

    # Gem fixtures
    sk = df["sport_key"].dropna().unique()
    if len(sk) == 1:
        out = FIXDIR / f"{sk[0]}_{season}.csv"
    else:
        out = FIXDIR / f"league_{league_id}_{season}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"✅ saved {len(df)} rows → {out}")

    # Standings
    std = _get("standings", {"league": league_id, "season": season})
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
    if std_rows:
        p = out.with_name(out.stem + "_standings.csv")
        pd.DataFrame(std_rows).to_csv(p, index=False)
        print(f"   ➕ standings → {p} ({len(std_rows)})")

    # Team statistics
    ts_rows = []
    teams_df = df[["home_id","home"]].drop_duplicates().rename(columns={"home_id":"team_id","home":"team_name"})
    teams_df = pd.concat([
        teams_df,
        df[["away_id","away"]].drop_duplicates().rename(columns={"away_id":"team_id","away":"team_name"})
    ], ignore_index=True).drop_duplicates("team_id")
    for tid, tname in teams_df.itertuples(index=False):
        resp_stats = _get("teams/statistics", {"league": league_id, "season": season, "team": tid})
        # API can return a list with one item or a dict directly — normalize to dict
        if not resp_stats:
            continue
        if isinstance(resp_stats, list):
            st = resp_stats[0] if len(resp_stats) > 0 else {}
        elif isinstance(resp_stats, dict):
            st = resp_stats
        else:
            st = {}
        if not st:
            continue

        goals = st.get("goals", {}).get("for", {}).get("average", {})
        against = st.get("goals", {}).get("against", {}).get("average", {})
        played = st.get("fixtures", {}).get("played", {}).get("total")
        wins = st.get("fixtures", {}).get("wins", {}).get("total")
        draws = st.get("fixtures", {}).get("draws", {}).get("total")
        losses = st.get("fixtures", {}).get("losses", {}).get("total")
        clean = st.get("clean_sheet", {}).get("total") if "clean_sheet" in st else None
        fts = st.get("failed_to_score", {}).get("total") if "failed_to_score" in st else None
        ts_rows.append({
            "team_id": tid,
            "played_total": played,
            "wins_total": wins,
            "draws_total": draws,
            "losses_total": losses,
            "goals_for_avg": (goals.get("total") if isinstance(goals, dict) else None),
            "goals_against_avg": (against.get("total") if isinstance(against, dict) else None),
            "clean_sheet": clean,
            "failed_to_score": fts,
        })
        time.sleep(0.15)
    if ts_rows:
        p = out.with_name(out.stem + "_teamstats.csv")
        pd.DataFrame(ts_rows).to_csv(p, index=False)
        print(f"   ➕ teamstats → {p} ({len(ts_rows)})")

    # Injuries (seneste 30 dage)
    inj_rows = []
    since = (datetime.now(timezone.utc) - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    injuries = _get("injuries", {"league": league_id, "season": season, "date": since})
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
    if inj_rows:
        p = out.with_name(out.stem + "_injuries_last30.csv")
        pd.DataFrame(inj_rows).to_csv(p, index=False)
        print(f"   ➕ injuries(30d) → {p} ({len(inj_rows)})")

    return df


def main():
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument("--season", type=int, default=datetime.now(timezone.utc).year, help="Season year, e.g. 2025")
    args = ap.parse_args()

    for lg in LEAGUES:
        fetch_fixtures_for_league(lg, args.season)


if __name__ == "__main__":
    _check_key()
    main()