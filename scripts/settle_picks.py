#!/usr/bin/env python3
import os, pandas as pd, numpy as np
from sqlalchemy import create_engine
from datetime import datetime, timezone

def _pg():
    url = os.getenv("POSTGRES_URL")
    if not url: raise SystemExit("POSTGRES_URL not set")
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg://", 1)
    elif url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return create_engine(url, pool_pre_ping=True)

def load_outcomes_from_parquet():
    p = "data/features/postmatch_flat.parquet"
    if not os.path.exists(p): return pd.DataFrame()
    df = pd.read_parquet(p)
    # transformer til outcomes schema
    df = df.rename(columns={
        "goals_h":"goals_h","goals_a":"goals_a",
        "closing_odds_h":"closing_odds_h",
        "closing_odds_d":"closing_odds_d",
        "closing_odds_a":"closing_odds_a",
        "home_expected_goals":"home_expected_goals",
        "away_expected_goals":"away_expected_goals",
        "home_shots_on_goal":"home_shots_on_goal",
        "away_shots_on_goal":"away_shots_on_goal",
        "home_ball_possession":"home_ball_possession",
        "away_ball_possession":"away_ball_possession",
    })
    # resultat (H/D/A)
    def res(r):
        try:
            h,a = int(r["goals_h"]), int(r["goals_a"])
            return "H" if h>a else ("A" if a>h else "D")
        except Exception:
            return None
    df["result"] = df.apply(res, axis=1)
    return df[[
        "fixture_id","league_id","season","date","result",
        "goals_h","goals_a",
        "closing_odds_h","closing_odds_d","closing_odds_a",
        "home_expected_goals","away_expected_goals",
        "home_shots_on_goal","away_shots_on_goal",
        "home_ball_possession","away_ball_possession",
    ]].rename(columns={"date":"kick_off"})

def upsert_outcomes(df, eng):
    if df.empty: return 0
    with eng.begin() as conn:
        tmp = "_outcomes_tmp"
        df.to_sql(tmp, conn, if_exists="replace", index=False)
        conn.exec_driver_sql("""
        insert into outcomes as o (
          fixture_id, league_id, season, kick_off, result,
          goals_h, goals_a,
          closing_odds_h, closing_odds_d, closing_odds_a,
          home_expected_goals, away_expected_goals,
          home_shots_on_goal, away_shots_on_goal,
          home_ball_possession, away_ball_possession
        )
        select * from _outcomes_tmp
        on conflict (fixture_id) do update
        set league_id=excluded.league_id,
            season=excluded.season,
            kick_off=excluded.kick_off,
            result=excluded.result,
            goals_h=excluded.goals_h,
            goals_a=excluded.goals_a,
            closing_odds_h=excluded.closing_odds_h,
            closing_odds_d=excluded.closing_odds_d,
            closing_odds_a=excluded.closing_odds_a,
            home_expected_goals=excluded.home_expected_goals,
            away_expected_goals=excluded.away_expected_goals,
            home_shots_on_goal=excluded.home_shots_on_goal,
            away_shots_on_goal=excluded.away_shots_on_goal,
            home_ball_possession=excluded.home_ball_possession,
            away_ball_possession=excluded.away_ball_possession,
            updated_at=now();
        drop table if exists _outcomes_tmp;
        """)
    return len(df)

def settle(eng):
    with eng.begin() as conn:
        picks = pd.read_sql("""select p.*, o.result,
          case p.selection
            when 'H' then o.closing_odds_h
            when 'D' then o.closing_odds_d
            when 'A' then o.closing_odds_a
          end as closing_line
          from picks p
          join outcomes o using (fixture_id)
          where p.pick_id not in (select pick_id from pick_performance)
            and o.result is not null
        """, conn)
        if picks.empty: return 0
        def pnl(row):
            if row["result"] == row["selection"]:
                return (row["closing_line"] - 1.0) * (row["stake_units"] or 1.0)
            else:
                return -(row["stake_units"] or 1.0)
        perf = picks.assign(
            won = picks["result"] == picks["selection"],
            pnl_units = picks.apply(pnl, axis=1),
            settled_at = datetime.now(timezone.utc)
        )[["pick_id","fixture_id","settled_at","won","pnl_units","closing_line"]]
        perf.to_sql("pick_performance", conn, if_exists="append", index=False)
        return len(perf)

if __name__ == "__main__":
    eng = _pg()
    new = upsert_outcomes(load_outcomes_from_parquet(), eng)
    closed = settle(eng)
    print(f"✅ outcomes upserted: {new}, picks settled: {closed}")