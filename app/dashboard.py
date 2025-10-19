import os
from datetime import datetime, timedelta
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine

# -------------------------
# Utilities
# -------------------------

def normalize_pg_url(url: str) -> str:
    if not url:
        return url
    u = url.strip().strip('"').strip("'")
    low = u.lower()
    if low.startswith("postgres://"):
        u = "postgresql+psycopg://" + u[len("postgres://"):]
    elif low.startswith("postgresql://") and "+psycopg" not in low:
        u = "postgresql+psycopg://" + u[len("postgresql://"):]
    return u

@st.cache_resource(show_spinner=False)
def get_engine():
    url = normalize_pg_url(os.getenv("POSTGRES_URL", ""))
    if not url:
        st.stop()
    return create_engine(url, pool_pre_ping=True)

@st.cache_data(ttl=60, show_spinner=False)
def load_data(days_back: int = 120):
    e = get_engine()

    # Join of picks + outcomes + performance (settled)
    sql = f"""
    SELECT
      p.pick_id, p.created_at, p.fixture_id, p.league_id, p.season, p.kick_off,
      p.home, p.away, p.market, p.selection, p.model_prob, p.best_odds, p.ev, p.kelly, p.stake_units, p.model_tag,
      o.result, o.goals_h, o.goals_a, o.closing_odds_h, o.closing_odds_d, o.closing_odds_a,
      pp.won, pp.pnl_units, pp.closing_line, pp.settled_at
    FROM picks p
    LEFT JOIN outcomes o USING (fixture_id)
    LEFT JOIN pick_performance pp USING (pick_id)
    WHERE pp.settled_at > now() - interval '{days_back} days'
    ORDER BY pp.settled_at DESC
    """
    perf = pd.read_sql(sql, e)

    # Training runs / calibration snapshots
    runs = pd.read_sql(
        """
        SELECT created_at, model_tag, n_test, accuracy, brier, log_loss
        FROM training_runs
        ORDER BY created_at DESC
        LIMIT 1000
        """,
        e,
    )

    return perf, runs

# -------------------------
# App
# -------------------------

st.set_page_config(page_title="Odds – Model Dashboard", layout="wide")

st.title("📊 Odds – Model Performance Dashboard")

with st.sidebar:
    st.subheader("Filters")
    days_back = st.slider("Days back", 14, 365, 120, step=7)
    default_markets = ["1x2", "ou25", "ahm05", "ahc", "totals"]
    market_filter = st.multiselect(
        "Markets",
        options=default_markets,
        default=["1x2", "ou25", "ahc", "totals"],
        help="Filter charts/tables to these markets (case-insensitive contains).",
    )
    model_tag_like = st.text_input("Model tag contains", "", help="Filter by substring in model_tag")

perf, runs = load_data(days_back)

if perf.empty:
    st.info("No settled picks in the selected window yet.")
    st.stop()

# Normalize types
for c in ("settled_at", "created_at", "kick_off"):
    if c in perf.columns:
        perf[c] = pd.to_datetime(perf[c], utc=True, errors="coerce")

# Apply market + model_tag filters
if market_filter:
    patt = tuple(x.lower() for x in market_filter)
    perf = perf[perf["market"].str.lower().str.contains("|".join(patt), na=False)]

if model_tag_like:
    perf = perf[perf["model_tag"].astype(str).str.contains(model_tag_like, case=False, na=False)]

# -------------------------
# KPI cards
# -------------------------

latest = perf[perf["settled_at"].notna()].copy()

last_30 = latest[latest["settled_at"] >= (latest["settled_at"].max() - pd.Timedelta(days=30))]

pnl_total = latest["pnl_units"].fillna(0).sum()
pnl_30d = last_30["pnl_units"].fillna(0).sum() if not last_30.empty else 0.0

hit = float(latest["won"].mean()) if latest["won"].notna().any() else np.nan
hit_30d = float(last_30["won"].mean()) if not last_30.empty and last_30["won"].notna().any() else np.nan

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total PnL (units)", f"{pnl_total:,.2f}")
col2.metric("PnL last 30d", f"{pnl_30d:,.2f}")
col3.metric("Hit rate (lifetime)", f"{hit*100:,.1f}%" if not np.isnan(hit) else "–")
col4.metric("Hit rate (30d)", f"{hit_30d*100:,.1f}%" if not np.isnan(hit_30d) else "–")

# -------------------------
# PnL over time
# -------------------------

daily = (
    latest.assign(day=lambda d: pd.to_datetime(d["settled_at"]).dt.date)
    .groupby("day", as_index=False)["pnl_units"].sum()
)
fig_pnl = px.bar(daily, x="day", y="pnl_units", title="PnL per day (units)")
st.plotly_chart(fig_pnl, use_container_width=True)

# -------------------------
# CLV (Closing Line Value)
# -------------------------

def pick_closing_for_row(r):
    m = str(r.get("market", "")).lower()
    sel = r.get("selection")
    if m in ("1x2", "1x2 "):
        if sel == "H":
            return r.get("closing_odds_h")
        if sel == "D":
            return r.get("closing_odds_d")
        if sel == "A":
            return r.get("closing_odds_a")
    # fallback
    return r.get("closing_line")

cl = latest.copy()
cl["closing_for_pick"] = cl.apply(pick_closing_for_row, axis=1)
clv_df = cl[["best_odds", "closing_for_pick"]].apply(pd.to_numeric, errors="coerce")
clv_df["clv_pct"] = (clv_df["best_odds"] / clv_df["closing_for_pick"] - 1.0) * 100.0
share_beating_close = float((clv_df["best_odds"] > clv_df["closing_for_pick"]).mean()) if clv_df.notna().all(axis=1).any() else np.nan

st.subheader("CLV – Closing Line Value")
c1, c2 = st.columns(2)
c1.metric("% beating closing line", f"{share_beating_close*100:,.1f}%" if not np.isnan(share_beating_close) else "–")
fig_clv = px.histogram(clv_df["clv_pct"].dropna(), nbins=40, title="Distribution of (entry/closing - 1) %")
c2.plotly_chart(fig_clv, use_container_width=True)

# -------------------------
# By league / market breakdowns
# -------------------------

st.subheader("Breakdowns")
left, right = st.columns(2)

if "league_id" in latest.columns:
    pnl_league = latest.groupby("league_id", as_index=False)["pnl_units"].sum()
    left.plotly_chart(px.bar(pnl_league, x="league_id", y="pnl_units", title="PnL by league"), use_container_width=True)

pnl_market = latest.groupby("market", as_index=False)["pnl_units"].sum()
right.plotly_chart(px.bar(pnl_market, x="market", y="pnl_units", title="PnL by market"), use_container_width=True)

# -------------------------
# Calibration snapshots (training_runs)
# -------------------------

st.subheader("Calibration – training_runs")
if not runs.empty:
    runs = runs.sort_values("created_at")
    a, b, c = st.columns(3)
    a.line_chart(runs.set_index("created_at")["accuracy"], height=160)
    b.line_chart(runs.set_index("created_at")["brier"], height=160)
    c.line_chart(runs.set_index("created_at")["log_loss"], height=160)
else:
    st.caption("No rows in training_runs yet.")

# -------------------------
# Latest settled picks table
# -------------------------

st.subheader("Latest settled picks")
show_cols = [
    "settled_at", "league_id", "market", "selection", "model_prob", "best_odds", "ev", "kelly",
    "pnl_units", "won", "home", "away", "closing_for_pick"
]

have = [c for c in show_cols if c in cl.columns]
st.dataframe(cl[have].head(300))

st.caption("Data source: Railway Postgres – picks, outcomes, pick_performance, training_runs.")
