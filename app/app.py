
import pandas as pd, numpy as np, joblib
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIXDIR = DATA / "fixtures"
FEAT = DATA / "features"
MODELS = ROOT / "models"

st.set_page_config(page_title="Odds Value AI", layout="wide")
st.title("🎯 Odds Value AI — EV & CLV demo")

fix_files = sorted(FIXDIR.glob("*.csv"))
if not fix_files:
    st.error("No fixtures found. Run scripts/fetch_fixtures.py first."); st.stop()
fx = pd.concat([pd.read_csv(f, parse_dates=["date"]) for f in fix_files], ignore_index=True)

featf = FEAT / "train_set.parquet"
if not featf.exists():
    st.warning("No features dataset yet. Build it after fetching odds history."); st.stop()
df = pd.read_parquet(featf)
df = df.dropna(subset=["p_entry_h","p_entry_d","p_entry_a"]).tail(300).copy()

try:
    clf = joblib.load(MODELS/'outcome_logreg.pkl')
    X = df[["p_entry_h","p_entry_d","p_entry_a","home_form5","away_form5"]].fillna(0)
    probs = clf.predict_proba(X)
    df["p_hat_h"], df["p_hat_d"], df["p_hat_a"] = probs[:,0], probs[:,1], probs[:,2]
    for side, colp, colo in [("H","p_hat_h","entry_h"),("D","p_hat_d","entry_d"),("A","p_hat_a","entry_a")]:
        df[f"EV_{side}"] = df[colp] * df[colo] - 1.0
    df["EV_best"] = df[["EV_H","EV_D","EV_A"]].max(axis=1)
    df["Pick"] = df[["EV_H","EV_D","EV_A"]].idxmax(axis=1).str[-1]

    st.sidebar.header("Filters")
    min_ev = st.sidebar.slider("Min EV (%)", 0.0, 10.0, 2.0, 0.5)
    picks = df[df["EV_best"] >= (min_ev/100.0)].sort_values("EV_best", ascending=False)
    st.subheader("Top EV picks (historical sample)")
    st.write(picks[["date","league_name","home","away","entry_h","entry_d","entry_a","p_hat_h","p_hat_d","p_hat_a","Pick","EV_best"]].tail(50))
except Exception as e:
    st.error(f"Model not trained yet or error: {e}")
