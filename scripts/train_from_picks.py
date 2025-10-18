import os
from typing import Dict, List
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

"""
Calibrate/evaluate the model using settled picks.

Joins picks → outcomes → pick_performance and computes per-market metrics:
- accuracy (win rate)
- brier score (binary, uses model_prob for the chosen side vs won)
- log loss (binary)
- counts (n)

Results are logged into training_runs/training_features.
Usage:
  export POSTGRES_URL=postgresql+psycopg://.../railway?sslmode=require
  python scripts/train_from_picks.py
"""

EPS = 1e-9

def _pg_engine():
    url = os.environ.get("POSTGRES_URL")
    if not url:
        raise SystemExit("POSTGRES_URL not set")
    return create_engine(url, pool_pre_ping=True)

def _ensure_runs_tables(e):
    sql = """
    CREATE TABLE IF NOT EXISTS training_runs (
      id bigserial PRIMARY KEY,
      created_at timestamptz NOT NULL,
      model_tag text NOT NULL,
      n_train integer,
      n_test integer,
      n_features integer,
      log_loss double precision,
      accuracy double precision,
      brier double precision,
      artifact_path text
    );
    CREATE TABLE IF NOT EXISTS training_features (
      run_id bigint REFERENCES training_runs(id) ON DELETE CASCADE,
      feature text
    );
    """
    with e.begin() as c:
        c.exec_driver_sql(sql)

def _load_pick_outcomes(e) -> pd.DataFrame:
    # Prefer a view if you already created it
    with e.connect() as c:
        has_view = c.execute(text("""
            SELECT EXISTS (
              SELECT 1
              FROM information_schema.views
              WHERE table_schema='public' AND table_name='v_pick_outcomes'
            );
        """)).scalar()
    if has_view:
        return pd.read_sql("SELECT * FROM v_pick_outcomes", e)

    # Fallback explicit join
    sql = """
    SELECT
      p.pick_id,
      p.created_at,
      p.fixture_id,
      p.league_id,
      p.season,
      p.kick_off,
      p.home,
      p.away,
      p.market,
      p.selection,
      p.model_prob,
      p.best_odds,
      p.ev,
      p.kelly,
      p.stake_units,
      p.model_tag,
      o.result,
      o.goals_h,
      o.goals_a,
      pp.won,
      pp.pnl_units,
      pp.closing_line,
      pp.settled_at
    FROM picks p
    LEFT JOIN outcomes o USING (fixture_id)
    LEFT JOIN pick_performance pp USING (pick_id)
    """
    return pd.read_sql(sql, e)

def _binary_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Compute binary metrics using y=won (1/0) and p=model_prob for the chosen side."""
    d = df.copy()
    d = d[(~d["model_prob"].isna()) & (~d["won"].isna())]
    if d.empty:
        return {"n": 0, "acc": np.nan, "brier": np.nan, "logloss": np.nan}

    y = d["won"].astype(int).to_numpy()
    p = np.clip(pd.to_numeric(d["model_prob"], errors="coerce").to_numpy(), EPS, 1 - EPS)

    acc = float((y == (p >= 0.5)).mean())
    brier = float(np.mean((p - y) ** 2))
    logloss = float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))
    return {"n": int(len(d)), "acc": acc, "brier": brier, "logloss": logloss}

def _insert_run(e, model_tag: str, m: Dict[str, float], used_features: List[str]):
    with e.begin() as c:
        run_id = c.execute(text("""
            INSERT INTO training_runs (
              created_at, model_tag, n_train, n_test, n_features,
              log_loss, accuracy, brier, artifact_path
            ) VALUES (
              now(), :model_tag, 0, :n_test, :n_features,
              :log_loss, :accuracy, :brier, NULL
            ) RETURNING id
        """), {
            "model_tag": model_tag,
            "n_test": int(m.get("n", 0)),
            "n_features": int(len(used_features)),
            "log_loss": float(m.get("logloss")) if m.get("logloss") is not None else np.nan,
            "accuracy": float(m.get("acc")) if m.get("acc") is not None else np.nan,
            "brier": float(m.get("brier")) if m.get("brier") is not None else np.nan,
        }).scalar_one()

        for f in used_features:
            c.execute(text("INSERT INTO training_features (run_id, feature) VALUES (:rid, :f)"),
                      {"rid": run_id, "f": f})

    print(f"[from_picks] logged run {model_tag} → id={run_id}")

def main():
    e = _pg_engine()
    _ensure_runs_tables(e)

    df = _load_pick_outcomes(e)
    if df.empty:
        print("ℹ️ no picks/outcomes to evaluate yet")
        return

    # Only settled picks with usable probabilities
    df = df[(~df["won"].isna()) & (~df["model_prob"].isna())]
    if df.empty:
        print("ℹ️ no settled picks with model_prob available")
        return

    for c in ("model_prob", "best_odds", "kelly", "ev", "stake_units"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    used = ["model_prob", "best_odds"]

    # Overall
    overall = _binary_metrics(df)
    _insert_run(e, model_tag="calib_from_picks_all", m=overall, used_features=used)

    # Per market
    if "market" in df.columns:
        for mkt, dsub in df.groupby("market"):
            metrics = _binary_metrics(dsub)
            tag = f"calib_from_picks_{str(mkt).lower()}"
            _insert_run(e, model_tag=tag, m=metrics, used_features=used)

    # 1X2 per selection (H/D/A)
    if set(["market", "selection"]).issubset(df.columns):
        d1x2 = df[df["market"].astype(str).str.lower().isin(["1x2"])].copy()
        for sel, dsel in d1x2.groupby("selection"):
            metrics = _binary_metrics(dsel)
            tag = f"calib_from_picks_1x2_{str(sel).upper()}"
            _insert_run(e, model_tag=tag, m=metrics, used_features=used)

    print("✅ calibration/eval from settled picks logged to training_runs/training_features")

if __name__ == "__main__":
    main()
