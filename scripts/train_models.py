#!/usr/bin/env python3
"""
Train models from data/features/train_set.parquet and write artifacts to models/.
- Multiclass (1X2) Logistic Regression for match outcome using available numeric features
- Binary models (optional) for:
    * OU 2.5 (label_ou25)
    * AH Home -0.5 (label_ah_home_m0_5)
Includes robust feature selection, scaling, clipping, and clear logging/metrics.
"""

from __future__ import annotations
import os, json, warnings
from pathlib import Path
from datetime import datetime
from typing import List, Tuple
import sqlite3

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score, brier_score_loss
from sklearn.model_selection import train_test_split
import joblib

# Optional: Postgres logging (if POSTGRES_URL is set)
try:
    from sqlalchemy import create_engine, text  # type: ignore
except Exception:  # pragma: no cover
    create_engine = None  # type: ignore
    text = None  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
FEAT_DIR = ROOT / "data" / "features"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# SQLite training DB
DB_DIR = ROOT / "data" / "db"
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / "odds_ai.db"

# -------------------- Utilities --------------------

def _now_iso():
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

# Columns we must not learn on (IDs/strings/targets/leakage)
IGNORE_COLS = {
    # identifiers
    "fixture_id","league_id","season","home_id","away_id",
    # strings / meta
    "league_name","sport_key","home","away","home_norm","away_norm",
    "status_short","status_long",
    # direct targets or labels
    "result","label_ou25","label_ah_home_m0_5",
}

# Additional prefixes that are unsafe (close odds etc.). Adjust as needed.
IGNORE_PREFIXES = (
    "p_close_",  # any post-match/closing probabilities
    # add more here if you later store close odds explicitly
)

# Odds columns we ALLOW (entry odds) — helpful signal, not considered leakage
ALLOW_ODDS_COLS = {"odds_h","odds_d","odds_a","p_entry_h","p_entry_d","p_entry_a","overround"}


def _candidate_numeric_columns(df: pd.DataFrame) -> List[str]:
    """Select usable numeric feature columns.
    - Excludes ID/label/meta columns
    - Excludes columns with unsafe prefixes (e.g., closing odds)
    - **Drops columns that are entirely NaN** to avoid imputer warnings
    - Ensures entry-odds columns (if numeric) are kept
    """
    num_cols = df.select_dtypes(include=[np.number, bool]).columns.tolist()

    cols: list[str] = []
    dropped_all_nan: list[str] = []

    for c in num_cols:
        if c in IGNORE_COLS:
            continue
        if any(c.startswith(pref) for pref in IGNORE_PREFIXES):
            continue
        s = df[c]
        if not s.notna().any():  # entirely NaN
            dropped_all_nan.append(c)
            continue
        cols.append(c)

    # ensure odds columns (if present as numeric) are included
    for c in ALLOW_ODDS_COLS:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            if c not in cols:
                cols.append(c)

    if dropped_all_nan:
        print(f"[feat] dropping {len(dropped_all_nan)} all-NaN columns (e.g., {dropped_all_nan[:5]})")
    return cols


def _clip_extremes(X: pd.DataFrame, zmax: float = 8.0) -> pd.DataFrame:
    """Clip extreme values column-wise to +/- zmax standard deviations to avoid overflow."""
    Xc = X.copy()
    # compute mean/std ignoring NaNs
    mu = Xc.mean(axis=0, skipna=True)
    sd = Xc.std(axis=0, skipna=True).replace(0, np.nan)
    upper = mu + zmax * sd
    lower = mu - zmax * sd
    # where sd is NaN (constant col), skip clipping
    for c in Xc.columns:
        hi = upper.get(c, np.nan)
        lo = lower.get(c, np.nan)
        if not np.isnan(hi) and not np.isnan(lo):
            Xc[c] = Xc[c].clip(lower=lo, upper=hi)
    return Xc

def _ensure_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS training_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            model_tag TEXT NOT NULL,
            n_train INTEGER,
            n_test INTEGER,
            n_features INTEGER,
            log_loss REAL,
            accuracy REAL,
            brier REAL,
            artifact_path TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS training_features (
            run_id INTEGER,
            feature TEXT,
            FOREIGN KEY(run_id) REFERENCES training_runs(id)
        )
        """
    )
    con.commit()
    con.close()


# -------------------- Optional Postgres logging --------------------
def _pg_engine():
    """Return a SQLAlchemy engine if POSTGRES_URL is set and sqlalchemy is available; else None."""
    url = os.environ.get("POSTGRES_URL")
    if not url or create_engine is None:
        return None
    try:
        eng = create_engine(url, pool_pre_ping=True)  # type: ignore
        return eng
    except Exception:
        return None

def _ensure_pg_tables():
    eng = _pg_engine()
    if eng is None:
        return
    with eng.begin() as c:
        c.execute(text("""
        CREATE TABLE IF NOT EXISTS training_runs (
            id bigserial PRIMARY KEY,
            created_at timestamptz NOT NULL,
            model_tag text NOT NULL,
            n_train int,
            n_test int,
            n_features int,
            log_loss double precision,
            accuracy double precision,
            brier double precision,
            artifact_path text
        );
        """))
        c.execute(text("""
        CREATE TABLE IF NOT EXISTS training_features (
            run_id bigint REFERENCES training_runs(id) ON DELETE CASCADE,
            feature text
        );
        """))

def _log_training_pg(model_tag: str, metrics: dict, artifact_path: Path, features: list[str]):
    """Mirror of _log_training but writes to Postgres if POSTGRES_URL is configured."""
    eng = _pg_engine()
    if eng is None:
        return
    _ensure_pg_tables()
    with eng.begin() as c:
        run = c.execute(text("""
            INSERT INTO training_runs (created_at, model_tag, n_train, n_test, n_features, log_loss, accuracy, brier, artifact_path)
            VALUES (:created_at, :model_tag, :n_train, :n_test, :n_features, :log_loss, :accuracy, :brier, :artifact_path)
            RETURNING id
        """), {
            "created_at": metrics.get("created_at"),
            "model_tag": model_tag,
            "n_train": metrics.get("n_train"),
            "n_test": metrics.get("n_test"),
            "n_features": metrics.get("n_features"),
            "log_loss": metrics.get("log_loss"),
            "accuracy": metrics.get("accuracy"),
            "brier": metrics.get("brier"),
            "artifact_path": str(artifact_path),
        }).scalar_one()
        if features:
            c.execute(text("""
                INSERT INTO training_features (run_id, feature)
                VALUES """ + ",".join(["(:rid, :f"+str(i)+")" for i in range(len(features))])
            ), dict({"rid": run}, **{("f"+str(i)): f for i, f in enumerate(features)}))


def _log_training(model_tag: str, metrics: dict, artifact_path: Path, features: list[str]):
    _ensure_db()
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        """
        INSERT INTO training_runs(created_at, model_tag, n_train, n_test, n_features, log_loss, accuracy, brier, artifact_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            metrics.get("created_at"),
            model_tag,
            metrics.get("n_train"),
            metrics.get("n_test"),
            metrics.get("n_features"),
            metrics.get("log_loss"),
            metrics.get("accuracy"),
            metrics.get("brier"),
            str(artifact_path),
        ),
    )
    run_id = cur.lastrowid
    # store feature list
    for f in (features or []):
        cur.execute("INSERT INTO training_features(run_id, feature) VALUES (?, ?)", (run_id, f))
    con.commit()
    con.close()

def _train_multiclass_1x2(df: pd.DataFrame, model_path: Path, metrics_path: Path) -> bool:
    if "result" not in df.columns:
        print("[1x2] ❌ 'result' column not found → skip")
        return False
    y_map = {"H":0, "D":1, "A":2}
    y = df["result"].map(y_map)
    mask = y.notna()

    X_cols = _candidate_numeric_columns(df)
    X = df.loc[mask, X_cols].copy()
    y = y.loc[mask].astype(int)

    # drop zero-variance columns (constant after clipping/imputation risk)
    nunique = X.nunique(dropna=False)
    zero_var = nunique[nunique <= 1].index.tolist()
    if zero_var:
        X = X.drop(columns=zero_var)
        X_cols = [c for c in X_cols if c not in zero_var]
        print(f"[1x2] dropping {len(zero_var)} zero-variance cols (e.g., {zero_var[:5]})")

    if len(X) < 100:
        print(f"[1x2] ❌ Not enough rows to train (n={len(X)}) → skip")
        return False

    # clip extremes to avoid overflow in solver
    X = _clip_extremes(X)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(max_iter=2000, multi_class="multinomial", class_weight="balanced")),
    ])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(X_tr, y_tr)

    proba_te = pipe.predict_proba(X_te)
    y_pred = proba_te.argmax(axis=1)

    metrics = {
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "n_features": len(X_cols),
        "features": X_cols,
        "log_loss": float(log_loss(y_te, proba_te)),
        "accuracy": float(accuracy_score(y_te, y_pred)),
        "created_at": _now_iso(),
        "model": "LogisticRegression(multinomial)",
    }

    joblib.dump({"model": pipe, "features": X_cols, "target": "result"}, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    # log to sqlite and export feature list
    feats_csv = model_path.with_suffix(".features.csv")
    pd.Series(X_cols, name="feature").to_csv(feats_csv, index=False)
    _log_training("1x2", metrics, model_path, X_cols)
    _log_training_pg("1x2", metrics, model_path, X_cols)
    print(f"[1x2] ✅ saved model → {model_path}")
    print(f"[1x2] 📊 metrics → {metrics_path}")
    return True

def _train_binary(df: pd.DataFrame, label_col: str, tag: str, model_path: Path, metrics_path: Path) -> bool:
    if label_col not in df.columns:
        print(f"[{tag}] ❌ '{label_col}' not found → skip")
        return False

    y = pd.to_numeric(df[label_col], errors="coerce")
    mask = y.isin([0,1])

    X_cols = _candidate_numeric_columns(df)
    X = df.loc[mask, X_cols].copy()
    y = y.loc[mask].astype(int)

    nunique = X.nunique(dropna=False)
    zero_var = nunique[nunique <= 1].index.tolist()
    if zero_var:
        X = X.drop(columns=zero_var)
        X_cols = [c for c in X_cols if c not in zero_var]
        print(f"[{tag}] dropping {len(zero_var)} zero-variance cols (e.g., {zero_var[:5]})")

    if len(X) < 100:
        print(f"[{tag}] ❌ Not enough rows to train (n={len(X)}) → skip")
        return False

    X = _clip_extremes(X)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(X_tr, y_tr)

    proba_te = pipe.predict_proba(X_te)[:,1]
    y_pred = (proba_te >= 0.5).astype(int)

    # build a 2-col probability for log_loss
    proba_mat = np.vstack([1-proba_te, proba_te]).T

    try:
        brier = float(brier_score_loss(y_te, proba_te))
    except Exception:
        brier = float("nan")

    metrics = {
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "n_features": len(X_cols),
        "features": X_cols,
        "log_loss": float(log_loss(y_te, proba_mat)),
        "brier": brier,
        "accuracy": float(accuracy_score(y_te, y_pred)),
        "created_at": _now_iso(),
        "model": "LogisticRegression(binary)",
        "target": label_col,
    }

    joblib.dump({"model": pipe, "features": X_cols, "target": label_col}, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    feats_csv = model_path.with_suffix(".features.csv")
    pd.Series(X_cols, name="feature").to_csv(feats_csv, index=False)
    _log_training(tag, metrics, model_path, X_cols)
    _log_training_pg(tag, metrics, model_path, X_cols)
    print(f"[{tag}] ✅ saved model → {model_path}")
    print(f"[{tag}] 📊 metrics → {metrics_path}")
    return True


# -------------------- Main --------------------

def main():
    print("🔎 loading:", FEAT_DIR / "train_set.parquet")
    _ensure_db()
    p = FEAT_DIR / "train_set.parquet"
    if not p.exists():
        print("❌ train_set.parquet not found. Run build_features.py first.")
        return

    df = pd.read_parquet(p)
    print(f"✅ train_set loaded: shape={df.shape}")

    pm = pd.read_parquet("data/features/postmatch_flat.parquet") if os.path.exists("data/features/postmatch_flat.parquet") else None
    if pm is not None and not pm.empty:
        keep = ["fixture_id","home_expected_goals","away_expected_goals",
                "home_shots_on_goal","away_shots_on_goal",
                "home_ball_possession","away_ball_possession"]
        pm = pm[keep].drop_duplicates("fixture_id")
        df = df.merge(pm, on="fixture_id", how="left")

    # Ensure date is parsed (useful for future time-based splits)
    if "date" in df.columns:
        try:
            df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        except Exception:
            pass

    # 1X2 model
    _train_multiclass_1x2(df, MODEL_DIR / "model_1x2.pkl", MODEL_DIR / "metrics_1x2.json")

    # OU 2.5 model (if label exists)
    _train_binary(df, "label_ou25", "ou25", MODEL_DIR / "model_ou25.pkl", MODEL_DIR / "metrics_ou25.json")

    # AH Home -0.5 model (if label exists)
    _train_binary(df, "label_ah_home_m0_5", "ah_home_m0_5", MODEL_DIR / "model_ah_home_m0_5.pkl", MODEL_DIR / "metrics_ah_home_m0_5.json")


if __name__ == "__main__":
    main()
