import os
from sqlalchemy import create_engine, text

"""
Normalize picks table in Postgres (idempotent):
  1) Ensure key columns exist
  2) Fill NULLs
  3) Remove duplicates (keep newest per key)
  4) Add UNIQUE constraint + helpful indexes

Usage:
  export POSTGRES_URL=postgresql+psycopg://.../railway?sslmode=require
  python scripts/fix_picks_uq.py
"""

POSTGRES_URL = os.environ.get("POSTGRES_URL")
if not POSTGRES_URL:
    raise SystemExit("POSTGRES_URL not set")

eng = create_engine(POSTGRES_URL, pool_pre_ping=True)

REQUIRED = {
    "fixture_id": "bigint",
    "market": "text",
    "selection": "text",
    "model_tag": "text",
    "created_at": "timestamptz NOT NULL DEFAULT now()",
}

with eng.begin() as conn:
    # 1) Ensure required columns exist
    cols = set(
        r[0]
        for r in conn.execute(
            text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema='public' AND table_name='picks'"
            )
        )
    )

    for col, typ in REQUIRED.items():
        if col not in cols:
            conn.execute(text(f"ALTER TABLE picks ADD COLUMN {col} {typ};"))

    # 2) Fill NULLs for key columns
    conn.execute(text("UPDATE picks SET market = COALESCE(market,'1X2');"))
    conn.execute(text("UPDATE picks SET selection = COALESCE(selection,'H');"))
    conn.execute(text("UPDATE picks SET model_tag = COALESCE(model_tag,'legacy');"))

    # 3) Drop duplicates, keep newest created_at
    conn.execute(
        text(
            """
            WITH ranked AS (
              SELECT ctid, created_at, fixture_id, market, selection, model_tag,
                     ROW_NUMBER() OVER (
                       PARTITION BY fixture_id, market, selection, model_tag
                       ORDER BY created_at DESC NULLS LAST
                     ) AS rn
              FROM picks
            ),
            dups AS (SELECT ctid FROM ranked WHERE rn > 1)
            DELETE FROM picks p USING dups d
            WHERE p.ctid = d.ctid;
            """
        )
    )

    # 4) Add UNIQUE constraint if missing
    exists = conn.execute(
        text(
            """
            SELECT EXISTS (
              SELECT 1 FROM pg_constraint
              WHERE conrelid='public.picks'::regclass AND contype='u' AND conname='uq_picks_key'
            );
            """
        )
    ).scalar()

    if not bool(exists):
        conn.execute(
            text(
                "ALTER TABLE picks ADD CONSTRAINT uq_picks_key "
                "UNIQUE (fixture_id, market, selection, model_tag);"
            )
        )

    # 5) Helpful indexes
    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_picks_created ON picks(created_at);"))
    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_picks_fixture ON picks(fixture_id);"))

print("✅ FIX: picks has unique key + indexes, and duplicates removed.")
