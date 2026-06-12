"""
tools/db.py — PostgreSQL interface for the Pharma MMM pipeline.

Schema: mmm (in portfolio_db)
  mmm.results   — one JSONB row per (freq, result_type); upserted on each run
  mmm.raw_data  — national raw/transformed CSV rows as JSONB
  mmm.geo_data  — geo CSV rows as JSONB (freq, date, territory)
  mmm.narratives — AI narrative reports (freq, report_type)
  mmm.run_log   — one row per pipeline stage execution

result_type values:
  ols | bayesian | budget_optimized |
  geo_ols | geo_bayesian | geo_hierarchical | geo_budget_optimized

All public functions return None (not raise) on connection failure so the app
gracefully falls back to JSON/CSV files.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from typing import Optional

import psycopg2
import psycopg2.extras
import pandas as pd

DSN = os.environ.get(
    "MMM_DB_URL",
    "postgresql://shubham:localdevpass@localhost:5432/portfolio_db",
)

_SCHEMA_SQL = """
CREATE SCHEMA IF NOT EXISTS mmm;

CREATE TABLE IF NOT EXISTS mmm.results (
    id          SERIAL PRIMARY KEY,
    freq        TEXT        NOT NULL,
    result_type TEXT        NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    data        JSONB       NOT NULL,
    UNIQUE (freq, result_type)
);

CREATE TABLE IF NOT EXISTS mmm.raw_data (
    id       SERIAL PRIMARY KEY,
    freq     TEXT  NOT NULL,
    date     DATE  NOT NULL,
    data     JSONB NOT NULL,
    UNIQUE (freq, date)
);

CREATE TABLE IF NOT EXISTS mmm.geo_data (
    id        SERIAL PRIMARY KEY,
    freq      TEXT  NOT NULL,
    date      DATE  NOT NULL,
    territory TEXT  NOT NULL,
    data      JSONB NOT NULL,
    UNIQUE (freq, date, territory)
);

CREATE TABLE IF NOT EXISTS mmm.narratives (
    id          SERIAL PRIMARY KEY,
    freq        TEXT        NOT NULL,
    report_type TEXT        NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    content     TEXT        NOT NULL,
    UNIQUE (freq, report_type)
);

CREATE TABLE IF NOT EXISTS mmm.run_log (
    id      SERIAL PRIMARY KEY,
    run_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    freq    TEXT,
    stage   TEXT,
    status  TEXT,
    notes   TEXT
);
"""


@contextmanager
def _conn():
    conn = psycopg2.connect(DSN)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_schema() -> bool:
    """Create mmm schema and all tables if they don't exist. Returns True on success."""
    try:
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute(_SCHEMA_SQL)
        return True
    except Exception as e:
        print(f"[db] init_schema failed: {e}")
        return False


# ── Results (OLS / Bayesian / optimizer) ─────────────────────────────────────

def upsert_result(freq: str, result_type: str, data: dict) -> bool:
    """Store or overwrite a pipeline result. Returns True on success."""
    try:
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO mmm.results (freq, result_type, data)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (freq, result_type) DO UPDATE
                      SET data = EXCLUDED.data,
                          created_at = NOW()
                    """,
                    (freq, result_type, json.dumps(data)),
                )
        return True
    except Exception as e:
        print(f"[db] upsert_result({freq}, {result_type}) failed: {e}")
        return False


def load_result(freq: str, result_type: str) -> Optional[dict]:
    """Load a pipeline result. Returns None if not found or on error."""
    try:
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT data FROM mmm.results WHERE freq = %s AND result_type = %s",
                    (freq, result_type),
                )
                row = cur.fetchone()
                return row[0] if row else None
    except Exception as e:
        print(f"[db] load_result({freq}, {result_type}) failed: {e}")
        return None


# ── Raw data ──────────────────────────────────────────────────────────────────

def _clean_for_json(d: dict) -> dict:
    """Replace NaN/inf with None so json.dumps produces valid JSON."""
    import math
    return {
        k: (None if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) else v)
        for k, v in d.items()
    }

def upsert_raw_data(freq: str, df: pd.DataFrame) -> bool:
    """Upsert all rows of a raw/transformed DataFrame. Returns True on success."""
    try:
        records = []
        for _, row in df.iterrows():
            d = row.to_dict()
            if hasattr(d.get("date"), "isoformat"):
                d["date"] = d["date"].isoformat()
            records.append((freq, str(d.get("date", "")), json.dumps(_clean_for_json(d))))

        with _conn() as conn:
            with conn.cursor() as cur:
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO mmm.raw_data (freq, date, data)
                    VALUES %s
                    ON CONFLICT (freq, date) DO UPDATE
                      SET data = EXCLUDED.data
                    """,
                    records,
                )
        return True
    except Exception as e:
        print(f"[db] upsert_raw_data({freq}) failed: {e}")
        return False


def load_raw_data(freq: str) -> Optional[pd.DataFrame]:
    """Load raw data rows for a freq. Returns DataFrame or None."""
    try:
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT data FROM mmm.raw_data WHERE freq = %s ORDER BY date",
                    (freq,),
                )
                rows = cur.fetchall()
        if not rows:
            return None
        records = [r[0] for r in rows]
        df = pd.DataFrame(records)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception as e:
        print(f"[db] load_raw_data({freq}) failed: {e}")
        return None


# ── Geo data ──────────────────────────────────────────────────────────────────

def upsert_geo_data(freq: str, df: pd.DataFrame) -> bool:
    """Upsert all rows of a geo DataFrame (must have 'territory' column)."""
    try:
        records = []
        for _, row in df.iterrows():
            d = row.to_dict()
            if hasattr(d.get("date"), "isoformat"):
                d["date"] = d["date"].isoformat()
            records.append((
                freq,
                str(d.get("date", "")),
                str(d.get("territory", "")),
                json.dumps(_clean_for_json(d)),
            ))

        with _conn() as conn:
            with conn.cursor() as cur:
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO mmm.geo_data (freq, date, territory, data)
                    VALUES %s
                    ON CONFLICT (freq, date, territory) DO UPDATE
                      SET data = EXCLUDED.data
                    """,
                    records,
                )
        return True
    except Exception as e:
        print(f"[db] upsert_geo_data({freq}) failed: {e}")
        return False


def load_geo_data(freq: str) -> Optional[pd.DataFrame]:
    """Load geo data rows for a freq. Returns DataFrame or None."""
    try:
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT data FROM mmm.geo_data WHERE freq = %s ORDER BY date, territory",
                    (freq,),
                )
                rows = cur.fetchall()
        if not rows:
            return None
        df = pd.DataFrame([r[0] for r in rows])
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception as e:
        print(f"[db] load_geo_data({freq}) failed: {e}")
        return None


# ── Narratives ────────────────────────────────────────────────────────────────

def upsert_narrative(freq: str, report_type: str, content: str) -> bool:
    """Store or overwrite an AI narrative. Returns True on success."""
    try:
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO mmm.narratives (freq, report_type, content)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (freq, report_type) DO UPDATE
                      SET content = EXCLUDED.content,
                          created_at = NOW()
                    """,
                    (freq, report_type, content),
                )
        return True
    except Exception as e:
        print(f"[db] upsert_narrative({freq}, {report_type}) failed: {e}")
        return False


def load_narrative(freq: str, report_type: str) -> Optional[str]:
    """Load a narrative. Returns string or None."""
    try:
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT content FROM mmm.narratives WHERE freq = %s AND report_type = %s",
                    (freq, report_type),
                )
                row = cur.fetchone()
                return row[0] if row else None
    except Exception as e:
        print(f"[db] load_narrative({freq}, {report_type}) failed: {e}")
        return None


# ── Run log ───────────────────────────────────────────────────────────────────

def log_run(freq: str, stage: str, status: str, notes: str = "") -> None:
    """Append a pipeline run log entry. Silently ignores errors."""
    try:
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO mmm.run_log (freq, stage, status, notes) VALUES (%s, %s, %s, %s)",
                    (freq, stage, status, notes),
                )
    except Exception:
        pass  # run_log is best-effort
