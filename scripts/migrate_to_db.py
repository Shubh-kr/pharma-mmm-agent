"""
scripts/migrate_to_db.py
========================
One-time migration of all existing pipeline outputs into PostgreSQL.

Run from the project root:
    python scripts/migrate_to_db.py

What it migrates:
  - Raw CSVs (weekly + monthly, national + geo) → mmm.raw_data / mmm.geo_data
  - All JSON result files → mmm.results
  - Existing AI narrative reports → mmm.narratives

Safe to re-run — all upserts are idempotent (ON CONFLICT DO UPDATE).
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from tools.db import (
    init_schema, upsert_raw_data, upsert_geo_data,
    upsert_result, upsert_narrative, log_run,
)

DATA_DIR    = "data/raw"
REPORTS_DIR = "reports"

# Maps result_type → filename suffix
RESULT_FILES = {
    "ols":                "_ols_results.json",
    "bayesian":           "_bayesian_results.json",
    "budget_optimized":   "_budget_optimized.json",
    "geo_ols":            "_geo_ols_results.json",
    "geo_bayesian":       "_geo_bayesian_results.json",
    "geo_hierarchical":   "_geo_hierarchical_results.json",
    "geo_budget_optimized": "_geo_budget_optimized.json",
}

NARRATIVE_FILES = {
    "national": "_insights.md",
    "geo":      "_geo_insights.md",
}


def _ok(label: str, success: bool) -> None:
    status = "✓" if success else "✗ FAILED"
    print(f"  {status}  {label}")


def migrate_raw_data(freq: str) -> None:
    path = os.path.join(DATA_DIR, f"mmm_{freq}.csv")
    if not os.path.exists(path):
        print(f"  —  {path} not found, skipping")
        return
    df = pd.read_csv(path, parse_dates=["date"])
    ok = upsert_raw_data(freq, df)
    _ok(f"raw_data [{freq}] — {len(df)} rows", ok)


def migrate_geo_data(freq: str) -> None:
    path = os.path.join(DATA_DIR, f"mmm_{freq}_geo.csv")
    if not os.path.exists(path):
        print(f"  —  {path} not found, skipping")
        return
    df = pd.read_csv(path, parse_dates=["date"])
    ok = upsert_geo_data(freq, df)
    _ok(f"geo_data [{freq}] — {len(df)} rows", ok)


def migrate_results(freq: str) -> None:
    for result_type, suffix in RESULT_FILES.items():
        path = os.path.join(DATA_DIR, f"mmm_{freq}{suffix}")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        ok = upsert_result(freq, result_type, data)
        _ok(f"results [{freq}/{result_type}]", ok)


def migrate_narratives(freq: str) -> None:
    for report_type, suffix in NARRATIVE_FILES.items():
        path = os.path.join(REPORTS_DIR, f"mmm_{freq}{suffix}")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            content = f.read()
        ok = upsert_narrative(freq, report_type, content)
        _ok(f"narrative [{freq}/{report_type}]", ok)


def main():
    print("=" * 55)
    print("Pharma MMM → PostgreSQL migration")
    print("Target: postgresql://…/portfolio_db  schema: mmm")
    print("=" * 55)

    print("\n[1] Initialising schema…")
    ok = init_schema()
    if not ok:
        print("  ✗ Schema init failed — check DB connection and permissions.")
        sys.exit(1)
    print("  ✓ mmm schema ready")

    for freq in ("weekly", "monthly"):
        print(f"\n[{freq.upper()}]")
        migrate_raw_data(freq)
        migrate_geo_data(freq)
        migrate_results(freq)
        migrate_narratives(freq)
        log_run(freq, "migration", "complete", "initial migrate_to_db.py run")

    print("\n" + "=" * 55)
    print("Migration complete. DB is now the primary data source.")
    print("The app will read from mmm.* tables; JSON files kept as")
    print("fallback. Re-run any pipeline step to refresh DB data.")
    print("=" * 55)


if __name__ == "__main__":
    main()
