"""
run.py — Main entry point for the Pharma MMM Agent pipeline.

Always run this from the project root:
    python run.py
    python run.py --freq monthly
    python run.py --no-insights

This script fixes Python's module resolution so agents/ and tools/
can find each other correctly.
"""

import sys
import os

# ── Fix module resolution ─────────────────────────────────────────────────────
# Add project root to Python path so 'agents' and 'tools' are always findable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Now safe to import ────────────────────────────────────────────────────────
from agents.planner_agent import run_mmm_pipeline
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pharma MMM Agent — full pipeline")
    parser.add_argument("--data",        default=None,     help="Path to MMM CSV")
    parser.add_argument("--config",      default="config/config.yaml")
    parser.add_argument("--freq",        default="weekly", choices=["weekly", "monthly"])
    parser.add_argument("--no-insights", action="store_true", help="Skip LLM narrative")
    parser.add_argument("--bayesian",    action="store_true", help="Also run Bayesian MMM (adds 2-5 min)")
    parser.add_argument("--geo",         action="store_true", help="Also run geo-level MMM + optimizer")
    parser.add_argument("--quiet",       action="store_true")
    args = parser.parse_args()

    results = run_mmm_pipeline(
        data_path=args.data,
        config_path=args.config,
        freq=args.freq,
        run_insights=not args.no_insights,
        run_bayesian=args.bayesian,
        verbose=not args.quiet
    )

    print("\n" + "=" * 60)
    print("ANALYTICS SUMMARY")
    print("=" * 60)
    print(results["analytics_summary"])

    if results.get("insights") and "skipped" not in results["insights"]:
        print("\n" + "=" * 60)
        print("STRATEGIC INSIGHTS")
        print("=" * 60)
        print(results["insights"])

    if args.geo:
        import json
        from tools.geo_mmm_tool import run_geo_ols_mmm_tool
        from tools.geo_optimizer_tool import run_geo_budget_optimizer_tool

        geo_path = f"data/raw/mmm_{args.freq}_geo.csv"
        geo_ols_path = f"data/raw/mmm_{args.freq}_geo_ols_results.json"

        print("\n" + "=" * 60)
        print("GEO MMM")
        print("=" * 60)
        print(run_geo_ols_mmm_tool.invoke(
            {"data_path": geo_path, "config_path": args.config, "freq": args.freq}
        ))

        with open(geo_ols_path) as f:
            geo_results = json.load(f)
        total_budget = round(sum(
            t.get("avg_period_spend_k", 0)
            for t in geo_results.get("territories", {}).values()
        ), 1)

        print("\n" + "=" * 60)
        print("GEO BUDGET OPTIMISER")
        print("=" * 60)
        print(run_geo_budget_optimizer_tool.invoke({
            "results_path":             geo_ols_path,
            "config_path":              args.config,
            "total_national_budget_k":  total_budget,
            "freq":                     args.freq,
        }))