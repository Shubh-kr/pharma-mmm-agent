"""
agents/planner_agent.py
========================
Planner Agent — the top-level orchestrator for the Pharma MMM pipeline.

This is the main entry point. It:
  1. Accepts a natural language query from the user
  2. Decomposes it into sub-tasks
  3. Delegates to the analytics agent (transforms + OLS + optimiser)
  4. Delegates to the insight agent (narrative generation)
  5. Returns a unified response

Usage:
    python agents/planner_agent.py --data data/raw/mmm_weekly.csv --freq weekly
    
    Or from Python:
        from agents.planner_agent import run_mmm_pipeline
        result = run_mmm_pipeline(freq="weekly")
"""

import os
import yaml
from dotenv import load_dotenv

load_dotenv()


# ── Pipeline orchestrator ─────────────────────────────────────────────────────

def run_mmm_pipeline(
    data_path: str = None,
    config_path: str = "config/config.yaml",
    freq: str = "weekly",
    run_insights: bool = True,
    verbose: bool = True
) -> dict:
    """
    Run the full Pharma MMM pipeline end-to-end.

    Pipeline:
        generate_dataset → transforms → OLS MMM → budget optimiser → insights

    Args:
        data_path    : path to MMM CSV. If None, uses config default.
        config_path  : path to config.yaml
        freq         : 'weekly' or 'monthly'
        run_insights : whether to run the insight agent (requires API key)
        verbose      : print progress

    Returns:
        dict with keys: analytics_summary, insights, output_files
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if data_path is None:
        data_path = (
            config["data"]["weekly_path"]
            if freq == "weekly"
            else config["data"]["monthly_path"]
        )

    if verbose:
        print("\n" + "=" * 60)
        print("  🔬 Pharma MMM Agent Pipeline")
        print("=" * 60)
        print(f"  Dataset   : {data_path}")
        print(f"  Frequency : {freq}")
        print(f"  Config    : {config_path}")
        print("=" * 60 + "\n")

    results = {}

    # ── Step 1: Check dataset exists, generate if not ─────────────────────────
    import os
    if not os.path.exists(data_path):
        if verbose:
            print("📊 Dataset not found — generating synthetic data...")
        os.system("python scripts/generate_dataset.py")

    # ── Step 2: Analytics pipeline ────────────────────────────────────────────
    if verbose:
        print("🤖 Step 1/3 — Running analytics agent (transforms + OLS + optimiser)...")

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")

    # Detect placeholder values — treat as no key
    _placeholders = ("your-openai-key-here", "your-key-here", "sk-...", "", None)
    has_real_key = api_key and not any(p in str(api_key) for p in _placeholders)

    if has_real_key and not run_insights is False:
        # Full LLM-orchestrated pipeline
        from agents.analytics_agent import run_analytics_pipeline
        analytics_output = run_analytics_pipeline(
            data_path=data_path,
            config_path=config_path,
            freq=freq,
            verbose=verbose
        )
    else:
        # Fallback: run tools directly without LLM orchestration
        if verbose:
            print("  ℹ️  Running in direct mode (no LLM) — add a real API key to .env to enable agent orchestration")
        analytics_output = _run_tools_directly(data_path, config_path, freq, verbose)

    results["analytics_summary"] = analytics_output

    # ── Step 3: Insight generation ────────────────────────────────────────────
    if run_insights and api_key:
        if verbose:
            print("\n💡 Step 2/3 — Running insight agent (narrative generation)...")
        from agents.insight_agent import run_insight_agent
        insights = run_insight_agent(
            data_dir=os.path.dirname(data_path),
            config_path=config_path,
            freq=freq
        )
        results["insights"] = insights
    else:
        results["insights"] = (
            "Insight narrative skipped — add OPENAI_API_KEY to .env to enable."
        )

    # ── Step 4: Collect output files ──────────────────────────────────────────
    prefix = os.path.basename(data_path).replace(".csv", "")
    data_dir = os.path.dirname(data_path)
    results["output_files"] = {
        "transformed_data": f"{data_dir}/{prefix}_transformed.csv",
        "ols_results":      f"{data_dir}/{prefix}_ols_results.json",
        "budget_optimized": f"{data_dir}/{prefix}_budget_optimized.json",
        "insight_report":   f"reports/{prefix}_insights.md",
    }

    if verbose:
        print("\n✅ Step 3/3 — Pipeline complete!")
        print("\nOutput files:")
        for k, v in results["output_files"].items():
            exists = "✓" if os.path.exists(v) else "○"
            print(f"  {exists} {v}")
        print()

    return results


def _run_tools_directly(
    data_path: str,
    config_path: str,
    freq: str,
    verbose: bool
) -> str:
    """
    Fallback: run transforms + OLS + optimiser directly without LLM.
    Useful for testing the pipeline before adding an API key.
    """
    import json
    from tools.transforms import apply_all_transforms_tool
    from tools.ols_mmm_tool import run_ols_mmm_tool
    from tools.optimizer_tool import run_budget_optimizer_tool

    outputs = []

    # Transforms
    if verbose:
        print("  → Applying adstock + saturation transforms...")
    transform_out = apply_all_transforms_tool.invoke({
        "data_path": data_path,
        "config_path": config_path
    })
    outputs.append(transform_out)

    # OLS MMM
    transformed_path = data_path.replace(".csv", "_transformed.csv")
    if verbose:
        print("  → Running OLS MMM model...")
    ols_out = run_ols_mmm_tool.invoke({
        "data_path": transformed_path,
        "config_path": config_path,
        "freq": freq
    })
    outputs.append(ols_out)

    # Budget optimiser
    ols_results_path = data_path.replace(".csv", "_ols_results.json")
    if verbose:
        print("  → Running budget optimiser...")

    # Get total budget from OLS results
    try:
        with open(ols_results_path) as f:
            ols_data = json.load(f)
        total_spend = sum(
            ch["total_spend_k"]
            for ch in ols_data.get("channels", {}).values()
        )
        n_periods = ols_data.get("n_observations", 104)
        avg_period_budget = total_spend / n_periods
    except Exception:
        avg_period_budget = 900.0

    opt_out = run_budget_optimizer_tool.invoke({
        "results_path": ols_results_path,
        "config_path": config_path,
        "total_budget_k": avg_period_budget,
        "freq": freq
    })
    outputs.append(opt_out)

    return "\n\n".join(outputs)


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Pharma MMM Agent — full pipeline runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python agents/planner_agent.py
  python agents/planner_agent.py --freq monthly
  python agents/planner_agent.py --data data/raw/mmm_weekly.csv --freq weekly
  python agents/planner_agent.py --no-insights   # skip LLM narrative
        """
    )
    parser.add_argument("--data", default=None, help="Path to MMM CSV")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--freq", default="weekly", choices=["weekly", "monthly"])
    parser.add_argument("--no-insights", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    results = run_mmm_pipeline(
        data_path=args.data,
        config_path=args.config,
        freq=args.freq,
        run_insights=not args.no_insights,
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