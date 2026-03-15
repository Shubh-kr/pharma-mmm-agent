"""
agents/insight_agent.py
========================
Insight Agent — converts raw MMM results into a clear, pharma-grade
narrative that a commercial strategy team can act on directly.

Reads OLS results and budget optimisation JSON, then writes:
  - Executive summary (3–5 sentences)
  - Channel-by-channel interpretation
  - Strategic recommendations
  - Budget reallocation rationale
"""

import os
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os, sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

load_dotenv(dotenv_path=os.path.join(_project_root, ".env"))

# ── System prompt ─────────────────────────────────────────────────────────────

INSIGHT_SYSTEM_PROMPT = """You are a Senior Pharma Commercial Strategy Analyst with deep expertise 
in vaccine marketing and Marketing Mix Modelling (MMM).

You receive structured MMM model output (channel ROIs, contributions, 
budget optimisation results) and convert it into clear, actionable 
strategic narrative for a pharma commercial leadership team.

Your output must follow this exact structure:

---
## Executive Summary
(3–5 sentences: overall model quality, biggest finding, headline recommendation)

## Channel Performance Analysis

### Top Performing Channels (HCP)
(For each top HCP channel: what the ROI means, why it performs, recommendation)

### Top Performing Channels (DTC/Patient)
(Same for patient channels)

### Underperforming Channels
(Channels with low ROI or over-saturation — explain why and what to do)

## Budget Reallocation Recommendation
(Specific reallocation: "$X from DTC TV → medical congress", projected uplift,
rationale grounded in the model results)

## Strategic Considerations
(2–3 pharma-specific observations: seasonality, congress timing, HCP vs DTC balance,
field force effectiveness)

## Caveats & Data Notes
(Model assumptions, what external factors aren't captured, recommended next steps)
---

Rules:
- Never invent numbers not present in the data you receive
- Use pharma commercial vocabulary: HCP, NRx, detailing, pull-through, SOV, co-pay
- Keep recommendations specific and actionable — avoid vague statements
- Quantify every claim where possible ("$2.3M shift", "+8.3% NRx", "0.52 ROI")
- Audience: VP Commercial, Medical Affairs lead, Brand team — smart, data-literate, time-poor
"""


# ── Core insight generation ───────────────────────────────────────────────────

def generate_insights(
    ols_results_path: str,
    optimizer_results_path: str,
    brand_name: str = "VaxBrand",
    model_name: str = "gpt-4o"
) -> str:
    """
    Generate pharma-grade narrative from MMM results.

    Args:
        ols_results_path       : path to _ols_results.json
        optimizer_results_path : path to _budget_optimized.json
        brand_name             : pharma brand name for the narrative
        model_name             : LLM model to use

    Returns:
        Formatted insight narrative string
    """
    # Load results
    with open(ols_results_path) as f:
        ols_results = json.load(f)

    with open(optimizer_results_path) as f:
        opt_results = json.load(f)

    # Build context for the LLM
    context = f"""
    Brand: {brand_name}
    Campaign type: Vaccine — HCP + Patient channels
    
    === OLS MMM Results ===
    Model fit: R² = {ols_results['r_squared']} | MAPE = {ols_results['mape_pct']}%
    Observations: {ols_results['n_observations']} {ols_results['frequency']} periods
    Baseline scripts: {ols_results['baseline_scripts']:,.0f}
    
    Channel Results (sorted by contribution):
    {json.dumps(ols_results['channels'], indent=2)}
    
    === Budget Optimisation Results ===
    Total budget: ${opt_results['total_budget_k']:,.0f}K
    Projected uplift: +{opt_results['projected_uplift_pct']}% scripts_written
    
    Channel Recommendations:
    {json.dumps(opt_results['allocations'], indent=2)}
    """

    llm = ChatOpenAI(
        model=model_name,
        temperature=0.2,
        api_key=os.getenv("OPENAI_API_KEY")
    )

    messages = [
        SystemMessage(content=INSIGHT_SYSTEM_PROMPT),
        HumanMessage(content=f"""
        Generate a complete strategic insight report for the pharma commercial 
        leadership team based on these MMM results:
        
        {context}
        """)
    ]

    response = llm.invoke(messages)
    return response.content


def run_insight_agent(
    data_dir: str = "data/raw",
    config_path: str = "config/config.yaml",
    freq: str = "weekly"
) -> str:
    """
    Run the insight agent using results from the analytics pipeline.

    Args:
        data_dir    : directory containing the results JSON files
        config_path : path to config.yaml
        freq        : 'weekly' or 'monthly'

    Returns:
        Full insight narrative
    """
    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f)

    brand_name = config.get("report", {}).get("pharma_brand_name", "VaxBrand")
    model_name = config.get("llm", {}).get("model", "gpt-4o")

    prefix = "mmm_weekly" if freq == "weekly" else "mmm_monthly"
    ols_path = f"{data_dir}/{prefix}_ols_results.json"
    opt_path = f"{data_dir}/{prefix}_budget_optimized.json"

    print(f"\n💡 Generating pharma insight narrative for {brand_name}...")

    narrative = generate_insights(
        ols_results_path=ols_path,
        optimizer_results_path=opt_path,
        brand_name=brand_name,
        model_name=model_name
    )

    # Save narrative
    out_path = f"reports/{prefix}_insights.md"
    os.makedirs("reports", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(f"# {brand_name} — MMM Insight Report\n\n")
        f.write(narrative)

    print(f"✅ Insight report saved to {out_path}")
    return narrative


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Pharma MMM Insight Agent")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--freq", default="weekly", choices=["weekly", "monthly"])
    args = parser.parse_args()

    output = run_insight_agent(
        data_dir=args.data_dir,
        config_path=args.config,
        freq=args.freq
    )
    print("\n" + output)