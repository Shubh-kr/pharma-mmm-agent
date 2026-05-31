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
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

load_dotenv(dotenv_path=os.path.join(_project_root, ".env"))

# ── System prompt ─────────────────────────────────────────────────────────────

INSIGHT_SYSTEM_PROMPT = """You are a Senior Pharma Commercial Strategy Analyst with deep expertise
in vaccine marketing and Marketing Mix Modelling (MMM).

You receive structured MMM model output — always an OLS Ridge model and optionally a
Bayesian PyMC model — plus budget optimisation results. Convert these into a clear,
actionable strategic narrative for a pharma commercial leadership team.

## How to use each model

OLS Ridge MMM:
  - R² and MAPE are the primary fit metrics
  - Channels marked "model" are directly identified from the data — high confidence
  - Channels marked "prior_estimate" could not be separated by Ridge (collinearity with
    seasonality dummies) — their contributions are estimated from config priors, not data
  - Use OLS contributions and the budget optimisation for all spend recommendations

Bayesian MMM (when provided):
  - Every channel has a posterior distribution — no prior-estimation fallback needed
  - Quote the 90% HDI (credible interval) when discussing channel uncertainty
  - Wide HDI = high uncertainty; narrow HDI = data strongly supports the estimate
  - The Bayesian competitor coefficient should be negative (competitive suppression) —
    if OLS shows a positive competitor coefficient, flag this and cite the Bayesian result
  - For channels that were "prior_estimated" in OLS, the Bayesian posterior is the
    more reliable estimate — cite Bayesian numbers for those channels
  - R² is typically lower in Bayesian than OLS (priors regularise more aggressively) —
    this is expected and does not mean the Bayesian model is worse

## Output structure

---
## Executive Summary
(3–5 sentences: model quality from both models if available, biggest finding,
headline recommendation with projected uplift)

## Channel Performance Analysis

### Top Performing Channels (HCP)
(For each: ROI, contribution %, Bayesian HDI if available, why it performs, recommendation)

### Top Performing Channels (DTC/Patient)
(Same structure)

### Underperforming Channels
(Low ROI channels — compare OLS vs Bayesian where they differ)

## Model Confidence & Uncertainty
(Which channels are model-identified vs prior-estimated in OLS; which have wide Bayesian
HDIs; what this means for decision risk. Include a 2-model comparison table if Bayesian
results are present.)

## Budget Reallocation Recommendation
(Specific reallocation with dollar amounts and projected % uplift, from the optimiser)

## Strategic Considerations
(2–3 pharma-specific observations: seasonality, congress timing, HCP vs DTC balance,
competitive dynamics if competitor coefficient is meaningful)

## Caveats & Data Notes
(Model assumptions, what external factors aren't captured, recommended next steps)
---

Rules:
- Never invent numbers not present in the data you receive
- Use pharma commercial vocabulary: HCP, NRx, detailing, pull-through, SOV, co-pay
- Keep recommendations specific and actionable — avoid vague statements
- Quantify every claim ("$2.3M shift", "+8.3% NRx", "0.52 ROI", "[+3.9K–+153.6K HDI]")
- Audience: VP Commercial, Medical Affairs lead, Brand team — smart, data-literate, time-poor
"""


# ── Core insight generation ───────────────────────────────────────────────────

def _build_llm(model_name: str, provider: str = "openai"):
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model_name,
            temperature=0.2,
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name,
            temperature=0.2,
            api_key=os.getenv("OPENAI_API_KEY")
        )


def generate_insights(
    ols_results_path: str,
    optimizer_results_path: str,
    bayesian_results_path: str = None,
    brand_name: str = "VaxBrand",
    model_name: str = "gpt-4o",
    provider: str = "openai"
) -> str:
    """
    Generate pharma-grade narrative from MMM results.

    Args:
        ols_results_path       : path to _ols_results.json
        optimizer_results_path : path to _budget_optimized.json
        bayesian_results_path  : path to _bayesian_results.json (optional)
        brand_name             : pharma brand name for the narrative
        model_name             : LLM model to use
        provider               : 'anthropic' or 'openai'

    Returns:
        Formatted insight narrative string
    """
    with open(ols_results_path) as f:
        ols = json.load(f)
    with open(optimizer_results_path) as f:
        opt = json.load(f)

    bayes = None
    if bayesian_results_path and os.path.exists(bayesian_results_path):
        with open(bayesian_results_path) as f:
            bayes = json.load(f)

    # ── Count OLS source split ────────────────────────────────────────────────
    n_model = sum(
        1 for ch in ols["channels"].values()
        if ch.get("contribution_source") == "model"
    )
    n_prior = len(ols["channels"]) - n_model

    # ── OLS section ───────────────────────────────────────────────────────────
    context = f"""
Brand: {brand_name}
Campaign type: Vaccine — HCP + Patient channels

=== OLS Ridge MMM ===
R² = {ols['r_squared']} | MAPE = {ols['mape_pct']}%
Observations: {ols['n_observations']} {ols['frequency']} periods
Baseline scripts/period: {ols['baseline_scripts']:,.0f}
Avg period spend: ${ols.get('avg_period_spend_k', 'N/A')}K
Channels identified by model: {n_model} of {len(ols['channels'])}
Channels prior-estimated (Ridge could not identify): {n_prior} of {len(ols['channels'])}

OLS Channel Results (contribution_source: model = data-identified, prior_estimate = config prior):
{json.dumps(ols['channels'], indent=2)}

OLS Control coefficients:
{json.dumps(ols.get('controls', {}), indent=2)}
"""

    # ── Bayesian section (if available) ───────────────────────────────────────
    if bayes:
        mcmc      = bayes["mcmc"]
        conv_tag  = "CONVERGED ✓" if mcmc.get("converged") else f"R̂={mcmc.get('max_rhat')} — CHECK"
        comp_coef = bayes["controls"].get("beta_competitor", "N/A")
        comp_note = ("correctly negative — competitor suppresses our scripts ✓"
                     if isinstance(comp_coef, (int, float)) and comp_coef < 0
                     else "unexpected sign — check model")

        context += f"""
=== Bayesian MMM (PyMC) ===
R² (posterior mean) = {bayes['r_squared_posterior_mean']} | MAPE = {bayes['mape_pct']}%
MCMC: {mcmc['chains']} chains × {mcmc['draws']} draws | R̂ max = {mcmc.get('max_rhat')} | {conv_tag}
Baseline scripts/period: {bayes['baseline_scripts']:,.0f}
Note: All 12 channels have posterior distributions — no prior-estimation fallback.

Bayesian Channel Posteriors (contribution_hdi_5 / contribution_hdi_95 = 90% credible interval):
{json.dumps(bayes['channels'], indent=2)}

Bayesian Control posteriors:
  competitor_spend beta = {comp_coef} ({comp_note})
  price_index beta      = {bayes['controls'].get('beta_price', 'N/A')} per 10 price-index points
  vaccine_season beta   = {bayes['controls'].get('beta_season', 'N/A')} scripts/period
  congress beta         = {bayes['controls'].get('beta_congress', 'N/A')} scripts/period
"""
    else:
        context += "\n=== Bayesian MMM === Not run for this analysis.\n"

    # ── Optimiser section ─────────────────────────────────────────────────────
    context += f"""
=== Budget Optimisation (based on OLS ROIs) ===
Total weekly budget: ${opt['total_budget_k']:,.1f}K
Projected script uplift: +{opt['projected_uplift_pct']}%

Channel Recommendations:
{json.dumps(opt['allocations'], indent=2)}
"""

    llm = _build_llm(model_name, provider)

    has_bayes = "Both OLS and Bayesian results are provided." if bayes else "Only OLS results are provided (Bayesian was not run)."
    messages = [
        SystemMessage(content=INSIGHT_SYSTEM_PROMPT),
        HumanMessage(content=f"""
Generate a complete strategic insight report for the pharma commercial leadership team.
{has_bayes}

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
    provider   = config.get("llm", {}).get("provider", "openai").lower()

    prefix       = "mmm_weekly" if freq == "weekly" else "mmm_monthly"
    ols_path     = f"{data_dir}/{prefix}_ols_results.json"
    opt_path     = f"{data_dir}/{prefix}_budget_optimized.json"
    bayes_path   = f"{data_dir}/{prefix}_bayesian_results.json"
    has_bayesian = os.path.exists(bayes_path)

    print(f"\n💡 Generating pharma insight narrative for {brand_name}...")
    if has_bayesian:
        print("  ℹ️  Bayesian results found — incorporating posterior credible intervals")
    else:
        print("  ℹ️  No Bayesian results found — run with --bayesian to enrich the narrative")

    narrative = generate_insights(
        ols_results_path=ols_path,
        optimizer_results_path=opt_path,
        bayesian_results_path=bayes_path if has_bayesian else None,
        brand_name=brand_name,
        model_name=model_name,
        provider=provider
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