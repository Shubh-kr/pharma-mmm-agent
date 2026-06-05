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
            max_tokens=8096,
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


# ── Geo insight system prompt ─────────────────────────────────────────────────

GEO_INSIGHT_SYSTEM_PROMPT = """You are a Senior Pharma Commercial Strategy Analyst with deep expertise
in vaccine marketing, regional field operations, and geo-level Marketing Mix Modelling (MMM).

You receive territory-disaggregated MMM results for a pharma vaccine brand across 6 US territories.
Each territory has its own Ridge MMM model, budget optimisation, and optionally a Bayesian MMM
with 90% credible intervals. When available, a Hierarchical Bayesian model pools information
across territories via shared national hyperpriors. Convert all of this into a clear, actionable
geo strategy narrative for a pharma commercial leadership team.

## How to interpret the geo data

Territory config parameters you will receive:
- market_size: relative NRx potential (higher = larger addressable market)
- spend_share: current fraction of national budget allocated to this territory
- hcp_mult: how much more (>1) or less (<1) responsive this territory's HCPs are vs national avg
- dtc_mult: same for DTC/patient channels
- season_str: vaccine season intensity (1.0 = national average; >1 = stronger flu season)

Key analytical lenses:
1. OVER/UNDER-INVESTMENT: Compare spend_share vs market_size share. A territory getting 20% of
   budget but only 15% of market is over-invested — and vice versa.
2. CHANNEL MIX FIT: Does the territory's top channel match its hcp_mult vs dtc_mult profile?
   A high-hcp_mult territory dominated by DTC suggests a channel mix problem.
3. ROI EFFICIENCY: Compare each territory's weighted avg ROI — which generates the most scripts
   per dollar? The geo optimizer's ROI efficiency score is your primary signal.
4. UNCERTAINTY (Bayesian): Wide HDI = the model is uncertain about that territory's response.
   These are the best candidates for geo holdout / lift test experiments.
5. BASELINE SCRIPTS: High baseline relative to market_size = strong brand equity in that territory.
   Low baseline = organic NRx underperforming market potential.
6. HIERARCHICAL MODEL (when provided): The national_hyperprior block gives the model's best estimate
   of each channel's ROI pooled across all territories.
   - national_roi_mean: the national consensus ROI for that channel — use this when comparing channels
     across the portfolio, as it is more stable than any single territory's Ridge estimate.
   - sigma_terr_mean: territory heterogeneity in channel response. High sigma (>2) = ROI varies
     widely by territory; the national average may mask large local differences. Low sigma (<1.5) =
     channel performs consistently everywhere — strong candidate for national scaling.
   - When a territory's Ridge ROI diverges from the national hyperprior ROI, the hierarchical model
     "shrinks" the territory estimate toward the national mean (partial pooling). This is especially
     valuable for Mountain (small territory) whose Ridge estimates are noisiest.

## Output structure

---
## Executive Summary
(4–5 sentences: overall geo model quality, biggest territory finding, headline reallocation
recommendation with projected uplift, and where uncertainty is highest)

## Territory Performance Snapshot
(A concise comparison table or ranked list: territory, ROI efficiency, spend share vs market share,
model R², top channel — annotated with over/under-investment flags)

## Territory Deep-Dives
For each territory (Northeast, Southeast, Midwest, Southwest, Mountain, Pacific):
  - 2–3 sentences: what's working, what's not, why (cite hcp/dtc_mult, season_str)
  - Specific recommendation (e.g. "shift $Xk from DTC TV to rep_visits" or "hold spend, run lift test")
  - If Bayesian: cite HDI width to flag confidence level

## National Geo Budget Reallocation
(Territory-level reallocation: which territories gain/lose budget, by how much, projected uplift.
Always cite the corridor constraint — e.g. "capped at +30% per cycle operational constraint")

## Lift Test & Uncertainty Priorities
(Which 1–2 territories should be geo holdout candidates based on: widest Bayesian HDI,
biggest Ridge vs Bayesian ROI disagreement, or largest over/under-investment gap)

## Cross-Territory Strategic Themes
(2–3 pharma-specific themes that cut across territories: e.g. HCP channel strength in academic
medical centre markets, DTC pull-through in Southern markets, seasonal variation in Mountain/Pacific)

## Hierarchical Model Insights (include only when hierarchical results are provided)
(National channel ROI consensus: rank channels by national_roi_mean and call out the top 3 and
bottom 3. Flag channels with sigma_terr_mean > 2.0 as "high heterogeneity — territory-specific
strategy required." Flag channels with sigma < 1.5 as "nationally consistent — candidate for
standardised playbook." Highlight where Mountain's hierarchical ROI differs most from its Ridge
estimate — this is where partial pooling added the most information.)

## Caveats & Next Steps
(Model independence limitation; if hierarchical results were NOT provided note what they would add;
recommended measurement actions such as geo holdout, lift test, or media experiment)
---

Rules:
- Never invent numbers not present in the data
- Use pharma commercial vocabulary: HCP, NRx, detailing, SOV, pull-through, KOL density
- Quantify every claim with territory name + dollar amount + ROI + % change
- Flag over/under-investment explicitly with the spend_share vs market_size gap
- Audience: VP Commercial, Field VP, Brand lead — data-literate, time-poor, regionally accountable
"""


# ── Geo insight generation ────────────────────────────────────────────────────

def _territory_context_block(terr_key: str, terr_cfg: dict, ols_td: dict,
                              opt_ta: dict, bayes_td) -> str:
    """Build a compact context block for one territory."""
    label      = terr_cfg.get("label", terr_key)
    top_chs    = sorted(
        ols_td.get("channels", {}).items(),
        key=lambda x: x[1]["contribution_pct"], reverse=True
    )[:3]
    top_ch_str = ", ".join(
        f"{v['label']} {v['contribution_pct']:.1f}% (ROI {v['estimated_roi']:.3f})"
        for _, v in top_chs
    )

    block = f"""
--- {label.upper()} ---
Config: market_size={terr_cfg.get('market_size')}, spend_share={terr_cfg.get('spend_share'):.0%}, \
hcp_mult={terr_cfg.get('hcp_mult')}, dtc_mult={terr_cfg.get('dtc_mult')}, season_str={terr_cfg.get('season_str')}
Ridge: R²={ols_td.get('r_squared_posterior_mean', ols_td.get('r_squared'))}, \
MAPE={ols_td.get('mape_pct')}%, baseline={ols_td.get('baseline_scripts'):,.0f} scripts/period
Avg period spend: ${ols_td.get('avg_period_spend_k', 0):,.1f}K
Top 3 channels: {top_ch_str}"""

    if bayes_td:
        mcmc      = bayes_td.get("mcmc", {})
        conv_tag  = "converged ✓" if mcmc.get("converged") else f"R̂={mcmc.get('max_rhat')} ⚠"
        # Widest HDI channel (highest uncertainty)
        widest    = max(
            bayes_td.get("channels", {}).items(),
            key=lambda x: x[1].get("contribution_hdi_95", 0) - x[1].get("contribution_hdi_5", 0),
            default=(None, {})
        )
        w_label   = widest[1].get("label", "—") if widest[0] else "—"
        w_width   = (widest[1].get("contribution_hdi_95", 0) - widest[1].get("contribution_hdi_5", 0)
                     if widest[0] else 0)
        block += f"""
Bayesian: R²={bayes_td.get('r_squared_posterior_mean')}, {conv_tag}
Widest HDI: {w_label} (±{w_width:,.0f} scripts — highest uncertainty in this territory)"""

    if opt_ta:
        block += f"""
Optimizer: current_share={opt_ta.get('current_share', 0):.1%} → optimal_share={opt_ta.get('optimal_share', 0):.1%}, \
action={opt_ta.get('action', '—')}, channel_uplift=+{opt_ta.get('projected_channel_uplift_pct', 0):.1f}%"""
        top_ch_opt = sorted(
            opt_ta.get("channel_allocations", []),
            key=lambda x: abs(x.get("change_pct", 0)), reverse=True
        )[:2]
        if top_ch_opt:
            ch_recs = "; ".join(
                f"{r['channel']} {r['change_pct']:+.0f}% (${r['change_k']:+.1f}K)"
                for r in top_ch_opt
            )
            block += f"\nBiggest channel moves: {ch_recs}"

    return block


def generate_geo_insights(
    geo_ols_path: str,
    geo_opt_path: str,
    config_path: str,
    geo_bayes_path: str = None,
    geo_hier_path: str = None,
    brand_name: str = "VaxBrand",
    model_name: str = "gpt-4o",
    provider: str = "openai",
) -> str:
    """
    Generate a territory-level strategic narrative from geo MMM results.

    Args:
        geo_ols_path   : path to *_geo_ols_results.json
        geo_opt_path   : path to *_geo_budget_optimized.json
        config_path    : path to config.yaml (for territory metadata)
        geo_bayes_path : path to *_geo_bayesian_results.json (optional)
        geo_hier_path  : path to *_geo_hierarchical_results.json (optional)
        brand_name     : brand name for the narrative
        model_name     : LLM model
        provider       : 'anthropic' or 'openai'

    Returns:
        Formatted geo insight narrative string
    """
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)
    with open(geo_ols_path) as f:
        geo_ols = json.load(f)
    with open(geo_opt_path) as f:
        geo_opt = json.load(f)

    geo_bayes = None
    if geo_bayes_path and os.path.exists(geo_bayes_path):
        with open(geo_bayes_path) as f:
            geo_bayes = json.load(f)

    geo_hier = None
    if geo_hier_path and os.path.exists(geo_hier_path):
        with open(geo_hier_path) as f:
            geo_hier = json.load(f)

    terr_config   = config.get("territories", {})
    ols_terrs     = geo_ols.get("territories", {})
    opt_terrs     = geo_opt.get("territory_allocation", {})
    bayes_terrs   = (geo_bayes or {}).get("territories", {})
    total_market  = sum(t.get("market_size", 0) for t in terr_config.values())

    # ── National geo summary ──────────────────────────────────────────────────
    avg_r2   = round(sum(t.get("r_squared_posterior_mean", t.get("r_squared", 0))
                         for t in ols_terrs.values()) / max(len(ols_terrs), 1), 3)
    uplift   = geo_opt.get("projected_territory_uplift_pct", 0)
    n_terrs  = len(ols_terrs)

    context = f"""
Brand: {brand_name}  |  Analysis: Geo-level MMM across {n_terrs} US territories
Bayesian per-territory: {"YES — 90% HDI included" if geo_bayes else "NO — Ridge only"}
National total budget: ${geo_opt.get('total_national_budget_k', 0):,.0f}K / period
Territory reallocation projected uplift: +{uplift:.1f}%
Average territory R²: {avg_r2}

National market size breakdown:
{chr(10).join(
    f"  {tc.get('label', tk):<14} market_size={tc.get('market_size', 0):,} "
    f"({tc.get('market_size', 0)/total_market:.1%} of total)  "
    f"spend_share={tc.get('spend_share', 0):.0%}  "
    f"hcp_mult={tc.get('hcp_mult')}  dtc_mult={tc.get('dtc_mult')}"
    for tk, tc in terr_config.items()
)}
"""

    # ── Per-territory context blocks ──────────────────────────────────────────
    context += "\n=== TERRITORY-BY-TERRITORY RESULTS ===\n"
    for tk in sorted(ols_terrs.keys()):
        context += _territory_context_block(
            tk,
            terr_config.get(tk, {}),
            ols_terrs[tk],
            opt_terrs.get(tk, {}),
            bayes_terrs.get(tk) if geo_bayes else None,
        )
        context += "\n"

    # ── Hierarchical hyperprior context block ─────────────────────────────────
    if geo_hier:
        hyperpriors = geo_hier.get("national_hyperpriors", {})
        hier_mcmc   = geo_hier.get("mcmc", {})
        conv_tag    = "converged ✓" if hier_mcmc.get("converged") else f"R̂={hier_mcmc.get('max_rhat')} ⚠"

        # Sort channels by national_roi_mean descending
        ranked = sorted(
            hyperpriors.items(),
            key=lambda x: x[1].get("national_roi_mean", 0),
            reverse=True,
        )
        context += f"""
=== HIERARCHICAL BAYESIAN MMM — NATIONAL HYPERPRIORS ===
Model: Single joint PyMC model, partial pooling across all 6 territories. {conv_tag}
R̂ max: {hier_mcmc.get('max_rhat', '—')}

Channels ranked by national consensus ROI (pooled across territories):
{"Channel":<35} {"Nat ROI":>8}  {"σ_terr":>7}  Heterogeneity
{"-" * 70}
"""
        for ck, hp in ranked:
            sigma = hp.get("sigma_terr_mean", 0)
            flag  = "HIGH — territory-specific strategy needed" if sigma > 2.0 else (
                    "low — nationally consistent" if sigma < 1.5 else "moderate")
            context += (
                f"  {hp.get('label', ck):<33} {hp.get('national_roi_mean', 0):>8.3f}"
                f"  {sigma:>7.3f}  {flag}\n"
            )

        # Mountain partial-pooling benefit: compare Ridge vs hierarchical ROI per channel
        mtn_ridge = (geo_ols.get("territories", {}).get("mountain", {})
                     .get("channels", {}))
        mtn_hier  = (geo_hier.get("territories", {}).get("mountain", {})
                     .get("channels", {}))
        if mtn_ridge and mtn_hier:
            context += "\nMountain — Ridge vs Hierarchical ROI (partial-pooling benefit):\n"
            for ck in sorted(mtn_ridge.keys()):
                r_roi = mtn_ridge[ck].get("estimated_roi", 0)
                h_roi = mtn_hier[ck].get("estimated_roi", 0)
                diff  = h_roi - r_roi
                context += (
                    f"  {mtn_ridge[ck].get('label', ck):<33}"
                    f" Ridge={r_roi:.3f}  Hier={h_roi:.3f}  Δ={diff:+.3f}\n"
                )

    has_bayes_str = (
        "Both Ridge and Bayesian (90% HDI) results are provided for all territories."
        if geo_bayes
        else "Only Ridge MMM results are provided (geo Bayesian was not run)."
    )
    has_hier_str = (
        "A Hierarchical Bayesian model (partial pooling across all territories) is also provided "
        "— use the national_hyperpriors block to anchor cross-territory channel comparisons and "
        "call out Mountain's partial-pooling corrections specifically."
        if geo_hier
        else "No Hierarchical Bayesian model results are available for this analysis."
    )

    llm = _build_llm(model_name, provider)
    messages = [
        SystemMessage(content=GEO_INSIGHT_SYSTEM_PROMPT),
        HumanMessage(content=f"""
Generate a complete geo strategy insight report for the pharma commercial leadership team.
{has_bayes_str}
{has_hier_str}

{context}
"""),
    ]

    response = llm.invoke(messages)
    return response.content


def run_geo_insight_agent(
    data_dir: str = "data/raw",
    config_path: str = "config/config.yaml",
    freq: str = "weekly",
) -> str:
    """
    Run the geo insight agent using territory-level MMM results.

    Args:
        data_dir    : directory containing the results JSON files
        config_path : path to config.yaml
        freq        : 'weekly' or 'monthly'

    Returns:
        Full geo insight narrative
    """
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)

    brand_name = config.get("report", {}).get("pharma_brand_name", "VaxBrand")
    model_name = config.get("llm", {}).get("model", "gpt-4o")
    provider   = config.get("llm", {}).get("provider", "openai").lower()

    prefix     = f"mmm_{freq}"
    geo_ols    = f"{data_dir}/{prefix}_geo_ols_results.json"
    geo_opt    = f"{data_dir}/{prefix}_geo_budget_optimized.json"
    geo_bayes  = f"{data_dir}/{prefix}_geo_bayesian_results.json"
    geo_hier   = f"{data_dir}/{prefix}_geo_hierarchical_results.json"

    if not os.path.exists(geo_ols):
        return f"Error: {geo_ols} not found. Run the geo pipeline first."
    if not os.path.exists(geo_opt):
        return f"Error: {geo_opt} not found. Run the geo optimizer first."

    has_bayes = os.path.exists(geo_bayes)
    has_hier  = os.path.exists(geo_hier)
    print(f"\n🌍 Generating geo insight narrative for {brand_name} ({freq})...")
    if has_bayes:
        print("  ℹ️  Geo Bayesian results found — HDI credible intervals included")
    else:
        print("  ℹ️  No geo Bayesian results — run with --geo-bayesian for richer narrative")
    if has_hier:
        print("  ℹ️  Hierarchical Bayesian results found — national hyperpriors included")
    else:
        print("  ℹ️  No hierarchical results — run with --geo-hierarchical for pooled ROI estimates")

    narrative = generate_geo_insights(
        geo_ols_path=geo_ols,
        geo_opt_path=geo_opt,
        config_path=config_path,
        geo_bayes_path=geo_bayes if has_bayes else None,
        geo_hier_path=geo_hier  if has_hier  else None,
        brand_name=brand_name,
        model_name=model_name,
        provider=provider,
    )

    out_path = f"reports/{prefix}_geo_insights.md"
    os.makedirs("reports", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(f"# {brand_name} — Geo MMM Insight Report\n\n")
        f.write(narrative)

    print(f"✅ Geo insight report saved to {out_path}")
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