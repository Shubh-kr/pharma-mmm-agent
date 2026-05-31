"""
agents/analytics_agent.py
==========================
Analytics Agent — runs the full MMM modelling pipeline:
  1. Applies adstock + saturation transforms to all 12 channels
  2. Fits OLS (frequentist) MMM model
  3. Runs budget optimiser
  4. Returns structured results for the insight agent

Can be used standalone or orchestrated by the planner agent.
"""

import os
import yaml
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from dotenv import load_dotenv

from tools.transforms import (
    apply_adstock_tool,
    apply_saturation_tool,
    apply_all_transforms_tool,
)
from tools.ols_mmm_tool import run_ols_mmm_tool
from tools.bayesian_mmm_tool import run_bayesian_mmm_tool
from tools.optimizer_tool import run_budget_optimizer_tool

load_dotenv()

# ── System prompt ─────────────────────────────────────────────────────────────

ANALYTICS_SYSTEM_PROMPT = """You are a Senior Pharma Marketing Analytics Agent specialising in
Marketing Mix Modelling (MMM) for vaccine and pharmaceutical campaigns.

Your job is to run a rigorous, step-by-step MMM analysis pipeline:

STEP 1 — Transforms
  Call apply_all_transforms_tool to apply adstock and saturation to ALL channels at once.
  This must always be the first step before any modelling.

STEP 2 — OLS MMM Model
  Call run_ols_mmm_tool on the transformed dataset.
  Use frequency matching: if data is mmm_weekly, use freq='weekly'.
  Interpret R² > 0.75 as a good fit. Flag anything below 0.65.
  Note channels marked 'prior_estimate' — these could not be identified by Ridge
  and were estimated from priors. Mention this in your summary.

STEP 3 — Budget Optimisation
  Call run_budget_optimizer_tool using the OLS results.
  Use the current total spend as the budget input (read from OLS results).

STEP 4 — Bayesian MMM (optional, run when explicitly requested)
  Call run_bayesian_mmm_tool on the same transformed dataset.
  This takes 2-5 minutes. Advantages over OLS:
    - All channels receive non-zero contributions via informative priors
    - Every contribution has a 90% credible interval
    - Competitor and price controls get proper negative posteriors
  Check R̂ (r-hat) convergence: values < 1.05 indicate good mixing.
  Compare Bayesian vs OLS contributions and flag any meaningful differences.

STEP 5 — Summarise
  After all tools have run, produce a clean structured summary:
  - Model fit quality (R², MAPE, convergence if Bayesian)
  - Top 3 highest-ROI channels with confidence
  - Top 3 underperforming channels
  - Budget reallocation headline (e.g. "+8.3% scripts with same budget")
  - How many channels were model-identified vs prior-estimated (OLS only)

Always be precise with numbers. Round spend to 1 decimal place, ROI to 2 decimal places,
percentages to 1 decimal place. Speak like a data scientist presenting to a pharma
commercial strategy team — confident, evidence-based, no fluff.
"""


# ── Agent factory ─────────────────────────────────────────────────────────────

def _build_llm(cfg: dict):
    provider = cfg.get("llm", {}).get("provider", "openai").lower()
    model = cfg.get("llm", {}).get("model", "gpt-4o")
    temperature = cfg.get("llm", {}).get("temperature", 0.1)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY")
        )


def create_analytics_agent(model_name: str = None, verbose: bool = True):
    """
    Create and return the analytics agent executor.

    Args:
        model_name : LLM model override (defaults to config or gpt-4o)
        verbose    : print agent reasoning steps

    Returns:
        AgentExecutor ready to invoke
    """
    config_path = "config/config.yaml"

    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
    except Exception:
        cfg = {"llm": {"provider": "openai", "model": "gpt-4o", "temperature": 0.1}}

    if model_name is not None:
        cfg["llm"]["model"] = model_name

    llm = _build_llm(cfg)

    tools = [
        apply_all_transforms_tool,
        apply_adstock_tool,
        apply_saturation_tool,
        run_ols_mmm_tool,
        run_bayesian_mmm_tool,
        run_budget_optimizer_tool,
    ]

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=ANALYTICS_SYSTEM_PROMPT),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=verbose, max_iterations=10)


# ── Convenience run function ──────────────────────────────────────────────────

def run_analytics_pipeline(
    data_path: str = "data/raw/mmm_weekly.csv",
    config_path: str = "config/config.yaml",
    freq: str = "weekly",
    verbose: bool = True
) -> str:
    """
    Run the full analytics pipeline end-to-end.

    Args:
        data_path   : path to raw MMM CSV
        config_path : path to config.yaml
        freq        : 'weekly' or 'monthly'
        verbose     : show agent reasoning

    Returns:
        Analytics summary string
    """
    agent = create_analytics_agent(verbose=verbose)

    query = f"""
    Run the full MMM analytics pipeline on this pharma vaccine campaign dataset:
    
    - Data file: {data_path}
    - Config file: {config_path}
    - Frequency: {freq}
    
    Execute all three steps: transforms → OLS MMM → budget optimisation.
    Then provide a structured summary of results.
    """

    result = agent.invoke({"input": query})
    output = result["output"]
    if isinstance(output, list):
        output = "\n".join(
            part["text"] for part in output if isinstance(part, dict) and "text" in part
        )
    return output


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Pharma MMM Analytics Agent")
    parser.add_argument("--data", default="data/raw/mmm_weekly.csv")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--freq", default="weekly", choices=["weekly", "monthly"])
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    print("\n🔬 Pharma MMM Analytics Agent Starting...\n")
    output = run_analytics_pipeline(
        data_path=args.data,
        config_path=args.config,
        freq=args.freq,
        verbose=not args.quiet
    )
    print("\n📊 Analytics Summary:\n")
    print(output)