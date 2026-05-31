# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Always run from the project root — `run.py` inserts the root into `sys.path` so `agents/` and `tools/` can import each other correctly.

```bash
# Install dependencies (venv already at .venv/)
pip install -r requirements.txt

# Generate synthetic dataset (if data/raw/*.csv are missing)
python scripts/generate_dataset.py

# Run the full pipeline (default: weekly, with LLM insight narrative)
python run.py

# Skip LLM narrative (no API key needed)
python run.py --no-insights

# Monthly data
python run.py --freq monthly

# Run individual agents directly
python agents/planner_agent.py --freq weekly
python agents/analytics_agent.py --data data/raw/mmm_weekly.csv
python agents/insight_agent.py --freq weekly

# Launch notebooks
jupyter lab notebooks/
```

### Environment

Copy `.env.example` to `.env` and set `OPENAI_API_KEY` (or `ANTHROPIC_API_KEY`). Without a real key the pipeline runs in **direct mode** — it calls the tools directly, skipping LLM orchestration.

## Architecture

### Agent flow

```
run.py / planner_agent.py
    └── run_mmm_pipeline()
          ├── (if real API key) analytics_agent → AgentExecutor (LLM orchestrates tools)
          │    OR
          │   _run_tools_directly() → calls tools sequentially without LLM
          │
          │   Both paths execute the same three-step sequence:
          │     1. apply_all_transforms_tool  → *_transformed.csv
          │     2. run_ols_mmm_tool           → *_ols_results.json
          │     3. run_budget_optimizer_tool  → *_budget_optimized.json
          │
          └── (if API key) insight_agent → LLM call → reports/*_insights.md
```

### Tools (`tools/`)

Each tool is a `@tool`-decorated LangChain function that reads/writes files directly:

| Tool | Input | Output |
|------|-------|--------|
| `apply_all_transforms_tool` | raw CSV + config | `*_transformed.csv` with `_adstocked` and `_saturated` columns per channel |
| `apply_adstock_tool` / `apply_saturation_tool` | single channel | same CSV, new column added |
| `run_ols_mmm_tool` | `*_transformed.csv` | `*_ols_results.json` with per-channel ROI and contribution |
| `run_budget_optimizer_tool` | `*_ols_results.json` | `*_budget_optimized.json` with reallocation recommendations |

**Math notes:**
- Adstock: geometric carryover `result[t] = spend[t] + decay * result[t-1]`
- Saturation: Hill/power curve `(x/x_max)^alpha`, output normalised 0–1
- OLS model: Ridge regression (alpha=10) with non-negativity constraint on channel coefficients; ROI is blended 60/40 between config prior and model estimate
- Optimizer: SLSQP constrained optimisation; channel spend bounded by `[min_channel_share, max_channel_share]` × total budget

### Configuration (`config/config.yaml`)

Single source of truth for all model parameters. Key sections:
- `llm`: provider, model name, temperature
- `data`: file paths, outcome column (`scripts_written`)
- `channels`: 12 channels with `adstock_decay`, `saturation`, `prior_roi`, `channel_type` (hcp/dtc)
- `ols_model`: seasonality dummies, congress control variable toggle
- `optimizer`: budget share bounds
- `report`: brand name used in insight narrative

### Data

- `data/raw/mmm_weekly.csv` — 104 rows × 22 cols (2-year weekly)
- `data/raw/mmm_monthly.csv` — 36 rows × 24 cols (3-year monthly)
- Derived files written alongside source: `*_transformed.csv`, `*_ols_results.json`, `*_budget_optimized.json`
- KPI / outcome: `scripts_written` (vaccine Rx written)

### LLM integration

Both `analytics_agent` and `insight_agent` default to OpenAI (`gpt-4o`). To switch to Claude, set `llm.provider: anthropic` and `llm.model: claude-3-5-sonnet-20241022` in `config/config.yaml` — but note `analytics_agent.py` hardcodes `ChatOpenAI`; it would need updating to use `ChatAnthropic`.
