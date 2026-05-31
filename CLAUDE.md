# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Always run from the project root — `run.py` inserts the root into `sys.path` so `agents/` and `tools/` can import each other correctly.

```bash
# Install dependencies (venv already at .venv/)
pip install -r requirements.txt

# Generate all datasets (national + geo, weekly + monthly)
python scripts/generate_dataset.py

# Run national pipeline (no API key needed)
python run.py --no-insights

# Run national + geo Ridge MMM + geo optimizer
python run.py --no-insights --geo

# Run geo + Bayesian per territory (~2 min)
python run.py --no-insights --geo-bayesian

# Run geo + AI narrative (needs ANTHROPIC_API_KEY or OPENAI_API_KEY in .env)
python run.py --geo --geo-insights

# Monthly data
python run.py --freq monthly --no-insights --geo

# Run individual agents directly
python agents/planner_agent.py --freq weekly
python agents/analytics_agent.py --data data/raw/mmm_weekly.csv
python agents/insight_agent.py --freq weekly

# Launch Streamlit dashboard
streamlit run app.py

# Launch notebooks
jupyter lab notebooks/
```

### Environment

Copy `.env.example` to `.env` and set `OPENAI_API_KEY` (or `ANTHROPIC_API_KEY`). Without a real key the pipeline runs in **direct mode** — it calls the tools directly, skipping LLM orchestration. The geo insight narrative always requires an API key.

## Architecture

### Full pipeline flow

```
run.py
  ├── run_mmm_pipeline()              (national)
  │     ├── apply_all_transforms_tool → *_transformed.csv
  │     ├── run_ols_mmm_tool          → *_ols_results.json
  │     ├── run_budget_optimizer_tool → *_budget_optimized.json
  │     ├── [--bayesian] run_bayesian_mmm_tool → *_bayesian_results.json
  │     └── [LLM] insight_agent       → reports/*_insights.md
  │
  ├── [--geo / --geo-bayesian / --geo-insights]
  │     ├── run_geo_ols_mmm_tool      → *_geo_ols_results.json
  │     ├── run_geo_budget_optimizer_tool → *_geo_budget_optimized.json
  │     ├── [--geo-bayesian] run_geo_bayesian_mmm_tool → *_geo_bayesian_results.json
  │     └── [--geo-insights] run_geo_insight_agent → reports/*_geo_insights.md
  │
  └── app.py (Streamlit, 6 tabs)
        Overview | Ridge MMM | Bayesian MMM | Budget | Geo | Insights
```

### Tools (`tools/`)

| Tool | Input | Output |
|------|-------|--------|
| `transforms.py` | raw CSV + config | `*_transformed.csv` (`_adstocked`, `_saturated` columns) |
| `ols_mmm_tool.py` | `*_transformed.csv` | `*_ols_results.json` — Ridge MMM, per-channel ROI + contribution |
| `bayesian_mmm_tool.py` | `*_transformed.csv` | `*_bayesian_results.json` — PyMC posteriors + 90% HDI |
| `optimizer_tool.py` | `*_ols_results.json` | `*_budget_optimized.json` — SLSQP channel reallocation |
| `geo_mmm_tool.py` | `*_geo.csv` (long format) | `*_geo_ols_results.json` — Ridge per territory |
| `geo_bayesian_mmm_tool.py` | `*_geo.csv` | `*_geo_bayesian_results.json` — PyMC per territory + HDI |
| `geo_optimizer_tool.py` | `*_geo_ols_results.json` | `*_geo_budget_optimized.json` — two-level optimizer |

**Math notes:**
- Adstock: geometric carryover `result[t] = spend[t] + decay * result[t-1]`
- Saturation: Hill/power curve `(x/x_max)^alpha`, output normalised 0–1
- OLS model: Ridge regression (alpha=1.0) with non-negativity constraint; ROI blended 60/40 prior/model
- Bayesian: PyMC HalfNormal priors on channel betas calibrated from `prior_roi × mean_y × 0.5`
- Optimizer: SLSQP; channel bounds = tighter of portfolio share limit and per-channel corridor
- Geo optimizer: two-level SLSQP — channel mix per territory (Level 1) + territory allocation (Level 2)

### Configuration (`config/config.yaml`)

Single source of truth. Key sections:
- `llm`: provider (anthropic/openai), model, temperature
- `data`: file paths, outcome column (`scripts_written`)
- `channels`: 12 channels × `adstock_decay`, `saturation`, `prior_roi`, `channel_type` (hcp/dtc)
- `ols_model`: `ridge_alpha`, seasonality dummies, congress/competitor/price controls, `prior_contribution_weight`
- `bayesian_model`: `draws`, `tune`, `chains`, `geo_draws`/`geo_tune`/`geo_chains` (faster for per-territory runs)
- `optimizer`: budget share bounds, `max_spend_increase_factor`/`decrease_factor` (channel corridors), `max_territory_increase_factor`/`decrease_factor` (tighter, 1.30×/0.80×)
- `territories`: 6 regions × `market_size`, `spend_share`, `hcp_mult`, `dtc_mult`, `season_str`, `states`
- `report`: brand name used in narratives

### Data files

**National:**
- `data/raw/mmm_weekly.csv` — 104 rows × 24 cols (2yr weekly)
- `data/raw/mmm_monthly.csv` — 36 rows × 26 cols (3yr monthly)

**Geo (long format, `territory` column):**
- `data/raw/mmm_weekly_geo.csv` — 624 rows (104 weeks × 6 territories)
- `data/raw/mmm_monthly_geo.csv` — 216 rows (36 months × 6 territories)

Derived files written alongside source: `*_transformed.csv`, `*_ols_results.json`, `*_budget_optimized.json`, `*_bayesian_results.json`, `*_geo_*.json`

KPI / outcome: `scripts_written` (vaccine Rx written)

### Geo data scaling (important)

Territory spend is scaled by `spend_share` (not `spend_share × n_territories`). Channel contribution `base` is also scaled by `spend_share` so that `effect = roi × saturation × base` is proportional to territory budget. Baseline scripts scale by `market_size / total_market`. This ensures territory scripts sum approximately to national totals.

### LLM integration

`insight_agent.py` supports both providers. Set `llm.provider: anthropic` and `llm.model: claude-sonnet-4-6` in `config/config.yaml`. `analytics_agent.py` hardcodes `ChatOpenAI` — would need updating for Anthropic.

Two narrative functions:
- `run_insight_agent()` — national MMM narrative
- `run_geo_insight_agent()` — geo/territory narrative (reads `*_geo_ols_results.json`, `*_geo_budget_optimized.json`, and optionally `*_geo_bayesian_results.json`)

## Session progress (as of 2026-05-31)

### Completed this session
- Geo dataset generator (6 US territories, long-format, correct spend/ROI/baseline scaling)
- Geo Ridge MMM tool (`tools/geo_mmm_tool.py`) — per-territory, in-memory transforms
- Two-level geo optimizer (`tools/geo_optimizer_tool.py`) — channel mix + territory allocation
- Separate territory corridors in config (1.30×/0.80× vs 2.50×/0.50× for channels)
- Streamlit Geo tab — choropleth, channel stacked bar, optimizer table, expanders
- Geo Bayesian MMM (`tools/geo_bayesian_mmm_tool.py`) — PyMC per territory, ~2 min total
- Dashboard: Ridge vs Bayesian scatter, convergence table, HDI error bars, uncertainty heatmap
- `PROJECT_STATUS.md` — full feature inventory and roadmap
- PR workflow established: always use `feat/...` branch, never push directly to main

### In progress (interrupted)
- `feat/geo-insight-narrative` branch — geo AI narrative is ~90% implemented
  - `GEO_INSIGHT_SYSTEM_PROMPT` written in `agents/insight_agent.py`
  - `generate_geo_insights()` and `run_geo_insight_agent()` written
  - `_territory_context_block()` helper written
  - `run.py --geo-insights` flag added
  - `app.py` sidebar checkbox + pipeline runner + loader + tab section added
  - **Blocked at**: Python 3.9 `dict | None` type hint syntax error — fixed with bare `bayes_td` param
  - **Next step**: verify imports (`python -c "from agents.insight_agent import run_geo_insight_agent"`), then run `python run.py --geo-insights --no-insights` to test end-to-end, then commit + PR

## Next steps (roadmap order)

1. **Finish geo insight narrative** (in-progress on `feat/geo-insight-narrative`)
2. **What-if geo simulator** — dashboard sliders to shift territory budget shares and preview script change
3. **Monthly geo pipeline in dashboard** — Geo tab currently only loads weekly geo files; add freq-aware loading
4. **Hierarchical geo model** — shared national trend via PyMC hierarchical priors (improves Mountain estimates)
5. **Territory × time interactions** — allow ROI multipliers to vary by season per territory
6. **Response curves per territory** — visualise adstock+saturation curves by territory in dashboard
