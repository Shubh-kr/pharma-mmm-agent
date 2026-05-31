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

`_build_llm()` sets `max_tokens=8096` for Anthropic — required because the geo narrative is long (6-territory deep-dives + 7 sections). Default LangChain Anthropic limit is 1024 and will truncate the report.

Two narrative functions:
- `run_insight_agent()` — national MMM narrative
- `run_geo_insight_agent()` — geo/territory narrative (reads `*_geo_ols_results.json`, `*_geo_budget_optimized.json`, and optionally `*_geo_bayesian_results.json`)

### What-if Budget Simulator (`app.py → _render_whatif_simulator`)

Rendered inside the Geo tab between the optimizer expanders and the AI narrative. No LLM or re-running the optimizer — uses `roi_efficiency` values already in `*_geo_budget_optimized.json` as a linear proxy.

Math: `Δ Scripts ≈ roi_efficiency[t] × Δ Budget[t]` summed across territories. Proportionally scales allocations to `total_national_budget_k` when the user over/under-allocates. Preset buttons seed `st.session_state[f"whatif_s_{tk}"]` before slider rendering — this is the correct Streamlit pattern for programmatic slider resets.

## Session progress (as of 2026-05-31)

### Completed — previous sessions (merged to main via PR #1)
- Geo dataset generator (6 US territories, long-format, correct spend/ROI/baseline scaling)
- Geo Ridge MMM tool (`tools/geo_mmm_tool.py`) — per-territory, in-memory transforms
- Two-level geo optimizer (`tools/geo_optimizer_tool.py`) — channel mix + territory allocation
- Separate territory corridors in config (1.30×/0.80× vs 2.50×/0.50× for channels)
- Streamlit Geo tab — choropleth, channel stacked bar, optimizer table, expanders
- Geo Bayesian MMM (`tools/geo_bayesian_mmm_tool.py`) — PyMC per territory, ~2 min total
- Dashboard: Ridge vs Bayesian scatter, convergence table, HDI error bars, uncertainty heatmap
- `PROJECT_STATUS.md` — full feature inventory and roadmap
- PR workflow established: always use `feat/...` branch, never push directly to main

### Completed — this session (on `feat/geo-insight-narrative`, PR #2 open)
- **Geo AI narrative** — `GEO_INSIGHT_SYSTEM_PROMPT`, `generate_geo_insights()`, `run_geo_insight_agent()`, `_territory_context_block()` in `agents/insight_agent.py`; `--geo-insights` flag in `run.py`; sidebar checkbox + pipeline runner + Markdown display in `app.py`
  - Bug fixed: `ChatAnthropic` default `max_tokens=1024` truncated the report → raised to `max_tokens=8096`
  - Verified end-to-end: `python run.py --geo-insights --no-insights` → `reports/mmm_weekly_geo_insights.md` (164 lines, all 7 sections)
- **What-if geo budget simulator** — `_render_whatif_simulator()` in `app.py`; 6 territory sliders, budget balance indicator, simulated NRx uplift KPI, overlay bar chart, delta table, preset buttons (reset / apply optimizer)
  - Verified in browser: slider interaction, both presets, chart/table rendering all confirmed

### In progress
- Nothing currently in progress. PR #2 is open and ready for review.

## Next steps (roadmap order)

1. **Monthly geo pipeline in dashboard** — Geo tab loads weekly geo files only; add freq-aware file loading so switching to "monthly" shows monthly geo results
2. **Hierarchical geo model** — shared national trend via PyMC hierarchical priors (improves Mountain territory estimates which have the least data)
3. **Territory × time interactions** — allow ROI multipliers to vary by season per territory
4. **Response curves per territory** — visualise adstock+saturation curves by territory in dashboard
