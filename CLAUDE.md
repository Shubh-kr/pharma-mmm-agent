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

# Run geo + hierarchical Bayesian across all territories (~2 min)
python run.py --no-insights --geo-hierarchical

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
  ├── [--geo / --geo-bayesian / --geo-hierarchical / --geo-insights]
  │     ├── run_geo_ols_mmm_tool          → *_geo_ols_results.json (+ season split)
  │     ├── run_geo_budget_optimizer_tool → *_geo_budget_optimized.json
  │     ├── [--geo-bayesian]     run_geo_bayesian_mmm_tool     → *_geo_bayesian_results.json
  │     ├── [--geo-hierarchical] run_geo_hierarchical_mmm_tool → *_geo_hierarchical_results.json
  │     └── [--geo-insights]     run_geo_insight_agent         → reports/*_geo_insights.md
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
| `geo_mmm_tool.py` | `*_geo.csv` (long format) | `*_geo_ols_results.json` — Ridge per territory + post-hoc season ROI split |
| `geo_bayesian_mmm_tool.py` | `*_geo.csv` | `*_geo_bayesian_results.json` — PyMC per territory + HDI |
| `geo_hierarchical_mmm_tool.py` | `*_geo.csv` | `*_geo_hierarchical_results.json` — single PyMC model, partial pooling |
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
- `ols_model`: `ridge_alpha`, seasonality dummies, congress/competitor/price controls, `prior_contribution_weight`, `season_interactions` (bool, default true)
- `bayesian_model`: `draws`, `tune`, `chains`, `geo_draws`/`geo_tune`/`geo_chains` (per-territory), `hier_draws`/`hier_tune`/`hier_chains` (hierarchical joint model)
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

## Phase 1 complete — all PRs merged to main (PRs #1–#8 + docs)

### PRs #1–#4 (previous sessions)
- Geo dataset generator, Geo Ridge MMM, two-level geo optimizer, Geo tab (choropleth, stacked bar, optimizer table, expanders)
- Geo Bayesian MMM per territory (PyMC, ~2 min), Ridge vs Bayesian scatter, convergence table, HDI error bars, uncertainty heatmap
- Geo AI narrative (`run_geo_insight_agent()`), what-if geo budget simulator
- Monthly geo pipeline + freq-aware dashboard labels
- Hierarchical Bayesian geo MMM (`tools/geo_hierarchical_mmm_tool.py`) — log-normal non-centred hyperpriors; weekly R̂=1.005, monthly R̂=1.004

### PRs #5–#8 (this session)

**PR #5 — Post-hoc seasonal ROI split + heatmap**
- `_season_roi_split()` rewritten to use marginal efficiency ratio (`avg_sat/avg_spend`) per season — original clip-based approach zeroed all lifts (0.0%)
- Field Rep Visits -21–23% in-season; Medical Congress +8–12%; `_render_season_interactions()` heatmap in Geo tab

**PR #6 — Response curves per territory**
- `_render_response_curves()`: channel dropdown + Hill saturation curve per territory; steady-state adstock proxy `x_ss = x_raw / (1 - decay)`; operating-point dots; 80% saturation reference line
- `geo_mmm_tool.py` stores `adstock_x_max_k` and `avg_saturated` per channel

**PR #7 — Hierarchical model in geo insight narrative**
- `insight_agent.py` auto-detects `*_geo_hierarchical_results.json`; injects national hyperprior ROI ranking, `sigma_terr_mean` flags, Mountain partial-pooling correction table into LLM prompt
- System prompt extended with interpretation lens #6 and dedicated output section

**PR #8 — Attribution decomposition tab**
- Both OLS tools now emit `dates`, `actuals`, `baseline_timeseries`, `contribution_timeseries` per period in JSON
- New Attribution tab: stacked area (baseline + 12 channels + actual overlay) + waterfall (spacer-bar technique — `go.Waterfall` lacks per-bar color support); geo territory selector
- Tab order: Overview | Ridge | Bayesian | Attribution | Budget | Geo | Insights

**Docs checkpoint**
- `README.md` fully rewritten: all Phase 1 features, 7-tab dashboard, Phase 1 ✅ / Phase 2 roadmap
- `PROJECT_STATUS.md` fully rewritten: architecture, file map, feature inventory tables, Phase 2 plans, key design decisions

## Next steps — Phase 2 candidates

1. **Incrementality testing planner** — rank territory × channel candidates for geo holdout / lift tests using Bayesian HDI width + Ridge vs Bayesian disagreement
2. **Scenario planner** — inverse optimizer: given a NRx target, solve for required budget and channel mix
3. **Budget scenario comparison** — save and compare named scenarios (current / optimizer / custom) side-by-side
4. **Real data ingestion** — CSV upload flow in the dashboard sidebar with schema auto-detection
