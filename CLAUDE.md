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

## Session progress (as of 2026-06-05)

### Completed — previous sessions (merged to main, PRs #1–#2)
- Geo dataset generator, Geo Ridge MMM, two-level geo optimizer, Geo tab (choropleth, stacked bar, optimizer table, expanders)
- Geo Bayesian MMM per territory (PyMC, ~2 min total), Ridge vs Bayesian scatter, convergence table, HDI error bars, uncertainty heatmap
- Geo AI narrative (`run_geo_insight_agent()`), what-if geo budget simulator with 6 territory sliders + preset buttons

### Completed — this session (merged to main, PRs #3–#4 + direct commits)

**PR #3 — Monthly geo pipeline in dashboard**
- Generated `mmm_monthly_geo_ols_results.json` and `mmm_monthly_geo_budget_optimized.json`
- Geo tab already loaded files by freq prefix; the monthly results just hadn't been generated
- Fixed hardcoded "weekly"/"Weeks" labels in `tab_overview` and `tab_ridge` — both functions now take `freq` param:
  - "Avg weekly scripts" → "Avg monthly/weekly scripts"
  - "Weeks" → "Months"/"Weeks"
  - "Baseline scripts/wk" → "Baseline scripts/mo" or "/wk"
  - "Avg wk spend $K" table header follows same pattern

**PR #4 — Hierarchical Bayesian geo MMM** (`tools/geo_hierarchical_mmm_tool.py`)
- Single joint PyMC model across all 6 territories with partial pooling via log-normal non-centred hyperpriors on channel betas
- Mountain territory borrows ROI estimates from larger territories (key benefit)
- **Critical design decision — log-normal baselines:** first run used Normal non-centred for territory baselines → Mountain and Midwest went negative. Switched to log-normal non-centred (`pt.exp(log_mu_bl + sigma_log_bl * z_bl)`) which forces positivity and handles the 3× range in market sizes naturally
- Output: `*_geo_hierarchical_results.json` with top-level `national_hyperpriors` block (μ_beta, σ_terr, national_roi_mean per channel) + same territory structure as per-territory Bayesian
- Dashboard: sidebar checkbox "Include Geo Hierarchical (~5 min)", new expander in Geo tab, `_render_geo_hierarchical()` shows hyperprior table then reuses `_render_geo_bayesian()`
- CLI: `python run.py --geo-hierarchical` (works for both weekly and monthly)
- Config: `hier_draws: 600`, `hier_tune: 400`, `hier_chains: 2` under `bayesian_model`
- Results: weekly R̂=1.005, monthly R̂=1.004, both converged, all baselines positive

**Direct commit — monthly hierarchical results**
- `data/raw/mmm_monthly_geo_hierarchical_results.json` generated and committed to main

### In progress — `feat/season-interactions` branch

**Territory × time interactions** — implementation started, branch open but not yet committed:
- **Design decision (important):** First attempted adding `sat_ch × vaccine_season` as explicit Ridge interaction features. This caused wildly unstable gammas on sparse territories (Mountain: -108604% lift for HCP Programmatic Digital) due to multicollinearity with month dummies and underdetermined system on 36 monthly observations.
- **Switched to post-hoc seasonal ROI split** (cleaner, no model change):
  - Keep Ridge model exactly as-is
  - After fitting, evaluate the same beta at in-season vs off-season avg saturation/spend levels
  - The ROI difference comes from saturation: more in-season spend → further along diminishing-returns curve → lower marginal ROI
  - Only computed for model-identified channels (`contribution_source == "model"`, `beta > 1e-6`) — prior-estimated channels have beta ≈ 0, split would be meaningless
  - Per-channel additions to JSON: `roi_in_season`, `roi_off_season`, `season_lift_pct`
  - New helper `_season_roi_split()` in `geo_mmm_tool.py`; `has_season_interactions: bool` flag on territory result
- Current state: `geo_mmm_tool.py` rewritten with post-hoc approach, **pipeline not yet re-run**, dashboard not yet updated, PR not yet opened
- **Next action:** run `python run.py --no-insights --geo` to verify season lifts are sensible, then add dashboard heatmap and open PR

## Next steps (exact, in order)

1. **Finish `feat/season-interactions`** (currently on this branch):
   - Run `python run.py --no-insights --geo` and verify `season_lift_pct` values are reasonable (expect ±5–30% for HCP channels, smaller for DTC)
   - Run `python run.py --freq monthly --no-insights --geo` for monthly results
   - Add `_render_season_interactions(geo_ols)` to `app.py`: heatmap territory (rows) × channel (cols) → `season_lift_pct`, diverging colorscale (green = better in season, red = worse)
   - Place heatmap in Geo tab after the channel stacked-bar section, before the Bayesian section
   - Commit result files + code changes, open PR

2. **Response curves per territory** — visualise adstock+saturation curves by territory in dashboard (next roadmap item after season interactions)

3. **Hierarchical model in geo narrative** — `run_geo_insight_agent()` currently reads only Ridge and per-territory Bayesian results; extend to include hierarchical national ROI summaries in the narrative context
