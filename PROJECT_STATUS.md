# Pharma MMM Agent — Project Status

> Last updated: 2026-05-31
> Branch: `main` | Latest commit: `a9001eb`

---

## What This Project Is

An end-to-end **Marketing Mix Modelling (MMM) agent** for a pharma vaccine brand ("VaxBrand"). It ingests synthetic spend and script data across 12 HCP + DTC channels, fits both frequentist (Ridge) and Bayesian models, optimises the budget, generates an AI narrative, and surfaces everything through a Streamlit dashboard. A geo layer disaggregates all of the above across 6 US territories.

The project is designed so that **no Python code changes are needed for routine adjustments** — channel priors, model settings, corridor constraints, and territory parameters all live in `config/config.yaml`.

---

## Architecture

```
run.py / planner_agent.py
    └── run_mmm_pipeline()
          ├── analytics_agent (LLM orchestrated, or direct if no API key)
          │     1. apply_all_transforms_tool   → *_transformed.csv
          │     2. run_ols_mmm_tool            → *_ols_results.json
          │     3. run_budget_optimizer_tool   → *_budget_optimized.json
          │    [4. run_bayesian_mmm_tool]       → *_bayesian_results.json  (--bayesian flag)
          │
          └── insight_agent (LLM call → reports/*_insights.md)

run.py --geo
    ├── run_geo_ols_mmm_tool     → *_geo_ols_results.json
    └── run_geo_budget_optimizer_tool → *_geo_budget_optimized.json

app.py (Streamlit)
    Sidebar: Run Pipeline | Run Geo Pipeline
    Tabs: Overview | Ridge MMM | Bayesian MMM | Budget | Geo | Insights
```

---

## File Map

```
pharma-mmm-agent/
├── config/
│   └── config.yaml              # Single source of truth: channels, model params, territories
│
├── scripts/
│   └── generate_dataset.py      # Synthetic data generator (national + geo, weekly + monthly)
│
├── agents/
│   ├── planner_agent.py         # Orchestrates the full pipeline
│   ├── analytics_agent.py       # LangChain agent: runs transforms → OLS → optimizer
│   └── insight_agent.py         # LLM narrative: OLS + Bayesian + optimizer → Markdown report
│
├── tools/
│   ├── transforms.py            # geometric_adstock + hill_saturation (LangChain tools)
│   ├── ols_mmm_tool.py          # Ridge MMM with prior-contribution floor
│   ├── bayesian_mmm_tool.py     # PyMC Bayesian MMM with MCMC + HDI
│   ├── optimizer_tool.py        # SLSQP channel-mix optimizer (national)
│   ├── geo_mmm_tool.py          # Ridge MMM per territory (geo layer)
│   └── geo_optimizer_tool.py    # Two-level geo optimizer (channels + territories)
│
├── app.py                       # Streamlit dashboard (6 tabs)
├── run.py                       # CLI entry point
│
├── data/raw/
│   ├── mmm_weekly.csv           # 104 rows × 24 cols (2yr weekly, national)
│   ├── mmm_monthly.csv          # 36 rows × 26 cols (3yr monthly, national)
│   ├── mmm_weekly_geo.csv       # 624 rows × 26 cols (104wk × 6 territories)
│   ├── mmm_monthly_geo.csv      # 216 rows × 25 cols (36mo × 6 territories)
│   └── [*_transformed, *_ols_results, *_budget_optimized, *_bayesian_results]
│
└── reports/
    ├── mmm_weekly_insights.md
    └── mmm_monthly_insights.md
```

---

## What Is Built — Feature Inventory

### 1. Synthetic Data Generator (`scripts/generate_dataset.py`)

- **National datasets** — weekly (104 rows) and monthly (36 rows) with 12 spend channels, competitor spend, price index, scripts_written KPI
- **Realistic pharma DGP** — geometric adstock carryover, Hill saturation, HCP detailing lag (2wk), vaccine seasonality (Sep–Nov +40%), staggered congress pulses per channel, competitive share erosion, price elasticity
- **v2 properties** — channels decorrelated via independent noise + staggered pulses; rep_visits correlation with outcome 0.45–0.60; HCP ROI > DTC ROI (vaccine brand)
- **Geo datasets** — long-format (date × territory) for 6 US territories; spend scaled by `spend_share`, channel contribution scaled by `hcp_mult`/`dtc_mult`, baseline by `market_size / total_market`, seasonality by `season_str`

### 2. Transforms (`tools/transforms.py`)

- `geometric_adstock(spend, decay)` — carryover model: `result[t] = spend[t] + decay × result[t-1]`
- `hill_saturation(x, alpha)` — diminishing returns: `(x/x_max)^alpha`, output normalised 0–1
- Both exposed as LangChain `@tool` wrappers (`apply_adstock_tool`, `apply_saturation_tool`, `apply_all_transforms_tool`)

### 3. Ridge MMM (`tools/ols_mmm_tool.py`)

- Ridge regression (L2) with `alpha=1.0` (configurable) — prevents multicollinearity sign flips
- Non-negativity constraint on channel coefficients (standard MMM practice)
- **Prior-contribution floor** — channels Ridge cannot identify (collinear with seasonality dummies) receive contribution estimated from `prior_roi` config; clearly flagged as `"prior_estimate"` in output
- Blended ROI: 60% config prior + 40% model estimate (prevents implausible values)
- Control variables: competitor spend (negative), price index (negative), seasonality dummies, congress flag
- Output: per-channel ROI, contribution%, R², MAPE, baseline scripts → `*_ols_results.json`

### 4. Bayesian MMM (`tools/bayesian_mmm_tool.py`)

- PyMC model with MCMC sampling (2000 draws, 4 chains, target_accept=0.90)
- Informative `HalfNormal` priors on channel betas (calibrated from `prior_roi` × mean outcome)
- Full posterior distributions → 90% HDI (5th–95th percentile) per channel
- R̂ convergence diagnostics reported
- Solves the zero-contribution problem that Ridge has with collinear seasonal channels
- Output: per-channel posterior mean, HDI bounds, R², MAPE → `*_bayesian_results.json`

### 5. Budget Optimizer (`tools/optimizer_tool.py`)

- SLSQP constrained optimisation maximising total script response
- Objective: `Σ_ch roi_ch × hill_sat(adstock(spend_ch)) × spend_ch`
- Uses **fitted ROI** from OLS results (not config priors)
- Corridor constraints per channel: `[max_spend_decrease_factor × curr, max_spend_increase_factor × curr]` — models operational reality (KOL capacity, media contracts)
- Portfolio bounds: `[min_channel_share, max_channel_share]` × total budget
- Output: current vs optimal spend, Δ%, action flags, projected uplift% → `*_budget_optimized.json`

### 6. Geo MMM (`tools/geo_mmm_tool.py`)

- Fits separate Ridge MMM for each of 6 US territories on long-format geo CSV
- Applies adstock + saturation transforms in-memory per territory slice (no temp files)
- Same prior-contribution floor logic as national model
- Territory parameters from `config.yaml` (`hcp_mult`, `dtc_mult`, `season_str`): Northeast and Pacific show higher HCP responsiveness; Southeast and Pacific show stronger DTC response
- Current model fit: R² 0.895–0.958, MAPE 2.8–3.5% across all territories
- Output: per-territory channel ROIs, contributions, model fit → `*_geo_ols_results.json`

### 7. Geo Budget Optimizer (`tools/geo_optimizer_tool.py`)

- **Level 1 — Channel mix**: SLSQP per territory, same logic as national optimizer
- **Level 2 — Territory allocation**: SLSQP across territories maximising weighted ROI efficiency; suggests shifting national budget toward higher-ROI territories
- Separate territory corridors (`max_territory_increase_factor: 1.30`, `max_territory_decrease_factor: 0.80`) tighter than channel corridors — field infrastructure can't be redeployed across geographies as quickly as media line-items
- Output: territory share recommendations, per-channel breakdown per territory → `*_geo_budget_optimized.json`

### 8. LLM Insight Narrative (`agents/insight_agent.py`)

- Reads OLS results + (optionally) Bayesian results + optimizer output
- Structured system prompt: pharma commercial strategy analyst persona
- Generates: executive summary, channel interpretation, strategic recommendations, budget rationale
- Supports both Anthropic (Claude) and OpenAI (GPT-4o) via `config.yaml` `llm.provider`
- Output: `reports/mmm_{freq}_insights.md`

### 9. Streamlit Dashboard (`app.py`)

Six tabs:

| Tab | Contents |
|-----|----------|
| **Overview** | KPI cards, spend mix donut, scripts time-series with vaccine season bands |
| **Ridge MMM** | Channel contributions bar chart, ROI bar chart, control variable coefficients, full results table |
| **Bayesian MMM** | Contributions with 90% HDI error bars, OLS vs Bayesian ROI scatter, control posteriors |
| **Budget** | Current vs recommended overlay bar chart, reallocation table |
| **Geo** | Territory KPI cards, US choropleth (HCP responsiveness), channel contributions by territory, two-level optimizer results, per-territory expanders |
| **Insights** | Rendered Markdown narrative, download button |

Sidebar: frequency selector (weekly/monthly), Bayesian + insights toggles, Run Pipeline button, Run Geo Pipeline button, LLM config display.

### 10. Configuration (`config/config.yaml`)

| Section | Key parameters |
|---------|---------------|
| `llm` | provider (anthropic/openai), model, temperature |
| `data` | file paths, outcome_col, date_col |
| `channels` | 12 channels × adstock_decay, saturation, prior_roi, channel_type, label |
| `ols_model` | ridge_alpha, seasonality_dummies, congress_control, competitor_control, price_control, prior_contribution_weight |
| `bayesian_model` | draws, tune, chains, target_accept, prior_sigma |
| `optimizer` | budget multipliers, min/max channel share, channel corridors, territory corridors |
| `territories` | 6 territories × market_size, spend_share, hcp_mult, dtc_mult, season_str, states |
| `report` | output_dir, brand_name |

---

## How To Run

```bash
# 1. Generate all datasets (national + geo)
python scripts/generate_dataset.py

# 2. Run national pipeline (no LLM needed)
python run.py --no-insights

# 3. Run national + geo pipeline
python run.py --no-insights --geo

# 4. Run with Bayesian MMM (adds ~2 min)
python run.py --bayesian --no-insights

# 5. Run with LLM insights (needs ANTHROPIC_API_KEY or OPENAI_API_KEY in .env)
python run.py

# 6. Monthly frequency
python run.py --freq monthly --no-insights --geo

# 7. Launch dashboard
streamlit run app.py
```

---

## Commit History

| Commit | Description |
|--------|-------------|
| `a9001eb` | feat: geo MMM — territory dataset, Ridge per territory, two-level optimizer, Geo tab |
| `85d52d2` | fix: per-channel spend corridor constraints in optimizer |
| `21a9d8c` | docs: GIF demo in README |
| `d326a66` | feat: Streamlit dashboard (app.py) |
| `03e5d95` | feat: wire Bayesian results into insight narrative |
| `edb026a` | fix: optimizer budget scale bug |
| `1e748ae` | feat: Bayesian MMM via PyMC |
| `85f4c49` | fix: zero-contribution channels in Ridge MMM (prior floor) |
| `c3ddadd` | feat: competitor_spend + price_index controls |
| `1d0308a` | feat: dataset generator v2 (realistic pharma DGP) |

---

## TODO / Roadmap

### Completed ✅

- [x] Synthetic pharma dataset generator (weekly + monthly, realistic DGP)
- [x] Adstock + saturation transforms (geometric + Hill)
- [x] Ridge MMM with prior-contribution floor for unidentifiable channels
- [x] Bayesian MMM (PyMC, MCMC, 90% HDI per channel)
- [x] SLSQP budget optimizer with per-channel corridor constraints
- [x] LLM insight narrative (Claude + GPT-4o)
- [x] Streamlit dashboard (Overview, Ridge, Bayesian, Budget, Insights tabs)
- [x] Geo dataset generator (6 US territories, long-format, correct spend + ROI scaling)
- [x] Geo Ridge MMM (per-territory, in-memory transforms)
- [x] Two-level geo optimizer (channel mix + territory allocation)
- [x] Geo tab in dashboard (choropleth, channel stacked bar, optimizer table)
- [x] Separate territory budget corridors (1.30×/0.80×) vs channel corridors (2.50×/0.50×)
- [x] Geo Bayesian MMM (PyMC per territory, 90% HDI, R̂ convergence diagnostics)
- [x] Geo Bayesian dashboard section (Ridge vs Bayesian scatter, HDI error bars, uncertainty heatmap)

---

### In Progress / Current State 🔄

The geo MMM layer (Ridge + Bayesian + optimizer) is fully live. All 6 territories fit at Ridge R² 0.895–0.958 and Bayesian R̂ ≤ 1.005. The Bayesian layer adds 90% HDI per channel and an uncertainty heatmap showing where the model is least confident (useful for lift test prioritisation). The two-level optimizer recommends shifting budget toward Southeast (highest ROI efficiency at 0.378) and Northeast (HCP multiplier 1.20), capped at ±30%/20% per planning cycle.

Known limitations:
- Each territory is modelled **independently** — no information sharing across territories. A hierarchical model would improve Mountain's estimates most.
- Territory reallocation uses a **single efficiency score** (weighted avg ROI) — doesn't account for diminishing returns at the territory level.

---

### Next Steps 🔜

#### Near-term (incremental improvements to existing geo layer)

- [x] **Geo Bayesian MMM** — PyMC per territory, 90% HDI per channel, R̂ convergence, Ridge vs Bayesian scatter, uncertainty heatmap. All 6 territories converge (R̂ ≤ 1.005) in ~2 min.
- [ ] **Geo insight narrative** — Extend `insight_agent` to consume geo results and write a territory-level narrative (e.g. "Southeast underperforms its DTC multiplier — investigate creative quality").
- [ ] **What-if geo simulator** — Dashboard sliders to manually shift territory budget shares and instantly preview projected script change without re-running the optimizer.
- [ ] **Monthly geo pipeline** — `run.py --freq monthly --geo` works end-to-end but monthly geo results are not yet wired into the dashboard Geo tab (currently only loads weekly geo files).

#### Medium-term (modelling improvements)

- [ ] **Partial pooling / hierarchical geo model** — Share a national trend component across territories via a hierarchical PyMC model. Improves Mountain estimates most (small market, noisy signal). Classic Bayesian MMM approach for sub-national modelling.
- [ ] **Territory × time interactions** — Allow ROI multipliers to vary by season per territory (e.g. Mountain flu season is stronger Oct–Dec than Sep). Currently season_str is a flat annual multiplier.
- [ ] **Response curves per territory** — Visualise adstock+saturation response curves by territory in the dashboard to show where each territory sits on the diminishing returns curve.
- [ ] **Geo optimizer: diminishing returns at territory level** — Level-2 territory allocation currently uses a linear ROI efficiency score. Replace with a proper response function that accounts for marginal returns as territory budget grows.

#### Longer-term (new capabilities)

- [ ] **Real data ingestion** — Replace synthetic generator with a CSV upload flow in the dashboard; auto-detect date format, channel columns, and outcome column.
- [ ] **Incremental MMM (iROAS)** — Add geo-based lift test simulation: hold out one territory as control, estimate incremental ROI from natural experiment variation.
- [ ] **Scenario planner** — Given a target NRx goal, work backwards to the required budget (inverse of the optimizer).
- [ ] **PDF / slide export** — Export the insight narrative + charts as a board-ready PDF or PowerPoint-ready slide deck.
- [ ] **Automated refresh** — Scheduled pipeline run (weekly/monthly) that updates results and re-runs insight narrative when new spend data is dropped into `data/raw/`.
