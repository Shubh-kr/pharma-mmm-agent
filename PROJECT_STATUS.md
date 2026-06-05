# Pharma MMM Agent — Project Status

> Last updated: 2026-06-05
> Branch: `main` | Phase 1 complete

---

## What This Project Is

An end-to-end **Marketing Mix Modelling (MMM) agent** for a pharma vaccine brand ("VaxBrand"). It ingests synthetic spend and script data across 12 HCP + DTC channels, fits frequentist (Ridge), Bayesian, and Hierarchical Bayesian models, optimises the budget at both national and territory level, generates AI narratives, and surfaces everything through a Streamlit dashboard.

The project is designed so that **no Python code changes are needed for routine adjustments** — channel priors, model settings, corridor constraints, and territory parameters all live in `config/config.yaml`.

---

## Architecture

```
run.py
  ├── run_mmm_pipeline()                     [national]
  │     ├── apply_all_transforms_tool        → *_transformed.csv
  │     ├── run_ols_mmm_tool                 → *_ols_results.json  (+ attribution timeseries)
  │     ├── run_budget_optimizer_tool        → *_budget_optimized.json
  │     ├── [--bayesian] run_bayesian_mmm_tool → *_bayesian_results.json
  │     └── [LLM] insight_agent             → reports/*_insights.md
  │
  └── [--geo / --geo-bayesian / --geo-hierarchical / --geo-insights]
        ├── run_geo_ols_mmm_tool             → *_geo_ols_results.json  (+ seasonal ROI split + attribution)
        ├── run_geo_budget_optimizer_tool    → *_geo_budget_optimized.json
        ├── [--geo-bayesian] run_geo_bayesian_mmm_tool       → *_geo_bayesian_results.json
        ├── [--geo-hierarchical] run_geo_hierarchical_mmm_tool → *_geo_hierarchical_results.json
        └── [--geo-insights] run_geo_insight_agent           → reports/*_geo_insights.md

app.py (Streamlit — 7 tabs)
  Overview | Ridge MMM | Bayesian MMM | Attribution | Budget | Geo | Insights
```

---

## File Map

```
pharma-mmm-agent/
├── config/
│   └── config.yaml                    # Single source of truth
│
├── scripts/
│   └── generate_dataset.py            # National + geo synthetic data generator
│
├── agents/
│   ├── planner_agent.py               # Pipeline orchestrator (LLM or direct mode)
│   ├── analytics_agent.py             # LangChain agent: transforms → OLS → optimizer
│   └── insight_agent.py               # LLM narrative: national + geo + hierarchical
│
├── tools/
│   ├── transforms.py                  # geometric_adstock + hill_saturation
│   ├── ols_mmm_tool.py                # Ridge MMM + prior floor + attribution timeseries
│   ├── bayesian_mmm_tool.py           # National Bayesian MMM (PyMC, MCMC, HDI)
│   ├── optimizer_tool.py              # National SLSQP optimizer
│   ├── geo_mmm_tool.py                # Geo Ridge MMM per territory + seasonal ROI split + attribution
│   ├── geo_bayesian_mmm_tool.py       # Geo Bayesian MMM per territory (PyMC)
│   ├── geo_hierarchical_mmm_tool.py   # Hierarchical Bayesian (single joint PyMC, partial pooling)
│   └── geo_optimizer_tool.py          # Two-level geo optimizer
│
├── app.py                             # Streamlit dashboard (7 tabs)
├── run.py                             # CLI entry point
│
├── data/raw/
│   ├── mmm_weekly.csv                 # 104 rows × 24 cols (national weekly)
│   ├── mmm_monthly.csv                # 36 rows × 26 cols (national monthly)
│   ├── mmm_weekly_geo.csv             # 624 rows (104wk × 6 territories)
│   ├── mmm_monthly_geo.csv            # 216 rows (36mo × 6 territories)
│   └── [pre-run result JSONs for all pipelines]
│
└── reports/
    ├── mmm_weekly_insights.md
    ├── mmm_monthly_insights.md
    └── [geo insight reports when run with API key]
```

---

## Phase 1 — Complete ✅

### National MMM

| Feature | Tool / File | Notes |
|---------|-------------|-------|
| Synthetic data generator | `scripts/generate_dataset.py` | Weekly + monthly; 12 channels; realistic pharma DGP (adstock carryover, Hill saturation, HCP detailing lag, vaccine seasonality, staggered congress pulses, competitive erosion, price elasticity) |
| Adstock + saturation transforms | `tools/transforms.py` | Geometric carryover (`decay`), Hill power curve (`alpha`); normalised 0–1 |
| Ridge MMM | `tools/ols_mmm_tool.py` | Non-negativity constraint; prior-contribution floor for collinear channels; blended ROI (60% prior / 40% model); **per-period attribution timeseries** stored in JSON |
| Bayesian MMM | `tools/bayesian_mmm_tool.py` | PyMC; HalfNormal channel priors; informed negative priors on competitor + price; 90% HDI; R̂ diagnostics; ~53 sec |
| Budget optimizer | `tools/optimizer_tool.py` | SLSQP; channel corridors (±50%/+150%); portfolio share bounds |
| LLM insight narrative | `agents/insight_agent.py` | National narrative; supports Anthropic + OpenAI |
| Attribution decomposition | `app.py → tab_attribution()` | Stacked area (baseline + channels + actual overlay) + contribution waterfall |

### Geo MMM (6 US territories)

| Feature | Tool / File | Notes |
|---------|-------------|-------|
| Geo dataset generator | `scripts/generate_dataset.py` | Long format; spend scaled by `spend_share`; ROI scaled by `hcp_mult`/`dtc_mult`; baseline by `market_size / total`; seasonality by `season_str` |
| Geo Ridge MMM | `tools/geo_mmm_tool.py` | Per-territory in-memory transforms; same prior floor; `adstock_x_max_k` + `avg_saturated` for response curves; **per-period attribution timeseries**; weekly R²=0.895–0.958, MAPE=2.8–3.5% |
| Seasonal ROI split | `tools/geo_mmm_tool.py` | Post-hoc marginal efficiency ratio per season; scales blended ROI preserving weighted mean; Field Rep Visits -21–23% in-season (saturation), Medical Congress +8–12% (event concentration) |
| Geo Bayesian MMM | `tools/geo_bayesian_mmm_tool.py` | Independent PyMC per territory; 90% HDI; R̂ convergence; ~20 min for 6 territories |
| Hierarchical Bayesian MMM | `tools/geo_hierarchical_mmm_tool.py` | Single joint PyMC model; log-normal non-centred hyperpriors (forced positivity, handles 3× market size range); `national_hyperpriors` block with `mu_beta_mean`, `sigma_terr_mean`, `national_roi_mean` per channel; weekly R̂=1.005, monthly R̂=1.004; ~5 min |
| Two-level geo optimizer | `tools/geo_optimizer_tool.py` | Level 1: channel mix per territory; Level 2: territory allocation (tighter corridors ±30%/20%); ROI efficiency ranking drives territory shifts |
| Geo LLM narrative | `agents/insight_agent.py` | Auto-detects hierarchical JSON; injects hyperprior ROI ranking, `sigma_terr_mean` heterogeneity flags, Mountain partial-pooling correction table into LLM context |

### Dashboard (7 tabs)

| Tab | Key visuals |
|-----|-------------|
| **Overview** | KPI cards; spend mix donut; scripts time-series with vaccine season bands |
| **Ridge MMM** | Channel contributions bar chart; ROI bar chart; control variable coefficients; full results table |
| **Bayesian MMM** | Contributions with 90% HDI error bars; OLS vs Bayesian ROI scatter; control posteriors |
| **Attribution** | Stacked area decomposition (baseline grey, 12 channel layers, dotted actual overlay); contribution waterfall (spacer-bar technique for per-channel colors); KPI metrics; geo territory selector with same charts |
| **Budget** | Current vs recommended horizontal bar chart; reallocation table with change% and action flags |
| **Geo** | Territory KPI cards; US choropleth (HCP responsiveness); channel stacked bar by territory; response curves per territory (channel dropdown, Hill saturation curves, operating-point dots, 80% reference line); seasonal ROI heatmap (territory × channel, diverging colorscale); Bayesian section (Ridge vs Bayesian scatter, HDI error bars, uncertainty heatmap); Hierarchical section (hyperprior table); two-level optimizer bar chart + table; what-if simulator (6 territory sliders + preset buttons); AI geo narrative |
| **Insights** | Rendered Markdown narrative; download button |

---

## Phase 2 — Planned

### Measurement & experimentation

- **Incrementality testing planner** — Translate Bayesian HDI width + Ridge vs Bayesian ROI disagreement into a ranked list of geo holdout / lift test candidates. Operationalises the uncertainty already in the model into experiment design recommendations.
- **iROAS estimator** — Use the hierarchical model's territory baselines as counterfactuals for a geo-based natural experiment estimator.

### Planning tools

- **Scenario planner** — Given a target NRx goal, work backwards to required total budget and channel mix (inverse of the current optimizer).
- **Budget scenario comparison** — Save and compare 2–3 named scenarios (current / optimizer / custom) side-by-side with projected scripts; useful for leadership budget cycle presentations.

### Data & operationalisation

- **Real data ingestion** — CSV upload flow in the Streamlit sidebar; auto-detect column schema; validation against expected format; allow custom outcome column name.
- **Automated refresh** — Scheduled pipeline run (weekly/monthly) that refits models and regenerates narratives when new spend data is dropped.

### Output & reporting

- **PDF / slide export** — Export insight narrative + key charts as a board-ready PDF or PowerPoint deck.
- **IQVIA / Symphony schema adapter** — Pre-built column mapping for standard pharma claims data providers.

---

## Key Design Decisions (worth preserving)

| Decision | Why |
|----------|-----|
| Prior-contribution floor in Ridge | Monthly seasonality dummies are collinear with seasonal HCP channels; Ridge attributes Sep-Nov variance to dummies, zeroing those channels. Bayesian solves properly with priors; Ridge uses a heuristic floor clearly flagged in output. |
| Post-hoc seasonal ROI split (not interaction features) | Adding `sat_ch × vaccine_season` features caused -108,604% gammas on Mountain (36 monthly obs, underdetermined). The ratio-based post-hoc approach captures the saturation-driven seasonal difference without any model change. |
| Ratio-based seasonal ROI scaling | Original approach re-clipped raw betas against `prior × 2.5` ceiling → both in/off-season values hit the same ceiling → all lifts were 0.0%. The marginal efficiency ratio `eff_in/eff_off` scales the already-blended ROI and survives the blending step. |
| Log-normal non-centred hyperpriors in hierarchical model | First attempt used Normal non-centred for territory baselines → Mountain and Midwest went negative. Log-normal forces positivity and handles the 3× market size range naturally. |
| Waterfall via stacked bars (not `go.Waterfall`) | `go.Waterfall` in Plotly only supports uniform `increasing/decreasing` marker colors. Used transparent spacer bars + `barmode=stack` to achieve per-channel coloring. |
| `max_tokens=8096` in `_build_llm()` | Default LangChain Anthropic limit is 1024, which truncated the geo narrative at ~27 lines. Always keep this override. |
| Steady-state adstock for response curves | Dashboard doesn't reload the geo CSV; uses `x_ss = x_raw / (1 - decay)` as an adstock proxy so curves can be rendered from JSON alone. |
