# 💊 Pharma MMM Agent — LangChain-Powered Marketing Mix Modelling for Life Sciences

[![CI](https://github.com/Shubh-kr/pharma-mmm-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/Shubh-kr/pharma-mmm-agent/actions/workflows/ci.yml)
[![Licence: MIT](https://img.shields.io/badge/Licence-MIT-yellow.svg)](#-licence)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](requirements.txt)

> **An enterprise-grade, agentic Marketing Mix Modelling (MMM) pipeline built specifically for pharmaceutical and vaccine campaign analytics.**
> Built by a Senior Data Scientist with 6+ years of real-world pharma analytics experience at a global consultancy.

![Pharma MMM Agent dashboard demo](assets/demo.gif)

---

## 🧬 What Is This?

Most MMM tools are built for e-commerce or CPG. This one is built for **pharma**.

This template gives you a fully working **LangChain multi-agent system** that ingests vaccine / drug campaign spend data across **12 HCP and patient channels**, runs both **frequentist (Ridge/OLS)** and **Bayesian (PyMC)** MMM models across **6 US territories**, and generates **plain-English insights and budget recommendations** — the kind your commercial strategy team can actually act on.

If you've ever spent weeks wrangling claims data, fitting adstock curves, and then writing a 40-slide deck to explain what it means — this agent does that end-to-end.

---

## ⚡ What You Get

### Core modelling

| Component | Description |
|---|---|
| `tools/transforms.py` | Geometric adstock + Hill saturation transforms, wrapped as LangChain tools |
| `tools/ols_mmm_tool.py` | Ridge MMM with non-negativity constraint, prior-contribution floor, per-period attribution time series |
| `tools/bayesian_mmm_tool.py` | Bayesian MMM via PyMC — informative priors, 90% HDI per channel, MCMC convergence diagnostics |
| `tools/optimizer_tool.py` | SLSQP budget reallocation optimiser using fitted ROIs and per-channel corridors |

### Geo layer (6 US territories)

| Component | Description |
|---|---|
| `tools/geo_mmm_tool.py` | Ridge MMM per territory — in-memory transforms, prior floor, post-hoc seasonal ROI split, response curve params |
| `tools/geo_bayesian_mmm_tool.py` | Bayesian MMM per territory (PyMC) — same as national, run independently for each territory |
| `tools/geo_hierarchical_mmm_tool.py` | Hierarchical Bayesian model — single joint PyMC model with partial pooling across all 6 territories via log-normal hyperpriors; Mountain territory borrows ROI estimates from larger markets |
| `tools/geo_optimizer_tool.py` | Two-level SLSQP — Level 1: channel mix per territory; Level 2: territory budget allocation |

### Agents & insights

| Component | Description |
|---|---|
| `agents/planner_agent.py` | Orchestrator — runs the full pipeline; falls back to direct mode when no API key |
| `agents/analytics_agent.py` | LangChain agent that calls transforms → Ridge MMM → Bayesian MMM → optimiser |
| `agents/insight_agent.py` | Converts OLS + Bayesian + Hierarchical results into a pharma-grade narrative with credible intervals, national hyperprior ROI rankings, and Mountain partial-pooling corrections |
| `scripts/generate_dataset.py` | Synthetic vaccine campaign dataset generator — 12 channels, realistic pharma seasonality, 6 geo territories |
| `tools/scenario_tool.py` | Calibrated response-curve solvers — target NRx → required budget (inverse SLSQP) and fixed budget → maximised scripts, plus efficiency-frontier sweep |
| `tools/incrementality_tool.py` | Scores every territory × channel pair for experiment priority (HDI uncertainty, Ridge/Bayesian disagreement, saturation headroom, spend materiality) and generates holdout test designs |
| `tools/db.py` | Optional PostgreSQL persistence layer — every pipeline run upserts results/raw data/narratives into `portfolio_db`; all functions fail soft to JSON/CSV files if no DB is reachable |

### Dashboard (9 tabs)

| Tab | Contents |
|-----|----------|
| **Overview** | KPI cards, spend mix donut, scripts time-series with vaccine season bands |
| **Ridge MMM** | Channel contributions bar chart, ROI bar chart, control variable coefficients, full results table |
| **Bayesian MMM** | Contributions with 90% HDI error bars, OLS vs Bayesian ROI scatter, control posteriors |
| **Attribution** | Per-period stacked area decomposition (baseline + channels + actual overlay), contribution waterfall |
| **Budget** | Current vs recommended overlay bar chart, reallocation table |
| **Geo** | Territory KPIs, US choropleth, channel contributions, response curves, seasonal ROI heatmap, Bayesian + Hierarchical sections, two-level optimizer, what-if simulator |
| **Scenario** | Target→Budget and Budget→Scripts solvers, channel allocation table, efficiency frontier chart |
| **Incrementality** | Territory × channel test-priority scoring, signal weight sliders, bubble/score-breakdown charts, holdout design cards |
| **Insights** | Rendered Markdown narrative, download button |

### Ops & tooling

| Component | Description |
|---|---|
| `Dockerfile` | Reproducible container image — ships with the sample dataset + model outputs baked in, so `docker run` opens straight into a populated dashboard |
| `.github/workflows/ci.yml` | GitHub Actions: lint (ruff), full pipeline smoke test (national + geo, direct mode), Docker build — runs on every push/PR to `main` |
| `tools/db.py` + `scripts/migrate_to_db.py` | Optional PostgreSQL persistence with a one-time migration script for existing JSON/CSV outputs |

---

## 🏥 Pharma-Specific Features

- **12-channel HCP + DTC split** — rep visits, medical congress, journal ads, speaker programs, samples/coupons, HCP digital, HCP email, DTC TV, DTC digital, OOH, patient email, patient advocacy
- **Vaccine seasonality** — Sep–Nov peak, summer trough, Q1 moderate; applied per-channel with calibrated strength
- **Staggered congress pulses** — rep visits pulse in Aug+Oct; medical congress in Feb+May; speaker programs 1 month post-congress — prevents artificial HCP channel collinearity
- **2-week detailing lag** — HCP contributions shifted 2 weeks in data-generating process, matching real pharma conversion dynamics
- **Competitor spend + price index** — two control variables included with informed negative priors in the Bayesian model
- **Prior-contribution floor** — for channels Ridge cannot separately identify, contributions estimated from `prior_roi` in config, clearly flagged `prior_estimate` vs `model` in all outputs
- **Seasonal ROI split** — post-hoc split of Ridge ROI into in-season / off-season using marginal efficiency ratios; surfaces saturation-driven ROI differences without adding collinear interaction features
- **Partial pooling** — hierarchical Bayesian model shares national hyperpriors across all territories; dramatically improves Mountain (small market) estimates that Ridge gets wrong
- **Attribution decomposition** — full per-period baseline + channel breakdown stored in JSON and visualised as stacked area + waterfall

---

## 🤖 Architecture

`run.py` is the single entrypoint. It orchestrates a **Planner Agent**, which either drives an LLM-based **Analytics Agent** (LangChain, tool-calling) or — with no API key configured — calls the same tools directly ("direct mode"). Every tool is a pure function that reads/writes JSON or CSV, so the pipeline, the dashboard, and the optional database are all reading from the same artefacts.

```mermaid
flowchart TD
    CLI["run.py --geo --geo-bayesian --geo-insights"] --> Planner

    subgraph Orchestration
        Planner["Planner Agent\n(orchestrates stages, falls back to direct mode)"]
        Analytics["Analytics Agent\n(LLM tool-calling, or direct-mode function calls)"]
        Insight["Insight Agent\n(LLM narrative writer)"]
        Planner --> Analytics
        Planner --> Insight
    end

    subgraph "Tools — national"
        T1["transforms.py\nadstock + saturation"]
        T2["ols_mmm_tool.py\nRidge MMM"]
        T3["bayesian_mmm_tool.py\nPyMC MMM"]
        T4["optimizer_tool.py\nSLSQP reallocation"]
    end

    subgraph "Tools — geo (6 territories)"
        G1["geo_mmm_tool.py"]
        G2["geo_bayesian_mmm_tool.py"]
        G3["geo_hierarchical_mmm_tool.py"]
        G4["geo_optimizer_tool.py"]
    end

    subgraph "Tools — planning"
        S1["scenario_tool.py\ntarget↔budget solvers"]
        S2["incrementality_tool.py\ntest-priority scoring"]
    end

    Analytics --> T1 --> T2 --> T4
    T2 --> T3
    Analytics --> G1 --> G4
    G1 --> G2 --> G3
    T2 -.-> S1
    T3 -.-> S2
    G1 -.-> S2

    T2 & T3 & T4 & G1 & G2 & G3 & G4 & S1 & S2 --> Files["data/raw/*.json / *.csv\n(ols_results, bayesian_results,\nbudget_optimized, geo_*, ...)"]
    Insight --> Reports["reports/*.md\n(national + geo narratives)"]

    Files -.->|optional, fails soft| DB[("PostgreSQL\nmmm.results / raw_data / geo_data\n/ narratives / run_log")]
    Files --> Dashboard["Streamlit Dashboard\n(app.py, 9 tabs)"]
    Reports --> Dashboard
    DB -.->|DB-first, file fallback| Dashboard
```

**Design principle:** the dashboard, the CI smoke test, and a data scientist debugging a channel's ROI are all reading the *same* JSON files the pipeline writes — there is no separate "serving" representation of the model output to drift out of sync with what was actually fit.

---

## 🚀 Quickstart

### 1. Clone and install

```bash
git clone https://github.com/Shubh-kr/pharma-mmm-agent.git
cd pharma-mmm-agent
pip install -r requirements.txt
```

### 2. Set your API key (optional)

```bash
# Anthropic (default — recommended)
ANTHROPIC_API_KEY=your-key-here
```

Without a key the pipeline runs in **direct mode** — all models fit and results are saved, but no LLM narrative is generated.

### 3. Run the pipeline

```bash
# National only — no API key needed, < 5 seconds
python run.py --no-insights

# National + geo Ridge MMM + geo optimizer
python run.py --no-insights --geo

# Add per-territory Bayesian MMM (~2 min)
python run.py --no-insights --geo-bayesian

# Add hierarchical Bayesian (~5 min)
python run.py --no-insights --geo-hierarchical

# Full run with AI narratives (needs API key)
python run.py --geo --geo-insights

# Monthly frequency
python run.py --freq monthly --no-insights --geo
```

### 4. Launch the dashboard

```bash
streamlit run app.py
```

### 5. Or run it in Docker (no local Python setup at all)

The image ships with the pre-generated sample dataset and model outputs already in `data/raw/`, so the dashboard is fully populated on first launch — no pipeline run required.

```bash
docker build -t pharma-mmm-agent .
docker run -p 8501:8501 pharma-mmm-agent
# → http://localhost:8501
```

To refit models against new data or config inside a running container:

```bash
docker exec -it <container-id> python run.py --no-insights --geo
```

To connect the container to a PostgreSQL instance instead of the file fallback, pass `MMM_DB_URL`:

```bash
docker run -p 8501:8501 -e MMM_DB_URL="postgresql://user:pass@host:5432/db" pharma-mmm-agent
```

---

## 📊 Sample Output

### Ridge MMM (national, weekly)

```
Ridge MMM Results (R²=0.951, MAPE=3.0%)
Observations: 104 weekly periods | Avg period spend: $938.6K
Channels: 5 model-identified, 7 prior-estimated

Channel                        Type  Spend $K   ROI    Contrib%  Source
-----------------------------------------------------------------------
Samples & Co-pay Coupons       hcp   $10,237   0.462   32.7%    model
Field Rep Visits               hcp   $21,414   0.532   27.5%    model
Speaker Bureau Programs        hcp    $4,139   0.630   12.9%    model
DTC Television                 dtc   $24,846   0.280   10.2%    model
Medical Congress & Symposia    hcp    $6,925   0.728    1.7%    model
...
```

### Hierarchical Bayesian (6 territories, national hyperpriors)

```
National channel ROI consensus (pooled across all territories):
Channel                          Nat ROI   σ_terr   Heterogeneity
----------------------------------------------------------------
Medical Congress & Symposia       0.910    1.472    low — national playbook
Speaker Bureau Programs           0.787    1.587    moderate
Field Rep Visits                  0.665    2.414    HIGH — territory-specific needed
Patient Advocacy Partnerships     0.613    1.442    low — national playbook
...

Mountain — Ridge vs Hierarchical ROI (partial-pooling benefit):
  Patient Advocacy     Ridge=0.210  Hier=0.613  Δ=+0.403
  HCP Digital          Ridge=0.168  Hier=0.490  Δ=+0.322
  Samples & Coupons    Ridge=0.198  Hier=0.578  Δ=+0.380
```

### Seasonal ROI split (geo, weekly)

```
Channel                   Territory   In-season ROI  Off-season   Lift %
------------------------------------------------------------------------
Field Rep Visits          Northeast      0.444          0.561     -20.9%
Medical Congress          Northeast      0.777          0.712      +9.1%
Samples & Coupons         Pacific        0.421          0.476     -11.5%
```

---

## 🗂️ Dataset Schema

### Weekly national (`data/raw/mmm_weekly.csv`) — 104 rows × 24 cols

| Column | Type | Description |
|---|---|---|
| `date` | date | Week start (Monday) |
| `rep_visits` … `patient_advocacy` | float ($K) | 12 brand channel spend columns |
| `competitor_spend` | float ($K) | Competing vaccine brand spend |
| `price_index` | float (100=base) | Co-pay/price index |
| `scripts_written` | int | Vaccine prescriptions written — outcome KPI |
| `vaccine_season` | binary | 1 = Sep/Oct/Nov |
| `congress_week` | binary | 1 = congress month |

### Geo (`data/raw/mmm_weekly_geo.csv`) — 624 rows × 26 cols

Long format (104 weeks × 6 territories). Same columns plus `territory`, `territory_label`, `territory_abbr`.

### Territories

| Key | Label | Market share | HCP mult | DTC mult |
|---|---|---|---|---|
| `northeast` | Northeast | 22% | 1.20 | 1.05 |
| `southeast` | Southeast | 18% | 1.05 | 1.15 |
| `midwest` | Midwest | 20% | 1.00 | 1.00 |
| `southwest` | Southwest | 16% | 0.90 | 1.10 |
| `mountain` | Mountain | 8% | 0.85 | 0.90 |
| `pacific` | Pacific | 16% | 1.15 | 1.20 |

---

## 🧠 Modelling Approaches

### Ridge MMM (frequentist)

- Geometric adstock per channel → Hill saturation → Ridge regression (α=1.0, configurable)
- Non-negativity constraint on channel coefficients
- Prior-contribution floor for channels collinear with seasonality dummies
- Blended ROI: 60% config prior + 40% model estimate
- **Attribution time series**: per-period `baseline_timeseries` + `contribution_timeseries` per channel stored in JSON for dashboard decomposition

### Bayesian MMM (PyMC)

- `HalfNormal` priors on channel betas calibrated from `prior_roi × mean_y × 0.5`
- Informed negative priors on competitor and price controls
- 90% HDI (5th–95th percentile) on every channel
- Runtime: ~53 seconds national, ~20 min for 6 territories

### Hierarchical Bayesian MMM (PyMC)

- Single joint model across all 6 territories
- Log-normal non-centred hyperpriors on channel betas — forces positivity, handles the 3× market size range naturally
- `national_hyperpriors` block: `mu_beta_mean`, `sigma_terr_mean`, `national_roi_mean` per channel
- `sigma_terr_mean` measures territory heterogeneity — HIGH (>2) means territory-specific strategy required
- Mountain borrows strength from all other territories; corrects large under-estimates from the independent Ridge model
- Runtime: ~5 minutes; weekly R̂=1.005, monthly R̂=1.004

### Seasonal ROI Split

- Post-hoc computation using marginal efficiency ratio `(avg_sat/avg_spend)` per season
- Scales the already-blended `estimated_roi` proportionally, preserving the observation-weighted mean
- Avoids the collinearity problems of explicit interaction features

---

## 🧪 Evaluation Methodology

MMM has no ground-truth label — nobody knows the "true" incremental scripts a channel produced. Because of that, this project leans on **model diagnostics + cross-model agreement + explicit uncertainty**, rather than a single accuracy number, to judge whether a result is trustworthy enough to act on.

### 1. Fit quality (per model)
- **Ridge MMM**: R² and MAPE reported on in-sample fit (`Ridge MMM Results (R²=0.951, MAPE=3.0%)`); non-negativity constraint checked; residuals inspected for autocorrelation from the adstock/seasonality specification.
- **Bayesian MMM**: MCMC convergence via **R̂** (Gelman-Rubin) per parameter — the hierarchical model targets R̂ < 1.01 (weekly R̂=1.005, monthly R̂=1.004) — plus effective sample size and 90% HDI width as a direct uncertainty measure per channel.

### 2. Cross-model agreement as a validity signal
The Ridge and Bayesian models are fit **independently**, on the same transformed data, with different assumptions (point estimate + regularisation vs. full posterior with informative priors). Where they agree, confidence in the channel's ROI is high. Where they disagree materially, that disagreement is itself surfaced as a signal — it drives the **Incrementality tab**'s ranking of which territory × channel pairs most need a real holdout experiment, rather than being hidden or averaged away.

### 3. Held-out and out-of-sample structure
- The synthetic data generator (`scripts/generate_dataset.py`) is **seeded and deterministic** — re-running it reproduces byte-identical CSVs (verified in CI), which makes every reported metric reproducible from a clean checkout.
- Territory-level models (`geo_mmm_tool.py`, per-territory Bayesian) are effectively an out-of-sample check on the national model: territory spend and baseline scripts are scaled so that territory-level totals should aggregate back to the national number, and systematic mismatches (e.g. Mountain's small-sample noise) are what motivated the hierarchical partial-pooling model in the first place.
- The **efficiency frontier** in the Scenario tab plots the current spend mix against the modelled frontier — the current point sitting *above* the curve is a built-in sanity check that the optimizer is finding a genuinely better allocation, not just a different one.

### 4. What this does *not* claim
- This is a **correlational, model-based estimate** of channel contribution, not a causal, experimentally-validated one. The Incrementality tab exists specifically to prioritise where a real lift test would most reduce that uncertainty.
- ROI numbers blend a config-defined prior with the model estimate (`prior-estimate` vs `model` is labelled in every output table) — channels the model can't separately identify are never silently reported as model-driven.
- All current data is **synthetic**, calibrated to plausible pharma spend/response ranges but not fit to real claims data. See [Responsible AI & Auditability](#-responsible-ai--auditability) for how this is meant to be used with real data.

---

## 🛡️ Responsible AI & Auditability

Pharma commercial analytics is a regulated context — budget and channel decisions trace back to claims data, and any AI-assisted recommendation needs to be explainable and reviewable after the fact. This project is built with that in mind:

### Decision support, not autonomous decisioning
The LLM never touches spend. The **Analytics Agent** calls fixed, deterministic tools (Ridge/Bayesian MMM, SLSQP optimizer) — the LLM's only role is orchestration (which tool to call next) and, separately, narrative generation. The **Insight Agent** is a pure text-generation step that reads already-computed JSON and writes a plain-English summary of it; it cannot alter a model coefficient, a ROI estimate, or a budget recommendation. Every number a stakeholder acts on comes from an auditable statistical model, not from the LLM.

### Every output is provenance-labelled
Channel contributions are explicitly tagged `model` vs. `prior_estimate` in every table and JSON payload — a reviewer can immediately see which numbers came from the fitted regression and which are a config-defined fallback for channels the model couldn't separately identify (`ols_mmm_tool.py`). The same applies to Ridge-vs-Bayesian disagreement, which is surfaced (Incrementality tab) rather than silently reconciled.

### Full run history, not just the latest result
`tools/db.py` (optional PostgreSQL layer) keeps `mmm.run_log` — one row per pipeline stage execution with timestamp, freq, stage, and status — plus every historical result upserted with a `created_at`. Nothing overwrites silently without a record that a run happened. When no database is configured, the same guarantee holds at the filesystem level: every JSON result includes its own generation parameters, so a result file is self-describing without needing to reconstruct which config produced it.

### Reproducibility by construction
- `config/config.yaml` is the single source of truth for every channel prior, adstock decay, saturation curve, and optimizer bound — there are no hidden magic numbers in the modelling code, and every parameter change is a version-controlled diff.
- The synthetic dataset generator is seeded; identical config + identical seed reproduces byte-identical input data (checked in CI), so a reported result can always be regenerated from the repo alone.
- The LLM narrative step (`insight_agent.py`) sets an explicit `max_tokens` for the Anthropic client specifically because the default (1024) was found to silently truncate the longer geo narrative mid-sentence — a truncated report failing loudly (or not being generated) is preferred over a silently incomplete one.

### No PII / PHI in this repository
All shipped data (`data/raw/*.csv`) is synthetically generated — there is no real patient, HCP, or claims data anywhere in this repo, and none is required to evaluate the modelling approach. The intended adoption path for real data is aggregate, de-identified claims (e.g. via an IQVIA/Symphony-style territory-week extract, see [Phase 2](#-phase-2--in-progress)) — patient-level data is out of scope for this pipeline by design; it operates on aggregated weekly/monthly territory totals only.

### Known limitations (read before using with real data)
- Estimates are **correlational**, not causal — see [Evaluation Methodology](#-evaluation-methodology) for why the Incrementality tab exists.
- The Bayesian priors (`prior_roi` per channel in `config.yaml`) encode analyst judgment; a materially wrong prior will bias the blended ROI even when the model technically converges (R̂ ≈ 1). Priors should be reviewed by a domain expert before results are used to justify a real budget shift.
- This template has not been validated against real-world claims data or submitted through any regulatory or medical-legal review process — it is a modelling framework and starting point, not a validated production system.

---

## 🔧 Configuration

All parameters live in `config/config.yaml`:

```yaml
llm:
  provider: anthropic        # openai | anthropic
  model: claude-sonnet-4-6

channels:
  rep_visits:
    adstock_decay: 0.60
    saturation: 0.55
    prior_roi: 0.55
    channel_type: hcp
    label: "Field Rep Visits"

ols_model:
  ridge_alpha: 1.0
  prior_contribution_weight: 0.15
  season_interactions: true      # post-hoc seasonal ROI split

bayesian_model:
  draws: 2000
  tune: 1000
  chains: 4
  geo_draws: 1000                # per-territory Bayesian
  hier_draws: 600                # hierarchical joint model
  hier_chains: 2

optimizer:
  max_spend_increase_factor: 2.5
  max_spend_decrease_factor: 0.5
  max_territory_increase_factor: 1.30   # tighter — field ops can't redeploy fast
  max_territory_decrease_factor: 0.80

territories:
  northeast:
    market_size: 4400
    spend_share: 0.22
    hcp_mult: 1.20
    dtc_mult: 1.05
    season_str: 1.05
    states: [NY, NJ, CT, MA, RI, VT, NH, ME, PA]
```

---

## 📦 Requirements

```
langchain>=0.2.0
langchain-anthropic>=0.1.0
langchain-openai>=0.1.0
pymc>=5.0.0
arviz>=0.17.0,<0.18
scipy>=1.9.0,<1.11
scikit-learn>=1.3.0
statsmodels>=0.14.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.18.0
streamlit>=1.35.0
pyyaml>=6.0
python-dotenv>=1.0.0
psycopg2-binary>=2.9.0   # optional — only used if MMM_DB_URL / PostgreSQL is configured
```

---

## 🗺️ Phase 1 — Completed ✅

**National MMM pipeline**
- [x] Synthetic pharma dataset generator (weekly + monthly, realistic DGP)
- [x] Geometric adstock + Hill saturation transforms
- [x] Ridge MMM with non-negativity constraint and prior-contribution floor
- [x] Bayesian MMM (PyMC, MCMC, 90% HDI per channel)
- [x] SLSQP budget optimizer with per-channel corridor constraints
- [x] LLM insight narrative (Claude + GPT-4o)
- [x] Attribution decomposition: per-period stacked area + contribution waterfall

**Geo MMM pipeline (6 US territories)**
- [x] Geo dataset generator (long format, territory-scaled spend + ROI)
- [x] Geo Ridge MMM per territory with in-memory transforms
- [x] Post-hoc seasonal ROI split (in-season vs off-season marginal efficiency)
- [x] Bayesian MMM per territory (PyMC, 90% HDI, R̂ convergence)
- [x] Hierarchical Bayesian MMM — partial pooling with national hyperpriors
- [x] Two-level geo optimizer (channel mix + territory allocation)
- [x] Geo LLM narrative with hierarchical hyperprior context

**Streamlit dashboard (7 tabs)**
- [x] Overview, Ridge MMM, Bayesian MMM, Attribution, Budget, Geo, Insights
- [x] Response curves per territory (Hill saturation curves, operating-point dots)
- [x] Seasonal ROI heatmap (territory × channel diverging colorscale)
- [x] What-if geo budget simulator (6 territory sliders + preset buttons)
- [x] Hierarchical Bayesian section in Geo tab

---

## 🔭 Phase 2 — In Progress

**Measurement & experimentation**
- [x] **Incrementality testing planner** — `tools/incrementality_tool.py` scores every territory × channel pair on Bayesian HDI uncertainty, Ridge vs Bayesian disagreement, saturation headroom, and spend materiality; generates ranked holdout test designs (approach, duration, depth, power note) in the Incrementality tab
- [ ] **iROAS estimator** — Geo-based lift test simulation using the hierarchical model's territory baselines as counterfactuals

**Planning tools**
- [x] **Scenario planner** — `tools/scenario_tool.py`: calibrated response-curve solvers for target NRx → required budget (inverse SLSQP) and fixed budget → maximised scripts (forward SLSQP), plus a 25-point efficiency frontier, in the Scenario tab
- [ ] **Budget scenario comparison** — Side-by-side view of 2–3 named scenarios (current / optimizer / custom) with projected scripts for each; useful for budget cycle presentations

**Data & operationalisation**
- [x] **PostgreSQL integration** — `tools/db.py`: DB-first data layer (`mmm` schema — results, raw_data, geo_data, narratives, run_log), idempotent upserts, transparent fallback to JSON/CSV files when no DB is reachable
- [x] **Production packaging** — Dockerfile + GitHub Actions CI (lint, pipeline smoke test, Docker build) so the pipeline runs reproducibly outside a local dev environment
- [ ] **Real data ingestion** — CSV upload flow in the dashboard; auto-detect date format, channel columns, outcome column; validation against expected schema
- [ ] **Automated refresh** — Scheduled pipeline run (weekly/monthly) that refit models and regenerates narratives when new spend data is dropped into `data/raw/`

**Output & reporting**
- [ ] **PDF / slide export** — Export the insight narrative + charts as a board-ready PDF or PowerPoint-ready slide deck
- [ ] **IQVIA / Symphony schema adapter** — Pre-built column mapping for standard pharma claims data providers

---

## 👤 About the Author

Built by **Shubham Kumar** — Senior Data Scientist at Deloitte with 6+ years building production ML systems for pharma and life sciences. This template is distilled from real-world MMM projects spanning 20M+ patient profiles, vaccine campaigns across 247 zip codes, and commercial strategy work for some of the largest pharma brands globally.

- 🔗 [LinkedIn](https://linkedin.com/in/shabam23)
- 🐙 [GitHub](https://github.com/Shubh-kr)
- 📧 shubham.mle@gmail.com

---

## 📄 Licence

MIT — use freely, modify, and build on top of this for your own projects.
If this saves you a week of work, consider leaving a ⭐ on GitHub.
