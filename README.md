# 💊 Pharma MMM Agent — LangChain-Powered Marketing Mix Modelling for Life Sciences

> **An enterprise-grade, agentic Marketing Mix Modelling (MMM) pipeline built specifically for pharmaceutical and vaccine campaign analytics.**
> Built by a Senior Data Scientist with 6+ years of real-world pharma analytics experience at a global consultancy.

---

## 🧬 What Is This?

Most MMM tools are built for e-commerce or CPG. This one is built for **pharma**.

This template gives you a fully working **LangChain multi-agent system** that ingests vaccine / drug campaign spend data across **12 HCP and patient channels**, runs both **frequentist (Ridge/OLS)** and **Bayesian (PyMC)** MMM models, and generates **plain-English insights and budget recommendations** — the kind your commercial strategy team can actually act on.

If you've ever spent weeks wrangling claims data, fitting adstock curves, and then writing a 40-slide deck to explain what it means — this agent does that end-to-end.

---

## ⚡ What You Get

| Component | Description |
|---|---|
| `scripts/generate_dataset.py` | Synthetic vaccine campaign dataset generator — 12 channels, realistic pharma seasonality, competitor spend and price controls |
| `agents/planner_agent.py` | Orchestrator — runs the full pipeline, LLM-orchestrated or direct mode |
| `agents/analytics_agent.py` | LangChain agent that calls transforms → Ridge MMM → Bayesian MMM → optimiser |
| `agents/insight_agent.py` | Converts OLS + Bayesian results into a pharma-grade narrative with credible intervals |
| `tools/transforms.py` | Geometric adstock + Hill saturation transforms, wrapped as LangChain tools |
| `tools/ols_mmm_tool.py` | Ridge-regularised MMM with non-negativity constraint and prior-contribution floor |
| `tools/bayesian_mmm_tool.py` | Bayesian MMM via PyMC — informative priors, 90% HDI per channel, MCMC convergence diagnostics |
| `tools/optimizer_tool.py` | SLSQP budget reallocation optimiser using fitted ROIs |
| `notebooks/01_mmm_pipeline_walkthrough.ipynb` | Full MMM pipeline walkthrough — transforms, OLS, optimiser |
| `notebooks/02_agent_deep_dive.ipynb` | End-to-end agent run with annotated outputs |
| `data/raw/` | Ready-to-use synthetic datasets (weekly + monthly) with pre-run results |
| `config/config.yaml` | Channel definitions, model hyperparameters, adstock priors — single config for everything |

---

## 🏥 Pharma-Specific Features

- **12-channel HCP + DTC split** — rep visits, medical congress, journal ads, speaker programs, samples/coupons, HCP digital, HCP email, DTC TV, DTC digital, OOH, patient email, patient advocacy
- **Vaccine seasonality** — Sep–Nov peak, summer trough, Q1 moderate; applied per-channel with calibrated strength
- **Staggered congress pulses** — rep visits pulse in Aug+Oct; medical congress in Feb+May; speaker programs 1 month post-congress (Mar/Jun/Nov) — prevents artificial HCP channel collinearity
- **2-week detailing lag** — HCP channel contributions are shifted 2 weeks in the data-generating process, matching real pharma conversion dynamics
- **Competitor spend + price index** — two control variables included as model controls; Bayesian model uses informed negative priors on both
- **Prior-contribution floor** — for channels Ridge cannot separately identify, contributions are estimated from `prior_roi` in config, clearly flagged as `prior_estimate` vs `model` in all outputs
- **Dual-model insight narrative** — Claude reads both OLS and Bayesian results, cites credible intervals, and flags where models agree vs diverge

---

## 🤖 Agent Architecture

```
python run.py [--bayesian] [--no-insights] [--freq monthly]
    │
    ▼
┌─────────────────────────────────────────────────────┐
│                  Planner Agent                      │
│  Orchestrates pipeline; falls back to direct mode   │
│  when no API key is set (zero cost, same outputs)   │
└────────────┬────────────────────────────────────────┘
             │
    ┌────────┴────────┐
    ▼                 ▼
┌──────────────────┐  ┌─────────────────────────────┐
│  Analytics Agent │  │       Insight Agent          │
│  (LLM-orchestr.) │  │  Reads OLS + Bayesian JSON   │
│                  │  │  Writes pharma narrative with │
│  Step 1: Transforms  │  credible intervals + HDI    │
│  Step 2: Ridge MMM   └─────────────────────────────┘
│  Step 3: Optimiser
│  Step 4: Bayesian MMM (--bayesian flag)
└──────────────────┘
         │
         ▼
    LangChain Tools
    ├── apply_all_transforms_tool   → *_transformed.csv
    ├── run_ols_mmm_tool            → *_ols_results.json
    ├── run_bayesian_mmm_tool       → *_bayesian_results.json
    └── run_budget_optimizer_tool   → *_budget_optimized.json
```

---

## 🚀 Quickstart

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/pharma-mmm-agent.git
cd pharma-mmm-agent
pip install -r requirements.txt
```

### 2. Set your API key

Create a `.env` file in the project root:

```bash
# Anthropic (default — recommended)
ANTHROPIC_API_KEY=your-key-here

# Or OpenAI
OPENAI_API_KEY=your-key-here
```

Switch providers in `config/config.yaml` via `llm.provider: anthropic` or `llm.provider: openai`.

### 3. Generate the dataset (optional — sample data already included)

```bash
python scripts/generate_dataset.py
```

### 4. Run the pipeline

```bash
# OLS only — free, no API key needed, < 5 seconds
python run.py --no-insights

# Full run — OLS + Claude insight narrative (~$0.02)
python run.py

# Add Bayesian MMM — OLS + Bayesian + Claude narrative (~$0.02, +53 sec MCMC)
python run.py --bayesian

# Monthly data
python run.py --freq monthly --bayesian
```

### 5. Or explore the notebooks

```bash
jupyter lab notebooks/
```

---

## 📊 Sample Output

### Ridge MMM (from a live run)

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
HCP Programmatic Digital       hcp    $6,170   0.168    2.6%    prior*
DTC Digital & Social           dtc    $9,238   0.150    2.4%    prior*
...
```

### Bayesian MMM (PyMC, same data)

```
Bayesian MMM Results (R²=0.825, MAPE=6.0%)
4 chains × 2000 draws | R̂ max=1.001 ✓

Channel                        ROI    Contrib%  90% HDI (2yr scripts)
----------------------------------------------------------------------
Field Rep Visits               0.665   34.8%   [+190.5K – +488.6K]
Samples & Co-pay Coupons       0.578   21.8%   [+74.1K  – +358.2K]
HCP Programmatic Digital       0.490    8.2%   [+11.4K  – +168.2K]
Speaker Bureau Programs        0.787    6.0%   [+3.9K   – +153.6K]
Medical Congress & Symposia    0.910    5.1%   [+3.4K   – +134.7K]
...

Controls: competitor: -397.6/mean-unit ✓  price: -461.9/10pts ✓
```

### Budget optimisation

```
Budget Optimisation — $938.6K/week (no budget increase)
Projected uplift: +59.4% scripts with same spend

Speaker Bureau Programs        $39.8K  →  $328.5K   +725%  Increase ↑
Medical Congress & Symposia    $66.6K  →  $328.5K   +393%  Increase ↑
DTC Television                $238.9K  →   $18.8K    -92%  Reduce ↓
```

---

## 🗂️ Dataset Schema

### Weekly (`data/raw/mmm_weekly.csv`) — 104 rows × 24 columns

| Column | Type | Description |
|---|---|---|
| `date` | date | Week start (Monday) |
| `rep_visits` | float ($K) | HCP: Field rep detailing visits spend |
| `medical_congress` | float ($K) | HCP: Medical congress & symposia spend |
| `journal_advertising` | float ($K) | HCP: Journal advertising spend |
| `hcp_email` | float ($K) | HCP: Permission email to HCPs spend |
| `hcp_digital` | float ($K) | HCP: Programmatic HCP digital display spend |
| `speaker_programs` | float ($K) | HCP: KOL speaker bureau programs spend |
| `samples_coupons` | float ($K) | HCP: Sample drops & co-pay coupon spend |
| `dtc_tv` | float ($K) | DTC: Television advertising spend |
| `dtc_digital` | float ($K) | DTC: Digital & social patient advertising spend |
| `dtc_ooh` | float ($K) | DTC: Out-of-home advertising spend |
| `patient_email` | float ($K) | DTC: Patient CRM email campaigns spend |
| `patient_advocacy` | float ($K) | DTC: Patient advocacy partnerships spend |
| `competitor_spend` | float ($K) | Competing vaccine brand spend — model control |
| `price_index` | float (100=base) | Co-pay/price index; quarterly step-changes |
| `total_spend` | float ($K) | Sum of all 12 brand channel spends |
| `scripts_written` | int | Vaccine prescriptions written — outcome KPI |
| `nrx_index` | float (0–100) | Normalised new Rx index |
| `vaccine_season` | binary | 1 = Sep/Oct/Nov vaccine season |
| `congress_week` | binary | 1 = congress month (Feb/May/Oct) |

### Monthly (`data/raw/mmm_monthly.csv`) — 36 rows × 26 columns
Same columns plus `hcp_total_spend`, `dtc_total_spend`, `month_name`, `congress_month`.

Full schema: `data/raw/data_dictionary.csv`

---

## 🧠 Modelling Approaches

### Ridge MMM (frequentist)

- Geometric adstock per channel (decay rates in config, e.g. rep visits 0.6, medical congress 0.75)
- Hill saturation transform (normalised 0–1, alpha per channel)
- Ridge regression (alpha=1.0) with non-negativity constraint on channel coefficients
- Month dummies + congress flag + competitor spend + price index as controls
- **Prior-contribution floor**: channels that Ridge cannot separately identify (collinear with seasonality dummies) receive a contribution estimated from their `prior_roi`, flagged as `prior_estimate` in all outputs
- Suitable for: interpretability-first use cases, fast iteration, budget optimisation input

### Bayesian MMM (PyMC)

- Uses pre-computed adstocked + saturated features from the Ridge transform step
- `HalfNormal` priors on channel betas calibrated to `prior_roi × mean_scripts × 0.5` — ensures all 12 channels receive non-zero posteriors without a heuristic floor
- Informed negative priors on competitor (`beta_competitor ~ Normal(-mean_y × 0.03, ...)`) and price — correctly recovers negative posteriors even with weak raw signal
- 90% credible intervals (5th–95th percentile) on every channel contribution
- MCMC convergence via R̂: values < 1.05 indicate good chain mixing
- Runtime: ~53 seconds on a laptop (4 chains × 2000 draws)
- Suitable for: uncertainty quantification, stakeholder communication, channels with weak signal

---

## 🔧 Configuration

All parameters live in `config/config.yaml` — no code changes needed for routine tuning:

```yaml
llm:
  provider: anthropic        # openai | anthropic
  model: claude-sonnet-4-6

channels:
  rep_visits:
    adstock_decay: 0.60      # carryover rate (0=none, 1=full)
    saturation: 0.55         # diminishing returns (lower=faster saturation)
    prior_roi: 0.55          # expected ROI — used as Bayesian prior + Ridge floor
    channel_type: hcp
    label: "Field Rep Visits"
  # ... all 12 channels

ols_model:
  ridge_alpha: 1.0                  # regularisation; lower = less shrinkage
  prior_contribution_weight: 0.15   # share reserved for unidentifiable channels
  seasonality_dummies: true
  congress_control: true
  competitor_control: true          # includes competitor_spend as control
  price_control: true               # includes price_index as control

bayesian_model:
  draws: 2000
  tune: 1000
  chains: 4
  target_accept: 0.90

optimizer:
  min_channel_share: 0.02    # floor: no channel < 2% of budget
  max_channel_share: 0.35    # cap: no channel > 35% of budget
```

---

## 📦 Requirements

```
langchain>=0.2.0
langchain-openai>=0.1.0
langchain-anthropic>=0.1.0
pymc>=5.0.0
arviz>=0.17.0,<0.18     # 0.18+ requires Python 3.10
scipy>=1.9.0,<1.11      # scipy 1.11+ removed signal.gaussian used by arviz 0.17
scikit-learn>=1.3.0
statsmodels>=0.14.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.18.0
pyyaml>=6.0
python-dotenv>=1.0.0
jupyter>=1.0.0
```

---

## 👤 About the Author

Built by **Shubham Kumar** — Senior Data Scientist at Deloitte with 6+ years building production ML systems for pharma and life sciences. This template is distilled from real-world MMM projects spanning 20M+ patient profiles, vaccine campaigns across 247 zip codes, and commercial strategy work for some of the largest pharma brands globally.

- 🔗 [LinkedIn](https://linkedin.com/in/YOUR_HANDLE)
- 🐙 [GitHub](https://github.com/YOUR_USERNAME)
- 📧 shubham.mle@gmail.com

---

## 📄 Licence

MIT — use freely, modify, and build on top of this for your own projects.
If this saves you a week of work, consider leaving a ⭐ on GitHub.

---

## 🗺️ Roadmap

- [x] Frequentist Ridge MMM with non-negativity constraint
- [x] Bayesian MMM (PyMC) with informative priors and 90% HDI
- [x] Dual-model insight narrative — OLS + Bayesian side-by-side
- [x] Competitor spend + price index as model controls
- [x] Prior-contribution floor for unidentifiable channels
- [x] MCMC convergence diagnostics (R̂)
- [x] Anthropic Claude / OpenAI provider switching
- [ ] Streamlit dashboard for non-technical stakeholders
- [ ] Multi-territory / geo-level MMM support
- [ ] Integration with IQVIA / Symphony claims data schema
- [ ] Bayesian adstock decay fitting (within-model, not pre-computed)
- [ ] Payer mix and formulary access as additional controls

---

*Built for pharma data scientists who are tired of explaining adstock to stakeholders.*
