# 💊 Pharma MMM Agent — LangChain-Powered Marketing Mix Modelling for Life Sciences

> **An enterprise-grade, agentic Marketing Mix Modelling (MMM) pipeline built specifically for pharmaceutical and vaccine campaign analytics.**
> Built by a Senior Data Scientist with 6+ years of real-world pharma analytics experience at a global consultancy.

---

## 🧬 What Is This?

Most MMM tools are built for e-commerce or CPG. This one is built for **pharma**.

This template gives you a fully working **LangChain multi-agent system** that ingests vaccine / drug campaign spend data across **12 HCP and patient channels**, runs both **frequentist (OLS/Panel)** and **Bayesian (PyMC)** MMM models, and generates **plain-English insights and budget recommendations** — the kind your commercial strategy team can actually act on.

If you've ever spent weeks wrangling claims data, fitting adstock curves, and then writing a 40-slide deck to explain what it means — this agent does that end-to-end.

---

## ⚡ What You Get

| Component | Description |
|---|---|
| `scripts/generate_dataset.py` | Synthetic vaccine campaign dataset generator (12 channels, weekly + monthly) |
| `agents/planner_agent.py` | Orchestrator — decomposes MMM task into sub-goals |
| `agents/analytics_agent.py` | Runs adstock, saturation, OLS and Bayesian MMM models |
| `agents/insight_agent.py` | Converts model output to pharma-grade narrative |
| `tools/` | Modular LangChain tools: adstock, saturation, regression, PyMC, Plotly |
| `notebooks/01_data_exploration.ipynb` | EDA walkthrough of the sample dataset |
| `notebooks/02_mmm_walkthrough.ipynb` | Step-by-step MMM modelling notebook |
| `notebooks/03_agent_demo.ipynb` | End-to-end agent run with annotated outputs |
| `data/raw/` | Ready-to-use synthetic datasets (weekly + monthly) |
| `reports/` | Sample auto-generated HTML/PDF report |
| `config/config.yaml` | Channel definitions, model hyperparameters, adstock priors |

---

## 🏥 Pharma-Specific Features

This is not a generic MMM template. It is built around how pharma marketing actually works:

- **12-channel HCP + DTC split** — rep visits, medical congress, journal ads, speaker programs, samples/coupons, HCP digital, HCP email, DTC TV, DTC digital, OOH, patient email, patient advocacy
- **Vaccine seasonality** — built-in Sep–Nov peak, summer trough, Q1 moderate modelling
- **Medical congress pulse** — automatic spend uplift windows for ACIP (Feb), ASHP (May), IDWeek (Oct)
- **Adstock + saturation transforms** — per-channel geometric decay and Hill saturation, calibrated to pharma norms
- **HCP vs DTC attribution split** — separate contribution analysis for your field force vs patient channels
- **Scripts written as KPI** — outcome modelled as vaccine prescriptions, mappable to your own Rx / NRx data

---

## 🤖 Agent Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────┐
│         Planner Agent               │
│  Decomposes task → routes sub-goals │
└────────────┬────────────────────────┘
             │
    ┌─────────┴──────────┐
    ▼                    ▼
┌──────────────┐   ┌─────────────────────┐
│  Analytics   │   │   Insight Agent     │
│  Agent       │   │   Plain-English     │
│  OLS/Bayesian│   │   pharma narrative  │
│  MMM models  │   │   + recommendations │
└──────┬───────┘   └────────┬────────────┘
       │                    │
       ▼                    ▼
  LangChain Tools       LLM (GPT-4 / Claude)
  ├── adstock_tool
  ├── saturation_tool
  ├── ols_mmm_tool
  ├── bayesian_mmm_tool
  ├── budget_optimizer_tool
  └── plotly_report_tool
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

```bash
cp .env.example .env
# Add your OpenAI or Anthropic key to .env
```

### 3. Generate the sample dataset

```bash
python scripts/generate_dataset.py
```

### 4. Run the agent

```bash
python agents/planner_agent.py \
  --data data/raw/mmm_weekly.csv \
  --freq weekly \
  --model ols
```

### 5. Or explore the notebooks

```bash
jupyter lab notebooks/
```

---

## 📊 Sample Output

The agent produces three outputs automatically:

**1. Channel ROI table**
```
Channel               | Spend ($K) | Contribution | ROI  | Recommendation
----------------------|------------|--------------|------|----------------
rep_visits            |   18,720   |    31.2%     | 0.38 | Maintain
medical_congress      |    6,240   |    12.4%     | 0.52 | Increase ↑
dtc_tv                |   22,880   |    18.6%     | 0.20 | Reduce ↓
speaker_programs      |    3,640   |     9.8%     | 0.45 | Increase ↑
samples_coupons       |    9,360   |    11.3%     | 0.33 | Maintain
...
```

**2. Budget optimisation recommendation**
> *"Reallocating 15% of DTC TV spend to medical congress and speaker programs is projected to increase scripts written by 8.3% with the same total budget, based on marginal ROI analysis across the 2-year weekly dataset."*

**3. Auto-generated HTML report** — saved to `reports/mmm_report.html`

---

## 🗂️ Dataset Schema

### Weekly (`data/raw/mmm_weekly.csv`) — 104 rows × 22 columns

| Column | Type | Description |
|---|---|---|
| `date` | date | Week start (Monday) |
| `rep_visits` | float | HCP rep detailing spend ($K) |
| `medical_congress` | float | Congress & symposia spend ($K) |
| `journal_advertising` | float | HCP journal ad spend ($K) |
| `hcp_email` | float | HCP permission email spend ($K) |
| `hcp_digital` | float | HCP programmatic display spend ($K) |
| `speaker_programs` | float | KOL speaker bureau spend ($K) |
| `samples_coupons` | float | Samples & co-pay coupons spend ($K) |
| `dtc_tv` | float | DTC television spend ($K) |
| `dtc_digital` | float | DTC digital/social spend ($K) |
| `dtc_ooh` | float | Out-of-home spend ($K) |
| `patient_email` | float | Patient CRM email spend ($K) |
| `patient_advocacy` | float | Patient advocacy partnership spend ($K) |
| `total_spend` | float | Sum of all channels ($K) |
| `scripts_written` | int | Vaccine Rx written (outcome KPI) |
| `nrx_index` | float | Normalised Rx index (0–100) |
| `vaccine_season` | binary | 1 = Sep/Oct/Nov peak season |
| `congress_week` | binary | 1 = congress month (Feb/May/Oct) |

### Monthly (`data/raw/mmm_monthly.csv`) — 36 rows × 24 columns
Same channel columns, plus `hcp_total_spend`, `dtc_total_spend`, `month_name`, `congress_month`.

Full schema: see `data/raw/data_dictionary.csv`

---

## 🧠 Modelling Approaches

### Frequentist MMM (OLS / Panel)
- Geometric adstock per channel (decay rates calibrated to pharma norms)
- Hill saturation transform
- Panel OLS with fixed effects for seasonality
- Suitable for: weekly data, 2+ years, interpretability-first use cases

### Bayesian MMM (PyMC)
- Hierarchical priors on adstock decay and saturation
- Posterior distributions over channel ROI (uncertainty quantification)
- MCMC sampling via PyMC / ArviZ
- Suitable for: monthly data, shorter history, uncertainty-aware planning

---

## 🔧 Configuration

All model parameters live in `config/config.yaml`:

```yaml
channels:
  rep_visits:
    adstock_decay: 0.6
    saturation_alpha: 0.55
    prior_roi_mean: 0.38
  medical_congress:
    adstock_decay: 0.75
    saturation_alpha: 0.45
    prior_roi_mean: 0.52
  # ... all 12 channels

model:
  ols:
    fit_intercept: true
    seasonality_dummies: true
  bayesian:
    draws: 2000
    tune: 1000
    chains: 4
    target_accept: 0.9
```

---

## 📦 Requirements

```
langchain>=0.2.0
langchain-openai>=0.1.0
pymc>=5.0.0
arviz>=0.17.0
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

- 🔗 [LinkedIn](https://linkedin.com/in/shabam23)
- 🐙 [GitHub](https://github.com/Shubh-kr)
- 📧 shubham.mle@gmail.com

---

## 📄 Licence

MIT — use freely, modify, and build on top of this for your own projects.
If this saves you a week of work, consider leaving a ⭐ on GitHub.

---

## 🗺️ Roadmap

- [ ] Multi-territory / geo-level MMM support
- [ ] CrewAI multi-agent upgrade (parallel model fitting)
- [ ] Streamlit dashboard for non-technical stakeholders
- [ ] Integration with IQVIA / Symphony claims data schema
- [ ] Payer mix and co-pay dynamics as control variables

---

*Built for pharma data scientists who are tired of explaining adstock to stakeholders.*
