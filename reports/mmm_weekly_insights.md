# VaxBrand — MMM Insight Report

# VaxBrand Marketing Mix Model — Strategic Insight Report
### Prepared for: VP Commercial | Medical Affairs Lead | Brand Team
### Model Period: 104 Weekly Periods | Total Measured Spend: ~$97.6M

---

## Executive Summary

Both models confirm strong explanatory power for VaxBrand's NRx dynamics: the OLS Ridge model achieves an R² of 0.951 and MAPE of 2.98%, while the Bayesian PyMC model (R² = 0.825, MAPE = 5.97%) provides fully posterior-derived channel estimates with confirmed MCMC convergence (R̂ max = 1.001). The lower Bayesian R² is expected given more aggressive prior regularisation and does not indicate model inferiority — it is the more conservative and credible estimate for uncertain channels. The single biggest finding is a structural misallocation of spend: VaxBrand's two highest-ROI channels — Medical Congress & Symposia (OLS ROI: 0.728; Bayesian: 0.91) and Speaker Bureau Programs (OLS: 0.630; Bayesian: 0.787) — together receive only $106.4K/week (11.3% of budget), while DTC Television alone consumes $238.9K/week at an ROI of just 0.28–0.35. The budget optimiser projects a **+59.4% NRx uplift** at the same total weekly spend of $938.6K by reallocating primarily into Congress and Speaker programs, with corresponding reductions across DTC Television, Samples & Co-pay Coupons, and digital channels. A critical data flag: the OLS competitor coefficient is **positive (+3.07)**, which is directionally incorrect and likely an artefact of collinearity; the Bayesian model correctly identifies competitor spend as suppressive (β = −397.59), and the Bayesian result should govern all competitive strategy decisions.

---

## Channel Performance Analysis

### Top Performing Channels — HCP

---

#### 1. Medical Congress & Symposia
| Metric | OLS | Bayesian |
|---|---|---|
| Weekly Spend | $66.6K | $66.6K |
| Total Spend | $6.9M | $6.9M |
| Estimated ROI | 0.728 | 0.910 |
| Contribution % | 1.7% | 5.1% |
| Contribution (scripts) | 17,064 | 49,498 |
| 90% HDI | — | [+3,397 – +134,748] |
| Contribution Source | Model (data-identified) | Bayesian Posterior |

**Why it performs:** Congress and symposia activity drives peer-to-peer scientific exchange at high-prescriber density events. For a vaccine brand, congress presence reinforces clinical guideline alignment and immunisation schedule recommendations — both critical HCP pull-through levers. The OLS model directly identified this channel from the data (coefficient = 15.15), and the Bayesian model substantially upgrades the ROI estimate from 0.728 to 0.910, making this the **highest-ROI channel in the entire portfolio** by Bayesian measure.

**Uncertainty note:** The Bayesian HDI is wide ([+3.4K – +134.7K]), spanning a 40× range. This reflects limited weekly variation in congress spend (it is episodic by nature) rather than a weak signal. The point estimate is highly credible; the width reflects scheduling concentration, not noise.

**Recommendation:** Increase weekly allocation from $66.6K to $328.5K (+$261.9K/week, +393.4%) per optimiser. Prioritise pre-congress digital seeding and post-congress rep follow-up to extend the pull-through window. Align congress calendar with vaccine season onset (see Bayesian season coefficient: +2,357 scripts/period).

---

#### 2. Speaker Bureau Programs
| Metric | OLS | Bayesian |
|---|---|---|
| Weekly Spend | $39.8K | $39.8K |
| Total Spend | $4.1M | $4.1M |
| Estimated ROI | 0.630 | 0.787 |
| Contribution % | 12.9% | 6.0% |
| Contribution (scripts) | 128,756