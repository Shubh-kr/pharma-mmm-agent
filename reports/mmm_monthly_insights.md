# VaxBrand — MMM Insight Report

# VaxBrand MMM Strategic Insight Report
### Vaccine Commercial Strategy | HCP + Patient Channels | 36-Month Analysis

---

## Executive Summary

Both models confirm strong explanatory power for VaxBrand's NRx drivers: the OLS Ridge model achieves an excellent fit (R² = 0.976, MAPE = 2.88% across 36 monthly periods), while the Bayesian model converges cleanly (R̂ max = 1.001, 4 chains × 2,000 draws) and provides full posterior distributions for all 12 channels — eliminating the need for prior-estimation fallbacks on three channels that OLS could not isolate. The single biggest finding is a structural misallocation of spend: the two highest-ROI HCP channels — Speaker Bureau Programs (OLS ROI: 0.63; Bayesian: 0.787) and Medical Congress & Symposia (OLS ROI: 0.728; Bayesian: 0.910) — are collectively receiving only $499.4K/week against a combined optimal allocation of $3,024.8K/week, while DTC Television ($1,090.3K/week, ROI: 0.28–0.35) and Field Rep Visits ($986.2K/week) absorb the majority of budget at lower marginal returns. A critical flag: the OLS competitor coefficient is **positive (+4.39)**, which is directionally incorrect and likely an artifact of collinearity; the Bayesian model correctly identifies competitor spend as suppressive (β = −216.77), confirming competitive headwinds are real and should inform SOV strategy. Executing the optimiser's reallocation within the current $4,321.2K/week envelope is projected to deliver a **+46.2% NRx uplift** without incremental budget.

---

## Channel Performance Analysis

### Top Performing Channels — HCP

---

#### 1. Medical Congress & Symposia
| Metric | OLS | Bayesian |
|---|---|---|
| ROI | 0.728 | 0.910 |
| Contribution % | 13.0% | 8.8% |
| Total Spend | $11,131.6K | $11,131.6K |
| Contribution Source | Model (data-identified) | Posterior |
| 90% HDI | — | [+4,299 – +199,165 scripts] |

**Why it performs:** Congress and symposia create concentrated, high-credibility HCP touchpoints — KOL-led presentations, satellite symposia, and peer-to-peer exchange drive belief change and NRx intent in ways that passive channels cannot replicate. The Bayesian ROI of 0.910 is the highest of any channel in the portfolio, and the OLS coefficient (537.4) is data-identified with high confidence. The wide HDI ([+4.3K–+199.2K]) reflects natural variability in congress timing and attendance across the 36-month window — not model weakness.

**Recommendation:** Increase weekly spend from $309.2K to $1,512.4K (+$1,203.2K/week, +389.1%). Prioritise major infectious disease and primary care congresses (e.g., IDWeek, ACIP advisory cycles). Ensure medical affairs alignment on symposia content to maximise scientific pull-through.

---

#### 2. Speaker Bureau Programs
| Metric | OLS | Bayesian |
|---|---|---|
| ROI | 0.630 | 0.787 |
| Contribution % | 31.8% | 8.2% |
| Total Spend | $6,846.9K | $6,846.9K |
| Contribution Source | Model (data-identified) | Posterior |
| 90% HDI | — | [+3,947 – +187,334 scripts] |

**Why it performs:** Speaker programs are the highest-volume NRx driver in the OLS model (31.8% of total incremental scripts), reflecting the power of peer-to-peer HCP influence in vaccine adoption. Trusted local champions speaking to community prescribers generate durable NRx behaviour change — particularly important for adult vaccine categories where HCP recommendation is the primary patient conversion lever. The Bayesian model corroborates strong performance (ROI 0.787), though the contribution share is lower (8.2%) due to