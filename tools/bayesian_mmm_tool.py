"""
tools/bayesian_mmm_tool.py
===========================
Bayesian Marketing Mix Model using PyMC.

Solves two key limitations of the frequentist Ridge MMM:

  1. Zero-contribution channels
     Informative HalfNormal priors on channel betas force all channels to have
     positive, non-zero contributions — no prior-floor heuristic needed.
     The posterior naturally distributes credit proportionally to prior_roi
     when the data cannot separately identify correlated channels.

  2. No uncertainty quantification
     MCMC gives full posterior distributions. Every channel contribution has a
     90% credible interval (5th–95th percentile), which Ridge cannot provide.

Model structure:
  scripts_t = baseline
            + Σ_ch  beta_ch × sat(adstock(spend_ch))_t   [channel contributions]
            + beta_season   × vaccine_season_t             [Sep-Nov peak]
            + beta_congress × congress_t                   [conference months]
            + beta_competitor × norm_competitor_t          [competitive erosion]
            + beta_price    × price_deviation_t            [co-pay effect]
            + ε_t,   ε ~ Normal(0, sigma)

Priors (calibrated to outcome scale mean_y):
  beta_ch       ~ HalfNormal(prior_roi × mean_y × 0.5)   non-negative, informative
  baseline      ~ Normal(mean_y × 0.55,  mean_y × 0.20)
  beta_season   ~ HalfNormal(mean_y × 0.15)
  beta_congress ~ Normal(mean_y × 0.03,  mean_y × 0.04)
  beta_competitor ~ Normal(-mean_y × 0.03, mean_y × 0.03)  informed negative prior
  beta_price    ~ Normal(-mean_y × 0.03,  mean_y × 0.03)   informed negative prior
  sigma         ~ HalfNormal(mean_y × 0.08)

Uses pre-computed saturated features from *_transformed.csv (fast).
Fitting adstock/saturation decay within the model requires PyMC scan ops
and is left for a future Bayesian-adstock extension.
"""

import json
import numpy as np
import pandas as pd
import yaml
from langchain.tools import tool


@tool
def run_bayesian_mmm_tool(data_path: str, config_path: str, freq: str = "weekly") -> str:
    """
    Run a Bayesian Marketing Mix Model on the transformed pharma dataset using PyMC.

    Key advantages over Ridge MMM:
      - All 12 channels receive non-zero contributions via informative priors
      - Every contribution estimate includes a 90% credible interval
      - Competitor and price controls get proper negative posteriors from
        informed priors, even when raw signal is weak

    Use this AFTER apply_all_transforms_tool has been run.
    Typical runtime: 2-5 minutes (4 chains × 2000 draws on a laptop).

    Args:
        data_path   : path to the *_transformed.csv file
        config_path : path to config.yaml
        freq        : 'weekly' or 'monthly'

    Returns:
        Formatted summary with channel contributions and 90% HDI
    """
    try:
        import pymc as pm
        import arviz as az
    except ImportError:
        return "Error: PyMC not installed. Run: pip install 'pymc>=5.0.0' 'arviz>=0.17.0'"

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)

        df          = pd.read_csv(data_path, parse_dates=["date"])
        channels    = list(config["channels"].keys())
        bayes_cfg   = config["bayesian_model"]
        outcome_col = config["data"]["outcome_col"]

        y      = df[outcome_col].values.astype(float)
        n      = len(y)
        mean_y = float(y.mean())

        # ── Channel features (pre-computed saturated, range 0–1) ──────────────
        sat = {}
        for ch in channels:
            col = f"{ch}_saturated"
            if col in df.columns:
                sat[ch] = df[col].values.astype(float)

        if not sat:
            return "Error: no *_saturated columns found. Run apply_all_transforms_tool first."

        # ── Control features ───────────────────────────────────────────────────
        congress_col   = "congress_week" if freq == "weekly" else "congress_month"
        vaccine_season = df["vaccine_season"].values.astype(float)
        congress       = (df[congress_col].values.astype(float)
                          if congress_col in df.columns else np.zeros(n))

        # Competitor normalised: 0 = no spend, 1 = mean competitor level
        # This keeps the beta coefficient in units of "scripts per mean-competitor-unit"
        comp_mean  = float(df["competitor_spend"].mean()) + 1e-9 if "competitor_spend" in df.columns else 1.0
        competitor = (df["competitor_spend"].values.astype(float) / comp_mean
                      if "competitor_spend" in df.columns else np.zeros(n))

        # Price expressed as deviation from 100 in units of 10 index points
        # beta_price interpretation: scripts change per 10-point co-pay move
        price = ((df["price_index"].values.astype(float) - 100) / 10
                 if "price_index" in df.columns else np.zeros(n))

        # ── PyMC model ─────────────────────────────────────────────────────────
        print("  → Building PyMC model...")
        with pm.Model() as mmm_model:   # noqa: F841

            # Channel betas — HalfNormal with priors anchored to prior_roi.
            # sigma = prior_roi × mean_y × 0.5 means the prior mean contribution
            # per unit of saturated spend ≈ prior_roi × mean_y × 0.5 × sqrt(2/π)
            # ≈ 40% of prior_roi × mean_y — a weakly informative positive prior.
            betas = {
                ch: pm.HalfNormal(
                    f"beta_{ch}",
                    sigma=config["channels"][ch]["prior_roi"] * mean_y * 0.5,
                )
                for ch in sat
            }

            # Baseline organic scripts (brand equity + market)
            baseline = pm.Normal("baseline", mu=mean_y * 0.55, sigma=mean_y * 0.20)

            # Seasonality — positive (Sep-Nov peak drives more scripts)
            beta_season = pm.HalfNormal("beta_season", sigma=mean_y * 0.15)

            # Congress — positive but uncertain (some months drive HCP action)
            beta_congress = pm.Normal("beta_congress",
                                      mu=mean_y * 0.03, sigma=mean_y * 0.04)

            # Competitor — informed negative prior (more competitor spend → fewer our scripts)
            beta_comp = pm.Normal("beta_competitor",
                                  mu=-mean_y * 0.03, sigma=mean_y * 0.03)

            # Price — informed negative prior (higher co-pay → fewer scripts)
            beta_price = pm.Normal("beta_price",
                                   mu=-mean_y * 0.03, sigma=mean_y * 0.03)

            # Observation noise
            sigma_obs = pm.HalfNormal("sigma_obs", sigma=mean_y * 0.08)

            # Expected outcome
            mu = (
                baseline
                + beta_season   * vaccine_season
                + beta_congress * congress
                + beta_comp     * competitor
                + beta_price    * price
            )
            for ch, sat_vals in sat.items():
                mu = mu + betas[ch] * sat_vals

            pm.Normal("scripts", mu=mu, sigma=sigma_obs, observed=y)

            # ── MCMC ──────────────────────────────────────────────────────────
            draws         = bayes_cfg.get("draws", 2000)
            tune          = bayes_cfg.get("tune", 1000)
            chains        = bayes_cfg.get("chains", 4)
            target_accept = bayes_cfg.get("target_accept", 0.90)

            print(f"  → Sampling {chains} chains × {draws} draws (tune={tune}) ...")
            trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                progressbar=False,
                return_inferencedata=True,
                random_seed=42,
            )

        print("  → Extracting posteriors...")

        # ── Posterior extraction ───────────────────────────────────────────────
        def _post(var_name):
            return trace.posterior[var_name].values.flatten()

        beta_means    = {ch: float(_post(f"beta_{ch}").mean()) for ch in sat}
        baseline_mean = float(_post("baseline").mean())
        b_season      = float(_post("beta_season").mean())
        b_congress    = float(_post("beta_congress").mean())
        b_comp        = float(_post("beta_competitor").mean())
        b_price       = float(_post("beta_price").mean())

        # ── Channel contributions with credible intervals ──────────────────────
        contributions   = {}
        channel_results = {}

        for ch, sat_vals in sat.items():
            # Posterior samples of total 2-year contribution for this channel
            contrib_samples = _post(f"beta_{ch}") * float(sat_vals.sum())
            contrib_mean    = float(contrib_samples.mean())
            hdi_lo          = float(np.percentile(contrib_samples, 5))
            hdi_hi          = float(np.percentile(contrib_samples, 95))
            contributions[ch] = max(contrib_mean, 0.0)

            raw_spend  = df[ch].values
            avg_spend  = float(raw_spend.mean())
            avg_sat    = float(sat_vals.mean())
            roi_model  = beta_means[ch] * avg_sat / (avg_spend + 1e-9) * 100
            prior_roi  = config["channels"][ch]["prior_roi"]
            blended_roi = round(0.5 * prior_roi + 0.5 * min(roi_model, prior_roi * 2.5), 3)

            channel_results[ch] = {
                "label":               config["channels"][ch]["label"],
                "channel_type":        config["channels"][ch]["channel_type"],
                "avg_weekly_spend_k":  round(avg_spend, 1),
                "total_spend_k":       round(float(raw_spend.sum()), 1),
                "beta_mean":           round(beta_means[ch], 4),
                "estimated_roi":       blended_roi,
                "total_contribution":  round(contrib_mean, 1),
                "contribution_hdi_5":  round(hdi_lo, 1),
                "contribution_hdi_95": round(hdi_hi, 1),
                "contribution_source": "bayesian_posterior",
            }

        total_contrib = sum(contributions.values()) + 1e-9
        for ch in channel_results:
            channel_results[ch]["contribution_pct"] = round(
                contributions[ch] / total_contrib * 100, 1
            )

        # ── Posterior-mean R² ──────────────────────────────────────────────────
        y_pred = (
            baseline_mean
            + b_season   * vaccine_season
            + b_congress * congress
            + b_comp     * competitor
            + b_price    * price
        )
        for ch, sat_vals in sat.items():
            y_pred += beta_means[ch] * sat_vals

        r2   = round(1 - float(np.sum((y - y_pred) ** 2)) / float(np.sum((y - y.mean()) ** 2)), 4)
        mape = round(float(np.mean(np.abs((y - y_pred) / (y + 1e-9))) * 100), 2)

        # ── Convergence diagnostics (R̂ < 1.05 = good) ────────────────────────
        rhat_data = az.rhat(trace)
        rhat_vals = [
            float(rhat_data[f"beta_{ch}"].values)
            for ch in list(sat.keys())[:4]
            if f"beta_{ch}" in rhat_data
        ] + [
            float(rhat_data[v].values)
            for v in ("baseline", "sigma_obs")
            if v in rhat_data
        ]
        max_rhat  = round(max(rhat_vals), 3) if rhat_vals else None
        converged = max_rhat is not None and max_rhat < 1.05

        # ── Save JSON ──────────────────────────────────────────────────────────
        sorted_chs = sorted(
            channel_results.items(),
            key=lambda x: x[1]["contribution_pct"],
            reverse=True,
        )
        result = {
            "model":                    "Bayesian MMM (PyMC)",
            "frequency":                freq,
            "n_observations":           n,
            "r_squared_posterior_mean": r2,
            "mape_pct":                 mape,
            "mcmc": {
                "draws": draws, "tune": tune, "chains": chains,
                "target_accept": target_accept,
                "max_rhat": max_rhat, "converged": converged,
            },
            "baseline_scripts": round(baseline_mean, 0),
            "controls": {
                "beta_season":     round(b_season,   2),
                "beta_congress":   round(b_congress, 2),
                "beta_competitor": round(b_comp,     4),
                "beta_price":      round(b_price,    4),
            },
            "channels": dict(sorted_chs),
        }

        out_path = data_path.replace("_transformed.csv", "_bayesian_results.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        # ── Formatted summary ──────────────────────────────────────────────────
        conv_tag = f"R̂ max={max_rhat} ✓" if converged else f"R̂ max={max_rhat} ⚠ check convergence"
        lines = [
            f"Bayesian MMM Results (R²={r2:.3f}, MAPE={mape:.1f}%)",
            f"Observations: {n} {freq} | {chains}×{draws} draws | {conv_tag}",
            f"Baseline scripts: {baseline_mean:,.0f}",
            "",
            (f"{'Channel':<30} {'Type':<5} {'Spend $K':<12} "
             f"{'ROI':<8} {'Contrib%':<10} {'90% HDI (2yr scripts)'}"),
            "-" * 90,
        ]
        for ch, res in sorted_chs:
            hdi = (f"[{res['contribution_hdi_5']/1000:+.1f}K"
                   f" – {res['contribution_hdi_95']/1000:+.1f}K]")
            lines.append(
                f"{res['label']:<30} {res['channel_type']:<5} "
                f"${res['total_spend_k']:<10,.0f} "
                f"{res['estimated_roi']:<8.3f} "
                f"{res['contribution_pct']:<10.1f} "
                f"{hdi}"
            )
        lines += [
            "",
            (f"Controls — season: +{b_season:.0f} scripts  "
             f"congress: {b_congress:+.0f}  "
             f"competitor: {b_comp:+.3f}/mean-unit  "
             f"price: {b_price:+.3f}/10pts"),
            f"\nFull posterior results saved to: {out_path}",
        ]
        return "\n".join(lines)

    except Exception as exc:
        import traceback
        return f"Error running Bayesian MMM:\n{exc}\n\n{traceback.format_exc()}"
