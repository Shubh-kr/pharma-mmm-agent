"""
tools/geo_hierarchical_mmm_tool.py
====================================
Hierarchical Bayesian MMM — all territories in one PyMC model with shared
national hyperpriors (partial pooling).

Advantage over the independent per-territory model:
  - Channel betas are drawn from a national log-normal distribution, so
    data-sparse markets (Mountain) borrow strength from larger territories.
  - National-level ROI summaries emerge naturally from the hyperpriors.
  - Convergence is faster than 6 × 2-chain runs because the chains explore
    the full joint posterior.

Model structure
---------------
For each channel ch:
    log_mu_ch    ~ Normal(log(prior_scale), 0.5)        # national log-mean
    log_sigma_ch ~ HalfNormal(0.4)                       # territory spread
    z_ch         ~ Normal(0, 1, shape=n_terr)            # non-centred offsets
    beta_ch      = exp(log_mu_ch + log_sigma_ch * z_ch)  # territory betas ≥ 0

Territory baselines (non-centred normal):
    mu_baseline    ~ Normal(global_mean_y * 0.55, global_mean_y * 0.25)
    sigma_baseline ~ HalfNormal(global_mean_y * 0.20)
    z_baseline     ~ Normal(0, 1, shape=n_terr)
    baseline       = mu_baseline + sigma_baseline * z_baseline

Shared across territories: sigma_obs, beta_season, beta_congress,
    beta_competitor, beta_price.

Likelihood (vectorised over stacked observations):
    mu[i] = baseline[tid[i]] + sum(beta_ch[tid[i]] * sat_ch[i]) + controls[i]
    y[i]  ~ Normal(mu[i], sigma_obs)

Output
------
JSON format mirrors geo_bayesian_mmm_tool.py so the dashboard can render it
with the same component, plus a top-level "national_hyperpriors" block.
"""

import json

import numpy as np
import pandas as pd
import yaml
from langchain.tools import tool

from tools.transforms import geometric_adstock, hill_saturation


def _prepare_stacked_data(df: pd.DataFrame, channels: list, config: dict, freq: str):
    """
    Return stacked numpy arrays and per-territory metadata for the model.

    Returns
    -------
    terr_keys  : sorted list of territory keys
    y          : (total_n,)  observed scripts
    tid        : (total_n,)  territory index per observation
    sat        : dict ch → (total_n,)  saturated spend
    controls   : dict name → (total_n,)
    terr_meta  : dict tk → {y, sat, n, mean_y, spend_raw}
    global_mean_y : float
    """
    outcome_col  = config["data"]["outcome_col"]
    congress_col = "congress_week" if freq == "weekly" else "congress_month"
    terr_keys    = sorted(df["territory"].unique())
    n_terr       = len(terr_keys)

    y_parts         = []
    tid_parts       = []
    sat_parts       = {ch: [] for ch in channels}
    vs_parts        = []   # vaccine_season
    cong_parts      = []
    comp_parts      = []
    price_parts     = []
    terr_meta       = {}

    for i, tk in enumerate(terr_keys):
        tdf = df[df["territory"] == tk].copy().reset_index(drop=True)
        n   = len(tdf)

        y_t = tdf[outcome_col].values.astype(float)
        y_parts.append(y_t)
        tid_parts.extend([i] * n)

        sat_t = {}
        for ch in channels:
            if ch in tdf.columns:
                ads = geometric_adstock(tdf[ch].values, config["channels"][ch]["adstock_decay"])
                sat = hill_saturation(ads, config["channels"][ch]["saturation"])
            else:
                sat = np.zeros(n)
            sat_parts[ch].append(sat.astype(float))
            sat_t[ch] = sat.astype(float)

        vs   = tdf["vaccine_season"].values.astype(float)
        cong = (tdf[congress_col].values.astype(float)
                if congress_col in tdf.columns else np.zeros(n))

        comp_mean = float(tdf["competitor_spend"].mean()) + 1e-9
        comp      = (tdf["competitor_spend"].values.astype(float) / comp_mean
                     if "competitor_spend" in tdf.columns else np.zeros(n))
        price     = ((tdf["price_index"].values.astype(float) - 100) / 10
                     if "price_index" in tdf.columns else np.zeros(n))

        vs_parts.append(vs)
        cong_parts.append(cong)
        comp_parts.append(comp)
        price_parts.append(price)

        terr_meta[tk] = {
            "y":         y_t,
            "sat":       sat_t,
            "n":         n,
            "mean_y":    float(y_t.mean()),
            "spend_raw": {ch: tdf[ch].values if ch in tdf.columns else np.zeros(n)
                          for ch in channels},
        }

    y_all   = np.concatenate(y_parts)
    tid_all = np.array(tid_parts, dtype=int)
    sat_all = {ch: np.concatenate(sat_parts[ch]) for ch in channels}
    ctrl    = {
        "vaccine_season":  np.concatenate(vs_parts),
        "congress":        np.concatenate(cong_parts),
        "competitor":      np.concatenate(comp_parts),
        "price":           np.concatenate(price_parts),
    }

    global_mean_y = float(y_all.mean())
    return terr_keys, y_all, tid_all, sat_all, ctrl, terr_meta, global_mean_y


def _fit_hierarchical(
    terr_keys,
    y_all,
    tid_all,
    sat_all,
    ctrl,
    terr_meta,
    global_mean_y,
    channels,
    config,
    freq,
):
    import pymc as pm
    import arviz as az
    import pytensor.tensor as pt

    bayes_cfg = config["bayesian_model"]
    draws     = bayes_cfg.get("hier_draws",  600)
    tune      = bayes_cfg.get("hier_tune",   400)
    chains    = bayes_cfg.get("hier_chains", 2)
    t_accept  = bayes_cfg.get("target_accept", 0.90)

    n_terr = len(terr_keys)
    print(f"  Fitting hierarchical model: {n_terr} territories, "
          f"{chains}×{draws} draws (tune={tune}) …")

    with pm.Model() as model:  # noqa: F841
        # ── Channel hyperpriors (log-normal non-centred) ────────────────────
        betas = {}   # ch → (n_terr,) tensor
        for ch in channels:
            prior_scale = (config["channels"][ch]["prior_roi"]
                           * global_mean_y * 0.5)
            prior_scale = max(prior_scale, 1e-3)

            log_mu    = pm.Normal(f"log_mu_{ch}",
                                  mu=np.log(prior_scale), sigma=0.5)
            log_sigma = pm.HalfNormal(f"log_sigma_{ch}", sigma=0.4)
            z         = pm.Normal(f"z_{ch}", 0.0, 1.0, shape=n_terr)
            betas[ch] = pm.Deterministic(f"beta_{ch}",
                                         pt.exp(log_mu + log_sigma * z))

        # ── Territory baselines (log-normal non-centred — always positive) ──
        # Log-normal handles the ~3× range in territory market sizes naturally.
        log_mu_bl    = pm.Normal("log_mu_baseline",
                                 mu=np.log(max(global_mean_y * 0.55, 1.0)),
                                 sigma=0.5)
        sigma_log_bl = pm.HalfNormal("sigma_log_baseline", sigma=0.6)
        z_bl         = pm.Normal("z_baseline", 0.0, 1.0, shape=n_terr)
        baseline     = pm.Deterministic("baseline",
                                        pt.exp(log_mu_bl + sigma_log_bl * z_bl))

        # ── Shared control coefficients ─────────────────────────────────────
        beta_season   = pm.HalfNormal("beta_season",    sigma=global_mean_y * 0.15)
        beta_congress = pm.Normal("beta_congress",       mu=global_mean_y * 0.03,
                                  sigma=global_mean_y * 0.04)
        beta_comp     = pm.Normal("beta_competitor",     mu=-global_mean_y * 0.03,
                                  sigma=global_mean_y * 0.03)
        beta_price    = pm.Normal("beta_price",          mu=-global_mean_y * 0.03,
                                  sigma=global_mean_y * 0.03)
        sigma_obs     = pm.HalfNormal("sigma_obs",       sigma=global_mean_y * 0.08)

        # ── Vectorised likelihood ───────────────────────────────────────────
        mu = (baseline[tid_all]
              + beta_season   * ctrl["vaccine_season"]
              + beta_congress * ctrl["congress"]
              + beta_comp     * ctrl["competitor"]
              + beta_price    * ctrl["price"])

        for ch in channels:
            mu = mu + betas[ch][tid_all] * sat_all[ch]

        pm.Normal("y_obs", mu=mu, sigma=sigma_obs, observed=y_all)

        trace = pm.sample(
            draws=draws, tune=tune, chains=chains,
            target_accept=t_accept,
            progressbar=False,
            return_inferencedata=True,
            random_seed=42,
        )

    # ── Extract posteriors ──────────────────────────────────────────────────
    post = trace.posterior

    def _flat(var):
        return post[var].values.reshape(-1)

    def _flat_terr(var, i):
        # var has shape (chains, draws, n_terr)
        return post[var].values[:, :, i].reshape(-1)

    # National hyperprior summaries
    national_hyperpriors = {}
    for ch in channels:
        log_mu_s    = _flat(f"log_mu_{ch}")
        log_sigma_s = _flat(f"log_sigma_{ch}")
        mu_beta_nat = float(np.exp(log_mu_s.mean()))
        sigma_nat   = float(np.exp(log_sigma_s.mean()))
        avg_spend_nat = float(np.mean([
            terr_meta[tk]["spend_raw"][ch].mean() for tk in terr_keys
        ]))
        avg_sat_nat = float(np.mean([
            terr_meta[tk]["sat"][ch].mean() for tk in terr_keys
        ]))
        roi_nat = mu_beta_nat * avg_sat_nat / (avg_spend_nat + 1e-9) * 100
        national_hyperpriors[ch] = {
            "label":          config["channels"][ch]["label"],
            "mu_beta_mean":   round(mu_beta_nat, 4),
            "sigma_terr_mean": round(sigma_nat, 4),
            "national_roi_mean": round(0.5 * config["channels"][ch]["prior_roi"]
                                       + 0.5 * min(roi_nat, config["channels"][ch]["prior_roi"] * 2.5), 3),
        }

    b_season_mean  = float(_flat("beta_season").mean())
    b_congress_mean = float(_flat("beta_congress").mean())
    b_comp_mean    = float(_flat("beta_competitor").mean())
    b_price_mean   = float(_flat("beta_price").mean())

    # ── R̂ for convergence ──────────────────────────────────────────────────
    rhat_data  = az.rhat(trace)
    rhat_vars  = (
        [f"log_mu_{ch}" for ch in list(channels)[:4] if f"log_mu_{ch}" in rhat_data]
        + ["log_mu_baseline", "sigma_obs"]
    )
    rhat_vals  = [float(rhat_data[v].values) for v in rhat_vars if v in rhat_data]
    max_rhat   = round(max(rhat_vals), 3) if rhat_vals else None
    converged  = max_rhat is not None and max_rhat < 1.05

    mcmc_meta = {
        "draws": draws, "tune": tune, "chains": chains,
        "target_accept": t_accept,
        "max_rhat": max_rhat, "converged": converged,
        "model_type": "hierarchical",
    }

    # ── Per-territory results ───────────────────────────────────────────────
    territories = {}
    for i, tk in enumerate(terr_keys):
        tmeta  = terr_meta[tk]
        label  = config.get("territories", {}).get(tk, {}).get("label", tk)
        y_t    = tmeta["y"]
        n      = tmeta["n"]

        baseline_s = _flat_terr("baseline", i)
        baseline_m = float(baseline_s.mean())

        channel_results = {}
        contributions   = {}
        y_pred = (baseline_m
                  + b_season_mean  * ctrl["vaccine_season"][tid_all == i]
                  + b_congress_mean * ctrl["congress"][tid_all == i]
                  + b_comp_mean    * ctrl["competitor"][tid_all == i]
                  + b_price_mean   * ctrl["price"][tid_all == i])

        for ch in channels:
            beta_s = _flat_terr(f"beta_{ch}", i)
            sat_t  = tmeta["sat"][ch]

            contrib_s = beta_s * float(sat_t.sum())
            contrib_m = float(contrib_s.mean())
            hdi_lo    = float(np.percentile(contrib_s, 5))
            hdi_hi    = float(np.percentile(contrib_s, 95))
            contributions[ch] = max(contrib_m, 0.0)

            avg_spend = float(tmeta["spend_raw"][ch].mean())
            avg_sat   = float(sat_t.mean())
            beta_mean = float(beta_s.mean())
            roi_model = beta_mean * avg_sat / (avg_spend + 1e-9) * 100
            prior_roi = config["channels"][ch]["prior_roi"]
            blended   = round(0.5 * prior_roi + 0.5 * min(roi_model, prior_roi * 2.5), 3)

            channel_results[ch] = {
                "label":               config["channels"][ch]["label"],
                "channel_type":        config["channels"][ch]["channel_type"],
                "avg_period_spend_k":  round(avg_spend, 1),
                "total_spend_k":       round(float(tmeta["spend_raw"][ch].sum()), 1),
                "beta_mean":           round(beta_mean, 4),
                "estimated_roi":       blended,
                "total_contribution":  round(contrib_m, 1),
                "contribution_hdi_5":  round(hdi_lo, 1),
                "contribution_hdi_95": round(hdi_hi, 1),
                "contribution_source": "hierarchical_bayesian",
            }

            y_pred += beta_mean * sat_t

        total_contrib = sum(contributions.values()) + 1e-9
        for ch in channel_results:
            channel_results[ch]["contribution_pct"] = round(
                contributions[ch] / total_contrib * 100, 1
            )

        r2   = round(1 - float(np.sum((y_t - y_pred) ** 2))
                     / float(np.sum((y_t - y_t.mean()) ** 2)), 4)
        mape = round(float(np.mean(np.abs((y_t - y_pred) / (y_t + 1e-9))) * 100), 2)

        territories[tk] = {
            "label":                    label,
            "r_squared_posterior_mean": r2,
            "mape_pct":                 mape,
            "n_observations":           n,
            "baseline_scripts":         round(baseline_m, 0),
            "mcmc":                     mcmc_meta,
            "controls": {
                "beta_season":     round(b_season_mean, 2),
                "beta_congress":   round(b_congress_mean, 2),
                "beta_competitor": round(b_comp_mean, 4),
                "beta_price":      round(b_price_mean, 4),
            },
            "channels": dict(sorted(
                channel_results.items(),
                key=lambda x: x[1]["contribution_pct"],
                reverse=True,
            )),
        }

    return {
        "model":               "Hierarchical Bayesian MMM (PyMC, shared national prior)",
        "frequency":           freq,
        "national_hyperpriors": national_hyperpriors,
        "mcmc":                mcmc_meta,
        "territories":         territories,
    }


@tool
def run_geo_hierarchical_mmm_tool(
    data_path: str, config_path: str, freq: str = "weekly"
) -> str:
    """
    Run hierarchical Bayesian MMM (PyMC) across all territories simultaneously.

    Uses partial pooling via log-normal national hyperpriors on channel betas.
    Data-sparse territories (Mountain) borrow strength from larger ones.

    Args:
        data_path   : path to *_geo.csv (long format with 'territory' column)
        config_path : path to config.yaml
        freq        : 'weekly' or 'monthly'

    Returns:
        Per-territory R², MAPE, R̂, national hyperprior ROI summary
    """
    try:
        import pymc   # noqa: F401
        import arviz  # noqa: F401
    except ImportError:
        return "Error: PyMC not installed. Run: pip install 'pymc>=5.0.0' 'arviz>=0.17.0'"

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)

        df = pd.read_csv(data_path, parse_dates=["date"])
        if "territory" not in df.columns:
            return "Error: 'territory' column not found. Use a *_geo.csv file."

        channels = list(config["channels"].keys())

        print("  Preparing stacked data …")
        terr_keys, y_all, tid_all, sat_all, ctrl, terr_meta, global_mean_y = (
            _prepare_stacked_data(df, channels, config, freq)
        )

        results = _fit_hierarchical(
            terr_keys, y_all, tid_all, sat_all, ctrl, terr_meta,
            global_mean_y, channels, config, freq,
        )

        out_path = data_path.replace("_geo.csv", "_geo_hierarchical_results.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

        # ── Summary printout ────────────────────────────────────────────────
        mcmc = results["mcmc"]
        lines = [
            f"Geo Hierarchical Bayesian MMM — {freq} data",
            f"MCMC: R̂ max={mcmc['max_rhat']}  converged={'✓' if mcmc['converged'] else '⚠'}",
            "",
            f"{'Territory':<14} {'R²':<7} {'MAPE':<8} {'Baseline':<12} Top channel",
            "-" * 65,
        ]
        for tk, td in results["territories"].items():
            top_ch   = next(iter(td["channels"]))
            top_lbl  = td["channels"][top_ch]["label"]
            lines.append(
                f"{td['label']:<14} {td['r_squared_posterior_mean']:<7.3f} "
                f"{td['mape_pct']:.1f}%    {td['baseline_scripts']:<12,.0f} {top_lbl}"
            )

        lines += [
            "",
            "National hyperprior ROI (blended):",
            f"  {'Channel':<36} {'Natl ROI':<10} {'μ_beta':<10} {'σ_terr':<10}",
            "  " + "-" * 56,
        ]
        for ch, hp in sorted(
            results["national_hyperpriors"].items(),
            key=lambda x: x[1]["national_roi_mean"],
            reverse=True,
        ):
            lines.append(
                f"  {hp['label']:<36} {hp['national_roi_mean']:<10.3f} "
                f"{hp['mu_beta_mean']:<10.4f} {hp['sigma_terr_mean']:<10.4f}"
            )

        lines.append(f"\nSaved to: {out_path}")
        return "\n".join(lines)

    except Exception as exc:
        import traceback
        return f"Error in hierarchical geo MMM:\n{exc}\n\n{traceback.format_exc()}"
