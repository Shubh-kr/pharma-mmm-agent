"""
tools/geo_bayesian_mmm_tool.py
================================
Bayesian MMM per territory using PyMC — geo layer counterpart to bayesian_mmm_tool.py.

Each territory is modelled independently (separate MCMC run on its slice of the geo CSV).
Adstock + saturation transforms are applied in-memory so no *_transformed.csv is needed.

Key advantages over Geo Ridge MMM:
  - Every territory's channel ROI comes with a 90% credible interval (HDI)
  - No prior-contribution floor heuristic: informative HalfNormal priors handle
    unidentifiable channels automatically via posterior shrinkage
  - Convergence diagnostics (R̂) per territory flag data-sparse markets (Mountain)
    where estimates should be treated with more caution

Runtime: ~2–4 min per territory × 6 territories = 12–25 min total.
Tune with geo_draws / geo_tune / geo_chains in config.yaml to trade speed vs accuracy.
"""

import json

import numpy as np
import pandas as pd
import yaml
from langchain.tools import tool

from tools.transforms import geometric_adstock, hill_saturation


def _fit_territory_bayes(
    terr_df: pd.DataFrame,
    channels: list,
    config: dict,
    freq: str,
    label: str,
) -> dict:
    """Apply transforms + fit PyMC model for one territory slice."""
    import pymc as pm
    import arviz as az

    bayes_cfg   = config["bayesian_model"]
    outcome_col = config["data"]["outcome_col"]

    # ── In-memory transforms ────────────────────────────────────────────────────
    sat = {}
    for ch in channels:
        if ch not in terr_df.columns:
            continue
        params    = config["channels"][ch]
        adstocked = geometric_adstock(terr_df[ch].values, params["adstock_decay"])
        saturated = hill_saturation(adstocked, params["saturation"])
        sat[ch]   = saturated.astype(float)

    n      = len(terr_df)
    y      = terr_df[outcome_col].values.astype(float)
    mean_y = float(y.mean())

    # ── Control features ────────────────────────────────────────────────────────
    congress_col   = "congress_week" if freq == "weekly" else "congress_month"
    vaccine_season = terr_df["vaccine_season"].values.astype(float)
    congress       = (terr_df[congress_col].values.astype(float)
                      if congress_col in terr_df.columns else np.zeros(n))

    comp_mean  = float(terr_df["competitor_spend"].mean()) + 1e-9
    competitor = (terr_df["competitor_spend"].values.astype(float) / comp_mean
                  if "competitor_spend" in terr_df.columns else np.zeros(n))

    price = ((terr_df["price_index"].values.astype(float) - 100) / 10
             if "price_index" in terr_df.columns else np.zeros(n))

    # ── PyMC model (same structure as national bayesian_mmm_tool) ───────────────
    draws         = bayes_cfg.get("geo_draws",  800)
    tune          = bayes_cfg.get("geo_tune",   400)
    chains        = bayes_cfg.get("geo_chains", 2)
    target_accept = bayes_cfg.get("target_accept", 0.90)

    print(f"    [{label}] {chains}×{draws} draws (tune={tune}) ...")

    with pm.Model():
        betas = {
            ch: pm.HalfNormal(
                f"beta_{ch}",
                sigma=config["channels"][ch]["prior_roi"] * mean_y * 0.5,
            )
            for ch in sat
        }
        baseline      = pm.Normal("baseline",      mu=mean_y * 0.55, sigma=mean_y * 0.20)
        beta_season   = pm.HalfNormal("beta_season",   sigma=mean_y * 0.15)
        beta_congress = pm.Normal("beta_congress",  mu=mean_y * 0.03, sigma=mean_y * 0.04)
        beta_comp     = pm.Normal("beta_competitor", mu=-mean_y * 0.03, sigma=mean_y * 0.03)
        beta_price    = pm.Normal("beta_price",      mu=-mean_y * 0.03, sigma=mean_y * 0.03)
        sigma_obs     = pm.HalfNormal("sigma_obs",   sigma=mean_y * 0.08)

        mu = (
            baseline
            + beta_season   * vaccine_season
            + beta_congress * congress
            + beta_comp     * competitor
            + beta_price    * price
        )
        for ch, sv in sat.items():
            mu = mu + betas[ch] * sv

        pm.Normal("scripts", mu=mu, sigma=sigma_obs, observed=y)

        trace = pm.sample(
            draws=draws, tune=tune, chains=chains,
            target_accept=target_accept,
            progressbar=False,
            return_inferencedata=True,
            random_seed=42,
        )

    # ── Posterior extraction ────────────────────────────────────────────────────
    def _post(var):
        return trace.posterior[var].values.flatten()

    beta_means    = {ch: float(_post(f"beta_{ch}").mean()) for ch in sat}
    baseline_mean = float(_post("baseline").mean())
    b_season      = float(_post("beta_season").mean())
    b_congress    = float(_post("beta_congress").mean())
    b_comp        = float(_post("beta_competitor").mean())
    b_price       = float(_post("beta_price").mean())

    contributions   = {}
    channel_results = {}

    for ch, sv in sat.items():
        contrib_samples = _post(f"beta_{ch}") * float(sv.sum())
        contrib_mean    = float(contrib_samples.mean())
        hdi_lo          = float(np.percentile(contrib_samples, 5))
        hdi_hi          = float(np.percentile(contrib_samples, 95))
        contributions[ch] = max(contrib_mean, 0.0)

        avg_spend  = float(terr_df[ch].values.mean())
        avg_sat    = float(sv.mean())
        roi_model  = beta_means[ch] * avg_sat / (avg_spend + 1e-9) * 100
        prior_roi  = config["channels"][ch]["prior_roi"]
        blended_roi = round(0.5 * prior_roi + 0.5 * min(roi_model, prior_roi * 2.5), 3)

        channel_results[ch] = {
            "label":               config["channels"][ch]["label"],
            "channel_type":        config["channels"][ch]["channel_type"],
            "avg_period_spend_k":  round(avg_spend, 1),
            "total_spend_k":       round(float(terr_df[ch].values.sum()), 1),
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

    # ── Posterior-mean R² ───────────────────────────────────────────────────────
    y_pred = (
        baseline_mean
        + b_season   * vaccine_season
        + b_congress * congress
        + b_comp     * competitor
        + b_price    * price
    )
    for ch, sv in sat.items():
        y_pred += beta_means[ch] * sv

    r2   = round(1 - float(np.sum((y - y_pred) ** 2)) / float(np.sum((y - y.mean()) ** 2)), 4)
    mape = round(float(np.mean(np.abs((y - y_pred) / (y + 1e-9))) * 100), 2)

    # ── R̂ convergence ──────────────────────────────────────────────────────────
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

    sorted_chs = dict(sorted(
        channel_results.items(),
        key=lambda x: x[1]["contribution_pct"],
        reverse=True,
    ))

    return {
        "r_squared_posterior_mean": r2,
        "mape_pct":                 mape,
        "n_observations":           n,
        "baseline_scripts":         round(baseline_mean, 0),
        "mcmc": {
            "draws": draws, "tune": tune, "chains": chains,
            "target_accept": target_accept,
            "max_rhat": max_rhat, "converged": converged,
        },
        "controls": {
            "beta_season":     round(b_season,   2),
            "beta_congress":   round(b_congress, 2),
            "beta_competitor": round(b_comp,     4),
            "beta_price":      round(b_price,    4),
        },
        "channels": sorted_chs,
    }


@tool
def run_geo_bayesian_mmm_tool(data_path: str, config_path: str, freq: str = "weekly") -> str:
    """
    Run Bayesian MMM (PyMC) independently for each territory in the geo dataset.

    Applies adstock + saturation transforms in-memory per territory slice.
    Uses geo_draws / geo_tune / geo_chains from config.yaml for speed control.

    Args:
        data_path   : path to *_geo.csv (long format with 'territory' column)
        config_path : path to config.yaml
        freq        : 'weekly' or 'monthly'

    Returns:
        Per-territory summary: R², MAPE, R̂ convergence, top channel with HDI
    """
    try:
        import pymc  # noqa: F401
        import arviz  # noqa: F401
    except ImportError:
        return "Error: PyMC not installed. Run: pip install 'pymc>=5.0.0' 'arviz>=0.17.0'"

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)

        df = pd.read_csv(data_path, parse_dates=["date"])
        if "territory" not in df.columns:
            return "Error: 'territory' column not found. Use a *_geo.csv file."

        channels    = list(config["channels"].keys())
        terr_config = config.get("territories", {})

        all_results = {
            "model":       "Bayesian MMM (PyMC, per territory)",
            "frequency":   freq,
            "territories": {},
        }

        summary_lines = [
            f"Geo Bayesian MMM — {freq} data",
            f"{'Territory':<14} {'R²':<7} {'MAPE':<8} {'R̂ max':<8} {'Conv':<6} Top channel (HDI width)",
            "-" * 75,
        ]

        for terr_key in sorted(df["territory"].unique()):
            terr_df = df[df["territory"] == terr_key].copy().reset_index(drop=True)
            label   = terr_config.get(terr_key, {}).get("label", terr_key)

            print(f"\n  ─── {label} ───")
            res = _fit_territory_bayes(terr_df, channels, config, freq, label)

            # Top channel with HDI width as uncertainty indicator
            top_ch      = next(iter(res["channels"]))
            top_data    = res["channels"][top_ch]
            hdi_width   = round(top_data["contribution_hdi_95"] - top_data["contribution_hdi_5"], 0)
            conv_icon   = "✓" if res["mcmc"]["converged"] else "⚠"
            rhat_str    = str(res["mcmc"]["max_rhat"]) if res["mcmc"]["max_rhat"] else "—"
            mape_str    = f"{res['mape_pct']:.1f}%"

            all_results["territories"][terr_key] = {"label": label, **res}

            summary_lines.append(
                f"{label:<14} {res['r_squared_posterior_mean']:<7.3f} {mape_str:<8}"
                f"{rhat_str:<8} {conv_icon:<6} {top_data['label']} (±{hdi_width:,.0f})"
            )

        out_path = data_path.replace("_geo.csv", "_geo_bayesian_results.json")
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)

        summary_lines.append(f"\nSaved to: {out_path}")
        return "\n".join(summary_lines)

    except Exception as exc:
        import traceback
        return f"Error in geo Bayesian MMM:\n{exc}\n\n{traceback.format_exc()}"
