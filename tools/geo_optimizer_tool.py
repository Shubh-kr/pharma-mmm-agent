"""
tools/geo_optimizer_tool.py
============================
Two-level geo budget optimiser:
  Level 1 — per-territory channel mix optimisation (same SLSQP logic as optimizer_tool.py)
  Level 2 — national territory allocation optimisation (shift budget toward high-ROI territories)
"""

import json

import numpy as np
import yaml
from langchain.tools import tool
from scipy.optimize import minimize

from tools.transforms import geometric_adstock, hill_saturation


def _territory_response(spend: float, decay: float, alpha: float, roi: float, n_periods: int) -> float:
    arr = np.full(n_periods, spend)
    ads = geometric_adstock(arr, decay)
    sat = hill_saturation(ads, alpha)
    return roi * sat.mean() * spend


def _optimize_territory_channels(terr_result: dict, config: dict, terr_budget_k: float,
                                   freq: str) -> dict:
    """Run SLSQP within a single territory to find the optimal channel mix."""
    channels   = config["channels"]
    ch_names   = list(channels.keys())
    n_periods  = 104 if freq == "weekly" else 36
    opt_cfg    = config["optimizer"]
    min_share  = opt_cfg["min_channel_share"]
    max_share  = opt_cfg["max_channel_share"]
    max_up     = opt_cfg.get("max_spend_increase_factor", 2.5)
    max_down   = opt_cfg.get("max_spend_decrease_factor", 0.5)

    fitted_roi    = {}
    current_spend = {}
    for ch in ch_names:
        ch_data = terr_result.get("channels", {}).get(ch, {})
        fitted_roi[ch]    = max(ch_data.get("estimated_roi", channels[ch]["prior_roi"]), 0.05)
        periods           = terr_result.get("n_observations", n_periods)
        current_spend[ch] = ch_data.get("total_spend_k", terr_budget_k / len(ch_names)) / periods

    def objective(x):
        return -sum(
            _territory_response(
                x[i], channels[ch]["adstock_decay"], channels[ch]["saturation"],
                fitted_roi[ch], n_periods
            )
            for i, ch in enumerate(ch_names)
        )

    bounds = []
    for ch in ch_names:
        curr = current_spend[ch]
        lo = max(terr_budget_k * min_share, curr * max_down)
        hi = min(terr_budget_k * max_share, curr * max_up)
        if lo > hi:
            lo = terr_budget_k * min_share
        bounds.append((lo, hi))

    constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - terr_budget_k}]
    x0 = np.array([current_spend[ch] for ch in ch_names])
    x0 = x0 / x0.sum() * terr_budget_k

    result  = minimize(objective, x0, method="SLSQP", bounds=bounds,
                       constraints=constraints, options={"maxiter": 2000, "ftol": 1e-10})
    optimal = result.x
    curr_resp = -objective(x0)
    opt_resp  = -objective(optimal)
    uplift    = ((opt_resp - curr_resp) / (curr_resp + 1e-9)) * 100

    rows = []
    for i, ch in enumerate(ch_names):
        curr  = current_spend[ch]
        optim = optimal[i]
        delta_pct = ((optim - curr) / (curr + 1e-9)) * 100
        action = "Increase ↑" if delta_pct > 5 else "Reduce ↓" if delta_pct < -5 else "Maintain →"
        rows.append({
            "channel":          channels[ch]["label"],
            "channel_key":      ch,
            "type":             channels[ch]["channel_type"],
            "current_spend_k":  round(curr, 1),
            "optimal_spend_k":  round(optim, 1),
            "change_k":         round(optim - curr, 1),
            "change_pct":       round(delta_pct, 1),
            "action":           action,
            "fitted_roi":       round(fitted_roi[ch], 3),
        })

    return {
        "territory_budget_k":          terr_budget_k,
        "current_response_index":      round(curr_resp, 4),
        "optimal_response_index":      round(opt_resp, 4),
        "projected_channel_uplift_pct": round(uplift, 1),
        "channel_allocations":         sorted(rows, key=lambda r: r["change_pct"], reverse=True),
    }


@tool
def run_geo_budget_optimizer_tool(
    results_path: str,
    config_path: str,
    total_national_budget_k: float,
    freq: str = "weekly",
) -> str:
    """
    Two-level geo budget optimiser.

    Level 1: Optimise channel mix within each territory (same SLSQP as the national optimizer).
    Level 2: Suggest national territory allocation — shift budget toward high-ROI territories.

    Args:
        results_path             : path to *_geo_ols_results.json
        config_path              : path to config.yaml
        total_national_budget_k  : total brand budget across all territories ($K / period)
        freq                     : 'weekly' or 'monthly'

    Returns:
        Territory-by-territory optimisation summary + national reallocation suggestion
    """
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        with open(results_path) as f:
            geo_ols = json.load(f)

        terr_config  = config.get("territories", {})
        territories  = geo_ols.get("territories", {})
        terr_keys    = list(territories.keys())
        n_periods    = 104 if freq == "weekly" else 36
        opt_cfg      = config["optimizer"]
        max_up       = opt_cfg.get("max_territory_increase_factor", 1.30)
        max_down     = opt_cfg.get("max_territory_decrease_factor", 0.80)

        # ── Level 1: per-territory channel optimisation ───────────────────────
        terr_opt = {}
        roi_efficiency = {}  # scripts-per-dollar index, used for level-2

        for terr_key in terr_keys:
            t_cfg     = terr_config.get(terr_key, {})
            share     = t_cfg.get("spend_share", 1 / len(terr_keys))
            t_budget  = total_national_budget_k * share
            t_result  = territories[terr_key]

            opt = _optimize_territory_channels(t_result, config, t_budget, freq)
            terr_opt[terr_key] = opt

            # Weighted avg ROI across channels (by current spend)
            total_spend = sum(r["current_spend_k"] for r in opt["channel_allocations"]) + 1e-9
            weighted_roi = sum(
                r["fitted_roi"] * r["current_spend_k"]
                for r in opt["channel_allocations"]
            ) / total_spend
            roi_efficiency[terr_key] = weighted_roi

        # ── Level 2: national territory allocation ────────────────────────────
        n_terr        = len(terr_keys)
        current_shares = np.array([
            terr_config.get(tk, {}).get("spend_share", 1 / n_terr)
            for tk in terr_keys
        ])
        roi_vals = np.array([roi_efficiency[tk] for tk in terr_keys])

        def territory_objective(shares):
            # Maximise weighted ROI efficiency
            return -float(np.dot(shares, roi_vals))

        share_bounds = []
        for i, tk in enumerate(terr_keys):
            curr = current_shares[i]
            lo = max(0.02, curr * max_down)
            hi = min(0.50, curr * max_up)
            if lo > hi:
                lo = 0.02
            share_bounds.append((lo, hi))

        constraints = [{"type": "eq", "fun": lambda s: s.sum() - 1.0}]
        result = minimize(territory_objective, current_shares, method="SLSQP",
                          bounds=share_bounds, constraints=constraints,
                          options={"maxiter": 2000, "ftol": 1e-12})
        optimal_shares = result.x

        territory_alloc = {}
        for i, tk in enumerate(terr_keys):
            curr_share = current_shares[i]
            opt_share  = optimal_shares[i]
            delta_pct  = ((opt_share - curr_share) / (curr_share + 1e-9)) * 100
            action = "Increase ↑" if delta_pct > 3 else "Reduce ↓" if delta_pct < -3 else "Maintain →"
            label  = terr_config.get(tk, {}).get("label", tk)
            territory_alloc[tk] = {
                "label":              label,
                "current_share":      round(float(curr_share), 4),
                "optimal_share":      round(float(opt_share), 4),
                "current_budget_k":   round(float(curr_share * total_national_budget_k), 1),
                "optimal_budget_k":   round(float(opt_share  * total_national_budget_k), 1),
                "change_pct":         round(delta_pct, 1),
                "action":             action,
                "roi_efficiency":     round(roi_efficiency[tk], 4),
                **terr_opt[tk],
            }

        overall_uplift = (
            (float(np.dot(optimal_shares, roi_vals)) - float(np.dot(current_shares, roi_vals)))
            / (float(np.dot(current_shares, roi_vals)) + 1e-9) * 100
        )

        out = {
            "total_national_budget_k":          total_national_budget_k,
            "frequency":                         freq,
            "projected_territory_uplift_pct":   round(overall_uplift, 1),
            "territory_allocation":              territory_alloc,
        }
        out_path = results_path.replace("_geo_ols_results.json", "_geo_budget_optimized.json")
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)

        lines = [
            f"Geo Budget Optimisation — National total: ${total_national_budget_k:,.0f}K ({freq})",
            f"Level-2 territory reallocation uplift: +{overall_uplift:.1f}%",
            "",
            f"{'Territory':<14} {'ROI eff':<10} {'Curr share':<12} {'Opt share':<11} {'Curr $K':<10} {'Opt $K':<10} {'Chan uplift':<13} Action",
            "-" * 90,
        ]
        for tk, ta in territory_alloc.items():
            lines.append(
                f"{ta['label']:<14} {ta['roi_efficiency']:<10.4f} "
                f"{ta['current_share']:<12.1%} {ta['optimal_share']:<11.1%} "
                f"${ta['current_budget_k']:<9,.1f} ${ta['optimal_budget_k']:<9,.1f} "
                f"+{ta['projected_channel_uplift_pct']:.1f}%{'':<7} {ta['action']}"
            )
        lines.append(f"\nSaved to: {out_path}")
        return "\n".join(lines)

    except Exception as e:
        return f"Error in geo budget optimiser: {str(e)}"
