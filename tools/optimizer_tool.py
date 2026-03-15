"""
tools/optimizer_tool.py
========================
Budget reallocation optimiser — uses fitted ROI from OLS/Ridge results
(not config priors) to find the spend mix that maximises scripts_written.
"""

import numpy as np
import pandas as pd
import yaml
import json
from langchain.tools import tool
from scipy.optimize import minimize
from tools.transforms import geometric_adstock, hill_saturation


def _response(spend, decay, alpha, roi, n_periods):
    arr = np.full(n_periods, spend)
    ads = geometric_adstock(arr, decay)
    sat = hill_saturation(ads, alpha)
    return roi * sat.mean() * spend


@tool
def run_budget_optimizer_tool(
    results_path: str,
    config_path: str,
    total_budget_k: float,
    freq: str = "weekly"
) -> str:
    """
    Optimise budget allocation using fitted ROI from the Ridge MMM results.

    Args:
        results_path   : path to _ols_results.json
        config_path    : path to config.yaml
        total_budget_k : total budget in $K per period
        freq           : 'weekly' or 'monthly'

    Returns:
        Budget reallocation table with current vs recommended spend
    """
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        with open(results_path) as f:
            ols = json.load(f)

        channels = config["channels"]
        ch_names = list(channels.keys())
        n_periods = 104 if freq == "weekly" else 36
        opt_cfg = config["optimizer"]
        min_share = opt_cfg["min_channel_share"]
        max_share = opt_cfg["max_channel_share"]

        # Use FITTED roi from OLS results — not config priors
        fitted_roi = {}
        current_spend = {}
        for ch in ch_names:
            ch_data = ols.get("channels", {}).get(ch, {})
            fitted_roi[ch] = max(ch_data.get("estimated_roi", channels[ch]["prior_roi"]), 0.05)
            periods = ols.get("n_observations", n_periods)
            current_spend[ch] = ch_data.get("total_spend_k", channels[ch]["base"] * n_periods) / periods

        def objective(x):
            total = 0.0
            for i, ch in enumerate(ch_names):
                total += _response(
                    x[i],
                    channels[ch]["adstock_decay"],
                    channels[ch]["saturation"],
                    fitted_roi[ch],
                    n_periods
                )
            return -total

        bounds = [(total_budget_k * min_share, total_budget_k * max_share) for _ in ch_names]
        constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - total_budget_k}]
        x0 = np.array([current_spend[ch] for ch in ch_names])
        x0 = x0 / x0.sum() * total_budget_k

        result = minimize(objective, x0, method="SLSQP", bounds=bounds,
                         constraints=constraints, options={"maxiter": 2000, "ftol": 1e-10})

        optimal = result.x
        curr_resp = -objective(x0)
        opt_resp = -objective(optimal)
        uplift = ((opt_resp - curr_resp) / (curr_resp + 1e-9)) * 100

        rows = []
        for i, ch in enumerate(ch_names):
            curr = current_spend[ch]
            optim = optimal[i]
            delta_pct = ((optim - curr) / (curr + 1e-9)) * 100
            action = "Increase ↑" if delta_pct > 5 else "Reduce ↓" if delta_pct < -5 else "Maintain →"
            rows.append({
                "channel": channels[ch]["label"],
                "type": channels[ch]["channel_type"],
                "current_spend_k": round(curr, 1),
                "optimal_spend_k": round(optim, 1),
                "change_k": round(optim - curr, 1),
                "change_pct": round(delta_pct, 1),
                "action": action,
                "fitted_roi": round(fitted_roi[ch], 3)
            })

        rows.sort(key=lambda x: x["change_pct"], reverse=True)

        out_path = results_path.replace("_ols_results.json", "_budget_optimized.json")
        with open(out_path, "w") as f:
            json.dump({
                "total_budget_k": total_budget_k,
                "current_response_index": round(curr_resp, 4),
                "optimal_response_index": round(opt_resp, 4),
                "projected_uplift_pct": round(uplift, 1),
                "allocations": rows
            }, f, indent=2)

        lines = [
            f"Budget Optimisation — Total: ${total_budget_k:,.0f}K ({freq})",
            f"Projected uplift in scripts_written: +{uplift:.1f}%",
            "",
            f"{'Channel':<30} {'ROI':<7} {'Curr $K':<10} {'Optim $K':<11} {'Change':<10} Action",
            "-" * 80,
        ]
        for r in rows:
            lines.append(
                f"{r['channel']:<30} {r['fitted_roi']:<7.3f} "
                f"${r['current_spend_k']:<9,.1f} "
                f"${r['optimal_spend_k']:<10,.1f} "
                f"{r['change_pct']:+.1f}%{'':<5} {r['action']}"
            )
        lines.append(f"\nSaved to: {out_path}")
        return "\n".join(lines)

    except Exception as e:
        return f"Error in budget optimiser: {str(e)}"