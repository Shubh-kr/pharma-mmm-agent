"""
Scenario planner — inverse budget optimiser.

Forward optimizer: given total_budget → maximise scripts.
Inverse (this tool): given target_scripts/period → minimise total_budget.

Response function is calibrated to the OLS results:
  scripts_i(spend) = (total_contribution_i / n_periods) × (spend / avg_spend_i)^alpha_i

This anchors at the actual fitted contribution and uses the saturation alpha to
capture diminishing returns, without depending on the blended estimated_roi field
(which is a regularised prior blend, not a scripts-per-$K value).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def _response_calibrated(
    spend_k_per_period: float,
    avg_spend_k: float,
    contrib_per_period: float,
    alpha: float,
) -> float:
    """
    Calibrated saturating response for one channel.

    Returns scripts/period for a given per-period spend, anchored to the
    actual OLS contribution at the actual average spend level.

    Model: scripts(s) = C_0 × (s / s_0)^alpha
      C_0   = contrib_per_period (actual, from OLS)
      s_0   = avg_spend_k (actual, from OLS)
      alpha = saturation exponent (< 1 → diminishing returns)
    """
    if avg_spend_k <= 0 or spend_k_per_period < 0:
        return 0.0
    ratio = max(spend_k_per_period / avg_spend_k, 1e-9)
    return float(contrib_per_period * (ratio ** alpha))


def _build_channel_params(ols: dict, config: dict, n_periods: int) -> list[dict]:
    """Extract per-channel parameters needed for the scenario optimizer."""
    channels_cfg = config["channels"]
    rows = []
    for ch, cd in ols.get("channels", {}).items():
        if ch not in channels_cfg:
            continue
        cfg         = channels_cfg[ch]
        total_spend = cd.get("total_spend_k", 0.0)
        total_contrib = cd.get("total_contribution", 0.0)
        rows.append({
            "key":              ch,
            "label":            cfg["label"],
            "channel_type":     cfg["channel_type"],
            "alpha":            cfg["saturation"],
            "avg_spend_k":      total_spend / n_periods,
            "contrib_per_period": total_contrib / n_periods,
            "current_spend_k":  total_spend / n_periods,
        })
    return rows


def _compute_bounds(ch_params: list[dict], opt_cfg: dict,
                    current_budget: float, relax: bool) -> list[tuple]:
    min_share = opt_cfg["min_channel_share"]
    max_share = opt_cfg["max_channel_share"]
    max_up    = 5.0 if relax else opt_cfg.get("max_spend_increase_factor", 2.5)
    max_down  = 0.1 if relax else opt_cfg.get("max_spend_decrease_factor", 0.5)
    bounds = []
    for p in ch_params:
        curr = p["current_spend_k"]
        lo   = max(current_budget * min_share, curr * max_down)
        hi   = min(current_budget * max_share * 3, curr * max_up)
        if lo > hi:
            lo = current_budget * min_share
        bounds.append((lo, hi))
    return bounds


def _total_scripts(x: np.ndarray, ch_params: list[dict],
                   baseline_per_period: float) -> float:
    total = baseline_per_period
    for i, p in enumerate(ch_params):
        total += _response_calibrated(
            x[i], p["avg_spend_k"], p["contrib_per_period"], p["alpha"]
        )
    return total


# ── Inverse optimiser ─────────────────────────────────────────────────────────

def solve_target_to_budget(
    target_scripts_per_period: float,
    ols: dict,
    config: dict,
    freq: str = "weekly",
    relax_corridors: bool = False,
) -> dict:
    """
    Find the minimum per-period budget that achieves target_scripts_per_period.

    Returns a dict with:
        feasible, infeasibility_reason,
        target_scripts, required_budget_k, current_budget_k,
        budget_delta_k, budget_delta_pct,
        baseline_per_period, achieved_scripts,
        channels (list[dict])
    """
    n_periods = 104 if freq == "weekly" else 36
    ch_params = _build_channel_params(ols, config, n_periods)
    opt_cfg   = config["optimizer"]

    baseline_per_period = ols.get("baseline_scripts", 0.0) / n_periods
    current_budget      = sum(p["current_spend_k"] for p in ch_params)
    bounds              = _compute_bounds(ch_params, opt_cfg, current_budget, relax_corridors)

    # Current scripts from calibrated model
    x_curr = np.array([p["current_spend_k"] for p in ch_params])
    current_scripts = _total_scripts(x_curr, ch_params, baseline_per_period)

    # Ceiling: scripts at max corridor spend
    x_max = np.array([b[1] for b in bounds])
    max_scripts = _total_scripts(x_max, ch_params, baseline_per_period)

    if target_scripts_per_period <= baseline_per_period:
        return {
            "feasible": True,
            "infeasibility_reason": None,
            "target_scripts":    target_scripts_per_period,
            "required_budget_k": sum(b[0] for b in bounds),
            "current_budget_k":  round(current_budget, 1),
            "budget_delta_k":    0.0,
            "budget_delta_pct":  0.0,
            "baseline_per_period": round(baseline_per_period, 1),
            "achieved_scripts":  round(baseline_per_period, 1),
            "channels": [],
        }

    if target_scripts_per_period > max_scripts:
        reason = (
            f"Target {target_scripts_per_period:,.0f} scripts/period exceeds corridor-max "
            f"achievable {max_scripts:,.0f}. "
            + ("Try enabling relaxed corridors." if not relax_corridors
               else "Target may be physically unreachable with current ROI estimates.")
        )
        return {
            "feasible": False,
            "infeasibility_reason": reason,
            "target_scripts":    target_scripts_per_period,
            "required_budget_k": None,
            "current_budget_k":  round(current_budget, 1),
            "budget_delta_k":    None,
            "budget_delta_pct":  None,
            "baseline_per_period": round(baseline_per_period, 1),
            "achieved_scripts":  None,
            "channels": [],
        }

    # Objective: minimise total spend
    def objective(x):
        return float(np.sum(x))

    def scripts_constraint(x):
        return _total_scripts(x, ch_params, baseline_per_period) - target_scripts_per_period

    # Warm start: scale current spend proportionally
    lift_needed = (target_scripts_per_period - baseline_per_period) / max(
        current_scripts - baseline_per_period, 1e-6
    )
    x0 = x_curr * max(lift_needed, 0.5)
    x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])

    result = minimize(
        objective, x0,
        method="SLSQP",
        bounds=bounds,
        constraints=[{"type": "ineq", "fun": scripts_constraint}],
        options={"maxiter": 5000, "ftol": 1e-10},
    )

    optimal = result.x
    required_budget = float(np.sum(optimal))
    achieved = _total_scripts(optimal, ch_params, baseline_per_period)

    channel_rows = []
    for i, p in enumerate(ch_params):
        curr  = p["current_spend_k"]
        opt_s = float(optimal[i])
        delta = opt_s - curr
        delta_pct = delta / max(curr, 0.001) * 100
        contrib = _response_calibrated(opt_s, p["avg_spend_k"], p["contrib_per_period"], p["alpha"])
        channel_rows.append({
            "channel_key":      p["key"],
            "channel_label":    p["label"],
            "channel_type":     p["channel_type"],
            "current_spend_k":  round(curr, 2),
            "scenario_spend_k": round(opt_s, 2),
            "delta_k":          round(delta, 2),
            "delta_pct":        round(delta_pct, 1),
            "contribution":     round(contrib, 1),
        })

    channel_rows.sort(key=lambda r: r["scenario_spend_k"], reverse=True)

    return {
        "feasible":             True,
        "infeasibility_reason": None,
        "target_scripts":       round(target_scripts_per_period, 1),
        "required_budget_k":    round(required_budget, 1),
        "current_budget_k":     round(current_budget, 1),
        "budget_delta_k":       round(required_budget - current_budget, 1),
        "budget_delta_pct":     round((required_budget - current_budget) / max(current_budget, 1) * 100, 1),
        "baseline_per_period":  round(baseline_per_period, 1),
        "achieved_scripts":     round(achieved, 1),
        "channels":             channel_rows,
    }


# ── Efficiency frontier ────────────────────────────────────────────────────────

def compute_efficiency_frontier(
    ols: dict,
    config: dict,
    freq: str = "weekly",
    n_points: int = 20,
    lift_range: tuple = (-0.20, 0.40),
) -> list[dict]:
    """
    Sweep NRx targets from (1+lift_min)× to (1+lift_max)× current scripts/period
    and return the (target, min_budget) efficiency curve.
    """
    n_periods = 104 if freq == "weekly" else 36
    ch_params = _build_channel_params(ols, config, n_periods)
    baseline_pp = ols.get("baseline_scripts", 0.0) / n_periods
    x_curr = np.array([p["current_spend_k"] for p in ch_params])
    current_scripts = _total_scripts(x_curr, ch_params, baseline_pp)

    lifts  = np.linspace(lift_range[0], lift_range[1], n_points)
    points = []
    for lift in lifts:
        target = current_scripts * (1.0 + lift)
        result = solve_target_to_budget(target, ols, config, freq)
        points.append({
            "lift_pct":          round(float(lift) * 100, 1),
            "target_scripts":    round(target, 0),
            "required_budget_k": result.get("required_budget_k"),
            "budget_delta_pct":  result.get("budget_delta_pct"),
            "feasible":          result.get("feasible", False),
        })
    return points


# ── Forward: budget → scripts (for comparison) ────────────────────────────────

def solve_budget_to_scripts(
    total_budget_k: float,
    ols: dict,
    config: dict,
    freq: str = "weekly",
) -> dict:
    """
    Maximise scripts/period for a given total per-period budget.
    Returns same structure as solve_target_to_budget.
    """
    n_periods = 104 if freq == "weekly" else 36
    ch_params = _build_channel_params(ols, config, n_periods)
    opt_cfg   = config["optimizer"]

    baseline_pp   = ols.get("baseline_scripts", 0.0) / n_periods
    current_budget = sum(p["current_spend_k"] for p in ch_params)
    bounds         = _compute_bounds(ch_params, opt_cfg, current_budget, False)

    def objective(x):
        return -_total_scripts(x, ch_params, baseline_pp)

    constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - total_budget_k}]
    x0 = np.array([p["current_spend_k"] for p in ch_params])
    x0 = x0 / x0.sum() * total_budget_k

    result = minimize(
        objective, x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 5000, "ftol": 1e-10},
    )

    optimal = result.x
    achieved = _total_scripts(optimal, ch_params, baseline_pp)
    channel_rows = []
    for i, p in enumerate(ch_params):
        curr  = p["current_spend_k"]
        opt_s = float(optimal[i])
        delta_pct = (opt_s - curr) / max(curr, 0.001) * 100
        contrib = _response_calibrated(opt_s, p["avg_spend_k"], p["contrib_per_period"], p["alpha"])
        channel_rows.append({
            "channel_key":      p["key"],
            "channel_label":    p["label"],
            "channel_type":     p["channel_type"],
            "current_spend_k":  round(curr, 2),
            "scenario_spend_k": round(opt_s, 2),
            "delta_k":          round(opt_s - curr, 2),
            "delta_pct":        round(delta_pct, 1),
            "contribution":     round(contrib, 1),
        })

    channel_rows.sort(key=lambda r: r["scenario_spend_k"], reverse=True)

    return {
        "feasible":             True,
        "infeasibility_reason": None,
        "target_scripts":       round(achieved, 1),
        "required_budget_k":    round(total_budget_k, 1),
        "current_budget_k":     round(current_budget, 1),
        "budget_delta_k":       round(total_budget_k - current_budget, 1),
        "budget_delta_pct":     round((total_budget_k - current_budget) / max(current_budget, 1) * 100, 1),
        "baseline_per_period":  round(baseline_pp, 1),
        "achieved_scripts":     round(achieved, 1),
        "channels":             channel_rows,
    }
