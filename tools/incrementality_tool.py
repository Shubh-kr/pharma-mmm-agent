"""
Incrementality testing planner.

Scores every (territory × channel) candidate for a geo holdout / lift test
using four signals derived from the geo OLS and Bayesian MMM outputs:

  1. hdi_uncertainty   — relative HDI width (Bayesian posterior spread)
  2. model_disagreement — |OLS ROI − Bayes ROI| / mean(ROIs)
  3. saturation_headroom — 1 − avg_saturated (room to move spend and see lift)
  4. spend_materiality  — normalised avg spend (bigger signal potential)

All four signals are min-max normalised across the candidate pool before
being combined into a composite score via user-supplied weights.
"""

from __future__ import annotations
from typing import Optional


def compute_incrementality_scores(
    geo_ols: dict,
    geo_bayes: Optional[dict],
    weights: Optional[dict] = None,
) -> list[dict]:
    """Return a ranked list of territory × channel candidates.

    Parameters
    ----------
    geo_ols   : parsed *_geo_ols_results.json
    geo_bayes : parsed *_geo_bayesian_results.json (may be None)
    weights   : dict with keys hdi_uncertainty, model_disagreement,
                saturation_headroom, spend_materiality (must sum to 1).
                Defaults to {0.35, 0.35, 0.20, 0.10}.

    Returns
    -------
    List of candidate dicts, sorted descending by composite_score.
    Each dict contains:
        territory_key, territory_label,
        channel_key, channel_label, channel_type,
        avg_spend_k, ols_roi, bayes_roi,
        hdi_width_pct, model_disagree_pct, saturation_headroom_pct,
        raw_hdi_uncertainty, raw_model_disagreement,
        raw_saturation_headroom, raw_spend_materiality,
        hdi_uncertainty, model_disagreement, saturation_headroom,
        spend_materiality, composite_score,
        missing_bayes (bool)
    """
    if weights is None:
        weights = {
            "hdi_uncertainty": 0.35,
            "model_disagreement": 0.35,
            "saturation_headroom": 0.20,
            "spend_materiality": 0.10,
        }

    ols_territories  = (geo_ols  or {}).get("territories", {})
    bayes_territories = (geo_bayes or {}).get("territories", {})

    rows: list[dict] = []

    for tk, td in ols_territories.items():
        bt = bayes_territories.get(tk, {})
        bt_channels = bt.get("channels", {})

        for ck, cd in td.get("channels", {}).items():
            bc = bt_channels.get(ck, {})

            ols_roi  = cd.get("estimated_roi",  0.0) or 0.0
            bayes_roi = bc.get("estimated_roi", None)
            missing_bayes = bayes_roi is None
            if bayes_roi is None:
                bayes_roi = ols_roi  # fallback — disagreement will be 0

            # HDI width as % of total contribution (relative uncertainty)
            contrib     = bc.get("total_contribution", 1.0) or 1.0
            hdi_5       = bc.get("contribution_hdi_5",  contrib * 0.8)
            hdi_95      = bc.get("contribution_hdi_95", contrib * 1.2)
            hdi_width   = max(hdi_95 - hdi_5, 0.0)
            hdi_width_pct = hdi_width / max(abs(contrib), 1.0) * 100

            # Model disagreement (relative)
            mean_roi = (abs(ols_roi) + abs(bayes_roi)) / 2.0
            disagree  = abs(ols_roi - bayes_roi) / max(mean_roi, 0.01)
            disagree_pct = disagree * 100

            # Saturation headroom (OLS)
            avg_sat  = cd.get("avg_saturated", 0.5) or 0.5
            headroom = max(1.0 - avg_sat, 0.0)

            # Raw spend (absolute)
            spend_k = cd.get("avg_period_spend_k", 0.0) or 0.0

            rows.append({
                "territory_key":   tk,
                "territory_label": td.get("label", tk),
                "channel_key":     ck,
                "channel_label":   cd.get("label", ck),
                "channel_type":    cd.get("channel_type", ""),
                "avg_spend_k":     spend_k,
                "ols_roi":         ols_roi,
                "bayes_roi":       bayes_roi,
                "hdi_width_pct":   hdi_width_pct,
                "model_disagree_pct": disagree_pct,
                "saturation_headroom_pct": headroom * 100,
                # raw signals (before normalisation)
                "raw_hdi_uncertainty":    hdi_width_pct,
                "raw_model_disagreement": disagree_pct,
                "raw_saturation_headroom": headroom,
                "raw_spend_materiality":   spend_k,
                "missing_bayes": missing_bayes,
            })

    if not rows:
        return []

    def _minmax(rows, key):
        vals = [r[key] for r in rows]
        lo, hi = min(vals), max(vals)
        rng = hi - lo
        for r in rows:
            r[key.replace("raw_", "")] = (r[key] - lo) / rng if rng > 0 else 0.5

    for signal in ("raw_hdi_uncertainty", "raw_model_disagreement",
                   "raw_saturation_headroom", "raw_spend_materiality"):
        _minmax(rows, signal)

    w = weights
    for r in rows:
        r["composite_score"] = (
            w["hdi_uncertainty"]    * r["hdi_uncertainty"]
            + w["model_disagreement"] * r["model_disagreement"]
            + w["saturation_headroom"] * r["saturation_headroom"]
            + w["spend_materiality"]   * r["spend_materiality"]
        )

    rows.sort(key=lambda r: r["composite_score"], reverse=True)
    for i, r in enumerate(rows):
        r["rank"] = i + 1

    return rows


def _holdout_design(row: dict, freq: str = "weekly") -> dict:
    """Generate a simple holdout design recommendation for a candidate."""
    headroom = row["saturation_headroom_pct"]
    spend_k  = row["avg_spend_k"]
    score    = row["composite_score"]

    # Recommended holdout duration
    if freq == "weekly":
        if score > 0.65:
            duration = "12–16 weeks"
        elif score > 0.40:
            duration = "8–12 weeks"
        else:
            duration = "6–8 weeks"
    else:
        duration = "4–6 months"

    # Hold-out depth (% spend reduction in test territory)
    if headroom > 60:
        depth = "50% spend reduction"
    elif headroom > 30:
        depth = "30–40% spend reduction"
    else:
        depth = "20–25% spend reduction (low headroom — channel near saturation)"

    # Power note
    if spend_k > 100:
        power = "High spend — strong signal potential; standard geo holdout should be adequately powered."
    elif spend_k > 30:
        power = "Moderate spend — consider a matched-market pair to improve statistical power."
    else:
        power = "Low spend — synthetic control or difference-in-differences recommended for power."

    return {
        "duration":   duration,
        "depth":      depth,
        "power_note": power,
        "approach":   "Geo holdout" if spend_k > 50 else "Matched market / synthetic control",
    }
