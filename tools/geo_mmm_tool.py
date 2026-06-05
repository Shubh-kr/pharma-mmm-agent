"""
tools/geo_mmm_tool.py
======================
Geo-level Ridge MMM — fits separate models per territory on long-format data.
Reuses the same Ridge + prior-floor logic as ols_mmm_tool.py.

Territory × season ROI split
-------------------------------
After fitting, each channel's beta is evaluated at the *in-season* and
*off-season* average saturation and spend levels separately.  This gives an
ROI estimate for each sub-period without adding collinear interaction features
to the model.

The seasonal effect on ROI arises from saturation: in vaccine season brands
typically spend more, pushing channels further along the diminishing-returns
curve and reducing the marginal ROI.  Channels with high adstock decay and
deep saturation show the largest seasonal ROI swing.

Per-channel additions to JSON output (when vaccine_season column is present):
  roi_in_season   — blended ROI during vaccine season (Sep–Nov)
  roi_off_season  — blended ROI outside vaccine season
  season_lift_pct — (roi_in - roi_off) / roi_off × 100
                    positive = higher ROI in vaccine season
"""

import json

import numpy as np
import pandas as pd
import yaml
from langchain.tools import tool
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from tools.transforms import geometric_adstock, hill_saturation


def _build_territory_features(df: pd.DataFrame, channels: list, config: dict, freq: str):
    """Build (X, feature_cols, n_channel_cols) for one territory's dataframe."""
    feature_cols = []
    X_parts      = []

    for ch in channels:
        col = f"{ch}_saturated"
        if col in df.columns:
            X_parts.append(df[col].values.reshape(-1, 1))
            feature_cols.append(ch)

    if not X_parts:
        raise ValueError("No saturated features found. Transforms must be applied first.")

    n_channel_cols = len(feature_cols)
    X = np.hstack(X_parts)

    if config["ols_model"].get("seasonality_dummies", True):
        month_dummies = pd.get_dummies(df["month"], prefix="month", drop_first=True)
        X = np.hstack([X, month_dummies.values.astype(float)])
        feature_cols += list(month_dummies.columns)

    congress_col = "congress_week" if freq == "weekly" else "congress_month"
    if config["ols_model"].get("congress_control", True) and congress_col in df.columns:
        X = np.hstack([X, df[congress_col].values.reshape(-1, 1).astype(float)])
        feature_cols.append("congress_flag")

    if config["ols_model"].get("competitor_control", True) and "competitor_spend" in df.columns:
        X = np.hstack([X, df["competitor_spend"].values.reshape(-1, 1).astype(float)])
        feature_cols.append("competitor_spend")

    if config["ols_model"].get("price_control", True) and "price_index" in df.columns:
        X = np.hstack([X, df["price_index"].values.reshape(-1, 1).astype(float)])
        feature_cols.append("price_index")

    return X, feature_cols, n_channel_cols


def _season_roi_split(
    sat_vals: np.ndarray,
    raw_spend: np.ndarray,
    in_mask: np.ndarray,
    blended_roi: float,
) -> dict:
    """
    Post-hoc seasonal ROI split via marginal efficiency ratio.

    Computes eff = avg_sat / avg_spend for each season.  The ratio captures the
    saturation-driven ROI difference (more in-season spend → deeper diminishing
    returns → lower eff → lower marginal ROI) without re-clipping or re-blending.

    Scales blended_roi so the observation-weighted mean is preserved:
      w_in * roi_in + w_off * roi_off = blended_roi

    Returns {} if either season is missing or avg spend is zero.
    """
    off_mask = ~in_mask
    if not (in_mask.any() and off_mask.any()):
        return {}

    avg_sat_in    = float(sat_vals[in_mask].mean())
    avg_sat_off   = float(sat_vals[off_mask].mean())
    avg_spend_in  = float(raw_spend[in_mask].mean())
    avg_spend_off = float(raw_spend[off_mask].mean())

    if avg_spend_in < 1e-6 or avg_spend_off < 1e-6:
        return {}

    eff_in  = avg_sat_in  / avg_spend_in
    eff_off = avg_sat_off / avg_spend_off

    if eff_off < 1e-9:
        return {}

    ratio = eff_in / eff_off  # < 1 → in-season marginal ROI lower (saturation)

    n_in  = int(in_mask.sum())
    w_in  = n_in / len(in_mask)
    w_off = 1.0 - w_in

    # roi_off * (w_in * ratio + w_off) = blended_roi  →  preserves weighted mean
    roi_off = blended_roi / (w_in * ratio + w_off)
    roi_in  = ratio * roi_off

    return {
        "roi_in_season":   round(roi_in,  3),
        "roi_off_season":  round(roi_off, 3),
        "season_lift_pct": round((ratio - 1.0) * 100, 1),
    }


def _fit_territory(terr_df: pd.DataFrame, channels: list, config: dict, freq: str) -> dict:
    """Apply transforms inline and fit Ridge MMM for one territory slice."""
    for ch in channels:
        if ch not in terr_df.columns:
            continue
        params = config["channels"][ch]
        adstocked = geometric_adstock(terr_df[ch].values, params["adstock_decay"])
        saturated = hill_saturation(adstocked, params["saturation"])
        terr_df[f"{ch}_adstocked"] = adstocked
        terr_df[f"{ch}_saturated"] = saturated

    outcome_col = config["data"]["outcome_col"]
    y = terr_df[outcome_col].values

    X, feature_cols, n_ch = _build_territory_features(terr_df, channels, config, freq)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    alpha = config["ols_model"].get("ridge_alpha", 1.0)
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X_scaled, y)
    y_pred = model.predict(X_scaled)

    r2   = r2_score(y, y_pred)
    mape = np.mean(np.abs((y - y_pred) / (y + 1e-9))) * 100

    ch_coefs  = np.maximum(model.coef_[:n_ch].copy(), 0)
    ch_scales = scaler.scale_[:n_ch]

    control_coefs = {
        fname: round(float(model.coef_[i] / (scaler.scale_[i] + 1e-9)), 4)
        for i, fname in enumerate(feature_cols)
        if fname in ("competitor_spend", "price_index")
    }

    contributions = {}
    for i, ch in enumerate(channels):
        if f"{ch}_saturated" not in terr_df.columns:
            continue
        sat_vals = terr_df[f"{ch}_saturated"].values
        contrib  = (ch_coefs[i] / (ch_scales[i] + 1e-9)) * sat_vals.sum()
        contributions[ch] = max(contrib, 0.0)

    prior_weight = config["ols_model"].get("prior_contribution_weight", 0.15)
    model_total  = sum(contributions.values())
    zero_chs     = [ch for ch, c in contributions.items() if c < 1e-3]
    contrib_source = {ch: "model" for ch in channels if ch in contributions}

    if prior_weight > 0 and zero_chs and model_total > 0:
        prior_pool = model_total * prior_weight / (1.0 - prior_weight)
        prior_rois = {ch: config["channels"][ch]["prior_roi"] for ch in zero_chs}
        prior_sum  = sum(prior_rois.values()) + 1e-9
        for ch in zero_chs:
            contributions[ch]  = prior_pool * (prior_rois[ch] / prior_sum)
            contrib_source[ch] = "prior_estimate"

    total_contribution = sum(contributions.values()) + 1e-9

    # Season masks for post-hoc split (requires vaccine_season column)
    in_mask = (
        terr_df["vaccine_season"].values == 1
        if "vaccine_season" in terr_df.columns else None
    )

    channel_results = {}
    for i, ch in enumerate(channels):
        if f"{ch}_saturated" not in terr_df.columns:
            continue
        raw_spend   = terr_df[ch].values
        sat_vals    = terr_df[f"{ch}_saturated"].values
        contrib     = contributions.get(ch, 0.0)
        contrib_pct = contrib / total_contribution * 100

        beta_unscaled = ch_coefs[i] / (ch_scales[i] + 1e-9)
        avg_sat       = sat_vals.mean()
        avg_spend     = raw_spend.mean()
        prior_roi     = config["channels"][ch]["prior_roi"]

        estimated_roi_raw = float(beta_unscaled * avg_sat / (avg_spend + 1e-9) * 100)
        blended_roi = round(0.6 * prior_roi + 0.4 * min(estimated_roi_raw, prior_roi * 2), 3)

        ch_result = {
            "label":               config["channels"][ch]["label"],
            "channel_type":        config["channels"][ch]["channel_type"],
            "avg_period_spend_k":  round(float(avg_spend), 1),
            "avg_weekly_spend_k":  round(float(avg_spend), 1),  # kept for dashboard compat
            "total_spend_k":       round(float(raw_spend.sum()), 1),
            "estimated_roi":       blended_roi,
            "total_contribution":  round(float(contrib), 1),
            "contribution_pct":    round(contrib_pct, 1),
            "contribution_source": contrib_source.get(ch, "model"),
        }

        # Only add season split for model-identified channels (prior-estimated ones
        # have beta ≈ 0 so the split adds no information)
        if (in_mask is not None
                and contrib_source.get(ch) == "model"
                and beta_unscaled > 1e-6):
            split = _season_roi_split(sat_vals, raw_spend, in_mask, blended_roi)
            ch_result.update(split)

        channel_results[ch] = ch_result

    total_spend = sum(r["total_spend_k"] for r in channel_results.values())
    has_season = any("season_lift_pct" in v for v in channel_results.values())

    return {
        "r_squared":               round(r2, 4),
        "mape_pct":                round(mape, 2),
        "n_observations":          len(y),
        "baseline_scripts":        round(float(model.intercept_), 0),
        "total_spend_k":           round(total_spend, 1),
        "avg_period_spend_k":      round(total_spend / len(y), 1),
        "has_season_interactions": has_season,
        "channels":                channel_results,
        "controls":                control_coefs,
    }


@tool
def run_geo_ols_mmm_tool(data_path: str, config_path: str, freq: str = "weekly") -> str:
    """
    Run Ridge MMM separately for each territory in a long-format geo dataset.

    Applies adstock + saturation transforms in-memory per territory slice,
    then fits Ridge regression with the same prior-floor logic as the national model.
    Adds post-hoc seasonal ROI split (roi_in_season / roi_off_season / season_lift_pct)
    for model-identified channels using vaccine_season grouping.

    Args:
        data_path   : path to *_geo.csv (long format, must have 'territory' column)
        config_path : path to config.yaml
        freq        : 'weekly' or 'monthly'

    Returns:
        Territory-by-territory summary with R², MAPE, and top channels
    """
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)

        df = pd.read_csv(data_path, parse_dates=["date"])
        if "territory" not in df.columns:
            return "Error: 'territory' column not found. Use a *_geo.csv file."

        channels    = list(config["channels"].keys())
        territories = config.get("territories", {})

        all_results = {"model": "Ridge MMM (per territory)", "frequency": freq, "territories": {}}
        summary_lines = [
            f"Geo Ridge MMM — {freq} data",
            f"{'Territory':<14} {'R²':<7} {'MAPE':<8} {'Baseline':<10} {'Avg spend $K':<14} Top channel",
            "-" * 75,
        ]

        for terr_key in sorted(df["territory"].unique()):
            terr_df  = df[df["territory"] == terr_key].copy().reset_index(drop=True)
            label    = territories.get(terr_key, {}).get("label", terr_key)
            res      = _fit_territory(terr_df, channels, config, freq)

            top_ch    = max(res["channels"].items(), key=lambda x: x[1]["contribution_pct"])
            top_label = top_ch[1]["label"]

            all_results["territories"][terr_key] = {"label": label, **res}
            mape_str = f"{res['mape_pct']:.1f}%"
            summary_lines.append(
                f"{label:<14} {res['r_squared']:<7.3f} {mape_str:<9}"
                f"{res['baseline_scripts']:<10,.0f} "
                f"${res['avg_period_spend_k']:<13,.1f} {top_label}"
            )

        out_path = data_path.replace("_geo.csv", "_geo_ols_results.json")
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)

        any_season = any(
            t.get("has_season_interactions", False)
            for t in all_results["territories"].values()
        )
        if any_season:
            summary_lines.append(
                "\n✓ Seasonal ROI split fitted. "
                "See roi_in_season / roi_off_season / season_lift_pct per channel in JSON."
            )

        summary_lines.append(f"\nSaved to: {out_path}")
        return "\n".join(summary_lines)

    except Exception as e:
        import traceback
        return f"Error in geo Ridge MMM: {str(e)}\n{traceback.format_exc()}"
