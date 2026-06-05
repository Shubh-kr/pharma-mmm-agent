"""
tools/ols_mmm_tool.py
======================
Frequentist Marketing Mix Model using Ridge Regression.
Ridge (L2 regularisation) prevents multicollinearity-driven sign flips —
a known issue with vanilla OLS when pharma channels are correlated.
Standard practice in production MMM.
"""

import numpy as np
import pandas as pd
import yaml
import json
from langchain.tools import tool
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


def _build_feature_matrix(df, channels, config, freq):
    """
    Returns (X, feature_cols, n_channel_cols).
    n_channel_cols marks where media channel features end in X —
    everything after is a control variable whose coefficient is allowed to be negative.
    """
    feature_cols = []
    X_parts = []

    for ch in channels:
        col = f"{ch}_saturated"
        if col in df.columns:
            X_parts.append(df[col].values.reshape(-1, 1))
            feature_cols.append(ch)

    if not X_parts:
        raise ValueError("No saturated features found. Run apply_all_transforms_tool first.")

    n_channel_cols = len(feature_cols)
    X = np.hstack(X_parts)

    if config["ols_model"].get("seasonality_dummies", True):
        month_dummies = pd.get_dummies(df["month"], prefix="month", drop_first=True)
        X = np.hstack([X, month_dummies.values])
        feature_cols += list(month_dummies.columns)

    congress_col = "congress_week" if freq == "weekly" else "congress_month"
    if config["ols_model"].get("congress_control", True) and congress_col in df.columns:
        X = np.hstack([X, df[congress_col].values.reshape(-1, 1)])
        feature_cols.append("congress_flag")

    # Competitor spend — should carry a negative coefficient (share erosion)
    if config["ols_model"].get("competitor_control", True) and "competitor_spend" in df.columns:
        X = np.hstack([X, df["competitor_spend"].values.reshape(-1, 1)])
        feature_cols.append("competitor_spend")

    # Price index — should carry a negative coefficient (higher price → fewer Rx)
    if config["ols_model"].get("price_control", True) and "price_index" in df.columns:
        X = np.hstack([X, df["price_index"].values.reshape(-1, 1)])
        feature_cols.append("price_index")

    return X, feature_cols, n_channel_cols


@tool
def run_ols_mmm_tool(data_path: str, config_path: str, freq: str = "weekly") -> str:
    """
    Run a Ridge-regularised Marketing Mix Model on the transformed pharma dataset.
    Ridge regression prevents multicollinearity-driven sign flips on correlated
    pharma channels — standard MMM practice.

    Use this AFTER apply_all_transforms_tool has been run.

    Args:
        data_path   : path to the *_transformed.csv file
        config_path : path to config.yaml
        freq        : 'weekly' or 'monthly'

    Returns:
        Formatted summary with channel ROIs and contributions
    """
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)

        df = pd.read_csv(data_path, parse_dates=["date"])
        channels = list(config["channels"].keys())
        outcome_col = config["data"]["outcome_col"]

        y = df[outcome_col].values
        X, feature_cols, n_ch = _build_feature_matrix(df, channels, config, freq)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        alpha = config["ols_model"].get("ridge_alpha", 1.0)
        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(X_scaled, y)
        y_pred = model.predict(X_scaled)

        r2 = r2_score(y, y_pred)
        mape = np.mean(np.abs((y - y_pred) / (y + 1e-9))) * 100

        # Media channel coefficients: non-negativity constraint (standard MMM practice)
        # Negative media coefs = multicollinearity artefact, not real negative ROI
        ch_coefs  = model.coef_[:n_ch].copy()
        ch_coefs  = np.maximum(ch_coefs, 0)
        ch_scales = scaler.scale_[:n_ch]

        # Control variable coefficients: kept as-is (competitor/price should be negative)
        control_coefs = {
            fname: round(float(model.coef_[i] / (scaler.scale_[i] + 1e-9)), 4)
            for i, fname in enumerate(feature_cols)
            if fname in ("competitor_spend", "price_index")
        }

        channel_results = {}
        contributions = {}

        for i, ch in enumerate(channels):
            if f"{ch}_saturated" not in df.columns:
                continue
            sat_vals  = df[f"{ch}_saturated"].values
            contrib   = (ch_coefs[i] / (ch_scales[i] + 1e-9)) * sat_vals.sum()
            contributions[ch] = max(contrib, 0.0)

        # ── Prior-contribution floor ───────────────────────────────────────────
        # When Ridge cannot separately identify a channel (zero coefficient),
        # we estimate its contribution from the config prior_roi. This is standard
        # MMM practice: model-identified channels take precedence; unidentifiable
        # channels get a prior-weighted share, clearly flagged in the output.
        #
        # Why channels get zero in Ridge:
        #   Monthly seasonality dummies (11 features) are collinear with seasonal
        #   channels. Ridge attributes Sep-Nov variance to the dummies, leaving
        #   nothing for those channels. Lowering alpha doesn't help — this is a
        #   collinearity problem, not a regularisation problem. The Bayesian model
        #   solves this properly with informative priors; here we apply a prior floor.
        prior_weight = config["ols_model"].get("prior_contribution_weight", 0.15)
        model_total  = sum(contributions.values())
        zero_chs     = [ch for ch, c in contributions.items() if c < 1e-3]
        contrib_source = {ch: "model" for ch in channels if ch in contributions}

        if prior_weight > 0 and zero_chs and model_total > 0:
            # Target: zero channels collectively get prior_weight fraction of total
            prior_pool   = model_total * prior_weight / (1.0 - prior_weight)
            prior_rois   = {ch: config["channels"][ch]["prior_roi"] for ch in zero_chs}
            prior_sum    = sum(prior_rois.values()) + 1e-9
            for ch in zero_chs:
                contributions[ch]  = prior_pool * (prior_rois[ch] / prior_sum)
                contrib_source[ch] = "prior_estimate"

        total_contribution = sum(contributions.values()) + 1e-9

        for i, ch in enumerate(channels):
            if f"{ch}_saturated" not in df.columns:
                continue
            raw_spend = df[ch].values
            contrib    = contributions.get(ch, 0.0)
            contrib_pct = contrib / total_contribution * 100

            coef_unscaled = ch_coefs[i] / (ch_scales[i] + 1e-9)
            avg_sat   = df[f"{ch}_saturated"].values.mean()
            avg_spend = raw_spend.mean()
            estimated_roi = round(
                float(coef_unscaled * avg_sat / (avg_spend + 1e-9) * 100), 3
            )
            prior_roi   = config["channels"][ch]["prior_roi"]
            blended_roi = round(0.6 * prior_roi + 0.4 * min(estimated_roi, prior_roi * 2), 3)

            channel_results[ch] = {
                "label":              config["channels"][ch]["label"],
                "channel_type":       config["channels"][ch]["channel_type"],
                "avg_weekly_spend_k": round(float(raw_spend.mean()), 1),
                "total_spend_k":      round(float(raw_spend.sum()), 1),
                "model_coefficient":  round(float(ch_coefs[i]), 4),
                "estimated_roi":      blended_roi,
                "total_contribution": round(float(contrib), 1),
                "contribution_pct":   round(contrib_pct, 1),
                "contribution_source": contrib_source.get(ch, "model"),
            }

        sorted_channels = sorted(
            channel_results.items(),
            key=lambda x: x[1]["contribution_pct"],
            reverse=True
        )

        # ── Per-period attribution time series ────────────────────────────────
        # model channels: beta_unscaled × sat[t]
        # prior-estimated channels: distribute total_contribution ∝ saturation
        # baseline: y_pred minus model-channel contributions (intercept + controls
        #           + seasonality dummies)
        contrib_ts = {}
        model_ch_sum = np.zeros(len(y))
        for i, ch in enumerate(channels):
            if f"{ch}_saturated" not in df.columns:
                continue
            sat_vals      = df[f"{ch}_saturated"].values
            coef_unscaled = ch_coefs[i] / (ch_scales[i] + 1e-9)
            if contrib_source.get(ch) == "model":
                ts            = coef_unscaled * sat_vals
                model_ch_sum += ts
            else:
                total_c = contributions.get(ch, 0.0)
                ts      = total_c * sat_vals / (sat_vals.sum() + 1e-9)
            contrib_ts[ch] = [round(float(v), 2) for v in ts]

        baseline_ts = y_pred - model_ch_sum

        result = {
            "model": "Ridge MMM",
            "frequency": freq,
            "n_observations": len(y),
            "r_squared": round(r2, 4),
            "mape_pct": round(mape, 2),
            "baseline_scripts": round(float(model.intercept_), 0),
            "total_brand_spend_k": round(float(sum(
                res["total_spend_k"] for _, res in sorted_channels
            )), 1),
            "avg_period_spend_k": round(float(sum(
                res["total_spend_k"] for _, res in sorted_channels
            ) / len(y)), 1),
            "channels": dict(sorted_channels),
            "controls": control_coefs,
            "dates":                [d.strftime("%Y-%m-%d") for d in df["date"]],
            "actuals":              [round(float(v), 1) for v in y],
            "baseline_timeseries":  [round(float(v), 1) for v in baseline_ts],
            "contribution_timeseries": contrib_ts,
        }

        out_path = data_path.replace("_transformed.csv", "_ols_results.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        n_model = sum(1 for _, r in sorted_channels if r["contribution_source"] == "model")
        n_prior = len(sorted_channels) - n_model

        avg_period_spend = round(
            sum(res["total_spend_k"] for _, res in sorted_channels) / len(y), 1
        )
        summary_lines = [
            f"Ridge MMM Results (R²={r2:.3f}, MAPE={mape:.1f}%)",
            f"Observations: {len(y)} {freq} periods",
            f"Baseline scripts: {model.intercept_:,.0f}",
            f"Avg period spend: ${avg_period_spend:,.1f}K  ← use this as total_budget_k for the optimiser",
            f"Channels: {n_model} model-identified, {n_prior} prior-estimated",
            "",
            f"{'Channel':<30} {'Type':<5} {'Spend $K':<12} {'ROI':<8} {'Contribution%':<16} {'Source'}",
            "-" * 85,
        ]
        for ch, res in sorted_channels:
            src = "model" if res["contribution_source"] == "model" else "prior*"
            summary_lines.append(
                f"{res['label']:<30} {res['channel_type']:<5} "
                f"${res['total_spend_k']:<10,.0f} "
                f"{res['estimated_roi']:<8.3f} "
                f"{res['contribution_pct']:<16.1f} "
                f"{src}"
            )
        if n_prior > 0:
            summary_lines.append(
                f"\n* prior-estimated: Ridge could not separately identify these channels"
                f" (collinear with seasonality dummies). Contribution estimated from"
                f" config prior_roi. See Bayesian model for full uncertainty quantification."
            )
        if control_coefs:
            summary_lines.append("\nControl variable coefficients (unscaled):")
            for ctrl, coef in control_coefs.items():
                direction = "↓ suppresses scripts" if coef < 0 else "↑ (check sign)"
                summary_lines.append(f"  {ctrl:<25} {coef:+.4f}  {direction}")

        summary_lines.append(f"\nFull results saved to: {out_path}")
        return "\n".join(summary_lines)

    except Exception as e:
        return f"Error running Ridge MMM: {str(e)}"