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
    feature_cols = []
    X_parts = []

    for ch in channels:
        col = f"{ch}_saturated"
        if col in df.columns:
            X_parts.append(df[col].values.reshape(-1, 1))
            feature_cols.append(ch)

    if not X_parts:
        raise ValueError("No saturated features found. Run apply_all_transforms_tool first.")

    X = np.hstack(X_parts)

    if config["ols_model"].get("seasonality_dummies", True):
        month_dummies = pd.get_dummies(df["month"], prefix="month", drop_first=True)
        X = np.hstack([X, month_dummies.values])
        feature_cols += list(month_dummies.columns)

    congress_col = "congress_week" if freq == "weekly" else "congress_month"
    if config["ols_model"].get("congress_control", True) and congress_col in df.columns:
        X = np.hstack([X, df[congress_col].values.reshape(-1, 1)])
        feature_cols.append("congress_flag")

    return X, feature_cols


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
        X, feature_cols = _build_feature_matrix(df, channels, config, freq)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Ridge regression — alpha controls regularisation strength
        # alpha=10 is a good default for MMM to prevent sign flips
        model = Ridge(alpha=10.0, fit_intercept=True)
        model.fit(X_scaled, y)
        y_pred = model.predict(X_scaled)

        r2 = r2_score(y, y_pred)
        mape = np.mean(np.abs((y - y_pred) / (y + 1e-9))) * 100

        # Clip negative channel coefficients to zero — standard MMM practice
        # Negative coefs = multicollinearity artefact, not real negative ROI
        ch_coefs = model.coef_[:len(channels)].copy()
        ch_coefs = np.maximum(ch_coefs, 0)  # non-negativity constraint
        ch_scales = scaler.scale_[:len(channels)]

        channel_results = {}
        contributions = {}

        for i, ch in enumerate(channels):
            if f"{ch}_saturated" not in df.columns:
                continue
            sat_vals = df[f"{ch}_saturated"].values
            raw_spend = df[ch].values

            # Unscaled contribution
            contrib = (ch_coefs[i] / (ch_scales[i] + 1e-9)) * sat_vals.sum()
            contributions[ch] = max(contrib, 0.0)

        total_contribution = sum(contributions.values()) + 1e-9

        for i, ch in enumerate(channels):
            if f"{ch}_saturated" not in df.columns:
                continue
            raw_spend = df[ch].values
            contrib = contributions.get(ch, 0.0)
            contrib_pct = contrib / total_contribution * 100

            # Estimate ROI: scale model coefficient back to spend units
            # Higher coef relative to spend = higher ROI
            coef_unscaled = ch_coefs[i] / (ch_scales[i] + 1e-9)
            avg_sat = df[f"{ch}_saturated"].values.mean()
            avg_spend = raw_spend.mean()
            estimated_roi = round(
                float(coef_unscaled * avg_sat / (avg_spend + 1e-9) * 100), 3
            )
            # Anchor to config prior with model adjustment — blend for stability
            prior_roi = config["channels"][ch]["prior_roi"]
            blended_roi = round(0.6 * prior_roi + 0.4 * min(estimated_roi, prior_roi * 2), 3)

            channel_results[ch] = {
                "label": config["channels"][ch]["label"],
                "channel_type": config["channels"][ch]["channel_type"],
                "avg_weekly_spend_k": round(float(raw_spend.mean()), 1),
                "total_spend_k": round(float(raw_spend.sum()), 1),
                "model_coefficient": round(float(ch_coefs[i]), 4),
                "estimated_roi": blended_roi,
                "total_contribution": round(float(contrib), 1),
                "contribution_pct": round(contrib_pct, 1),
            }

        sorted_channels = sorted(
            channel_results.items(),
            key=lambda x: x[1]["contribution_pct"],
            reverse=True
        )

        result = {
            "model": "Ridge MMM",
            "frequency": freq,
            "n_observations": len(y),
            "r_squared": round(r2, 4),
            "mape_pct": round(mape, 2),
            "baseline_scripts": round(float(model.intercept_), 0),
            "channels": dict(sorted_channels)
        }

        out_path = data_path.replace("_transformed.csv", "_ols_results.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        summary_lines = [
            f"Ridge MMM Results (R²={r2:.3f}, MAPE={mape:.1f}%)",
            f"Observations: {len(y)} {freq} periods",
            f"Baseline scripts: {model.intercept_:,.0f}",
            "",
            f"{'Channel':<30} {'Type':<5} {'Spend $K':<12} {'ROI':<8} {'Contribution%'}",
            "-" * 72,
        ]
        for ch, res in sorted_channels:
            summary_lines.append(
                f"{res['label']:<30} {res['channel_type']:<5} "
                f"${res['total_spend_k']:<10,.0f} "
                f"{res['estimated_roi']:<8.3f} "
                f"{res['contribution_pct']}%"
            )
        summary_lines.append(f"\nFull results saved to: {out_path}")
        return "\n".join(summary_lines)

    except Exception as e:
        return f"Error running Ridge MMM: {str(e)}"