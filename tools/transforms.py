"""
tools/transforms.py
====================
Adstock and saturation transforms used by the MMM analytics agent.
These are wrapped as LangChain tools so the agent can call them autonomously.
"""

import numpy as np
import pandas as pd
from langchain.tools import tool


# ── Core math ────────────────────────────────────────────────────────────────

def geometric_adstock(spend: np.ndarray, decay: float) -> np.ndarray:
    """
    Geometric adstock: carries over past spend at a decaying rate.
    Models how advertising effects linger after the spend stops.

    Args:
        spend : raw spend array (weekly or monthly)
        decay : carryover rate between 0 (no carryover) and 1 (full carryover)
                pharma norms: rep visits ~0.6, TV ~0.5, email ~0.35

    Returns:
        adstocked spend array, same shape as input
    """
    result = np.zeros_like(spend, dtype=float)
    result[0] = spend[0]
    for t in range(1, len(spend)):
        result[t] = spend[t] + decay * result[t - 1]
    return result


def hill_saturation(x: np.ndarray, alpha: float) -> np.ndarray:
    """
    Hill / power saturation: models diminishing returns on spend.
    Higher alpha = faster saturation (channel hits ceiling quickly).

    Args:
        x     : adstocked spend array
        alpha : saturation exponent (0 < alpha < 1)
                pharma norms: TV ~0.75 (saturates fast), email ~0.80

    Returns:
        saturated array, normalised 0–1
    """
    x_norm = x / (x.max() + 1e-9)
    return x_norm ** alpha


# ── LangChain tool wrappers ───────────────────────────────────────────────────

@tool
def apply_adstock_tool(channel_name: str, decay: float, data_path: str) -> str:
    """
    Apply geometric adstock transform to a single channel's spend data.

    Use this tool when you need to account for the carryover effect of
    advertising spend in a pharma campaign dataset.

    Args:
        channel_name : name of the spend column in the CSV (e.g. 'rep_visits')
        decay        : adstock decay rate between 0 and 1
        data_path    : path to the MMM CSV dataset

    Returns:
        Summary string with before/after spend statistics
    """
    try:
        df = pd.read_csv(data_path, parse_dates=["date"])
        if channel_name not in df.columns:
            return f"Error: column '{channel_name}' not found. Available: {list(df.columns)}"

        raw = df[channel_name].values
        adstocked = geometric_adstock(raw, decay)

        df[f"{channel_name}_adstocked"] = np.round(adstocked, 2)
        df.to_csv(data_path.replace(".csv", "_transformed.csv"), index=False)

        return (
            f"Adstock applied to '{channel_name}' (decay={decay}):\n"
            f"  Raw mean spend:      ${raw.mean():.1f}K\n"
            f"  Adstocked mean:      ${adstocked.mean():.1f}K\n"
            f"  Carryover uplift:    {((adstocked.mean() / raw.mean()) - 1) * 100:.1f}%\n"
            f"  Saved to: {data_path.replace('.csv', '_transformed.csv')}"
        )
    except Exception as e:
        return f"Error applying adstock: {str(e)}"


@tool
def apply_saturation_tool(channel_name: str, alpha: float, data_path: str) -> str:
    """
    Apply Hill saturation transform to a single channel's adstocked spend.

    Use this tool after adstock has been applied, to model diminishing
    returns — the point where doubling spend no longer doubles impact.

    Args:
        channel_name : base name of the channel (e.g. 'rep_visits')
                       will look for '{channel_name}_adstocked' column first
        alpha        : saturation alpha (lower = more saturation)
        data_path    : path to the transformed MMM CSV

    Returns:
        Summary string with saturation statistics
    """
    try:
        df = pd.read_csv(data_path, parse_dates=["date"])
        adstocked_col = f"{channel_name}_adstocked"
        source_col = adstocked_col if adstocked_col in df.columns else channel_name

        if source_col not in df.columns:
            return f"Error: neither '{adstocked_col}' nor '{channel_name}' found."

        x = df[source_col].values
        saturated = hill_saturation(x, alpha)
        df[f"{channel_name}_saturated"] = np.round(saturated, 4)
        df.to_csv(data_path, index=False)

        return (
            f"Saturation applied to '{channel_name}' (alpha={alpha}):\n"
            f"  Max saturated value: {saturated.max():.4f}\n"
            f"  Mean saturated value:{saturated.mean():.4f}\n"
            f"  Effective range:     {saturated.min():.4f} – {saturated.max():.4f}\n"
            f"  Interpretation: channel reaches ~{saturated.mean()*100:.0f}% of ceiling on average"
        )
    except Exception as e:
        return f"Error applying saturation: {str(e)}"


@tool
def apply_all_transforms_tool(data_path: str, config_path: str) -> str:
    """
    Apply adstock and saturation transforms to ALL channels at once,
    using decay and saturation parameters from config.yaml.

    Use this as the first analytical step — transforms all 12 channels
    in one call before running the MMM model.

    Args:
        data_path   : path to raw MMM CSV (weekly or monthly)
        config_path : path to config.yaml

    Returns:
        Summary of transforms applied across all channels
    """
    import yaml

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)

        df = pd.read_csv(data_path, parse_dates=["date"])
        channels = config["channels"]
        results = []

        for ch, params in channels.items():
            if ch not in df.columns:
                continue
            raw = df[ch].values
            adstocked = geometric_adstock(raw, params["adstock_decay"])
            saturated = hill_saturation(adstocked, params["saturation"])
            df[f"{ch}_adstocked"] = np.round(adstocked, 2)
            df[f"{ch}_saturated"] = np.round(saturated, 4)
            results.append(
                f"  {ch:<25} decay={params['adstock_decay']}  "
                f"sat={params['saturation']}  "
                f"avg_saturated={saturated.mean():.3f}"
            )

        out_path = data_path.replace(".csv", "_transformed.csv")
        df.to_csv(out_path, index=False)

        return (
            f"All channel transforms complete ({len(results)} channels):\n"
            + "\n".join(results)
            + f"\n\nTransformed dataset saved to: {out_path}"
        )
    except Exception as e:
        return f"Error in bulk transform: {str(e)}"