"""
Synthetic Pharma MMM Dataset Generator
=======================================
Vaccine Campaign — HCP + Patient Channels
Generates enterprise-grade synthetic data for Marketing Mix Modelling.

Outputs:
  - data/raw/mmm_weekly.csv   : 2 years of weekly data (104 rows)
  - data/raw/mmm_monthly.csv  : 3 years of monthly data (36 rows)

Channels (12 total):
  HCP Channels (7):
    1.  rep_visits          — Field sales rep detailing visits
    2.  medical_congress    — Conference sponsorships & symposia
    3.  journal_advertising — Print/digital journal ads (HCP-targeted)
    4.  hcp_email           — Permission-based HCP email campaigns
    5.  hcp_digital         — Programmatic HCP-targeted display
    6.  speaker_programs    — KOL-led speaker bureau programs
    7.  samples_coupons     — Sample drops & co-pay coupon programs

  Patient / DTC Channels (5):
    8.  dtc_tv              — Direct-to-consumer television
    9.  dtc_digital         — Patient-facing digital / social
    10. dtc_ooh             — Out-of-home (pharmacy, clinic)
    11. patient_email       — CRM patient email / reminder programs
    12. patient_advocacy    — Partnership with patient advocacy groups

Outcome:
    - scripts_written       — Weekly/monthly vaccine prescriptions (proxy)
    - nrx_index             — Normalised new Rx index (0–100)

Seasonality:
    - Flu/vaccine season peak: Sep–Nov
    - Summer trough: Jun–Aug
    - Congress surge: Feb (ACIP), May (ASHP), Oct (IDWeek)

Usage:
    python scripts/generate_dataset.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── Reproducibility ──────────────────────────────────────────────────────────
np.random.seed(42)

# ── Output paths ─────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Channel config ────────────────────────────────────────────────────────────
# Each channel: (base_spend_weekly_$K, roi_coefficient, adstock_decay, saturation_alpha)
CHANNELS = {
    # HCP channels
    "rep_visits":         dict(base=180, roi=0.38, decay=0.6,  sat=0.55),
    "medical_congress":   dict(base=60,  roi=0.52, decay=0.75, sat=0.45),
    "journal_advertising":dict(base=40,  roi=0.22, decay=0.5,  sat=0.65),
    "hcp_email":          dict(base=15,  roi=0.18, decay=0.35, sat=0.80),
    "hcp_digital":        dict(base=55,  roi=0.28, decay=0.4,  sat=0.70),
    "speaker_programs":   dict(base=35,  roi=0.45, decay=0.7,  sat=0.50),
    "samples_coupons":    dict(base=90,  roi=0.33, decay=0.55, sat=0.60),
    # Patient / DTC channels
    "dtc_tv":             dict(base=220, roi=0.20, decay=0.5,  sat=0.75),
    "dtc_digital":        dict(base=85,  roi=0.25, decay=0.38, sat=0.72),
    "dtc_ooh":            dict(base=45,  roi=0.15, decay=0.45, sat=0.68),
    "patient_email":      dict(base=12,  roi=0.16, decay=0.30, sat=0.82),
    "patient_advocacy":   dict(base=20,  roi=0.35, decay=0.65, sat=0.55),
}

CHANNEL_NAMES = list(CHANNELS.keys())


# ── Helper functions ──────────────────────────────────────────────────────────

def adstock(spend: np.ndarray, decay: float) -> np.ndarray:
    """Geometric adstock transform — carries-over effect of past spend."""
    adstocked = np.zeros_like(spend, dtype=float)
    adstocked[0] = spend[0]
    for t in range(1, len(spend)):
        adstocked[t] = spend[t] + decay * adstocked[t - 1]
    return adstocked


def saturation(x: np.ndarray, alpha: float) -> np.ndarray:
    """Hill / diminishing returns saturation transform."""
    x_norm = x / (x.max() + 1e-9)
    return x_norm ** alpha


def vaccine_seasonality(dates: pd.DatetimeIndex) -> np.ndarray:
    """
    Vaccine-specific seasonal index:
      - Flu season peak Sep–Nov (+40%)
      - Summer trough Jun–Aug (-20%)
      - Q1 moderate (+10%)
    """
    month = dates.month
    idx = np.ones(len(dates))
    idx = np.where((month >= 9) & (month <= 11), idx * 1.40, idx)
    idx = np.where((month >= 6) & (month <= 8),  idx * 0.80, idx)
    idx = np.where((month >= 1) & (month <= 3),  idx * 1.10, idx)
    return idx


def congress_pulse(dates: pd.DatetimeIndex) -> np.ndarray:
    """
    Spike rep_visits / congress spend around key medical conferences:
    Feb (ACIP), May (ASHP), Oct (IDWeek).
    """
    month = dates.month
    pulse = np.ones(len(dates))
    pulse = np.where(month == 2,  pulse * 1.55, pulse)   # ACIP
    pulse = np.where(month == 5,  pulse * 1.30, pulse)   # ASHP
    pulse = np.where(month == 10, pulse * 1.45, pulse)   # IDWeek
    return pulse


def spend_noise(n: int, cv: float = 0.15) -> np.ndarray:
    """Realistic spend variability — campaigns don't run at exactly budget."""
    return np.random.lognormal(0, cv, n)


def generate_spend(dates: pd.DatetimeIndex, ch: str, cfg: dict,
                   freq: str = "W") -> np.ndarray:
    """Generate realistic spend series for a single channel."""
    n = len(dates)
    scale = 1.0 if freq == "W" else 4.3   # monthly ≈ 4.3 × weekly

    base = cfg["base"] * scale * spend_noise(n)

    # Congress uplift for HCP channels
    if ch in ("rep_visits", "medical_congress", "speaker_programs"):
        base *= congress_pulse(dates)

    # Vaccine seasonality for patient channels
    if ch in ("dtc_tv", "dtc_digital", "dtc_ooh", "patient_advocacy"):
        base *= vaccine_seasonality(dates) * 0.7 + 0.3

    # Add occasional campaign bursts (2–3 per year)
    burst_prob = 3 / 52 if freq == "W" else 3 / 12
    bursts = np.random.binomial(1, burst_prob, n) * np.random.uniform(1.3, 2.0, n)
    base *= (1 + bursts * 0.4)

    return np.round(base, 2)


def build_outcome(spend_df: pd.DataFrame, dates: pd.DatetimeIndex,
                  freq: str = "W") -> np.ndarray:
    """
    Simulate scripts_written from adstocked + saturated spend,
    plus seasonality, trend, and noise.
    """
    n = len(dates)
    contribution = np.zeros(n)

    for ch, cfg in CHANNELS.items():
        raw = spend_df[ch].values
        ads = adstock(raw, cfg["decay"])
        sat = saturation(ads, cfg["sat"])
        contribution += cfg["roi"] * sat * cfg["base"]

    # Baseline scripts (market size / brand equity)
    baseline = 8000 if freq == "W" else 34000

    # Long-run brand trend (+15% over period)
    trend = np.linspace(1.0, 1.15, n)

    # Vaccine seasonality on outcomes
    season = vaccine_seasonality(dates)

    # Outcome noise
    noise = np.random.normal(1.0, 0.04, n)

    scripts = (baseline * trend * season + contribution * 10) * noise
    return np.round(scripts).astype(int)


# ── Weekly dataset (2 years = 104 weeks) ─────────────────────────────────────

def generate_weekly() -> pd.DataFrame:
    dates = pd.date_range(start="2022-01-03", periods=104, freq="W-MON")
    df = pd.DataFrame({"date": dates})
    df["year"] = dates.year
    df["week"] = dates.isocalendar().week.astype(int)
    df["month"] = dates.month
    df["quarter"] = dates.quarter

    for ch, cfg in CHANNELS.items():
        df[ch] = generate_spend(dates, ch, cfg, freq="W")

    df["total_spend"] = df[CHANNEL_NAMES].sum(axis=1).round(2)
    df["scripts_written"] = build_outcome(df, dates, freq="W")
    df["nrx_index"] = (
        (df["scripts_written"] - df["scripts_written"].min()) /
        (df["scripts_written"].max() - df["scripts_written"].min()) * 100
    ).round(2)

    # Vaccine season flag
    df["vaccine_season"] = df["month"].isin([9, 10, 11]).astype(int)

    # Congress week flag
    df["congress_week"] = df["month"].isin([2, 5, 10]).astype(int)

    return df


# ── Monthly dataset (3 years = 36 months) ────────────────────────────────────

def generate_monthly() -> pd.DataFrame:
    dates = pd.date_range(start="2021-01-01", periods=36, freq="MS")
    df = pd.DataFrame({"date": dates})
    df["year"] = dates.year
    df["month"] = dates.month
    df["month_name"] = dates.strftime("%b")
    df["quarter"] = dates.quarter

    for ch, cfg in CHANNELS.items():
        df[ch] = generate_spend(dates, ch, cfg, freq="M")

    df["total_spend"] = df[CHANNEL_NAMES].sum(axis=1).round(2)
    df["scripts_written"] = build_outcome(df, dates, freq="M")
    df["nrx_index"] = (
        (df["scripts_written"] - df["scripts_written"].min()) /
        (df["scripts_written"].max() - df["scripts_written"].min()) * 100
    ).round(2)

    df["vaccine_season"] = df["month"].isin([9, 10, 11]).astype(int)
    df["congress_month"] = df["month"].isin([2, 5, 10]).astype(int)

    # HCP vs DTC split summary columns (useful for quick analysis)
    hcp_cols = ["rep_visits", "medical_congress", "journal_advertising",
                "hcp_email", "hcp_digital", "speaker_programs", "samples_coupons"]
    dtc_cols = ["dtc_tv", "dtc_digital", "dtc_ooh", "patient_email", "patient_advocacy"]

    df["hcp_total_spend"] = df[hcp_cols].sum(axis=1).round(2)
    df["dtc_total_spend"] = df[dtc_cols].sum(axis=1).round(2)

    return df


# ── Data dictionary ───────────────────────────────────────────────────────────

def save_data_dictionary():
    meta = {
        "column": (
            ["date", "year", "week/month", "quarter"] +
            CHANNEL_NAMES +
            ["total_spend", "scripts_written", "nrx_index",
             "vaccine_season", "congress_week/month"]
        ),
        "type": (
            ["date", "int", "int", "int"] +
            ["float ($K spend)"] * 12 +
            ["float ($K)", "int", "float (0-100)", "binary", "binary"]
        ),
        "description": (
            [
                "Week start date (Mon) or month start date",
                "Calendar year",
                "ISO week number / calendar month",
                "Fiscal quarter (1-4)"
            ] +
            [
                "HCP: Field rep detailing visits spend ($K)",
                "HCP: Medical congress & symposia spend ($K)",
                "HCP: Journal advertising spend ($K)",
                "HCP: Permission email to HCPs spend ($K)",
                "HCP: Programmatic HCP digital display spend ($K)",
                "HCP: KOL speaker bureau programs spend ($K)",
                "HCP: Sample drops & co-pay coupon spend ($K)",
                "DTC: Television advertising spend ($K)",
                "DTC: Digital & social patient advertising spend ($K)",
                "DTC: Out-of-home advertising spend ($K)",
                "DTC: Patient CRM email campaigns spend ($K)",
                "DTC: Patient advocacy partnerships spend ($K)",
            ] +
            [
                "Sum of all 12 channel spends ($K)",
                "Vaccine prescriptions written (proxy outcome)",
                "Normalised new Rx index, 0=min, 100=max over period",
                "1 = Sep/Oct/Nov vaccine season, 0 = off-season",
                "1 = congress month (Feb/May/Oct), 0 = otherwise"
            ]
        )
    }
    pd.DataFrame(meta).to_csv(OUTPUT_DIR / "data_dictionary.csv", index=False)
    print("  ✓ data_dictionary.csv saved")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🔬 Pharma MMM Dataset Generator")
    print("=" * 45)

    print("\n[1/3] Generating weekly dataset (104 weeks, 2022–2023)...")
    weekly = generate_weekly()
    weekly.to_csv(OUTPUT_DIR / "mmm_weekly.csv", index=False)
    print(f"  ✓ mmm_weekly.csv saved  — {weekly.shape[0]} rows × {weekly.shape[1]} cols")
    print(f"  ✓ Spend range: ${weekly['total_spend'].min():.0f}K – ${weekly['total_spend'].max():.0f}K/week")
    print(f"  ✓ Scripts range: {weekly['scripts_written'].min():,} – {weekly['scripts_written'].max():,}")

    print("\n[2/3] Generating monthly dataset (36 months, 2021–2023)...")
    monthly = generate_monthly()
    monthly.to_csv(OUTPUT_DIR / "mmm_monthly.csv", index=False)
    print(f"  ✓ mmm_monthly.csv saved — {monthly.shape[0]} rows × {monthly.shape[1]} cols")
    print(f"  ✓ Spend range: ${monthly['total_spend'].min():.0f}K – ${monthly['total_spend'].max():.0f}K/month")
    print(f"  ✓ HCP vs DTC split: {monthly['hcp_total_spend'].mean():.0f}K vs {monthly['dtc_total_spend'].mean():.0f}K avg/month")

    print("\n[3/3] Saving data dictionary...")
    save_data_dictionary()

    print("\n✅ All datasets ready in data/raw/")
    print("   → mmm_weekly.csv")
    print("   → mmm_monthly.csv")
    print("   → data_dictionary.csv\n")
