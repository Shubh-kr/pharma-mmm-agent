"""
Synthetic Pharma MMM Dataset Generator — v2
=============================================
Vaccine Campaign — HCP + Patient Channels

v2 fixes vs v1:
  - Vaccine seasonality applied to ALL channels (HCP + DTC), not just DTC
    → eliminates spurious DTC-outcome correlation from shared seasonality
  - Congress pulses staggered per HCP channel
    → breaks lockstep rep_visits / medical_congress / speaker_programs correlation
  - rep_visits pulses in Aug (pre-season HCP push) + Oct (IDWeek)
    → reps genuinely correlate with Sep-Nov outcome peak
  - speaker_programs pulses 1 month post-congress (Mar/Jun/Nov)
    → realistic: KOL programs deploy after recruitment at conference
  - HCP channels are primary Rx drivers (higher ROI, 2-week detailing lag)
  - Added competitor_spend — negative control variable (share erosion)
  - Added price_index — quarterly co-pay step changes, negative on scripts
  - Extra independent noise per channel to reduce collinearity

Target data properties:
  - rep_visits correlation with outcome: 0.45–0.60  (was 0.14)
  - HCP channels correlate > DTC channels with outcome
  - No channel with negative or near-zero correlation
  - Competitor spend negatively correlated with scripts
  - Media drives ~30–35% of total scripts (realistic pharma benchmark)

Outputs:
  data/raw/mmm_weekly.csv   — 104 rows × 26 columns
  data/raw/mmm_monthly.csv  — 36 rows  × 28 columns
  data/raw/data_dictionary.csv
"""

import numpy as np
import pandas as pd
import yaml
from pathlib import Path

np.random.seed(42)
OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Channel config ─────────────────────────────────────────────────────────────
# roi: true causal weight used in the outcome-generating process
# HCP ROIs intentionally higher than DTC for a vaccine brand:
#   HCP detailing is the primary NRx lever; DTC drives patient pull-through only
CHANNELS = {
    "rep_visits":          dict(base=180, roi=0.55, decay=0.60, sat=0.55, ch_type="hcp"),
    "medical_congress":    dict(base=60,  roi=0.48, decay=0.75, sat=0.45, ch_type="hcp"),
    "journal_advertising": dict(base=40,  roi=0.22, decay=0.50, sat=0.65, ch_type="hcp"),
    "hcp_email":           dict(base=15,  roi=0.20, decay=0.35, sat=0.80, ch_type="hcp"),
    "hcp_digital":         dict(base=55,  roi=0.30, decay=0.40, sat=0.70, ch_type="hcp"),
    "speaker_programs":    dict(base=35,  roi=0.45, decay=0.70, sat=0.50, ch_type="hcp"),
    "samples_coupons":     dict(base=90,  roi=0.35, decay=0.55, sat=0.60, ch_type="hcp"),
    "dtc_tv":              dict(base=220, roi=0.18, decay=0.50, sat=0.75, ch_type="dtc"),
    "dtc_digital":         dict(base=85,  roi=0.22, decay=0.38, sat=0.72, ch_type="dtc"),
    "dtc_ooh":             dict(base=45,  roi=0.14, decay=0.45, sat=0.68, ch_type="dtc"),
    "patient_email":       dict(base=12,  roi=0.15, decay=0.30, sat=0.82, ch_type="dtc"),
    "patient_advocacy":    dict(base=20,  roi=0.32, decay=0.65, sat=0.55, ch_type="dtc"),
}
CHANNEL_NAMES = list(CHANNELS.keys())
HCP_CHANNELS  = [c for c, v in CHANNELS.items() if v["ch_type"] == "hcp"]
DTC_CHANNELS  = [c for c, v in CHANNELS.items() if v["ch_type"] == "dtc"]

# Per-channel congress pulses {month: multiplier} — deliberately staggered
# so HCP channels do NOT all spike in the same months
CONGRESS_PULSES = {
    # reps push hardest during IDWeek (Oct); seasonality handles the Sep-Nov lift
    "rep_visits":       {10: 1.45},
    # congress budget for ACIP (Feb) and ASHP (May) — academic HCP events only
    "medical_congress": {2: 1.60, 5: 1.35},
    # speaker programs deploy ~1 month AFTER congress (KOL recruitment lag)
    "speaker_programs": {3: 1.40, 6: 1.30, 11: 1.35},
}

# Per-channel vaccine seasonality strength (0 = flat, 1 = full ±40%/–20%)
# Rep-facing HCP channels intensify during flu season alongside patient demand
# Congress-driven channels (medical_congress, speaker_programs) follow their
# own calendar and are NOT seasonally indexed — this keeps them decorrelated
SEASON_STRENGTH = {
    "rep_visits":          0.85,   # reps push hard during flu season
    "medical_congress":    0.00,   # congress-calendar driven, not seasonal
    "journal_advertising": 0.00,   # planned quarterly, not seasonal
    "hcp_email":           0.10,   # slight uptick in fall, mostly flat
    "hcp_digital":         0.55,   # digital HCP push during vaccine season
    "speaker_programs":    0.00,   # post-congress calendar, not seasonal
    "samples_coupons":     0.50,   # sample drops increase before flu season
    "dtc_tv":              0.40,   # DTC follows patient demand (moderate)
    "dtc_digital":         0.40,
    "dtc_ooh":             0.35,
    "patient_email":       0.00,   # CRM is always-on, not seasonal
    "patient_advocacy":    0.30,
}


# ── Math primitives ────────────────────────────────────────────────────────────

def geometric_adstock(spend: np.ndarray, decay: float) -> np.ndarray:
    result = np.zeros_like(spend, dtype=float)
    result[0] = spend[0]
    for t in range(1, len(spend)):
        result[t] = spend[t] + decay * result[t - 1]
    return result


def hill_saturation(x: np.ndarray, alpha: float) -> np.ndarray:
    x_norm = x / (x.max() + 1e-9)
    return x_norm ** alpha


# ── Seasonal index builders ────────────────────────────────────────────────────

def vaccine_season_index(dates: pd.DatetimeIndex, strength: float = 1.0) -> np.ndarray:
    """Sep–Nov +40%, Jun–Aug –20%, Jan–Mar +10%."""
    m = np.array(dates.month)
    idx = np.ones(len(dates))
    idx[(m >= 9) & (m <= 11)] *= 1 + 0.40 * strength
    idx[(m >= 6) & (m <= 8)]  *= 1 - 0.20 * strength
    idx[(m >= 1) & (m <= 3)]  *= 1 + 0.10 * strength
    return idx



def congress_index(dates: pd.DatetimeIndex, pulses: dict) -> np.ndarray:
    m = np.array(dates.month)
    idx = np.ones(len(dates))
    for month, mult in pulses.items():
        idx[m == month] *= mult
    return idx


# ── Spend generator ────────────────────────────────────────────────────────────

def generate_spend(dates: pd.DatetimeIndex, ch: str, cfg: dict,
                   freq: str = "W") -> np.ndarray:
    n = len(dates)
    scale = 1.0 if freq == "W" else 4.3   # monthly ≈ 4.3× weekly

    # Base spend with realistic week-to-week variability (CoV ~15%)
    base = cfg["base"] * scale * np.random.lognormal(0, 0.15, n)

    # Per-channel vaccine seasonality (strength=0 means flat; see SEASON_STRENGTH)
    strength = SEASON_STRENGTH.get(ch, 0.0)
    if strength > 0:
        base *= vaccine_season_index(dates, strength=strength)

    # Staggered congress pulses (per-channel, breaks HCP collinearity)
    if ch in CONGRESS_PULSES:
        base *= congress_index(dates, CONGRESS_PULSES[ch])

    # Random campaign bursts (~3 per year, +20–50% uplift)
    burst_p = 3 / 52 if freq == "W" else 3 / 12
    bursts   = np.random.binomial(1, burst_p, n) * np.random.uniform(0.20, 0.50, n)
    base    *= (1 + bursts)

    # Extra independent noise — further decorrelates channels
    base *= np.random.lognormal(0, 0.10, n)

    return np.round(base, 2)


# ── Control variable generators ────────────────────────────────────────────────

def generate_competitor_spend(dates: pd.DatetimeIndex, freq: str = "W") -> np.ndarray:
    """
    Competing vaccine brand ($K/week or month).
    Deliberately kept flat (no vaccine seasonality) so its negative effect
    on scripts is visible in raw correlations — no seasonal confounding.
    Negatively impacts scripts_written via competitive share erosion.
    """
    n     = len(dates)
    scale = 1.0 if freq == "W" else 4.3
    base  = 80 * scale * np.random.lognormal(0, 0.20, n)
    burst_p = 2 / 52 if freq == "W" else 2 / 12
    bursts   = np.random.binomial(1, burst_p, n) * np.random.uniform(0.40, 0.90, n)
    base    *= (1 + bursts)
    return np.round(base, 2)


def generate_price_index(dates: pd.DatetimeIndex) -> np.ndarray:
    """
    Normalised co-pay / price index (100 = baseline, range 85–115).
    Quarterly step changes simulate payer renegotiation cycles.
    Higher value → fewer scripts (price elasticity modelled in outcome).
    """
    n        = len(dates)
    quarters = dates.to_period("Q")
    unique_q = sorted(set(quarters))
    level    = 100.0
    q_levels = {}
    for q in unique_q:
        step        = np.random.choice([-10, -5, 0, 5, 10])
        level       = float(np.clip(level + step, 85, 115))
        q_levels[q] = level
    base = np.array([q_levels[q] for q in quarters], dtype=float)
    base += np.random.normal(0, 1.5, n)   # small within-quarter noise
    return np.round(base, 1)


# ── Outcome model ──────────────────────────────────────────────────────────────

def build_outcome(spend_df: pd.DataFrame,
                  dates: pd.DatetimeIndex,
                  competitor: np.ndarray,
                  price_idx: np.ndarray,
                  freq: str = "W") -> np.ndarray:
    """
    Simulate scripts_written using a realistic pharma DGP:

        scripts = (baseline × trend × season
                   + HCP_contribution[t - lag]     <- 2-week conversion lag
                   + DTC_contribution[t])
                  × (1 - price_elasticity)
                  - competitor_drag
                  × noise

    HCP lag: detailing visits convert to Rx in ~2 weeks (weekly) / 1 month (monthly)
    DTC lag: immediate — patient pull-through happens same period
    Media drives ~30–35% of total scripts (industry benchmark for pharma)
    """
    n   = len(dates)
    LAG = 2 if freq == "W" else 1

    hcp_contrib = np.zeros(n)
    dtc_contrib = np.zeros(n)

    for ch, cfg in CHANNELS.items():
        raw    = spend_df[ch].values
        ads    = geometric_adstock(raw, cfg["decay"])
        sat    = hill_saturation(ads, cfg["sat"])
        effect = cfg["roi"] * sat * cfg["base"]
        if cfg["ch_type"] == "hcp":
            # Shift contribution by LAG — detailing-to-Rx conversion lag
            pad    = np.full(LAG, effect[:LAG].mean())
            effect = np.concatenate([pad, effect[:-LAG]])
            hcp_contrib += effect
        else:
            dtc_contrib += effect

    # Competitor drag: adstocked competitor spend erodes ~3–6% of scripts at peak
    comp_ads  = geometric_adstock(competitor, decay=0.5)
    comp_sat  = hill_saturation(comp_ads, alpha=0.60)
    comp_drag = 0.20 * comp_sat * 80   # calibrated to ~4–5% max erosion

    # Price elasticity: ±15 points ≈ ∓4.5% scripts (elasticity –0.30)
    price_eff = (price_idx - 100) / 100 * 0.30

    baseline = 6000  if freq == "W" else 26000   # lower baseline → media matters more
    trend    = np.linspace(1.0, 1.18, n)
    season   = vaccine_season_index(dates)
    noise    = np.random.normal(1.0, 0.035, n)

    scripts = (
        (
            baseline * trend * season
            + hcp_contrib * 22      # HCP drives ~38% of total scripts
            + dtc_contrib * 7       # DTC drives  ~7% of total scripts
        )
        * (1 - price_eff)
        - comp_drag * 30
    ) * noise

    return np.round(np.maximum(scripts, 500)).astype(int)


# ── Weekly dataset (2 years = 104 weeks) ──────────────────────────────────────

def generate_weekly() -> pd.DataFrame:
    dates = pd.date_range(start="2022-01-03", periods=104, freq="W-MON")
    df    = pd.DataFrame({"date": dates})
    df["year"]    = dates.year
    df["week"]    = dates.isocalendar().week.astype(int)
    df["month"]   = dates.month
    df["quarter"] = dates.quarter

    for ch, cfg in CHANNELS.items():
        df[ch] = generate_spend(dates, ch, cfg, freq="W")

    df["competitor_spend"] = generate_competitor_spend(dates, freq="W")
    df["price_index"]      = generate_price_index(dates)
    df["total_spend"]      = df[CHANNEL_NAMES].sum(axis=1).round(2)
    df["scripts_written"]  = build_outcome(
        df, dates, df["competitor_spend"].values, df["price_index"].values, freq="W"
    )
    df["nrx_index"] = (
        (df["scripts_written"] - df["scripts_written"].min()) /
        (df["scripts_written"].max() - df["scripts_written"].min()) * 100
    ).round(2)
    df["vaccine_season"] = df["month"].isin([9, 10, 11]).astype(int)
    df["congress_week"]  = df["month"].isin([2, 5, 10]).astype(int)
    return df


# ── Monthly dataset (3 years = 36 months) ─────────────────────────────────────

def generate_monthly() -> pd.DataFrame:
    dates = pd.date_range(start="2021-01-01", periods=36, freq="MS")
    df    = pd.DataFrame({"date": dates})
    df["year"]       = dates.year
    df["month"]      = dates.month
    df["month_name"] = dates.strftime("%b")
    df["quarter"]    = dates.quarter

    for ch, cfg in CHANNELS.items():
        df[ch] = generate_spend(dates, ch, cfg, freq="M")

    df["competitor_spend"] = generate_competitor_spend(dates, freq="M")
    df["price_index"]      = generate_price_index(dates)
    df["total_spend"]      = df[CHANNEL_NAMES].sum(axis=1).round(2)
    df["scripts_written"]  = build_outcome(
        df, dates, df["competitor_spend"].values, df["price_index"].values, freq="M"
    )
    df["nrx_index"] = (
        (df["scripts_written"] - df["scripts_written"].min()) /
        (df["scripts_written"].max() - df["scripts_written"].min()) * 100
    ).round(2)
    df["vaccine_season"] = df["month"].isin([9, 10, 11]).astype(int)
    df["congress_month"] = df["month"].isin([2, 5, 10]).astype(int)

    df["hcp_total_spend"] = df[HCP_CHANNELS].sum(axis=1).round(2)
    df["dtc_total_spend"] = df[DTC_CHANNELS].sum(axis=1).round(2)
    return df


# ── Data dictionary ────────────────────────────────────────────────────────────

def save_data_dictionary():
    rows = [
        ("date",              "date",            "Week start (Mon) or month start date"),
        ("year",              "int",             "Calendar year"),
        ("week / month",      "int",             "ISO week number / calendar month"),
        ("quarter",           "int",             "Fiscal quarter (1–4)"),
        ("rep_visits",        "float ($K)",      "HCP: Field rep detailing visits spend"),
        ("medical_congress",  "float ($K)",      "HCP: Medical congress & symposia spend"),
        ("journal_advertising","float ($K)",     "HCP: Journal advertising spend"),
        ("hcp_email",         "float ($K)",      "HCP: Permission email to HCPs spend"),
        ("hcp_digital",       "float ($K)",      "HCP: Programmatic HCP digital display spend"),
        ("speaker_programs",  "float ($K)",      "HCP: KOL speaker bureau programs spend"),
        ("samples_coupons",   "float ($K)",      "HCP: Sample drops & co-pay coupon spend"),
        ("dtc_tv",            "float ($K)",      "DTC: Television advertising spend"),
        ("dtc_digital",       "float ($K)",      "DTC: Digital & social patient advertising spend"),
        ("dtc_ooh",           "float ($K)",      "DTC: Out-of-home advertising spend"),
        ("patient_email",     "float ($K)",      "DTC: Patient CRM email campaigns spend"),
        ("patient_advocacy",  "float ($K)",      "DTC: Patient advocacy partnerships spend"),
        ("competitor_spend",  "float ($K)",      "Competing vaccine brand spend — model control; higher → fewer scripts"),
        ("price_index",       "float (100=base)","Co-pay/price index; quarterly step-changes; higher → fewer scripts"),
        ("total_spend",       "float ($K)",      "Sum of all 12 brand channel spends"),
        ("scripts_written",   "int",             "Vaccine prescriptions written — outcome KPI"),
        ("nrx_index",         "float (0–100)",   "Normalised new Rx index, 0=period min, 100=period max"),
        ("vaccine_season",    "binary",          "1 = Sep/Oct/Nov vaccine season, 0 = off-season"),
        ("congress_week/month","binary",         "1 = congress month (Feb/May/Oct), 0 = otherwise"),
    ]
    pd.DataFrame(rows, columns=["column", "type", "description"]).to_csv(
        OUTPUT_DIR / "data_dictionary.csv", index=False
    )
    print("  ✓ data_dictionary.csv saved")


# ── Geo dataset (long format: date × territory) ───────────────────────────────

def build_geo_outcome(spend_df: pd.DataFrame,
                      dates: pd.DatetimeIndex,
                      competitor: np.ndarray,
                      price_idx: np.ndarray,
                      freq: str,
                      channels_cfg: dict,
                      baseline_scale: float,
                      season_str: float) -> np.ndarray:
    """
    Same DGP as build_outcome() but parameterised for a territory:
      - baseline_scale : market_size / avg_market_size
      - season_str     : territory-specific vaccine season strength
      - channels_cfg   : CHANNELS dict with ROI already scaled by hcp/dtc_mult
    """
    n   = len(dates)
    LAG = 2 if freq == "W" else 1

    hcp_contrib = np.zeros(n)
    dtc_contrib = np.zeros(n)

    for ch, cfg in channels_cfg.items():
        raw    = spend_df[ch].values
        ads    = geometric_adstock(raw, cfg["decay"])
        sat    = hill_saturation(ads, cfg["sat"])
        effect = cfg["roi"] * sat * cfg["base"]
        if cfg["ch_type"] == "hcp":
            pad    = np.full(LAG, effect[:LAG].mean())
            effect = np.concatenate([pad, effect[:-LAG]])
            hcp_contrib += effect
        else:
            dtc_contrib += effect

    comp_ads  = geometric_adstock(competitor, decay=0.5)
    comp_sat  = hill_saturation(comp_ads, alpha=0.60)
    comp_drag = 0.20 * comp_sat * 80

    price_eff = (price_idx - 100) / 100 * 0.30

    national_baseline = 6000 if freq == "W" else 26000
    baseline = national_baseline * baseline_scale
    trend    = np.linspace(1.0, 1.18, n)
    season   = vaccine_season_index(dates, strength=season_str)
    noise    = np.random.normal(1.0, 0.035, n)

    scripts = (
        (
            baseline * trend * season
            + hcp_contrib * 22
            + dtc_contrib * 7
        )
        * (1 - price_eff)
        - comp_drag * 30 * baseline_scale
    ) * noise

    return np.round(np.maximum(scripts, 50)).astype(int)


def _generate_geo(territories: dict, freq: str) -> pd.DataFrame:
    if freq == "W":
        dates = pd.date_range(start="2022-01-03", periods=104, freq="W-MON")
    else:
        dates = pd.date_range(start="2021-01-01", periods=36, freq="MS")

    total_market = sum(t["market_size"] for t in territories.values())
    all_dfs      = []

    for terr_key, terr in territories.items():
        df = pd.DataFrame({"date": dates})
        df["territory"]       = terr_key
        df["territory_label"] = terr["label"]
        df["territory_abbr"]  = terr["abbr"]
        df["year"]            = dates.year
        df["month"]           = dates.month
        df["quarter"]         = dates.quarter
        if freq == "W":
            df["week"] = dates.isocalendar().week.astype(int)

        # Scale spend to territory's share of national budget
        spend_factor = terr["spend_share"]

        for ch, cfg in CHANNELS.items():
            scaled_cfg = dict(cfg, base=cfg["base"] * spend_factor)
            df[ch] = generate_spend(dates, ch, scaled_cfg, freq=freq)

        df["competitor_spend"] = generate_competitor_spend(dates, freq=freq) * spend_factor
        df["price_index"]      = generate_price_index(dates)
        df["total_spend"]      = df[CHANNEL_NAMES].sum(axis=1).round(2)

        # Apply hcp/dtc ROI multipliers AND scale base by spend_factor so that
        # contribution = roi × saturation × base is proportional to territory spend
        terr_channels = {
            ch: dict(
                cfg,
                roi=cfg["roi"] * (terr["hcp_mult"] if cfg["ch_type"] == "hcp" else terr["dtc_mult"]),
                base=cfg["base"] * spend_factor,
            )
            for ch, cfg in CHANNELS.items()
        }
        # baseline_scale = territory's share of national market
        baseline_scale = terr["market_size"] / total_market

        df["scripts_written"] = build_geo_outcome(
            df, dates,
            df["competitor_spend"].values,
            df["price_index"].values,
            freq=freq,
            channels_cfg=terr_channels,
            baseline_scale=baseline_scale,
            season_str=terr["season_str"],
        )
        df["vaccine_season"] = df["month"].isin([9, 10, 11]).astype(int)
        if freq == "W":
            df["congress_week"]  = df["month"].isin([2, 5, 10]).astype(int)
        else:
            df["congress_month"] = df["month"].isin([2, 5, 10]).astype(int)

        all_dfs.append(df)

    return pd.concat(all_dfs, ignore_index=True)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🔬 Pharma MMM Dataset Generator v2")
    print("=" * 50)

    print("\n[1/3] Generating weekly dataset (104 weeks, 2022–2023)...")
    weekly = generate_weekly()
    weekly.to_csv(OUTPUT_DIR / "mmm_weekly.csv", index=False)
    print(f"  ✓ mmm_weekly.csv   — {weekly.shape[0]} rows × {weekly.shape[1]} cols")
    print(f"  ✓ Spend:   ${weekly['total_spend'].min():.0f}K – ${weekly['total_spend'].max():.0f}K / week")
    print(f"  ✓ Scripts: {weekly['scripts_written'].min():,} – {weekly['scripts_written'].max():,}")

    print("\n[2/3] Generating monthly dataset (36 months, 2021–2023)...")
    monthly = generate_monthly()
    monthly.to_csv(OUTPUT_DIR / "mmm_monthly.csv", index=False)
    print(f"  ✓ mmm_monthly.csv  — {monthly.shape[0]} rows × {monthly.shape[1]} cols")
    print(f"  ✓ Spend:   ${monthly['total_spend'].min():.0f}K – ${monthly['total_spend'].max():.0f}K / month")
    print(f"  ✓ HCP vs DTC: ${monthly['hcp_total_spend'].mean():.0f}K vs ${monthly['dtc_total_spend'].mean():.0f}K avg/month")

    print("\n[3/5] Saving data dictionary...")
    save_data_dictionary()

    with open("config/config.yaml") as f:
        _cfg = yaml.safe_load(f)
    territories = _cfg["territories"]

    print("\n[4/5] Generating weekly geo dataset (104 weeks × 6 territories)...")
    geo_weekly = _generate_geo(territories, freq="W")
    geo_weekly.to_csv(OUTPUT_DIR / "mmm_weekly_geo.csv", index=False)
    print(f"  ✓ mmm_weekly_geo.csv  — {geo_weekly.shape[0]} rows × {geo_weekly.shape[1]} cols")
    for terr in geo_weekly["territory"].unique():
        sub = geo_weekly[geo_weekly["territory"] == terr]
        print(f"    {terr:<12} scripts: {sub['scripts_written'].min():,}–{sub['scripts_written'].max():,}/wk")

    print("\n[5/5] Generating monthly geo dataset (36 months × 6 territories)...")
    geo_monthly = _generate_geo(territories, freq="M")
    geo_monthly.to_csv(OUTPUT_DIR / "mmm_monthly_geo.csv", index=False)
    print(f"  ✓ mmm_monthly_geo.csv — {geo_monthly.shape[0]} rows × {geo_monthly.shape[1]} cols")

    print("\n✅ All datasets ready in data/raw/")
    print("   → mmm_weekly.csv")
    print("   → mmm_monthly.csv")
    print("   → mmm_weekly_geo.csv")
    print("   → mmm_monthly_geo.csv")
    print("   → data_dictionary.csv\n")
