"""
app.py — Streamlit dashboard for Pharma MMM Agent
Run:  streamlit run app.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from tools.incrementality_tool import compute_incrementality_scores, _holdout_design
from tools.scenario_tool import (
    solve_target_to_budget, solve_budget_to_scripts,
    compute_efficiency_frontier, _build_channel_params, _total_scripts,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pharma MMM Agent",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Colour palette ────────────────────────────────────────────────────────────
HCP_CLR      = "#2563EB"
DTC_CLR      = "#EA580C"
HCP_CLR_FADE = "rgba(37,99,235,0.4)"
DTC_CLR_FADE = "rgba(234,88,12,0.4)"
UP_CLR    = "#16A34A"
DOWN_CLR  = "#DC2626"
NEUT_CLR  = "#6B7280"

HCP_CHANNELS = [
    "rep_visits", "medical_congress", "journal_advertising",
    "hcp_email", "hcp_digital", "speaker_programs", "samples_coupons",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data
def load_config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)

@st.cache_data
def load_dataset(freq):
    path = f"data/raw/mmm_{freq}.csv"
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, parse_dates=["date"])

@st.cache_data
def load_geo_dataset(freq):
    path = f"data/raw/mmm_{freq}_geo.csv"
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, parse_dates=["date"])

@st.cache_data
def load_geo_bayesian(freq):
    path = f"data/raw/mmm_{freq}_geo_bayesian_results.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

@st.cache_data
def load_geo_hierarchical(freq):
    path = f"data/raw/mmm_{freq}_geo_hierarchical_results.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def load_geo_narrative(freq):
    path = f"reports/mmm_{freq}_geo_insights.md"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return f.read()

def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

def ch_color(ch_name, source=None):
    if source == "prior_estimate":
        return HCP_CLR_FADE if ch_name in HCP_CHANNELS else DTC_CLR_FADE
    return HCP_CLR if ch_name in HCP_CHANNELS else DTC_CLR

def fmt_k(v):
    return f"${v:,.0f}K"

def fmt_pct(v):
    return f"{v:+.1f}%"

# ── Sidebar ───────────────────────────────────────────────────────────────────

def sidebar(config):
    st.sidebar.image(
        "https://img.shields.io/badge/Pharma%20MMM-Agent-2563EB?style=for-the-badge",
        use_container_width=True,
    )
    st.sidebar.title("⚙️ Pipeline Controls")

    freq = st.sidebar.selectbox("Data frequency", ["weekly", "monthly"], index=0)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Run options**")
    run_bayesian = st.sidebar.checkbox("Include Bayesian MMM (~53s)", value=False)
    run_insights = st.sidebar.checkbox(
        "Include AI insight narrative",
        value=False,
        help="Requires ANTHROPIC_API_KEY or OPENAI_API_KEY in .env",
    )

    from dotenv import load_dotenv
    load_dotenv()
    provider = config.get("llm", {}).get("provider", "openai")
    key_var  = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
    has_key  = bool(os.getenv(key_var, "").strip())
    if run_insights and not has_key:
        st.sidebar.warning(f"Set {key_var} in .env to enable insights")

    run_btn         = st.sidebar.button("▶ Run Pipeline", type="primary", use_container_width=True)
    run_geo_btn     = st.sidebar.button("🗺️ Run Geo Pipeline", use_container_width=True)
    run_geo_bayes   = st.sidebar.checkbox(
        "Include Geo Bayesian (~20 min)",
        value=False,
        help="Runs PyMC per territory after Geo Ridge. Adds HDI credible intervals.",
    )
    run_geo_hier = st.sidebar.checkbox(
        "Include Geo Hierarchical (~5 min)",
        value=False,
        help="Single PyMC model over all territories with partial pooling. "
             "Mountain territory benefits most from shared national prior.",
    )
    run_geo_insights = st.sidebar.checkbox(
        "Include Geo AI narrative",
        value=False,
        help="Requires ANTHROPIC_API_KEY or OPENAI_API_KEY in .env",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**LLM config**")
    st.sidebar.code(
        f"provider : {config['llm']['provider']}\n"
        f"model    : {config['llm']['model']}",
        language=None,
    )
    return freq, run_bayesian, run_insights, run_btn, run_geo_btn, run_geo_bayes, run_geo_hier, run_geo_insights


def run_pipeline(freq, run_bayesian, run_insights):
    with st.status("Running Pharma MMM pipeline…", expanded=True) as status:
        from tools.transforms import apply_all_transforms_tool
        from tools.ols_mmm_tool import run_ols_mmm_tool
        from tools.optimizer_tool import run_budget_optimizer_tool

        data_path       = f"data/raw/mmm_{freq}.csv"
        transformed     = data_path.replace(".csv", "_transformed.csv")
        ols_results     = data_path.replace(".csv", "_ols_results.json")
        prefix          = f"data/raw/mmm_{freq}"

        st.write("🔄 Applying adstock + saturation transforms…")
        apply_all_transforms_tool.invoke(
            {"data_path": data_path, "config_path": "config/config.yaml"}
        )

        st.write("📊 Running Ridge MMM…")
        run_ols_mmm_tool.invoke(
            {"data_path": transformed, "config_path": "config/config.yaml", "freq": freq}
        )

        st.write("💰 Running budget optimiser…")
        with open(ols_results) as f:
            ols = json.load(f)
        budget = round(ols.get("avg_period_spend_k", 900.0), 1)
        run_budget_optimizer_tool.invoke({
            "results_path": ols_results,
            "config_path": "config/config.yaml",
            "total_budget_k": budget,
            "freq": freq,
        })

        if run_bayesian:
            from tools.bayesian_mmm_tool import run_bayesian_mmm_tool
            st.write("🧮 Running Bayesian MMM (this takes ~1 min)…")
            run_bayesian_mmm_tool.invoke(
                {"data_path": transformed, "config_path": "config/config.yaml", "freq": freq}
            )

        if run_insights:
            from agents.insight_agent import run_insight_agent
            st.write("✍️ Generating Claude insight narrative…")
            run_insight_agent(
                data_dir="data/raw", config_path="config/config.yaml", freq=freq
            )

        status.update(label="Pipeline complete!", state="complete")
    st.cache_data.clear()


# ── Tab helpers ───────────────────────────────────────────────────────────────

def tab_overview(df, config, freq="weekly"):
    channels = list(config["channels"].keys())
    period   = "monthly" if freq == "monthly" else "weekly"
    n_years  = len(df) // (12 if freq == "monthly" else 52)
    span_lbl = f"{n_years}yr" if n_years > 0 else f"{len(df)}-period"

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    total_spend = df[channels].values.sum() / 1000
    c1.metric(f"Total {span_lbl} spend", f"${total_spend:.1f}M")
    c2.metric(f"Avg {period} scripts", f"{df['scripts_written'].mean():,.0f}")
    c3.metric("Peak / trough ratio",
              f"{df['scripts_written'].max()/df['scripts_written'].min():.2f}×")
    c4.metric("Months" if freq == "monthly" else "Weeks", str(len(df)))

    col1, col2 = st.columns([1, 2])

    # Spend mix donut
    with col1:
        st.subheader("Spend mix")
        spends = {ch: df[ch].sum() for ch in channels}
        labels = [config["channels"][ch]["label"] for ch in channels]
        colors = [HCP_CLR if ch in HCP_CHANNELS else DTC_CLR for ch in channels]
        fig = go.Figure(go.Pie(
            labels=labels,
            values=list(spends.values()),
            marker_colors=colors,
            hole=0.45,
            textinfo="label+percent",
            textfont_size=11,
        ))
        fig.update_layout(
            margin=dict(t=10, b=10, l=10, r=10),
            showlegend=False,
            height=340,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🔵 HCP channels  🟠 DTC channels")

    # Scripts time series
    with col2:
        st.subheader("Scripts written over time")
        fig = go.Figure()
        # Vaccine season bands
        df["date_end"] = df["date"] + pd.Timedelta(weeks=1)
        season_wks = df[df["vaccine_season"] == 1]
        for _, row in season_wks.iterrows():
            fig.add_vrect(
                x0=row["date"], x1=row["date_end"],
                fillcolor="#2563EB", opacity=0.07, line_width=0,
            )
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["scripts_written"],
            mode="lines", line=dict(color=HCP_CLR, width=2),
            name="NRx written",
        ))
        fig.add_trace(go.Scatter(
            x=df["date"],
            y=df["scripts_written"].rolling(4).mean(),
            mode="lines", line=dict(color=DTC_CLR, width=2, dash="dot"),
            name="4-week MA",
        ))
        fig.update_layout(
            height=340,
            margin=dict(t=10, b=10, l=0, r=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            xaxis_title=None,
            yaxis_title="Scripts written",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Shaded bands = vaccine season (Sep–Nov)")


def tab_ridge(ols, config, freq="weekly"):
    if ols is None:
        st.info("No OLS results yet — run the pipeline from the sidebar.")
        return

    period_abbr = "mo" if freq == "monthly" else "wk"

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    n_model = sum(1 for ch in ols["channels"].values()
                  if ch.get("contribution_source") == "model")
    c1.metric("R²", f"{ols['r_squared']:.3f}",
              delta="Excellent" if ols["r_squared"] > 0.9 else None)
    c2.metric("MAPE", f"{ols['mape_pct']:.1f}%")
    c3.metric(f"Baseline scripts/{period_abbr}", f"{ols['baseline_scripts']:,.0f}")
    c4.metric("Model-identified channels", f"{n_model} / {len(ols['channels'])}")

    col1, col2 = st.columns(2)

    channels_df = pd.DataFrame(ols["channels"]).T.reset_index()
    channels_df.columns = ["channel"] + list(channels_df.columns[1:])
    channels_df["label"]            = channels_df.apply(lambda r: r.get("label", r["channel"]), axis=1)
    channels_df["contribution_pct"] = channels_df["contribution_pct"].astype(float)
    channels_df["estimated_roi"]    = channels_df["estimated_roi"].astype(float)
    channels_df["source"]           = channels_df.get("contribution_source", pd.Series(["model"]*len(channels_df)))
    channels_df["color"]            = channels_df.apply(
        lambda r: ch_color(r["channel"], r["source"]), axis=1)
    channels_df = channels_df.sort_values("contribution_pct", ascending=True)

    with col1:
        st.subheader("Channel contributions")
        fig = go.Figure(go.Bar(
            x=channels_df["contribution_pct"],
            y=channels_df["label"],
            orientation="h",
            marker_color=channels_df["color"],
            text=channels_df.apply(
                lambda r: f"{r['contribution_pct']:.1f}%"
                          + (" *" if r["source"] == "prior_estimate" else ""),
                axis=1,
            ),
            textposition="outside",
        ))
        fig.update_layout(
            height=420, margin=dict(t=10, b=10, l=0, r=60),
            xaxis_title="Contribution %", yaxis_title=None,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🔵 HCP  🟠 DTC  |  Lighter = prior-estimated  |  * = prior-estimated")

    with col2:
        st.subheader("ROI by channel")
        roi_df = channels_df.sort_values("estimated_roi", ascending=True)
        fig = go.Figure(go.Bar(
            x=roi_df["estimated_roi"],
            y=roi_df["label"],
            orientation="h",
            marker_color=roi_df["color"],
            text=roi_df["estimated_roi"].apply(lambda v: f"{v:.3f}"),
            textposition="outside",
        ))
        fig.update_layout(
            height=420, margin=dict(t=10, b=10, l=0, r=60),
            xaxis_title="Estimated ROI", yaxis_title=None,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Controls
    st.subheader("Control variable coefficients")
    ctrl = ols.get("controls", {})
    if ctrl:
        c_df = pd.DataFrame([
            {"Control": k,
             "Coefficient (unscaled)": round(v, 4),
             "Direction": "↓ suppresses scripts" if v < 0 else "↑ check sign"}
            for k, v in ctrl.items()
        ])
        st.dataframe(c_df, use_container_width=True, hide_index=True)

    # Full channel table
    with st.expander("Full channel results table"):
        disp = channels_df[["label", "channel_type", "avg_weekly_spend_k",
                             "total_spend_k", "estimated_roi", "contribution_pct",
                             "source"]].copy()
        disp.columns = ["Channel", "Type", f"Avg {period_abbr} spend $K", "Total spend $K",
                        "ROI", "Contrib %", "Source"]
        st.dataframe(disp.sort_values("Contrib %", ascending=False),
                     use_container_width=True, hide_index=True)


def tab_bayesian(bayes, ols, config):
    if bayes is None:
        st.info("No Bayesian results yet — re-run the pipeline with **Include Bayesian MMM** checked.")
        return

    mcmc = bayes["mcmc"]
    conv_ok = mcmc.get("converged", False)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R² (posterior mean)", f"{bayes['r_squared_posterior_mean']:.3f}")
    c2.metric("MAPE", f"{bayes['mape_pct']:.1f}%")
    c3.metric("R̂ max (convergence)", f"{mcmc.get('max_rhat', '—')}",
              delta="Converged ✓" if conv_ok else "Check chains",
              delta_color="normal" if conv_ok else "inverse")
    c4.metric("MCMC draws",
              f"{mcmc['chains']}×{mcmc['draws']}")

    col1, col2 = st.columns(2)

    b_df = pd.DataFrame(bayes["channels"]).T.reset_index()
    b_df.columns = ["channel"] + list(b_df.columns[1:])
    b_df["label"]            = b_df.apply(lambda r: r.get("label", r["channel"]), axis=1)
    b_df["contribution_pct"] = b_df["contribution_pct"].astype(float)
    b_df["estimated_roi"]    = b_df["estimated_roi"].astype(float)
    b_df["hdi_lo"]           = b_df["contribution_hdi_5"].astype(float)
    b_df["hdi_hi"]           = b_df["contribution_hdi_95"].astype(float)
    b_df["contrib"]          = b_df["total_contribution"].astype(float)
    b_df["color"]            = b_df["channel"].apply(
        lambda ch: HCP_CLR if ch in HCP_CHANNELS else DTC_CLR)
    b_df = b_df.sort_values("contribution_pct", ascending=True)

    with col1:
        st.subheader("Contributions with 90% HDI")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=b_df["contribution_pct"],
            y=b_df["label"],
            orientation="h",
            marker_color=b_df["color"],
            error_x=dict(
                type="data",
                symmetric=False,
                array=(b_df["hdi_hi"] - b_df["contrib"]) / (b_df["contrib"].abs() + 1e-9) * b_df["contribution_pct"],
                arrayminus=(b_df["contrib"] - b_df["hdi_lo"]) / (b_df["contrib"].abs() + 1e-9) * b_df["contribution_pct"],
                color="#9CA3AF",
            ),
            text=b_df["contribution_pct"].apply(lambda v: f"{v:.1f}%"),
            textposition="outside",
        ))
        fig.update_layout(
            height=420, margin=dict(t=10, b=10, l=0, r=60),
            xaxis_title="Contribution %", yaxis_title=None,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Error bars = 90% credible interval (5th–95th percentile)")

    with col2:
        st.subheader("OLS vs Bayesian ROI")
        if ols:
            ols_df = pd.DataFrame(ols["channels"]).T.reset_index()
            ols_df.columns = ["channel"] + list(ols_df.columns[1:])
            merged = b_df.merge(
                ols_df[["channel", "estimated_roi"]],
                on="channel", suffixes=("_bayes", "_ols")
            )
            merged["estimated_roi_ols"]   = merged["estimated_roi_ols"].astype(float)
            merged["estimated_roi_bayes"] = merged["estimated_roi_bayes"].astype(float)

            fig = go.Figure()
            lim = max(merged["estimated_roi_ols"].max(),
                      merged["estimated_roi_bayes"].max()) * 1.1
            fig.add_trace(go.Scatter(
                x=[0, lim], y=[0, lim],
                mode="lines",
                line=dict(color="#D1D5DB", dash="dash"),
                showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=merged["estimated_roi_ols"],
                y=merged["estimated_roi_bayes"],
                mode="markers+text",
                marker=dict(
                    color=[HCP_CLR if c in HCP_CHANNELS else DTC_CLR
                           for c in merged["channel"]],
                    size=10,
                ),
                text=merged["label"],
                textposition="top center",
                textfont_size=9,
            ))
            fig.update_layout(
                height=420,
                margin=dict(t=10, b=10, l=0, r=10),
                xaxis_title="OLS ROI",
                yaxis_title="Bayesian ROI",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Points above diagonal = Bayesian assigns higher ROI than OLS")

    # Controls comparison
    st.subheader("Control posteriors")
    ctrl = bayes.get("controls", {})
    if ctrl:
        rows = []
        ols_ctrl = (ols or {}).get("controls", {})
        name_map = {
            "beta_season":     "Vaccine season",
            "beta_congress":   "Congress month",
            "beta_competitor": "Competitor spend",
            "beta_price":      "Price index",
        }
        for k, v in ctrl.items():
            label = name_map.get(k, k)
            ols_v = ols_ctrl.get(k.replace("beta_", ""), None)
            expected = "negative" if k in ("beta_competitor", "beta_price") else "positive"
            actual    = "↓ negative" if v < 0 else "↑ positive"
            match     = "✓" if (expected == "negative") == (v < 0) else "⚠️"
            rows.append({
                "Control": label,
                "Bayesian posterior": round(v, 3),
                "OLS coef": round(ols_v, 4) if ols_v is not None else "—",
                "Expected direction": expected,
                "Actual": actual,
                "": match,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def tab_budget(opt, config):
    if opt is None:
        st.info("No optimiser results yet — run the pipeline from the sidebar.")
        return

    alloc = opt["allocations"]
    alloc_df = pd.DataFrame(alloc)

    c1, c2, c3 = st.columns(3)
    c1.metric("Projected NRx uplift", f"+{opt['projected_uplift_pct']:.1f}%",
              delta="same total budget")
    c2.metric("Total budget/period", fmt_k(opt["total_budget_k"]))
    top_inc = alloc_df.sort_values("change_pct", ascending=False).iloc[0]
    c3.metric(f"Biggest increase: {top_inc['channel']}",
              f"{top_inc['change_pct']:+.1f}%")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("Current vs recommended spend")
        plot_df = alloc_df.copy()
        plot_df["color"] = [
            UP_CLR if v > 5 else (DOWN_CLR if v < -5 else NEUT_CLR)
            for v in plot_df["change_pct"]
        ]
        plot_df = plot_df.sort_values("optimal_spend_k", ascending=True)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Current",
            y=plot_df["channel"],
            x=plot_df["current_spend_k"],
            orientation="h",
            marker_color="#CBD5E1",
        ))
        fig.add_trace(go.Bar(
            name="Recommended",
            y=plot_df["channel"],
            x=plot_df["optimal_spend_k"],
            orientation="h",
            marker_color=plot_df["color"],
        ))
        fig.update_layout(
            barmode="overlay",
            height=440,
            margin=dict(t=10, b=10, l=0, r=10),
            xaxis_title="Spend $K / period",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🟢 Increase  🔴 Reduce  ⚪ Maintain  (grey = current)")

    with col2:
        st.subheader("Reallocation table")
        disp = alloc_df[["channel", "fitted_roi", "current_spend_k",
                          "optimal_spend_k", "change_pct", "action"]].copy()
        disp.columns = ["Channel", "ROI", "Current $K", "Optimal $K", "Δ%", "Action"]
        disp["Δ%"] = disp["Δ%"].apply(lambda v: f"{v:+.1f}%")
        st.dataframe(
            disp.sort_values("ROI", ascending=False),
            use_container_width=True,
            hide_index=True,
            height=440,
        )


def run_geo_pipeline(freq, run_bayesian=False, run_hierarchical=False, run_insights=False):
    with st.status("Running Geo MMM pipeline…", expanded=True) as status:
        from tools.geo_mmm_tool import run_geo_ols_mmm_tool
        from tools.geo_optimizer_tool import run_geo_budget_optimizer_tool

        geo_path = f"data/raw/mmm_{freq}_geo.csv"
        geo_ols  = f"data/raw/mmm_{freq}_geo_ols_results.json"

        if not os.path.exists(geo_path):
            st.error(
                f"`{geo_path}` not found. "
                "Run `python scripts/generate_dataset.py` first to create geo data."
            )
            status.update(label="Geo pipeline failed.", state="error")
            return

        st.write("📊 Running Geo Ridge MMM (per territory)…")
        run_geo_ols_mmm_tool.invoke(
            {"data_path": geo_path, "config_path": "config/config.yaml", "freq": freq}
        )

        st.write("💰 Running Geo budget optimiser…")
        with open(geo_ols) as f:
            import json as _json
            geo_results = _json.load(f)
        total_budget = sum(
            t.get("avg_period_spend_k", 0)
            for t in geo_results.get("territories", {}).values()
        )
        run_geo_budget_optimizer_tool.invoke({
            "results_path":             geo_ols,
            "config_path":              "config/config.yaml",
            "total_national_budget_k":  round(total_budget, 1),
            "freq":                     freq,
        })

        if run_bayesian:
            from tools.geo_bayesian_mmm_tool import run_geo_bayesian_mmm_tool
            st.write("🧮 Running Geo Bayesian MMM (per territory, ~20 min)…")
            run_geo_bayesian_mmm_tool.invoke(
                {"data_path": geo_path, "config_path": "config/config.yaml", "freq": freq}
            )

        if run_hierarchical:
            from tools.geo_hierarchical_mmm_tool import run_geo_hierarchical_mmm_tool
            st.write("🔗 Running Geo Hierarchical Bayesian MMM (~5 min)…")
            run_geo_hierarchical_mmm_tool.invoke(
                {"data_path": geo_path, "config_path": "config/config.yaml", "freq": freq}
            )

        if run_insights:
            from agents.insight_agent import run_geo_insight_agent
            st.write("✍️ Generating geo AI narrative…")
            run_geo_insight_agent(
                data_dir="data/raw", config_path="config/config.yaml", freq=freq
            )

        status.update(label="Geo pipeline complete!", state="complete")
    st.cache_data.clear()


def _render_whatif_simulator(geo_opt: dict):
    """Interactive territory budget simulator using ROI-weighted linear approximation."""
    st.subheader("🎛️ What-if Budget Simulator")
    st.caption(
        "Shift budget between territories to preview estimated NRx impact. "
        "Uses territory ROI efficiency as a linear approximation — saturation effects "
        "mean actual returns diminish at very high spend."
    )

    alloc   = geo_opt.get("territory_allocation", {})
    total_k = geo_opt.get("total_national_budget_k", 0.0)
    if not alloc or total_k == 0:
        st.info("Run the geo optimizer first to enable the simulator.")
        return

    terr_keys    = list(alloc.keys())
    curr_budgets = {tk: alloc[tk]["current_budget_k"]  for tk in terr_keys}
    roi_eff      = {tk: alloc[tk]["roi_efficiency"]     for tk in terr_keys}
    labels       = {tk: alloc[tk]["label"]              for tk in terr_keys}
    opt_budgets  = {tk: alloc[tk]["optimal_budget_k"]   for tk in terr_keys}

    # ── Preset buttons — must run BEFORE sliders to seed session_state ─────────
    b1, b2, _ = st.columns([1.3, 2.2, 4])
    if b1.button("↺ Reset to current", key="whatif_btn_reset"):
        for tk in terr_keys:
            st.session_state[f"whatif_s_{tk}"] = float(round(curr_budgets[tk] / 5) * 5)
    if b2.button("→ Apply optimizer recommendation", key="whatif_btn_opt"):
        for tk in terr_keys:
            st.session_state[f"whatif_s_{tk}"] = float(round(opt_budgets[tk] / 5) * 5)

    # ── Sliders (3 columns × 2 rows) ──────────────────────────────────────────
    sim_budgets = {}
    slider_cols = st.columns(3)
    for i, tk in enumerate(terr_keys):
        col  = slider_cols[i % 3]
        curr = curr_budgets[tk]
        lo   = float(max(10.0, round(curr * 0.30 / 5) * 5))
        hi   = float(round(min(total_k * 0.55, curr * 2.5) / 5) * 5)
        lo   = min(lo, hi - 5)
        default = float(round(curr / 5) * 5)
        sim_budgets[tk] = col.slider(
            labels[tk],
            min_value=lo, max_value=hi,
            value=default,
            step=5.0,
            format="$%.0fK",
            key=f"whatif_s_{tk}",
        )

    # ── Budget balance indicator ───────────────────────────────────────────────
    total_sim   = sum(sim_budgets.values())
    remaining_k = total_k - total_sim
    if abs(remaining_k) < 5:
        st.success(f"Budget balanced: ${total_sim:,.0f}K allocated  ✓")
    elif remaining_k > 0:
        st.warning(
            f"${remaining_k:,.0f}K unallocated — increase a territory or the "
            "model will scale allocations proportionally."
        )
    else:
        st.error(f"Over-budget by ${-remaining_k:,.0f}K — reduce a territory.")

    # ── Scale to total_k for projection math (proportional if unbalanced) ─────
    scale    = total_k / (total_sim + 1e-9)
    norm_sim = {tk: sim_budgets[tk] * scale for tk in terr_keys}

    # ── ROI-weighted linear response ──────────────────────────────────────────
    base_resp = sum(roi_eff[tk] * curr_budgets[tk] for tk in terr_keys)
    sim_resp  = sum(roi_eff[tk] * norm_sim[tk]      for tk in terr_keys)
    uplift    = (sim_resp - base_resp) / (base_resp + 1e-9) * 100
    opt_ceil  = geo_opt.get("projected_territory_uplift_pct", 0)

    # ── KPI row ────────────────────────────────────────────────────────────────
    m1, m2, m3 = st.columns(3)
    m1.metric("Simulated NRx uplift", f"{uplift:+.1f}%", delta="vs current allocation")
    m2.metric("Optimizer ceiling (reference)", f"+{opt_ceil:.1f}%",
              delta="SLSQP optimal", delta_color="off")
    m3.metric(
        "Unallocated budget",
        f"${abs(remaining_k):,.0f}K",
        delta="over-budget" if remaining_k < -5 else ("balanced" if abs(remaining_k) < 5 else "under-allocated"),
        delta_color="inverse" if remaining_k < -5 else "off",
    )

    # ── Chart + table ─────────────────────────────────────────────────────────
    rows = []
    for tk in terr_keys:
        delta_k       = norm_sim[tk] - curr_budgets[tk]
        delta_scripts = roi_eff[tk] * delta_k
        rows.append({
            "Territory":      labels[tk],
            "Current $K":     curr_budgets[tk],
            "Simulated $K":   norm_sim[tk],
            "delta_k":        delta_k,
            "Est. Δ Scripts": round(delta_scripts),
        })
    sim_df = pd.DataFrame(rows)

    chart_col, table_col = st.columns([3, 2])
    with chart_col:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=sim_df["Territory"], x=sim_df["Current $K"],
            name="Current", orientation="h", marker_color="#CBD5E1",
        ))
        bar_colors = [
            UP_CLR if d > 2 else (DOWN_CLR if d < -2 else NEUT_CLR)
            for d in sim_df["delta_k"]
        ]
        fig.add_trace(go.Bar(
            y=sim_df["Territory"], x=sim_df["Simulated $K"],
            name="Simulated", orientation="h", marker_color=bar_colors,
        ))
        fig.update_layout(
            barmode="overlay", height=300,
            margin=dict(t=10, b=10, l=0, r=0),
            xaxis_title="Budget $K / period",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)

    with table_col:
        disp = sim_df[["Territory", "Current $K", "Simulated $K", "delta_k", "Est. Δ Scripts"]].copy()
        disp.columns = ["Territory", "Current $K", "Simulated $K", "Δ $K", "Est. Δ Scripts"]
        disp["Current $K"]     = disp["Current $K"].apply(lambda v: f"${v:,.0f}K")
        disp["Simulated $K"]   = disp["Simulated $K"].apply(lambda v: f"${v:,.0f}K")
        disp["Δ $K"]           = disp["Δ $K"].apply(lambda v: f"{v:+,.0f}K")
        disp["Est. Δ Scripts"] = disp["Est. Δ Scripts"].apply(lambda v: f"{v:+,.0f}")
        st.dataframe(disp, use_container_width=True, hide_index=True)

    st.caption(
        "Linear approximation: Δ Scripts ≈ ROI_efficiency × Δ Budget. "
        "Saturation means gains slow at very high spend — use the SLSQP optimizer for rigorous bounds."
    )


def _render_geo_bayesian(geo_bayes: dict, config: dict):
    """Bayesian sub-section rendered inside tab_geo."""
    territories_cfg = config.get("territories", {})
    terr_data       = geo_bayes.get("territories", {})

    if not terr_data:
        st.info("Geo Bayesian results are empty.")
        return

    # ── Convergence summary ────────────────────────────────────────────────────
    st.subheader("Convergence & model fit")
    conv_rows = []
    for tk, td in terr_data.items():
        mcmc = td.get("mcmc", {})
        conv_rows.append({
            "Territory":  td.get("label", tk),
            "R²":         td["r_squared_posterior_mean"],
            "MAPE":       f"{td['mape_pct']:.1f}%",
            "R̂ max":     mcmc.get("max_rhat", "—"),
            "Converged":  "✓" if mcmc.get("converged") else "⚠ check chains",
            "Draws":      f"{mcmc.get('chains', '?')}×{mcmc.get('draws', '?')}",
        })
    st.dataframe(pd.DataFrame(conv_rows), use_container_width=True, hide_index=True)

    # ── ROI + 90% HDI by territory — channel selector ─────────────────────────
    st.subheader("Channel ROI with 90% credible interval")

    # Build channel list from first territory
    first_td  = next(iter(terr_data.values()))
    ch_labels = {ck: cv["label"] for ck, cv in first_td["channels"].items()}
    sel_labels = st.multiselect(
        "Channels to display",
        options=list(ch_labels.values()),
        default=list(ch_labels.values())[:5],
    )
    sel_keys = [k for k, v in ch_labels.items() if v in sel_labels]

    if sel_keys:
        hdi_rows = []
        for tk, td in terr_data.items():
            label = td.get("label", tk)
            for ck in sel_keys:
                ch = td["channels"].get(ck, {})
                if not ch:
                    continue
                hdi_rows.append({
                    "territory":  label,
                    "channel":    ch["label"],
                    "roi":        ch["estimated_roi"],
                    "hdi_lo":     ch["contribution_hdi_5"],
                    "hdi_hi":     ch["contribution_hdi_95"],
                    "contrib":    ch["total_contribution"],
                    "ch_type":    ch["channel_type"],
                })
        hdi_df = pd.DataFrame(hdi_rows)

        fig = go.Figure()
        for ch_label in sel_labels:
            sub = hdi_df[hdi_df["channel"] == ch_label]
            if sub.empty:
                continue
            color = HCP_CLR if sub.iloc[0]["ch_type"] == "hcp" else DTC_CLR
            fig.add_trace(go.Scatter(
                x=sub["territory"],
                y=sub["roi"],
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=(sub["hdi_hi"] - sub["contrib"]) / (sub["contrib"].abs() + 1e-9) * sub["roi"],
                    arrayminus=(sub["contrib"] - sub["hdi_lo"]) / (sub["contrib"].abs() + 1e-9) * sub["roi"],
                    color="#9CA3AF",
                    thickness=1.5,
                    width=4,
                ),
                mode="markers+lines",
                marker=dict(color=color, size=8),
                line=dict(color=color, width=1, dash="dot"),
                name=ch_label,
            ))
        fig.update_layout(
            height=400,
            margin=dict(t=10, b=10, l=0, r=10),
            xaxis_title=None,
            yaxis_title="Blended ROI",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Error bars = 90% credible interval (5th–95th percentile). "
                   "Wider bars = more posterior uncertainty.")

    # ── HDI width heatmap — uncertainty across territories × channels ──────────
    st.subheader("Posterior uncertainty heatmap (HDI width)")
    st.caption("Wider HDI = model less confident. Useful for deciding where to run lift tests.")

    heat_rows = []
    ch_order_keys = list(first_td["channels"].keys())
    for tk, td in terr_data.items():
        row = {"Territory": td.get("label", tk)}
        for ck in ch_order_keys:
            ch = td["channels"].get(ck, {})
            if ch:
                width = round(ch["contribution_hdi_95"] - ch["contribution_hdi_5"], 0)
                row[ch["label"]] = width
        heat_rows.append(row)

    heat_df = pd.DataFrame(heat_rows).set_index("Territory")
    fig = px.imshow(
        heat_df,
        color_continuous_scale=[[0, "#EFF6FF"], [0.5, "#60A5FA"], [1, "#1E3A5F"]],
        labels={"color": "HDI width\n(scripts)"},
        aspect="auto",
        height=280,
    )
    fig.update_layout(margin=dict(t=10, b=10, l=0, r=0),
                      xaxis_tickangle=-35)
    st.plotly_chart(fig, use_container_width=True)

    # ── Ridge vs Bayesian ROI scatter per territory ────────────────────────────
    st.subheader("Ridge vs Bayesian ROI — all territories")
    scatter_rows = []
    for tk, td in terr_data.items():
        terr_label = td.get("label", tk)
        for ck, cv in td["channels"].items():
            scatter_rows.append({
                "territory": terr_label,
                "channel":   cv["label"],
                "bayes_roi": cv["estimated_roi"],
                "ch_type":   cv["channel_type"],
            })
    scatter_df = pd.DataFrame(scatter_rows)

    # We need Ridge ROI from geo_ols — pass it via st.session_state or rely on caller
    st.caption("Ridge ROI is not shown here; load geo OLS results alongside Bayesian to compare.")


def _render_geo_hierarchical(geo_hier: dict, config: dict):
    """Hierarchical Bayesian sub-section rendered inside tab_geo."""
    terr_data = geo_hier.get("territories", {})
    if not terr_data:
        st.info("Hierarchical results are empty.")
        return

    mcmc = geo_hier.get("mcmc", {})
    st.subheader("Hierarchical model — national hyperpriors")
    st.caption(
        f"Single joint model | R̂ max={mcmc.get('max_rhat', '—')} | "
        f"{'✓ converged' if mcmc.get('converged') else '⚠ check chains'} | "
        f"{mcmc.get('chains', '?')}×{mcmc.get('draws', '?')} draws"
    )

    # National hyperprior ROI table
    hp = geo_hier.get("national_hyperpriors", {})
    if hp:
        hp_rows = [
            {
                "Channel":       v["label"],
                "National ROI":  v["national_roi_mean"],
                "μ_beta (mean)": v["mu_beta_mean"],
                "σ_terr (spread)": v["sigma_terr_mean"],
                "CV (σ/μ)":      round(v["sigma_terr_mean"] / (v["mu_beta_mean"] + 1e-9), 2),
            }
            for v in sorted(hp.values(), key=lambda x: x["national_roi_mean"], reverse=True)
        ]
        st.dataframe(
            pd.DataFrame(hp_rows),
            use_container_width=True,
            hide_index=True,
            column_config={
                "National ROI":      st.column_config.NumberColumn(format="%.3f"),
                "μ_beta (mean)":     st.column_config.NumberColumn(format="%.4f"),
                "σ_terr (spread)":   st.column_config.NumberColumn(format="%.4f"),
                "CV (σ/μ)":          st.column_config.NumberColumn(
                    help="Coefficient of variation — high value = ROI varies a lot across territories",
                    format="%.2f",
                ),
            },
        )
        st.caption(
            "**σ_terr** = posterior mean of the territory-level spread in log-beta space. "
            "Low CV ≈ ROI is consistent across territories; high CV ≈ strong regional variation."
        )

    # Reuse the per-territory convergence + HDI rendering
    _render_geo_bayesian(geo_hier, config)


def _render_response_curves(geo_ols: dict, config: dict):
    """
    Per-channel saturation response curves by territory.

    For a chosen channel, plots the Hill saturation curve for each territory
    using steady-state adstock as the x-axis proxy, normalised by each
    territory's historical adstock max.  A dot marks where each territory
    currently operates on the curve.
    """
    territories = (geo_ols or {}).get("territories", {})
    if not territories:
        return

    ch_config = config.get("channels", {})

    # Build channel list in contribution-descending order (averaged across territories)
    ch_avg_contrib = {}
    for td in territories.values():
        for ck, cv in td.get("channels", {}).items():
            ch_avg_contrib[ck] = ch_avg_contrib.get(ck, 0) + cv.get("contribution_pct", 0)
    ch_keys_sorted = sorted(ch_avg_contrib, key=lambda c: ch_avg_contrib[c], reverse=True)
    ch_labels      = {ck: ch_config[ck]["label"] for ck in ch_keys_sorted if ck in ch_config}

    selected_ch = st.selectbox(
        "Channel",
        options=list(ch_labels.keys()),
        format_func=lambda k: ch_labels[k],
        key="resp_curve_channel",
    )

    decay = ch_config[selected_ch]["adstock_decay"]
    alpha = ch_config[selected_ch]["saturation"]

    TERR_COLORS = [
        "#2563EB", "#EA580C", "#16A34A", "#9333EA", "#DC2626", "#0891B2"
    ]

    fig = go.Figure()

    for (tk, td), color in zip(sorted(territories.items()), TERR_COLORS):
        cv = td.get("channels", {}).get(selected_ch)
        if cv is None:
            continue

        adstock_x_max = cv.get("adstock_x_max_k")
        avg_spend     = cv.get("avg_period_spend_k", 0)
        avg_sat       = cv.get("avg_saturated")
        terr_label    = td.get("label", tk)

        if not adstock_x_max or adstock_x_max < 1e-6:
            continue

        # x: raw spend from 0 to 2× historical average (or 1.5× max adstock equiv)
        x_max_plot = max(avg_spend * 2.5, 1.0)
        x_raw = np.linspace(0, x_max_plot, 300)

        # Steady-state adstock: x_ss = x / (1 - decay)
        x_ss  = x_raw / (1.0 - decay + 1e-9)
        y_sat = np.clip((x_ss / adstock_x_max) ** alpha, 0, 1)

        # Current operating point — use stored avg_saturated if available, else compute
        if avg_sat is not None:
            y_dot = avg_sat
        else:
            x_ss_dot = avg_spend / (1.0 - decay + 1e-9)
            y_dot    = float(np.clip((x_ss_dot / adstock_x_max) ** alpha, 0, 1))

        fig.add_trace(go.Scatter(
            x=x_raw, y=y_sat * 100,
            mode="lines",
            name=terr_label,
            line=dict(color=color, width=2),
            hovertemplate=(
                f"<b>{terr_label}</b><br>"
                "Spend: $%{x:.1f}K<br>"
                "Saturation: %{y:.1f}%<extra></extra>"
            ),
        ))
        fig.add_trace(go.Scatter(
            x=[avg_spend], y=[y_dot * 100],
            mode="markers",
            name=f"{terr_label} (current)",
            marker=dict(color=color, size=10, symbol="circle",
                        line=dict(color="white", width=2)),
            showlegend=False,
            hovertemplate=(
                f"<b>{terr_label} — current avg</b><br>"
                f"Spend: ${avg_spend:.1f}K<br>"
                f"Saturation: {y_dot*100:.1f}%<extra></extra>"
            ),
        ))

    fig.add_hline(
        y=80, line_dash="dot", line_color="#9CA3AF", line_width=1,
        annotation_text="80% saturation", annotation_position="right",
    )
    fig.update_layout(
        height=400,
        margin=dict(t=10, b=10, l=0, r=80),
        xaxis_title="Raw spend per period ($K)",
        yaxis=dict(title="Saturation (%)", range=[0, 102]),
        legend=dict(title="Territory", orientation="v", x=1.06),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Curve = Hill saturation (α={alpha}, adstock decay={decay}). "
        "Dots mark each territory's current average spend. "
        "Territories with higher historical spend have a larger x_max "
        "— the same dollar amount represents a smaller fraction of their capacity."
    )


def _render_season_interactions(geo_ols: dict, config: dict):
    """Heatmap: territory × channel → season_lift_pct (diverging, green=better in-season)."""
    territories = (geo_ols or {}).get("territories", {})
    if not territories:
        return

    rows = []
    for tk, td in territories.items():
        for ck, cv in td.get("channels", {}).items():
            if "season_lift_pct" in cv:
                rows.append({
                    "territory": td.get("label", tk),
                    "channel":   cv["label"],
                    "lift":      cv["season_lift_pct"],
                    "roi_in":    cv["roi_in_season"],
                    "roi_off":   cv["roi_off_season"],
                })
    if not rows:
        return

    df = pd.DataFrame(rows)
    pivot = df.pivot_table(index="territory", columns="channel", values="lift", aggfunc="mean")

    # Order channels by mean absolute lift descending
    ch_order = pivot.abs().mean(axis=0).sort_values(ascending=False).index.tolist()
    pivot = pivot[ch_order]

    # Build hover text matrix (roi_in / roi_off per cell)
    hover_df = df.pivot_table(index="territory", columns="channel", values="roi_in", aggfunc="mean")
    hover_off = df.pivot_table(index="territory", columns="channel", values="roi_off", aggfunc="mean")
    hover_df = hover_df.reindex(columns=ch_order).reindex(pivot.index)
    hover_off = hover_off.reindex(columns=ch_order).reindex(pivot.index)

    hover_text = []
    for terr in pivot.index:
        row_text = []
        for ch in ch_order:
            lift = pivot.loc[terr, ch] if ch in pivot.columns else None
            ri   = hover_df.loc[terr, ch]   if ch in hover_df.columns   else None
            ro   = hover_off.loc[terr, ch]  if ch in hover_off.columns  else None
            if pd.notna(lift):
                row_text.append(
                    f"<b>{ch}</b><br>{terr}<br>"
                    f"In-season ROI: {ri:.3f}<br>"
                    f"Off-season ROI: {ro:.3f}<br>"
                    f"Lift: {lift:+.1f}%"
                )
            else:
                row_text.append("")
        hover_text.append(row_text)

    abs_max = max(abs(df["lift"].min()), abs(df["lift"].max()), 5)

    fig = go.Figure(go.Heatmap(
        z=pivot.values.tolist(),
        x=ch_order,
        y=list(pivot.index),
        text=[[f"{v:+.1f}%" if pd.notna(v) else "" for v in row] for row in pivot.values.tolist()],
        hovertext=hover_text,
        hovertemplate="%{hovertext}<extra></extra>",
        texttemplate="%{text}",
        colorscale=[
            [0.0,  "#B91C1C"],
            [0.5,  "#F9FAFB"],
            [1.0,  "#15803D"],
        ],
        zmid=0,
        zmin=-abs_max,
        zmax=abs_max,
        colorbar=dict(
            title="Season lift %",
            ticksuffix="%",
            thickness=14,
            len=0.8,
        ),
        xgap=2,
        ygap=2,
    ))
    fig.update_layout(
        height=max(280, 60 * len(pivot) + 80),
        margin=dict(t=10, b=10, l=0, r=0),
        xaxis=dict(side="bottom", tickangle=-30, title=None),
        yaxis=dict(title=None, autorange="reversed"),
        font=dict(size=11),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Green = higher in-season ROI (channel punches above weight during vaccine season). "
        "Red = lower in-season ROI (channel saturates with increased seasonal spend). "
        "Only model-identified channels shown."
    )


def tab_geo(geo_df, geo_ols, geo_opt, geo_bayes, geo_hier, geo_narrative, config):
    territories_cfg = config.get("territories", {})

    if geo_df is None:
        st.info(
            "Geo dataset not found. "
            "Run `python scripts/generate_dataset.py` to create `mmm_*_geo.csv`."
        )
        return

    # ── Run geo pipeline button ────────────────────────────────────────────────
    if geo_ols is None:
        st.info("Geo MMM results not found. Click **▶ Run Geo Pipeline** in the sidebar.")

    if geo_ols is None and geo_opt is None:
        return

    # ── Build territory summary DF ─────────────────────────────────────────────
    terr_rows = []
    for tk, td in (geo_ols or {}).get("territories", {}).items():
        t_cfg = territories_cfg.get(tk, {})
        ch_data = td.get("channels", {})
        top_ch_key = max(ch_data, key=lambda c: ch_data[c]["contribution_pct"], default=None)
        top_ch_label = ch_data[top_ch_key]["label"] if top_ch_key else "—"
        hcp_contrib = sum(
            v["contribution_pct"] for v in ch_data.values() if v["channel_type"] == "hcp"
        )
        terr_rows.append({
            "key":           tk,
            "label":         td.get("label", tk),
            "abbr":          t_cfg.get("abbr", tk[:2].upper()),
            "r_squared":     td["r_squared"],
            "mape_pct":      td["mape_pct"],
            "baseline":      td["baseline_scripts"],
            "total_spend_k": td["total_spend_k"],
            "avg_spend_k":   td["avg_period_spend_k"],
            "hcp_share_pct": round(hcp_contrib, 1),
            "top_channel":   top_ch_label,
            "market_size":   t_cfg.get("market_size", 0),
            "hcp_mult":      t_cfg.get("hcp_mult", 1.0),
            "dtc_mult":      t_cfg.get("dtc_mult", 1.0),
            "season_str":    t_cfg.get("season_str", 1.0),
            "states":        t_cfg.get("states", []),
        })
    terr_df = pd.DataFrame(terr_rows) if terr_rows else pd.DataFrame()

    # ── Territory comparison KPIs ──────────────────────────────────────────────
    if not terr_df.empty:
        st.subheader("Territory overview")
        cols = st.columns(len(terr_rows))
        for col, row in zip(cols, terr_rows):
            col.metric(
                row["label"],
                f"R²={row['r_squared']:.3f}",
                delta=f"HCP {row['hcp_share_pct']:.0f}%",
            )

        col1, col2 = st.columns(2)

        # Model fit per territory
        with col1:
            st.subheader("Model fit by territory")
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=terr_df["label"],
                y=terr_df["r_squared"],
                name="R²",
                marker_color=HCP_CLR,
                text=terr_df["r_squared"].apply(lambda v: f"{v:.3f}"),
                textposition="outside",
            ))
            fig.update_layout(
                height=300, margin=dict(t=10, b=10, l=0, r=0),
                yaxis=dict(range=[0, 1.05], title="R²"),
                xaxis_title=None,
            )
            st.plotly_chart(fig, use_container_width=True)

        # US choropleth — ROI multiplier (HCP) as proxy for territory responsiveness
        with col2:
            st.subheader("HCP responsiveness (US territories)")
            state_rows = []
            for _, row in terr_df.iterrows():
                for state in row["states"]:
                    state_rows.append({
                        "state":    state,
                        "hcp_mult": row["hcp_mult"],
                        "label":    row["label"],
                        "r2":       row["r_squared"],
                    })
            if state_rows:
                sdf = pd.DataFrame(state_rows)
                fig = px.choropleth(
                    sdf,
                    locations="state",
                    locationmode="USA-states",
                    color="hcp_mult",
                    scope="usa",
                    hover_name="label",
                    hover_data={"state": True, "hcp_mult": ":.2f", "r2": ":.3f"},
                    color_continuous_scale=[[0, "#DBEAFE"], [0.5, "#3B82F6"], [1, "#1E3A5F"]],
                    labels={"hcp_mult": "HCP mult"},
                    title=None,
                )
                fig.update_layout(
                    height=300,
                    margin=dict(t=0, b=0, l=0, r=0),
                    coloraxis_colorbar=dict(title="HCP ROI<br>multiplier", thickness=12),
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Colour = territory HCP ROI multiplier vs national average (1.0 = avg)")

    # ── Per-territory channel contributions ────────────────────────────────────
    st.subheader("Channel contributions by territory")
    geo_ols_terrs = (geo_ols or {}).get("territories", {})
    if geo_ols_terrs:
        terr_keys_sorted = sorted(geo_ols_terrs.keys())
        ch_labels = {}
        for td in geo_ols_terrs.values():
            for ck, cv in td.get("channels", {}).items():
                ch_labels[ck] = cv["label"]

        contrib_rows = []
        for tk in terr_keys_sorted:
            td = geo_ols_terrs[tk]
            label = td.get("label", tk)
            for ck, cv in td.get("channels", {}).items():
                contrib_rows.append({
                    "territory": label,
                    "channel": ch_labels.get(ck, ck),
                    "channel_key": ck,
                    "contribution_pct": cv["contribution_pct"],
                    "channel_type": cv["channel_type"],
                })
        contrib_df = pd.DataFrame(contrib_rows)

        ch_order = (
            contrib_df.groupby("channel")["contribution_pct"]
            .mean()
            .sort_values(ascending=False)
            .index.tolist()
        )
        fig = px.bar(
            contrib_df,
            x="territory",
            y="contribution_pct",
            color="channel",
            category_orders={"channel": ch_order},
            labels={"contribution_pct": "Contribution %", "territory": ""},
            height=380,
        )
        fig.update_layout(
            margin=dict(t=10, b=10, l=0, r=10),
            legend=dict(title="Channel", orientation="h", yanchor="bottom",
                        y=1.02, xanchor="left", x=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Response curves ────────────────────────────────────────────────────────
    if geo_ols_terrs:
        st.subheader("Adstock + saturation response curves by territory")
        _render_response_curves(geo_ols, config)

    # ── Seasonal ROI split heatmap ─────────────────────────────────────────────
    any_season = any(
        td.get("has_season_interactions", False)
        for td in (geo_ols or {}).get("territories", {}).values()
    )
    if any_season:
        st.subheader("In-season vs off-season ROI (vaccine season lift)")
        _render_season_interactions(geo_ols, config)

    # ── Bayesian per-territory section ────────────────────────────────────────
    if geo_bayes:
        st.divider()
        st.header("🧮 Bayesian MMM — per territory")

        # Ridge vs Bayesian ROI scatter (requires both results)
        if geo_ols and (geo_ols or {}).get("territories"):
            st.subheader("Ridge vs Bayesian ROI — all territories")
            scatter_rows = []
            for tk, td_bayes in geo_bayes.get("territories", {}).items():
                td_ridge = (geo_ols.get("territories") or {}).get(tk, {})
                terr_label = td_bayes.get("label", tk)
                for ck, cv_b in td_bayes["channels"].items():
                    cv_r = td_ridge.get("channels", {}).get(ck, {})
                    if cv_r:
                        scatter_rows.append({
                            "territory":  terr_label,
                            "channel":    cv_b["label"],
                            "ridge_roi":  cv_r.get("estimated_roi", 0),
                            "bayes_roi":  cv_b["estimated_roi"],
                            "ch_type":    cv_b["channel_type"],
                        })
            if scatter_rows:
                sc_df = pd.DataFrame(scatter_rows)
                lim   = max(sc_df["ridge_roi"].max(), sc_df["bayes_roi"].max()) * 1.1
                fig   = go.Figure()
                fig.add_trace(go.Scatter(
                    x=[0, lim], y=[0, lim], mode="lines",
                    line=dict(color="#D1D5DB", dash="dash"), showlegend=False,
                ))
                for terr_label in sc_df["territory"].unique():
                    sub = sc_df[sc_df["territory"] == terr_label]
                    fig.add_trace(go.Scatter(
                        x=sub["ridge_roi"], y=sub["bayes_roi"],
                        mode="markers", name=terr_label,
                        marker=dict(size=8),
                        text=sub["channel"],
                        hovertemplate="%{text}<br>Ridge: %{x:.3f}<br>Bayesian: %{y:.3f}",
                    ))
                fig.update_layout(
                    height=380, margin=dict(t=10, b=10, l=0, r=10),
                    xaxis_title="Ridge ROI", yaxis_title="Bayesian ROI",
                    legend=dict(title="Territory", orientation="v"),
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Points above diagonal = Bayesian assigns higher ROI than Ridge. "
                           "Tight cluster = models agree; dispersion = more uncertainty.")

        _render_geo_bayesian(geo_bayes, config)
    elif geo_df is not None:
        st.info(
            "Geo Bayesian results not yet available. "
            "Check **Include Geo Bayesian** and click **🗺️ Run Geo Pipeline** (~20 min)."
        )

    # ── Hierarchical Bayesian section ──────────────────────────────────────────
    if geo_hier:
        st.divider()
        with st.expander("🔗 Hierarchical Bayesian MMM — partial pooling across territories",
                         expanded=False):
            _render_geo_hierarchical(geo_hier, config)
    elif geo_df is not None:
        st.info(
            "Hierarchical Bayesian results not yet available. "
            "Check **Include Geo Hierarchical** and click **🗺️ Run Geo Pipeline** (~5 min)."
        )

    # ── Geo budget optimiser results ───────────────────────────────────────────
    if geo_opt:
        st.subheader("Geo budget optimiser")
        alloc = geo_opt.get("territory_allocation", {})
        terr_alloc_rows = []
        for tk, ta in alloc.items():
            terr_alloc_rows.append({
                "Territory":        ta["label"],
                "ROI efficiency":   ta["roi_efficiency"],
                "Current share":    f"{ta['current_share']:.1%}",
                "Optimal share":    f"{ta['optimal_share']:.1%}",
                "Current $K":       ta["current_budget_k"],
                "Optimal $K":       ta["optimal_budget_k"],
                "Δ share":          f"{ta['change_pct']:+.1f}%",
                "Chan uplift":      f"+{ta['projected_channel_uplift_pct']:.1f}%",
                "Action":           ta["action"],
            })

        ta_df = pd.DataFrame(terr_alloc_rows)

        col1, col2 = st.columns([3, 2])
        with col1:
            fig = go.Figure()
            colors = [
                UP_CLR if a["change_pct"] > 3 else (DOWN_CLR if a["change_pct"] < -3 else NEUT_CLR)
                for a in alloc.values()
            ]
            labels_sorted = [a["label"] for a in alloc.values()]
            curr_budgets  = [a["current_budget_k"] for a in alloc.values()]
            opt_budgets   = [a["optimal_budget_k"] for a in alloc.values()]
            fig.add_trace(go.Bar(name="Current",     x=curr_budgets, y=labels_sorted,
                                 orientation="h", marker_color="#CBD5E1"))
            fig.add_trace(go.Bar(name="Recommended", x=opt_budgets,  y=labels_sorted,
                                 orientation="h", marker_color=colors))
            fig.update_layout(
                barmode="overlay", height=320,
                margin=dict(t=10, b=10, l=0, r=0),
                xaxis_title="Budget $K / period",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.dataframe(ta_df, use_container_width=True, hide_index=True, height=320)

        c1, c2 = st.columns(2)
        c1.metric(
            "Territory reallocation uplift",
            f"+{geo_opt['projected_territory_uplift_pct']:.1f}%",
            delta="same national budget",
        )
        c2.metric("National budget", fmt_k(geo_opt["total_national_budget_k"]))

        # Per-territory channel detail expanders
        st.subheader("Per-territory channel recommendations")
        for tk, ta in alloc.items():
            with st.expander(f"{ta['label']} — channel uplift +{ta['projected_channel_uplift_pct']:.1f}%"):
                ch_rows = ta.get("channel_allocations", [])
                if ch_rows:
                    ch_df = pd.DataFrame(ch_rows)[
                        ["channel", "type", "fitted_roi",
                         "current_spend_k", "optimal_spend_k", "change_pct", "action"]
                    ].copy()
                    ch_df.columns = ["Channel", "Type", "ROI", "Current $K", "Optimal $K", "Δ%", "Action"]
                    ch_df["Δ%"] = ch_df["Δ%"].apply(lambda v: f"{v:+.1f}%")
                    st.dataframe(ch_df.sort_values("ROI", ascending=False),
                                 use_container_width=True, hide_index=True)

    # ── What-if simulator ─────────────────────────────────────────────────────
    if geo_opt:
        st.divider()
        _render_whatif_simulator(geo_opt)

    # ── Geo AI narrative ───────────────────────────────────────────────────────
    st.divider()
    st.header("📝 Geo AI Narrative")
    if geo_narrative:
        st.download_button(
            "⬇ Download geo report (Markdown)",
            data=geo_narrative,
            file_name=f"geo_insights.md",
            mime="text/markdown",
        )
        st.divider()
        st.markdown(geo_narrative)
    else:
        st.info(
            "Geo AI narrative not yet generated. "
            "Check **Include Geo AI narrative** and click **🗺️ Run Geo Pipeline**."
        )


def tab_attribution(ols, geo_ols, config, freq):
    """Attribution decomposition: stacked area + waterfall for national and geo."""
    if ols is None:
        st.info("Run the pipeline first to see attribution results.")
        return

    ch_config = config.get("channels", {})

    # Distinct color palette: HCP = blues, DTC = warm tones, baseline = grey
    CH_COLORS = {
        "rep_visits":         "#1D4ED8",
        "medical_congress":   "#2563EB",
        "journal_advertising":"#3B82F6",
        "hcp_email":          "#60A5FA",
        "hcp_digital":        "#93C5FD",
        "speaker_programs":   "#1E40AF",
        "samples_coupons":    "#BFDBFE",
        "dtc_tv":             "#EA580C",
        "dtc_digital":        "#F97316",
        "dtc_ooh":            "#FB923C",
        "patient_email":      "#FCA5A1",
        "patient_advocacy":   "#DC2626",
    }
    BASELINE_COLOR = "#D1D5DB"

    def _stacked_area(dates, actuals, baseline_ts, contrib_ts, channel_meta, title, mape):
        """Stacked area (model decomposition) + actual scripts overlay."""
        fig = go.Figure()

        # Baseline layer
        fig.add_trace(go.Scatter(
            x=dates, y=baseline_ts,
            name="Baseline",
            stackgroup="one",
            fillcolor=BASELINE_COLOR,
            line=dict(color=BASELINE_COLOR, width=0),
            hovertemplate="Baseline: %{y:,.0f}<extra></extra>",
        ))

        # Channel layers — contribution-sorted order
        ch_order = sorted(
            contrib_ts.keys(),
            key=lambda c: channel_meta.get(c, {}).get("contribution_pct", 0),
        )
        for ch in ch_order:
            ts    = contrib_ts[ch]
            label = channel_meta.get(ch, {}).get("label", ch)
            color = CH_COLORS.get(ch, "#94A3B8")
            fig.add_trace(go.Scatter(
                x=dates, y=ts,
                name=label,
                stackgroup="one",
                fillcolor=color,
                line=dict(color=color, width=0),
                hovertemplate=f"{label}: %{{y:,.0f}}<extra></extra>",
            ))

        # Actual scripts overlay
        fig.add_trace(go.Scatter(
            x=dates, y=actuals,
            name="Actual scripts",
            mode="lines",
            line=dict(color="#111827", width=1.5, dash="dot"),
            hovertemplate="Actual: %{y:,.0f}<extra></extra>",
        ))

        fig.update_layout(
            title=dict(text=title, font=dict(size=13)),
            height=380,
            margin=dict(t=40, b=10, l=0, r=0),
            xaxis_title=None,
            yaxis_title="Scripts written",
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="left", x=0, font=dict(size=10)),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"Stacked area = model decomposition (y_pred). "
            f"Dotted line = actual scripts. Gap = model residual (MAPE {mape:.1f}%)."
        )

    def _waterfall(channel_meta, baseline_scripts, title):
        """Waterfall: baseline → each channel's total contribution → grand total.

        Built from go.Bar with invisible spacers so each segment gets its own
        channel color (go.Waterfall only supports uniform increasing/decreasing colors).
        """
        ch_sorted = sorted(
            [(ck, cv) for ck, cv in channel_meta.items()],
            key=lambda x: x[1]["total_contribution"],
            reverse=True,
        )
        total = baseline_scripts + sum(cv["total_contribution"] for _, cv in ch_sorted)
        labels  = ["Baseline"] + [cv["label"] for _, cv in ch_sorted] + ["Total"]
        heights = ([baseline_scripts]
                   + [cv["total_contribution"] for _, cv in ch_sorted]
                   + [total])
        bar_colors = (
            [BASELINE_COLOR]
            + [CH_COLORS.get(ck, "#94A3B8") for ck, _ in ch_sorted]
            + ["#374151"]
        )

        # Cumulative bottom for each bar (spacer height)
        bottoms = [0.0]
        running = baseline_scripts
        for _, cv in ch_sorted:
            bottoms.append(running)
            running += cv["total_contribution"]
        bottoms.append(0.0)  # Total bar starts at 0

        fig = go.Figure()
        # Invisible spacer bars that push each segment to the right height
        fig.add_trace(go.Bar(
            x=labels, y=bottoms,
            marker_color="rgba(0,0,0,0)",
            showlegend=False,
            hoverinfo="skip",
        ))
        # Visible segment bars
        fig.add_trace(go.Bar(
            x=labels, y=heights,
            marker_color=bar_colors,
            text=[f"{v:,.0f}" for v in heights],
            textposition="outside",
            showlegend=False,
            hovertemplate="%{x}: %{y:,.0f}<extra></extra>",
        ))
        fig.update_layout(
            barmode="stack",
            title=dict(text=title, font=dict(size=13)),
            height=420,
            margin=dict(t=40, b=10, l=0, r=0),
            xaxis=dict(tickangle=-30),
            yaxis_title="Scripts written",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── National attribution ───────────────────────────────────────────────────
    st.header("National attribution")

    freq_label = freq.capitalize()
    dates      = ols.get("dates", [])
    actuals    = ols.get("actuals", [])
    baseline   = ols.get("baseline_timeseries", [])
    contrib_ts = ols.get("contribution_timeseries", {})
    ch_meta    = ols.get("channels", {})

    if dates and actuals and baseline:
        col1, col2 = st.columns([3, 1])
        with col1:
            _stacked_area(dates, actuals, baseline, contrib_ts, ch_meta,
                          f"{freq_label} scripts — channel decomposition",
                          ols.get("mape_pct", 0))
        with col2:
            total_ch = sum(cv["total_contribution"] for cv in ch_meta.values())
            total_all = ols.get("baseline_scripts", 0) + total_ch
            st.metric("Baseline scripts", f"{ols.get('baseline_scripts', 0):,.0f}",
                      help="Model intercept + control + seasonality effects")
            st.metric("Channel-driven scripts", f"{total_ch:,.0f}",
                      delta=f"{total_ch / max(total_all, 1) * 100:.0f}% of total")
            st.metric("Avg scripts / period",
                      f"{sum(actuals) / max(len(actuals), 1):,.0f}")

        st.subheader("Contribution waterfall")
        _waterfall(ch_meta, ols.get("baseline_scripts", 0),
                   f"{freq_label} — total scripts decomposed by channel")
    else:
        st.info("Re-run the pipeline to generate time-series attribution data.")

    # ── Geo attribution ────────────────────────────────────────────────────────
    if geo_ols and geo_ols.get("territories"):
        st.divider()
        st.header("Geo attribution")

        terr_items   = sorted(geo_ols["territories"].items())
        terr_options = {tk: td.get("label", tk) for tk, td in terr_items}
        selected_tk  = st.selectbox(
            "Territory",
            options=list(terr_options.keys()),
            format_func=lambda k: terr_options[k],
            key="attr_geo_terr",
        )
        td = geo_ols["territories"][selected_tk]

        g_dates    = td.get("dates", [])
        g_actuals  = td.get("actuals", [])
        g_baseline = td.get("baseline_timeseries", [])
        g_contrib  = td.get("contribution_timeseries", {})
        g_ch_meta  = td.get("channels", {})

        if g_dates and g_actuals and g_baseline:
            col1, col2 = st.columns([3, 1])
            with col1:
                _stacked_area(g_dates, g_actuals, g_baseline, g_contrib, g_ch_meta,
                              f"{terr_options[selected_tk]} — {freq_label} channel decomposition",
                              td.get("mape_pct", 0))
            with col2:
                total_ch = sum(cv["total_contribution"] for cv in g_ch_meta.values())
                total_all = td.get("baseline_scripts", 0) + total_ch
                st.metric("Baseline scripts", f"{td.get('baseline_scripts', 0):,.0f}")
                st.metric("Channel-driven", f"{total_ch:,.0f}",
                          delta=f"{total_ch / max(total_all, 1) * 100:.0f}% of total")
                st.metric("Model R²", f"{td.get('r_squared', 0):.3f}")

            st.subheader("Contribution waterfall")
            _waterfall(g_ch_meta, td.get("baseline_scripts", 0),
                       f"{terr_options[selected_tk]} — total scripts decomposed by channel")
        else:
            st.info("Re-run the geo pipeline to generate territory time-series data.")


def tab_scenario(ols: dict, config: dict, freq: str = "weekly"):
    st.header("🎯 Scenario Planner")
    st.caption(
        "Two complementary views: **Target → Budget** finds the minimum spend required to hit "
        "a NRx goal; **Budget → Scripts** shows the maximum NRx achievable from a given budget. "
        "Both include an efficiency frontier and channel mix breakdown."
    )

    if not ols:
        st.info("Run the national pipeline first (OLS results required).")
        return

    import numpy as np

    n_periods = 104 if freq == "weekly" else 36
    ch_params = _build_channel_params(ols, config, n_periods)
    baseline_pp = ols.get("baseline_scripts", 0.0) / n_periods
    x_curr = np.array([p["current_spend_k"] for p in ch_params])
    current_scripts = _total_scripts(x_curr, ch_params, baseline_pp)
    current_budget  = float(x_curr.sum())

    period_label = "week" if freq == "weekly" else "month"

    # ── Mode selector ──────────────────────────────────────────────────────────
    mode = st.radio(
        "Planning mode",
        ["🎯 Target → find minimum budget",
         "💰 Budget → find maximum NRx"],
        horizontal=True,
        key="scenario_mode",
    )

    st.divider()

    if mode.startswith("🎯"):
        # ── TARGET → BUDGET ────────────────────────────────────────────────────
        st.subheader("🎯 Target NRx → Minimum budget")

        col_ctrl, col_metric = st.columns([3, 2])
        with col_ctrl:
            lift_pct = st.slider(
                "NRx target lift vs current",
                min_value=-20, max_value=50, value=10, step=1,
                format="%d%%", key="scen_lift_pct",
            )
            target_scripts = current_scripts * (1 + lift_pct / 100)
            st.caption(
                f"Target: **{target_scripts:,.0f} scripts/{period_label}** "
                f"(current: {current_scripts:,.0f})"
            )
            relax = st.checkbox("Relax spend corridors (allow larger channel shifts)",
                                key="scen_relax")

        result = solve_target_to_budget(target_scripts, ols, config, freq, relax_corridors=relax)

        with col_metric:
            if result["feasible"]:
                delta_k   = result["budget_delta_k"]
                delta_pct = result["budget_delta_pct"]
                req_k     = result["required_budget_k"]
                st.metric("Required budget / period",
                          f"${req_k:,.0f}K",
                          delta=f"{delta_pct:+.1f}% vs current",
                          delta_color="inverse")
                st.metric("Budget change",
                          f"${delta_k:+,.0f}K / {period_label}",
                          delta=None)
                st.metric("Achieved NRx",
                          f"{result['achieved_scripts']:,.0f} scripts/{period_label}")
            else:
                st.error(f"⚠️ Infeasible: {result['infeasibility_reason']}")

        if result["feasible"] and result["channels"]:
            st.divider()

            # Channel breakdown table + bar chart side by side
            tc1, tc2 = st.columns([2, 3])

            with tc1:
                st.subheader("Channel allocation")
                tbl = []
                for ch in result["channels"]:
                    action = ("↑ Increase" if ch["delta_pct"] > 5
                              else "↓ Reduce" if ch["delta_pct"] < -5
                              else "→ Hold")
                    tbl.append({
                        "Channel":      ch["channel_label"],
                        "Type":         ch["channel_type"].upper(),
                        "Current $K":   f"{ch['current_spend_k']:.1f}",
                        "Scenario $K":  f"{ch['scenario_spend_k']:.1f}",
                        "Change":       f"{ch['delta_pct']:+.1f}%",
                        "Action":       action,
                    })
                st.dataframe(pd.DataFrame(tbl), use_container_width=True, hide_index=True)

            with tc2:
                st.subheader("Current vs scenario spend")
                chs = result["channels"]
                labels = [c["channel_label"] for c in chs]
                curr_vals = [c["current_spend_k"]  for c in chs]
                scen_vals = [c["scenario_spend_k"] for c in chs]
                fig_cmp = go.Figure()
                fig_cmp.add_trace(go.Bar(
                    name="Current", x=labels, y=curr_vals,
                    marker_color="#94A3B8",
                    hovertemplate="%{x}<br>Current: $%{y:.1f}K<extra></extra>",
                ))
                fig_cmp.add_trace(go.Bar(
                    name="Scenario", x=labels, y=scen_vals,
                    marker_color="#2563EB",
                    hovertemplate="%{x}<br>Scenario: $%{y:.1f}K<extra></extra>",
                ))
                fig_cmp.update_layout(
                    barmode="group", height=370,
                    yaxis_title="Spend $K / period",
                    xaxis_tickangle=-35,
                    legend=dict(orientation="h", yanchor="bottom", y=1.01),
                    margin=dict(t=20, b=80),
                )
                st.plotly_chart(fig_cmp, use_container_width=True)

    else:
        # ── BUDGET → SCRIPTS ──────────────────────────────────────────────────
        st.subheader("💰 Budget → Maximum NRx")

        col_ctrl2, col_metric2 = st.columns([3, 2])
        with col_ctrl2:
            budget_delta_pct = st.slider(
                "Total budget vs current",
                min_value=-30, max_value=50, value=0, step=5,
                format="%d%%", key="scen_budget_pct",
            )
            scenario_budget = current_budget * (1 + budget_delta_pct / 100)
            st.caption(
                f"Budget: **${scenario_budget:,.0f}K / {period_label}** "
                f"(current: ${current_budget:,.0f}K)"
            )

        result2 = solve_budget_to_scripts(scenario_budget, ols, config, freq)

        with col_metric2:
            achieved = result2["achieved_scripts"]
            scripts_delta_pct = (achieved - current_scripts) / max(current_scripts, 1) * 100
            st.metric("Projected NRx / period",
                      f"{achieved:,.0f}",
                      delta=f"{scripts_delta_pct:+.1f}% vs current")
            st.metric("Budget / period", f"${scenario_budget:,.0f}K",
                      delta=f"{budget_delta_pct:+.0f}% vs current",
                      delta_color="off")

        if result2["channels"]:
            st.divider()
            tc1b, tc2b = st.columns([2, 3])

            with tc1b:
                st.subheader("Optimal channel mix")
                tbl2 = []
                for ch in result2["channels"]:
                    action = ("↑ Increase" if ch["delta_pct"] > 5
                              else "↓ Reduce" if ch["delta_pct"] < -5
                              else "→ Hold")
                    tbl2.append({
                        "Channel":     ch["channel_label"],
                        "Type":        ch["channel_type"].upper(),
                        "Current $K":  f"{ch['current_spend_k']:.1f}",
                        "Optimal $K":  f"{ch['scenario_spend_k']:.1f}",
                        "Change":      f"{ch['delta_pct']:+.1f}%",
                        "Action":      action,
                    })
                st.dataframe(pd.DataFrame(tbl2), use_container_width=True, hide_index=True)

            with tc2b:
                st.subheader("Current vs optimised spend")
                chs2 = result2["channels"]
                labels2   = [c["channel_label"] for c in chs2]
                curr_vals2 = [c["current_spend_k"]  for c in chs2]
                scen_vals2 = [c["scenario_spend_k"] for c in chs2]
                fig_cmp2 = go.Figure()
                fig_cmp2.add_trace(go.Bar(
                    name="Current", x=labels2, y=curr_vals2,
                    marker_color="#94A3B8",
                    hovertemplate="%{x}<br>Current: $%{y:.1f}K<extra></extra>",
                ))
                fig_cmp2.add_trace(go.Bar(
                    name="Optimised", x=labels2, y=scen_vals2,
                    marker_color="#16A34A",
                    hovertemplate="%{x}<br>Optimised: $%{y:.1f}K<extra></extra>",
                ))
                fig_cmp2.update_layout(
                    barmode="group", height=370,
                    yaxis_title="Spend $K / period",
                    xaxis_tickangle=-35,
                    legend=dict(orientation="h", yanchor="bottom", y=1.01),
                    margin=dict(t=20, b=80),
                )
                st.plotly_chart(fig_cmp2, use_container_width=True)

    # ── Efficiency frontier (shared by both modes) ─────────────────────────────
    st.divider()
    st.subheader("📈 Efficiency frontier — NRx vs required budget")
    st.caption(
        "How much does each extra NRx point cost? The curve flattens as saturation "
        "kicks in — the steeper sections show where incremental spend is most efficient."
    )

    with st.spinner("Computing frontier…"):
        frontier = compute_efficiency_frontier(ols, config, freq, n_points=25)

    feasible_pts = [p for p in frontier if p["feasible"] and p["required_budget_k"] is not None]
    infeasible_pts = [p for p in frontier if not p["feasible"]]

    if feasible_pts:
        f_lifts   = [p["lift_pct"]          for p in feasible_pts]
        f_budgets = [p["required_budget_k"]  for p in feasible_pts]
        f_targets = [p["target_scripts"]     for p in feasible_pts]

        fig_front = go.Figure()
        fig_front.add_trace(go.Scatter(
            x=f_lifts,
            y=f_budgets,
            mode="lines+markers",
            name="Min budget to hit target",
            line=dict(color="#2563EB", width=2.5),
            marker=dict(size=6),
            customdata=list(zip(f_targets, [p["budget_delta_pct"] for p in feasible_pts])),
            hovertemplate=(
                "NRx lift: %{x:+.0f}%<br>"
                "Target: %{customdata[0]:,.0f} scripts<br>"
                "Budget: $%{y:,.0f}K<br>"
                "vs current: %{customdata[1]:+.1f}%<extra></extra>"
            ),
        ))
        # Mark current point
        fig_front.add_trace(go.Scatter(
            x=[0], y=[current_budget],
            mode="markers",
            name="Current",
            marker=dict(color="#EA580C", size=12, symbol="diamond"),
            hovertemplate="Current: $%{y:,.0f}K<extra></extra>",
        ))
        # Mark selected target
        if mode.startswith("🎯"):
            selected_lift = lift_pct
        else:
            selected_lift = (result2["achieved_scripts"] / current_scripts - 1) * 100

        fig_front.add_vline(
            x=selected_lift, line_dash="dot", line_color="gray",
            annotation_text="Selected", annotation_position="top right",
        )
        fig_front.update_layout(
            xaxis_title="NRx lift vs current (%)",
            yaxis_title="Required budget $K / period",
            height=380,
            legend=dict(orientation="h", yanchor="bottom", y=1.01),
            margin=dict(t=30, b=40),
        )
        st.plotly_chart(fig_front, use_container_width=True)

        if infeasible_pts:
            max_lift = max(p["lift_pct"] for p in feasible_pts)
            st.caption(
                f"Corridor constraints prevent targets above +{max_lift:.0f}% lift "
                f"({len(infeasible_pts)} points outside feasible range). "
                "Enable 'relax corridors' to extend the frontier."
            )


def tab_incrementality(geo_ols: dict, geo_bayes: dict, config: dict, freq: str = "weekly"):
    st.header("🧪 Incrementality Testing Planner")
    st.caption(
        "Ranks every territory × channel pair as a candidate for a geo holdout / lift test. "
        "Four signals drive the score: **Bayesian HDI uncertainty**, **Ridge vs Bayesian model disagreement**, "
        "**saturation headroom**, and **spend materiality**. Higher score = higher priority to test."
    )

    if not geo_ols:
        st.info("Run the geo pipeline first to enable the incrementality planner.")
        return

    # ── Signal weight controls ─────────────────────────────────────────────────
    with st.expander("⚙️ Signal weights", expanded=False):
        st.caption("Adjust how much each signal contributes to the composite score (must sum to 1.0).")
        wc1, wc2, wc3, wc4 = st.columns(4)
        w_hdi  = wc1.slider("HDI uncertainty",     0.0, 1.0, 0.35, 0.05, key="inc_w_hdi")
        w_dis  = wc2.slider("Model disagreement",  0.0, 1.0, 0.35, 0.05, key="inc_w_dis")
        w_sat  = wc3.slider("Saturation headroom", 0.0, 1.0, 0.20, 0.05, key="inc_w_sat")
        w_spd  = wc4.slider("Spend materiality",   0.0, 1.0, 0.10, 0.05, key="inc_w_spd")
        total_w = w_hdi + w_dis + w_sat + w_spd
        if abs(total_w - 1.0) > 0.01:
            st.warning(f"Weights sum to {total_w:.2f} — they will be normalised automatically.")
        total_w = total_w or 1.0
        weights = {
            "hdi_uncertainty":    w_hdi  / total_w,
            "model_disagreement": w_dis  / total_w,
            "saturation_headroom": w_sat / total_w,
            "spend_materiality":   w_spd / total_w,
        }

    candidates = compute_incrementality_scores(geo_ols, geo_bayes, weights)
    if not candidates:
        st.warning("No candidate data found — ensure geo OLS results are available.")
        return

    missing_bayes = all(c["missing_bayes"] for c in candidates)
    if missing_bayes:
        st.info(
            "Bayesian geo results not found — HDI uncertainty and model disagreement signals "
            "are unavailable. Run the geo pipeline with **Bayesian** enabled for richer scoring."
        )

    # ── Summary metrics ────────────────────────────────────────────────────────
    top5   = candidates[:5]
    n_terr = len({c["territory_key"] for c in candidates})
    n_ch   = len({c["channel_key"]   for c in candidates})
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Total candidates",  f"{len(candidates)}")
    mc2.metric("Territories",       f"{n_terr}")
    mc3.metric("Channels",          f"{n_ch}")

    st.divider()

    # ── Ranked table ───────────────────────────────────────────────────────────
    st.subheader("📋 Ranked candidates")

    table_rows = []
    for c in candidates:
        table_rows.append({
            "Rank":          c["rank"],
            "Territory":     c["territory_label"],
            "Channel":       c["channel_label"],
            "Type":          c["channel_type"].upper(),
            "Score":         round(c["composite_score"], 3),
            "HDI Uncert.":   f"{c['hdi_width_pct']:.0f}%",
            "Model Disagree":f"{c['model_disagree_pct']:.0f}%",
            "Sat. Headroom": f"{c['saturation_headroom_pct']:.0f}%",
            "Avg Spend $K":  f"{c['avg_spend_k']:.0f}",
            "OLS ROI":       f"{c['ols_roi']:.2f}",
            "Bayes ROI":     f"—" if c["missing_bayes"] else f"{c['bayes_roi']:.2f}",
        })

    df_table = pd.DataFrame(table_rows)

    def _score_color(val):
        if isinstance(val, float):
            g = int(val * 200)
            return f"background-color: rgba(37,99,235,{val:.2f}); color: {'white' if val > 0.5 else 'black'}"
        return ""

    styled = (
        df_table.style
        .applymap(_score_color, subset=["Score"])
        .set_properties(subset=["Score"], **{"font-weight": "bold"})
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # ── Scatter: HDI uncertainty vs model disagreement ─────────────────────────
    st.divider()
    st.subheader("🔵 Signal map — HDI uncertainty vs model disagreement")
    st.caption(
        "Each bubble is a territory × channel pair. "
        "Candidates in the **upper-right** have both high uncertainty AND high model disagreement "
        "— the best targets for incrementality tests. Bubble size = avg spend."
    )

    scatter_df = pd.DataFrame([{
        "Territory":       c["territory_label"],
        "Channel":         c["channel_label"],
        "HDI Uncertainty": c["hdi_width_pct"],
        "Model Disagree":  c["model_disagree_pct"],
        "Spend $K":        max(c["avg_spend_k"], 5.0),
        "Score":           c["composite_score"],
        "Rank":            c["rank"],
    } for c in candidates])

    terr_colors = px.colors.qualitative.Set2
    terr_list   = scatter_df["Territory"].unique().tolist()
    color_map   = {t: terr_colors[i % len(terr_colors)] for i, t in enumerate(terr_list)}

    fig_scatter = go.Figure()
    for terr in terr_list:
        sub = scatter_df[scatter_df["Territory"] == terr]
        fig_scatter.add_trace(go.Scatter(
            x=sub["HDI Uncertainty"],
            y=sub["Model Disagree"],
            mode="markers+text",
            name=terr,
            text=sub["Channel"].str.split().str[0],  # first word for brevity
            textposition="top center",
            textfont=dict(size=9),
            marker=dict(
                size=sub["Spend $K"].apply(lambda v: max(8, min(45, v / 5))),
                color=color_map[terr],
                opacity=0.75,
                line=dict(width=1, color="white"),
            ),
            customdata=sub[["Territory", "Channel", "Score", "Rank"]].values,
            hovertemplate=(
                "<b>%{customdata[0]} — %{customdata[1]}</b><br>"
                "HDI uncertainty: %{x:.0f}%<br>"
                "Model disagree: %{y:.0f}%<br>"
                "Score: %{customdata[2]:.3f}  (rank #%{customdata[3]})<extra></extra>"
            ),
        ))

    # Quadrant lines at medians
    med_x = float(scatter_df["HDI Uncertainty"].median())
    med_y = float(scatter_df["Model Disagree"].median())
    for val, is_x in [(med_x, True), (med_y, False)]:
        fig_scatter.add_shape(
            type="line",
            x0=val if is_x else scatter_df["HDI Uncertainty"].min(),
            x1=val if is_x else scatter_df["HDI Uncertainty"].max(),
            y0=scatter_df["Model Disagree"].min() if is_x else val,
            y1=scatter_df["Model Disagree"].max() if is_x else val,
            line=dict(color="gray", width=1, dash="dot"),
        )
    fig_scatter.add_annotation(
        x=scatter_df["HDI Uncertainty"].max() * 0.95,
        y=scatter_df["Model Disagree"].max() * 0.97,
        text="High priority zone",
        showarrow=False,
        font=dict(size=11, color="#2563EB"),
        bgcolor="rgba(219,234,254,0.7)",
    )
    fig_scatter.update_layout(
        xaxis_title="HDI Uncertainty (%)",
        yaxis_title="Model Disagreement (%)",
        height=460,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=30, b=40),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Score breakdown bar chart ──────────────────────────────────────────────
    st.divider()
    st.subheader("📊 Score breakdown — top 15 candidates")
    top15 = candidates[:15]
    labels_top = [f"#{c['rank']} {c['territory_label'][:4]}·{c['channel_label'].split()[0]}"
                  for c in top15]
    signals = [
        ("HDI Uncertainty",    "hdi_uncertainty",    "#2563EB"),
        ("Model Disagreement", "model_disagreement", "#EA580C"),
        ("Sat. Headroom",      "saturation_headroom","#16A34A"),
        ("Spend Materiality",  "spend_materiality",  "#9333EA"),
    ]
    fig_bar = go.Figure()
    for sig_label, sig_key, color in signals:
        w_key = sig_key
        w_val = weights.get(w_key, 0.25)
        fig_bar.add_trace(go.Bar(
            name=sig_label,
            x=labels_top,
            y=[c[sig_key] * w_val for c in top15],
            marker_color=color,
            hovertemplate=f"{sig_label}: %{{y:.3f}}<extra></extra>",
        ))
    fig_bar.update_layout(
        barmode="stack",
        height=380,
        yaxis_title="Weighted contribution to score",
        xaxis_tickangle=-35,
        legend=dict(orientation="h", yanchor="bottom", y=1.01),
        margin=dict(t=30, b=80),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Top candidates with holdout design cards ───────────────────────────────
    st.divider()
    st.subheader("🗂️ Top candidates — holdout design recommendations")

    n_show = st.slider("Show top N candidates", 3, min(15, len(candidates)), 5,
                       key="inc_top_n")
    for c in candidates[:n_show]:
        design = _holdout_design(c, freq)
        with st.expander(
            f"#{c['rank']}  {c['territory_label']} — {c['channel_label']}"
            f"  (score {c['composite_score']:.3f})",
            expanded=(c["rank"] <= 3),
        ):
            dc1, dc2, dc3 = st.columns(3)
            dc1.metric("OLS ROI",   f"{c['ols_roi']:.2f}x")
            dc2.metric("Bayes ROI", f"—" if c["missing_bayes"] else f"{c['bayes_roi']:.2f}x")
            dc3.metric("Avg spend", f"${c['avg_spend_k']:.0f}K / period")

            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("HDI uncertainty",    f"{c['hdi_width_pct']:.0f}%")
            sc2.metric("Model disagreement", f"{c['model_disagree_pct']:.0f}%")
            sc3.metric("Sat. headroom",      f"{c['saturation_headroom_pct']:.0f}%")
            sc4.metric("Composite score",    f"{c['composite_score']:.3f}")

            st.markdown(
                f"**Recommended approach:** {design['approach']}  \n"
                f"**Test duration:** {design['duration']}  \n"
                f"**Holdout depth:** {design['depth']}  \n"
                f"**Power note:** {design['power_note']}"
            )


def tab_insights(freq):
    report_path = f"reports/mmm_{freq}_insights.md"
    if not os.path.exists(report_path):
        st.info(
            "No insight report yet — run the pipeline with "
            "**Include AI insight narrative** checked."
        )
        return

    with open(report_path) as f:
        content = f.read()

    st.download_button(
        "⬇ Download report (Markdown)",
        data=content,
        file_name=f"mmm_{freq}_insights.md",
        mime="text/markdown",
    )
    st.divider()
    st.markdown(content)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    config = load_config()
    freq, run_bayesian, run_insights, run_btn, run_geo_btn, run_geo_bayes, run_geo_hier, run_geo_insights = sidebar(config)

    if run_btn:
        run_pipeline(freq, run_bayesian, run_insights)
        st.rerun()

    if run_geo_btn:
        run_geo_pipeline(freq, run_bayesian=run_geo_bayes,
                         run_hierarchical=run_geo_hier, run_insights=run_geo_insights)
        st.rerun()

    prefix = f"data/raw/mmm_{freq}"
    df     = load_dataset(freq)
    geo_df = load_geo_dataset(freq)
    ols    = load_json(f"{prefix}_ols_results.json")
    bayes  = load_json(f"{prefix}_bayesian_results.json")
    opt    = load_json(f"{prefix}_budget_optimized.json")
    geo_ols       = load_json(f"{prefix}_geo_ols_results.json")
    geo_opt       = load_json(f"{prefix}_geo_budget_optimized.json")
    geo_bayes     = load_geo_bayesian(freq)
    geo_hier      = load_geo_hierarchical(freq)
    geo_narrative = load_geo_narrative(freq)

    st.title("💊 Pharma MMM Agent")
    st.caption(
        f"**{freq.capitalize()} dataset** — "
        + (f"{len(df)} periods | " if df is not None else "")
        + (f"OLS R²={ols['r_squared']:.3f} | " if ols else "")
        + (f"Bayesian R̂={bayes['mcmc'].get('max_rhat','—')} | " if bayes else "")
        + (f"+{opt['projected_uplift_pct']:.1f}% projected uplift" if opt else "")
        + (f"| Geo: {len((geo_ols or {}).get('territories', {}))} territories" if geo_ols else "")
    )

    tab_names = ["📈 Overview", "📊 Ridge MMM", "🧮 Bayesian MMM",
                 "🔍 Attribution", "💰 Budget", "🗺️ Geo",
                 "🎯 Scenario", "🧪 Incrementality", "📝 Insights"]
    tabs = st.tabs(tab_names)

    with tabs[0]:
        if df is None:
            st.info("Dataset not found. Run `python scripts/generate_dataset.py` first.")
        else:
            tab_overview(df, config, freq)

    with tabs[1]:
        tab_ridge(ols, config, freq)

    with tabs[2]:
        tab_bayesian(bayes, ols, config)

    with tabs[3]:
        tab_attribution(ols, geo_ols, config, freq)

    with tabs[4]:
        tab_budget(opt, config)

    with tabs[5]:
        tab_geo(geo_df, geo_ols, geo_opt, geo_bayes, geo_hier, geo_narrative, config)

    with tabs[6]:
        tab_scenario(ols, config, freq)

    with tabs[7]:
        tab_incrementality(geo_ols, geo_bayes, config, freq)

    with tabs[8]:
        tab_insights(freq)


if __name__ == "__main__":
    main()
