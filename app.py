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

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pharma MMM Agent",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Colour palette ────────────────────────────────────────────────────────────
HCP_CLR   = "#2563EB"
DTC_CLR   = "#EA580C"
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

def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

def ch_color(ch_name, source=None):
    base = HCP_CLR if ch_name in HCP_CHANNELS else DTC_CLR
    if source == "prior_estimate":
        return base + "80"   # 50% alpha in hex
    return base

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

    run_btn = st.sidebar.button("▶ Run Pipeline", type="primary", use_container_width=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**LLM config**")
    st.sidebar.code(
        f"provider : {config['llm']['provider']}\n"
        f"model    : {config['llm']['model']}",
        language=None,
    )
    return freq, run_bayesian, run_insights, run_btn


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

def tab_overview(df, config):
    channels = list(config["channels"].keys())

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    total_spend = df[channels].values.sum() / 1000
    c1.metric("Total 2yr spend", f"${total_spend:.1f}M")
    c2.metric("Avg weekly scripts", f"{df['scripts_written'].mean():,.0f}")
    c3.metric("Peak / trough ratio",
              f"{df['scripts_written'].max()/df['scripts_written'].min():.2f}×")
    c4.metric("Weeks", str(len(df)))

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


def tab_ridge(ols, config):
    if ols is None:
        st.info("No OLS results yet — run the pipeline from the sidebar.")
        return

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    n_model = sum(1 for ch in ols["channels"].values()
                  if ch.get("contribution_source") == "model")
    c1.metric("R²", f"{ols['r_squared']:.3f}",
              delta="Excellent" if ols["r_squared"] > 0.9 else None)
    c2.metric("MAPE", f"{ols['mape_pct']:.1f}%")
    c3.metric("Baseline scripts/wk", f"{ols['baseline_scripts']:,.0f}")
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
        disp.columns = ["Channel", "Type", "Avg wk spend $K", "Total spend $K",
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
                    color=merged["channel"].apply(
                        lambda c: HCP_CLR if c in HCP_CHANNELS else DTC_CLR),
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
        plot_df["color"] = plot_df["change_pct"].apply(
            lambda v: UP_CLR if v > 5 else (DOWN_CLR if v < -5 else NEUT_CLR))
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
    freq, run_bayesian, run_insights, run_btn = sidebar(config)

    if run_btn:
        run_pipeline(freq, run_bayesian, run_insights)
        st.rerun()

    prefix = f"data/raw/mmm_{freq}"
    df    = load_dataset(freq)
    ols   = load_json(f"{prefix}_ols_results.json")
    bayes = load_json(f"{prefix}_bayesian_results.json")
    opt   = load_json(f"{prefix}_budget_optimized.json")

    st.title("💊 Pharma MMM Agent")
    st.caption(
        f"**{freq.capitalize()} dataset** — "
        + (f"{len(df)} periods | " if df is not None else "")
        + (f"OLS R²={ols['r_squared']:.3f} | " if ols else "")
        + (f"Bayesian R̂={bayes['mcmc'].get('max_rhat','—')} | " if bayes else "")
        + (f"+{opt['projected_uplift_pct']:.1f}% projected uplift" if opt else "")
    )

    tab_names = ["📈 Overview", "📊 Ridge MMM", "🧮 Bayesian MMM",
                 "💰 Budget", "📝 Insights"]
    tabs = st.tabs(tab_names)

    with tabs[0]:
        if df is None:
            st.info("Dataset not found. Run `python scripts/generate_dataset.py` first.")
        else:
            tab_overview(df, config)

    with tabs[1]:
        tab_ridge(ols, config)

    with tabs[2]:
        tab_bayesian(bayes, ols, config)

    with tabs[3]:
        tab_budget(opt, config)

    with tabs[4]:
        tab_insights(freq)


if __name__ == "__main__":
    main()
