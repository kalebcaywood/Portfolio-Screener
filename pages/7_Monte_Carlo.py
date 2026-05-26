"""Monte Carlo simulation of portfolio forward paths."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import monte_carlo as MC
from data import require_portfolio
from theme import inject_css

st.set_page_config(page_title="Monte Carlo", layout="wide")
inject_css()
st.title("Monte Carlo Simulation")
st.caption("Project portfolio paths via multivariate-normal or historical bootstrap simulation.")

tickers, weights, prices, returns, bench_prices, bench_returns, rf = require_portfolio()

st.sidebar.header("Simulation settings")
method = st.sidebar.radio("Method",
                           ["Multivariate normal", "Historical bootstrap",
                            "Block bootstrap (preserves serial dep.)"], index=0)
horizon = st.sidebar.slider("Horizon (trading days)", 21, 1260, 252, step=21,
                              help="252 = 1 year, 1260 = 5 years")
n_sims = st.sidebar.slider("Number of simulations", 500, 20000, 5000, step=500)
initial = st.sidebar.number_input(
    "Initial portfolio value ($)", 1000.0, 1e12,
    float(st.session_state.get("aum", 100000.0)),
    step=1000.0, help="Defaults to fund AUM from the Home page",
)
block_size = 5
if method.startswith("Block"):
    block_size = st.sidebar.slider("Block size (days)", 2, 30, 5)

run = st.sidebar.button("Run simulation", type="primary", width="stretch")

if run:
    with st.spinner(f"Running {n_sims:,} simulations over {horizon} days..."):
        if method == "Multivariate normal":
            paths = MC.simulate_normal(returns, weights, horizon, n_sims, initial)
        elif method == "Historical bootstrap":
            paths = MC.simulate_bootstrap(returns, weights, horizon, n_sims, initial, block_size=1)
        else:
            paths = MC.simulate_bootstrap(returns, weights, horizon, n_sims, initial, block_size=block_size)

    st.session_state["mc_paths"] = paths
    st.session_state["mc_initial"] = initial

if "mc_paths" not in st.session_state:
    st.info("Configure simulation settings in the sidebar and click **Run simulation**.")
    st.stop()

paths = st.session_state["mc_paths"]
initial = st.session_state["mc_initial"]
summary = MC.summarize_paths(paths, initial)

# ─── Summary cards ────────────────────────────────────────────────────────────
st.header("Simulation results")
c = st.columns(5)
c[0].metric("Expected terminal", f"${summary['expected_terminal']:,.0f}",
              f"{(summary['expected_terminal']/initial - 1):.1%}")
c[1].metric("Median terminal", f"${summary['median_terminal']:,.0f}",
              f"{(summary['median_terminal']/initial - 1):.1%}")
c[2].metric("Prob. of loss", f"{summary['prob_loss']:.1%}")
c[3].metric("VaR 95% (return)", f"{summary['var_95']:.1%}")
c[4].metric("CVaR 95%", f"{summary['cvar_95']:.1%}")

c2 = st.columns(5)
c2[0].metric("Prob. > 10% loss", f"{summary['prob_10pct_loss']:.1%}")
c2[1].metric("Prob. > 25% loss", f"{summary['prob_25pct_loss']:.1%}")
c2[2].metric("Prob. > 50% loss", f"{summary['prob_50pct_loss']:.1%}")
c2[3].metric("Prob. doubling", f"{summary['prob_double']:.1%}")
c2[4].metric("Worst-case path", f"${summary['min_terminal']:,.0f}")

# ─── Fan chart ────────────────────────────────────────────────────────────────
st.subheader("Fan chart of percentile paths")
pct = MC.percentile_paths(paths)
days = np.arange(paths.shape[1])
fig = go.Figure()
fig.add_scatter(x=days, y=pct[95], mode="lines", line=dict(width=0), showlegend=False)
fig.add_scatter(x=days, y=pct[5], mode="lines", fill="tonexty",
                  line=dict(width=0), fillcolor="rgba(0,100,255,0.1)", name="5th–95th pctile")
fig.add_scatter(x=days, y=pct[75], mode="lines", line=dict(width=0), showlegend=False)
fig.add_scatter(x=days, y=pct[25], mode="lines", fill="tonexty",
                  line=dict(width=0), fillcolor="rgba(0,100,255,0.25)", name="25th–75th pctile")
fig.add_scatter(x=days, y=pct[50], mode="lines", line=dict(color="blue", width=2),
                  name="Median")
fig.update_layout(title=f"Portfolio value paths (n={paths.shape[0]:,})",
                   xaxis_title="Days from now", yaxis_title="Portfolio value ($)",
                   height=500)
st.plotly_chart(fig, width="stretch")

# ─── Terminal distribution ────────────────────────────────────────────────────
st.subheader("Terminal value distribution")
terminal = paths[:, -1]
fig = px.histogram(terminal, nbins=80, title="Distribution of terminal portfolio value")
fig.add_vline(x=initial, line_dash="dash", line_color="red",
                annotation_text="Initial value")
fig.add_vline(x=summary["median_terminal"], line_dash="dot", line_color="green",
                annotation_text="Median")
fig.update_layout(showlegend=False, xaxis_title="Terminal value ($)")
st.plotly_chart(fig, width="stretch")

# ─── Path-level max drawdown ──────────────────────────────────────────────────
st.subheader("Per-path maximum drawdown distribution")
dds = MC.path_max_drawdowns(paths)
fig = px.histogram(dds, nbins=60, title="Worst drawdown over horizon (across all sims)")
fig.update_xaxes(tickformat=".0%")
fig.update_layout(showlegend=False)
st.plotly_chart(fig, width="stretch")

c1, c2, c3 = st.columns(3)
c1.metric("Median max drawdown", f"{np.median(dds):.1%}")
c2.metric("95th pctile max drawdown", f"{np.percentile(dds, 5):.1%}")
c3.metric("Worst max drawdown", f"{dds.min():.1%}")

# ─── Percentile table ─────────────────────────────────────────────────────────
st.subheader("Terminal value percentiles")
pct_df = pd.DataFrame([{
    "5th": f"${summary['p5']:,.0f}",
    "25th": f"${summary['p25']:,.0f}",
    "50th": f"${summary['p50']:,.0f}",
    "75th": f"${summary['p75']:,.0f}",
    "95th": f"${summary['p95']:,.0f}",
}])
st.dataframe(pct_df, hide_index=True, width="stretch")
