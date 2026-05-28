"""Risk metrics: VaR/CVaR, component risk, diversification."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import risk as R
from data import benchmark_picker_and_data, portfolio_returns, require_portfolio
from theme import inject_css

inject_css()
st.title("Risk Metrics")
st.caption("Value-at-Risk, Conditional VaR, component risk contribution, diversification.")

tickers, weights, prices, returns, _, _, rf = require_portfolio()
bench_name, bench_prices, bench_returns = benchmark_picker_and_data()
port_ret = portfolio_returns(returns, weights)

st.sidebar.header("Settings")
alphas_str = st.sidebar.text_input("Confidence levels (alphas, comma-separated)", "0.01, 0.05, 0.10")
try:
    alphas = tuple(float(x) for x in alphas_str.split(","))
except Exception:
    alphas = (0.01, 0.05, 0.10)
notional = st.sidebar.number_input(
    "Portfolio notional ($)", 1000.0, 1e12,
    float(st.session_state.get("aum", 100000.0)),
    step=1000.0, help="Defaults to the fund AUM set on the Home page",
)
horizon = st.sidebar.number_input("VaR horizon (days)", 1, 252, 1)

# ─── VaR table ────────────────────────────────────────────────────────────────
st.header("Value at Risk")
st.caption("Negative numbers = loss thresholds. Multiply by notional for $ amounts.")

var_df = R.var_summary(port_ret, alphas=alphas)
# Add dollar columns
dollar_df = var_df.copy()
for col in dollar_df.columns:
    if col != "confidence":
        dollar_df[col] = dollar_df[col].apply(
            lambda v: f"{v:.3%} (${v*notional*np.sqrt(horizon):,.0f})" if pd.notna(v) else "—"
        )
st.dataframe(dollar_df, hide_index=True, width="stretch")

# Visualize
fig = go.Figure()
fig.add_histogram(x=port_ret, nbinsx=80, name="Daily returns",
                   marker_color="lightgray", opacity=0.7)
for a in alphas:
    v = R.var_historical(port_ret, a)
    fig.add_vline(x=v, line_dash="dash", annotation_text=f"VaR {1-a:.0%}",
                    annotation_position="top")
fig.update_layout(title="Daily return distribution with VaR thresholds",
                   xaxis_tickformat=".1%", showlegend=False)
st.plotly_chart(fig, width="stretch")

# ─── Risk contribution ────────────────────────────────────────────────────────
st.markdown("---")
st.header("Component risk contribution")
st.caption("How each asset's weight × correlation × vol contributes to total portfolio risk.")

rc = R.risk_contribution(returns, weights)
if not rc.empty:
    disp = rc.copy()
    disp["weight"] = disp["weight"].apply(lambda x: f"{x:.2%}")
    disp["marginal_vol_contribution"] = disp["marginal_vol_contribution"].apply(lambda x: f"{x:.4f}")
    disp["component_vol_contribution"] = disp["component_vol_contribution"].apply(lambda x: f"{x:.4f}")
    disp["pct_of_total_risk"] = disp["pct_of_total_risk"].apply(lambda x: f"{x:.2%}")
    st.dataframe(disp, width="stretch")

    bar_df = rc.reset_index().rename(columns={"index": "ticker"})
    fig = px.bar(bar_df, x="ticker", y=["weight", "pct_of_total_risk"], barmode="group",
                  title="Capital weight vs share of total portfolio risk")
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, width="stretch")

# ─── Component VaR ────────────────────────────────────────────────────────────
st.markdown("---")
st.header("Component VaR (95%)")
cvar_df = R.component_var(returns, weights, alpha=0.05)
if not cvar_df.empty:
    disp = cvar_df.copy()
    disp["weight"] = disp["weight"].apply(lambda x: f"{x:.2%}")
    disp["marginal_var"] = disp["marginal_var"].apply(lambda x: f"{x:.4f}")
    disp["component_var"] = disp["component_var"].apply(lambda x: f"{x:.4f}")
    disp["pct_of_var"] = disp["pct_of_var"].apply(lambda x: f"{x:.2%}")
    st.dataframe(disp, width="stretch")

# ─── Diversification metrics ──────────────────────────────────────────────────
st.markdown("---")
st.header("Diversification")
c1, c2, c3 = st.columns(3)
dr = R.diversification_ratio(returns, weights)
en = R.effective_n_assets(weights)
c1.metric("Diversification Ratio", f"{dr:.3f}", help="(weighted-avg asset vol) / (portfolio vol). >1 = some diversification benefit.")
c2.metric("Effective # Assets", f"{en:.2f}", help="1 / Herfindahl index of weights.")
c3.metric("Nominal # Assets", len(weights))
