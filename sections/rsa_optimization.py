"""Mean-variance optimization, efficient frontier, risk parity."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import optimization as O
from data import require_portfolio
from theme import inject_css, ut_sidebar_brand

inject_css()
ut_sidebar_brand()
st.title("Portfolio Optimization")
st.caption("Modern Portfolio Theory: efficient frontier, max-Sharpe, min-variance, Equal Risk Contribution.")

tickers, weights, prices, returns, bench_prices, bench_returns, rf = require_portfolio()

st.sidebar.header("Optimization settings")
allow_short = st.sidebar.checkbox("Allow short positions", value=False)
n_frontier = st.sidebar.slider("Frontier points", 10, 80, 40, step=5)

# ─── Strategy comparison ──────────────────────────────────────────────────────
with st.spinner("Optimizing across strategies..."):
    strategies = O.all_strategies(returns, rf=rf, allow_short=allow_short)
    frontier = O.efficient_frontier(returns, n_points=n_frontier, rf=rf, allow_short=allow_short)

st.header("Strategy comparison")
rows = []
weight_dict = {}
for name, res in strategies.items():
    rows.append({
        "strategy": name,
        "annual_return": res["return"],
        "annual_vol": res["vol"],
        "sharpe": res["sharpe"],
    })
    weight_dict[name] = res["weights"]
comp_df = pd.DataFrame(rows)
disp = comp_df.copy()
disp["annual_return"] = disp["annual_return"].apply(lambda x: f"{x:.2%}")
disp["annual_vol"] = disp["annual_vol"].apply(lambda x: f"{x:.2%}")
disp["sharpe"] = disp["sharpe"].apply(lambda x: f"{x:.3f}")
st.dataframe(disp, width="stretch", hide_index=True)

# ─── Weight comparison ────────────────────────────────────────────────────────
st.header("Weights by strategy")
w_df = pd.DataFrame(weight_dict)
w_df.index.name = "ticker"
plot_data = w_df.reset_index().melt(id_vars="ticker", var_name="strategy", value_name="weight")
fig = px.bar(plot_data, x="ticker", y="weight", color="strategy", barmode="group",
              title="Portfolio weights across strategies")
fig.update_yaxes(tickformat=".0%")
st.plotly_chart(fig, width="stretch")

# Tabular form
disp_w = w_df.copy()
for c in disp_w.columns:
    disp_w[c] = disp_w[c].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "—")
st.dataframe(disp_w, width="stretch")

# ─── Efficient frontier ───────────────────────────────────────────────────────
st.markdown("---")
st.header("Efficient frontier")
if not frontier.empty:
    fig = go.Figure()
    fig.add_scatter(x=frontier["vol"], y=frontier["return"], mode="lines+markers",
                      name="Efficient frontier", line=dict(color="blue"))

    # Mark strategy portfolios
    color_map = {"Max Sharpe": "red", "Min Variance": "green", "Risk Parity (ERC)": "purple",
                  "Equal Weight": "orange", "Inverse Vol": "brown"}
    for name, res in strategies.items():
        fig.add_scatter(x=[res["vol"]], y=[res["return"]], mode="markers+text",
                          marker=dict(size=14, color=color_map.get(name, "black"), symbol="star"),
                          text=[name], textposition="top center", name=name)

    # Individual assets
    mean = returns.mean() * 252
    vol = returns.std() * np.sqrt(252)
    fig.add_scatter(x=vol, y=mean, mode="markers+text",
                      marker=dict(size=8, color="lightgray"),
                      text=returns.columns, textposition="bottom center",
                      name="Individual assets")

    # Capital Market Line from Max Sharpe
    max_sr = strategies["Max Sharpe"]
    if max_sr["vol"] > 0:
        x_cml = np.linspace(0, vol.max() * 1.2, 50)
        y_cml = rf + (max_sr["return"] - rf) / max_sr["vol"] * x_cml
        fig.add_scatter(x=x_cml, y=y_cml, mode="lines", line=dict(dash="dash", color="gray"),
                          name="Capital Market Line")

    fig.update_layout(title="Efficient frontier with strategy portfolios",
                       xaxis_title="Annual volatility", yaxis_title="Annual return",
                       xaxis_tickformat=".0%", yaxis_tickformat=".0%", height=600)
    st.plotly_chart(fig, width="stretch")

# ─── Custom target return ─────────────────────────────────────────────────────
st.markdown("---")
st.header("Target-return optimizer")
target_pct = st.slider("Target annual return", -0.10, 0.50, 0.10, step=0.01, format="%.2f")
res = O.target_return(returns, target_pct, rf=rf, allow_short=allow_short)
c = st.columns(4)
c[0].metric("Target return", f"{target_pct:.2%}")
c[1].metric("Achieved return", f"{res['return']:.2%}")
c[2].metric("Volatility", f"{res['vol']:.2%}")
c[3].metric("Sharpe", f"{res['sharpe']:.3f}" if not pd.isna(res['sharpe']) else "—")

w = res["weights"]
fig = px.pie(values=w[w > 0.001].values, names=w[w > 0.001].index, hole=0.4,
              title="Optimal weights for target return")
st.plotly_chart(fig, width="stretch")
