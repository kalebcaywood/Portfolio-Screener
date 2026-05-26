"""Stress test the portfolio against historical crisis episodes and custom shocks."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import risk as R
from data import require_portfolio
from theme import inject_css

st.set_page_config(page_title="Stress Tests", layout="wide")
inject_css()
st.title("Stress Tests")
st.caption("Replay historical crises and apply custom return shocks to gauge portfolio resilience.")

tickers, weights, prices, returns, bench_prices, bench_returns, rf = require_portfolio()

tab_hist, tab_custom, tab_factor = st.tabs(["Historical scenarios", "Custom asset shocks", "Beta-driven market shock"])

# ─── Historical replay ───────────────────────────────────────────────────────
with tab_hist:
    st.subheader("Historical crisis windows")
    st.caption("Computes portfolio return over each window using current weights and asset prices that existed at that time.")
    stress = R.stress_test_historical(prices, weights)
    if stress.empty:
        st.warning("No price overlap with stress scenarios. Use longer lookback on the Home page.")
    else:
        disp = stress.copy()
        disp["portfolio_return"] = disp["portfolio_return"].apply(
            lambda v: f"{v:.2%}" if pd.notna(v) else "—"
        )
        disp["worst_asset_return"] = disp["worst_asset_return"].apply(
            lambda v: f"{v:.2%}" if pd.notna(v) else "—"
        )
        disp["best_asset_return"] = disp["best_asset_return"].apply(
            lambda v: f"{v:.2%}" if pd.notna(v) else "—"
        )
        st.dataframe(disp, hide_index=True, width="stretch")

        plot_df = stress.dropna(subset=["portfolio_return"])
        if not plot_df.empty:
            fig = px.bar(plot_df, x="scenario", y="portfolio_return",
                           color="portfolio_return", color_continuous_scale="RdYlGn",
                           color_continuous_midpoint=0,
                           title="Portfolio return under historical stress events")
            fig.update_yaxes(tickformat=".0%")
            fig.update_xaxes(tickangle=-30)
            st.plotly_chart(fig, width="stretch")

# ─── Custom shock ────────────────────────────────────────────────────────────
with tab_custom:
    st.subheader("Apply custom return shocks per asset")
    st.caption("Enter a hypothetical return shock (e.g., -0.20 = -20%) for each asset. Portfolio impact is the weighted sum.")

    shocks = {}
    cols = st.columns(min(len(tickers), 5))
    for i, t in enumerate(tickers):
        with cols[i % len(cols)]:
            shocks[t] = st.number_input(f"{t}", -1.0, 5.0, 0.0, step=0.01, format="%.3f", key=f"shock_{t}")

    res = R.custom_shock_test(returns, weights, shocks)
    st.metric("Portfolio impact", f"{res['portfolio_shock']:.2%}")

    contrib_df = pd.DataFrame([
        {"ticker": t, "weight": float(weights.get(t, 0)), "shock": s, "contribution": res["contributions"][t]}
        for t, s in shocks.items()
    ])
    fig = px.bar(contrib_df, x="ticker", y="contribution",
                   color="contribution", color_continuous_scale="RdYlGn",
                   color_continuous_midpoint=0,
                   title="Per-asset contribution to portfolio impact")
    fig.update_yaxes(tickformat=".1%")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(contrib_df.assign(
        weight=lambda d: d["weight"].apply(lambda x: f"{x:.2%}"),
        shock=lambda d: d["shock"].apply(lambda x: f"{x:.2%}"),
        contribution=lambda d: d["contribution"].apply(lambda x: f"{x:.3%}"),
    ), hide_index=True, width="stretch")

# ─── Beta-driven market shock ────────────────────────────────────────────────
with tab_factor:
    st.subheader("Market shock propagated by beta")
    st.caption("Shock the benchmark by X%; each asset's implied shock = β × market shock.")
    market_shock = st.slider("Market shock", -0.50, 0.50, -0.20, step=0.01, format="%.2f")

    res = R.factor_shock_test(returns, weights, bench_returns, market_shock)
    st.metric("Implied portfolio impact", f"{res['portfolio_shock']:.2%}")

    betas_df = pd.DataFrame([
        {"ticker": t, "beta": b, "implied_shock": b * market_shock,
          "weight": float(weights.get(t, 0)),
          "contribution": res["contributions"][t]}
        for t, b in res["betas"].items()
    ])
    disp = betas_df.copy()
    disp["beta"] = disp["beta"].apply(lambda x: f"{x:.2f}")
    disp["implied_shock"] = disp["implied_shock"].apply(lambda x: f"{x:.2%}")
    disp["weight"] = disp["weight"].apply(lambda x: f"{x:.2%}")
    disp["contribution"] = disp["contribution"].apply(lambda x: f"{x:.3%}")
    st.dataframe(disp, hide_index=True, width="stretch")

    fig = px.bar(betas_df.sort_values("contribution"), x="ticker", y="contribution",
                   color="beta", color_continuous_scale="Viridis",
                   title="Asset-level contribution to market-shock impact")
    fig.update_yaxes(tickformat=".1%")
    st.plotly_chart(fig, width="stretch")
