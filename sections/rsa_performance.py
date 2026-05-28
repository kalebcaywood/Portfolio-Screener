"""Portfolio performance analytics: returns, ratios, drawdowns, attribution."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import analytics as A
from data import portfolio_returns, require_portfolio
from theme import inject_css

inject_css()
st.title("Portfolio Performance")
st.caption("Risk-adjusted returns, drawdown analysis, and benchmark-relative metrics.")

tickers, weights, prices, returns, bench_prices, bench_returns, rf = require_portfolio()

port_ret = portfolio_returns(returns, weights)
port_cum = (1 + port_ret).cumprod()
bench_cum = (1 + bench_returns.reindex(port_ret.index).fillna(0)).cumprod()

# ─── Summary metrics ──────────────────────────────────────────────────────────
st.header("Summary statistics")
summary = A.summary_stats(port_ret, bench_returns.reindex(port_ret.index), rf)

def card(label, val, fmt_str="{:.2%}"):
    if val is None or pd.isna(val):
        return st.metric(label, "—")
    return st.metric(label, fmt_str.format(val))

r1 = st.columns(5)
with r1[0]: card("Total return", summary["total_return"])
with r1[1]: card("CAGR", summary["cagr"])
with r1[2]: card("Ann. return", summary["ann_return"])
with r1[3]: card("Ann. volatility", summary["ann_vol"])
with r1[4]: card("Max drawdown", summary["max_drawdown"])

r2 = st.columns(5)
with r2[0]: card("Sharpe", summary["sharpe"], "{:.2f}")
with r2[1]: card("Sortino", summary["sortino"], "{:.2f}")
with r2[2]: card("Calmar", summary["calmar"], "{:.2f}")
with r2[3]: card("Omega", summary["omega"], "{:.2f}")
with r2[4]: card("Ulcer Index", summary["ulcer_index"], "{:.4f}")

r3 = st.columns(5)
with r3[0]: card("Alpha (vs SPX)", summary.get("alpha"))
with r3[1]: card("Beta", summary.get("beta"), "{:.2f}")
with r3[2]: card("R²", summary.get("r_squared"), "{:.3f}")
with r3[3]: card("Info ratio", summary.get("info_ratio"), "{:.2f}")
with r3[4]: card("Tracking error", summary.get("tracking_error"))

r4 = st.columns(5)
with r4[0]: card("Treynor", summary.get("treynor"), "{:.4f}")
with r4[1]: card("M²", summary.get("m_squared"))
with r4[2]: card("Up capture", summary.get("up_capture"), "{:.2f}")
with r4[3]: card("Down capture", summary.get("down_capture"), "{:.2f}")
with r4[4]: card("% positive days", summary["pct_positive_days"])

r5 = st.columns(5)
with r5[0]: card("Skewness", summary["skew"], "{:.3f}")
with r5[1]: card("Excess kurtosis", summary["kurtosis"], "{:.3f}")
with r5[2]: card("Best day", summary["best_day"])
with r5[3]: card("Worst day", summary["worst_day"])
with r5[4]: card("Win/Loss ratio", summary.get("win_loss_ratio"), "{:.2f}")

st.markdown("---")

# ─── Charts ───────────────────────────────────────────────────────────────────
tabs = st.tabs(["Equity curve", "Drawdowns", "Rolling metrics", "Period returns", "Distribution"])

with tabs[0]:
    df = pd.DataFrame({"Portfolio": port_cum, "Benchmark (SPX)": bench_cum})
    fig = px.line(df, title="Cumulative growth of $1")
    st.plotly_chart(fig, width="stretch")

    # Excess return
    excess = port_cum / bench_cum - 1
    fig = px.line(excess, title="Cumulative excess return vs benchmark")
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width="stretch")

with tabs[1]:
    dd = A.drawdown_series(port_ret)
    fig = px.area(dd, title="Portfolio drawdown")
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width="stretch")

    st.subheader("Top drawdown episodes")
    ep = A.drawdown_episodes(port_ret, top_n=10)
    if not ep.empty:
        disp = ep.copy()
        disp["depth"] = disp["depth"].apply(lambda v: f"{v:.2%}")
        st.dataframe(disp, hide_index=True, width="stretch")

with tabs[2]:
    window = st.slider("Rolling window (days)", 21, 252, 63, step=21)
    rm = A.rolling_metrics(port_ret, window=window, rf=rf)
    rb = A.rolling_beta(port_ret, bench_returns, window=window)
    rc = A.rolling_correlation(port_ret, bench_returns, window=window)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(rm[["rolling_return", "rolling_vol"]],
                       title=f"Rolling annualized return & vol ({window}d)")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, width="stretch")
        fig = px.line(rm["rolling_sharpe"], title=f"Rolling Sharpe ({window}d)")
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig = px.line(rb, title=f"Rolling Beta vs SPX ({window}d)")
        st.plotly_chart(fig, width="stretch")
        fig = px.line(rc, title=f"Rolling correlation with SPX ({window}d)")
        st.plotly_chart(fig, width="stretch")

with tabs[3]:
    monthly = A.monthly_returns(port_ret)
    annual = A.annual_returns(port_ret)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Annual returns")
        ann_df = pd.DataFrame({"year": annual.index.year, "return": annual.values})
        fig = px.bar(ann_df, x="year", y="return", color="return",
                      color_continuous_scale="RdYlGn", color_continuous_midpoint=0)
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.subheader("Monthly heatmap")
        hm = A.monthly_heatmap(port_ret)
        fig = px.imshow(hm, text_auto=".1%", color_continuous_scale="RdYlGn",
                         color_continuous_midpoint=0, aspect="auto",
                         labels={"x": "Month", "y": "Year"})
        st.plotly_chart(fig, width="stretch")

with tabs[4]:
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(port_ret, nbins=60, title="Daily return distribution",
                            marginal="box")
        fig.update_xaxes(tickformat=".1%")
        st.plotly_chart(fig, width="stretch")
    with c2:
        # QQ plot
        from scipy import stats as sps
        qq = sps.probplot(port_ret.dropna(), dist="norm")
        x = qq[0][0]
        y = qq[0][1]
        slope, intercept = qq[1][0], qq[1][1]
        fig = go.Figure()
        fig.add_scatter(x=x, y=y, mode="markers", name="Empirical")
        fig.add_scatter(x=x, y=slope * x + intercept, mode="lines", name="Normal fit")
        fig.update_layout(title="Q-Q plot vs normal", xaxis_title="Theoretical quantiles",
                            yaxis_title="Sample quantiles")
        st.plotly_chart(fig, width="stretch")
