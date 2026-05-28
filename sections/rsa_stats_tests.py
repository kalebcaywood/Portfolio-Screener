"""Statistical hypothesis tests for return series."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats as sps

import stats_tests as ST
from data import portfolio_returns, require_portfolio
from theme import inject_css, ut_sidebar_brand

inject_css()
ut_sidebar_brand()
st.title("Statistical Tests")
st.caption("Distribution, stationarity, autocorrelation, heteroscedasticity, and random-walk diagnostics.")

tickers, weights, prices, returns, bench_prices, bench_returns, rf = require_portfolio()
port_ret = portfolio_returns(returns, weights)

st.sidebar.header("Test target")
target = st.sidebar.radio("Series to test", ["Portfolio", "Individual asset", "Benchmark"], index=0)
if target == "Individual asset":
    sym = st.sidebar.selectbox("Asset", tickers)
    series = returns[sym]
    label = sym
elif target == "Benchmark":
    series = bench_returns
    label = "SPX"
else:
    series = port_ret
    label = "Portfolio"

series = series.dropna()
st.subheader(f"Testing: **{label}** — n = {len(series)} observations")

# ─── Descriptive ──────────────────────────────────────────────────────────────
desc = ST.describe_distribution(series)
c = st.columns(6)
c[0].metric("Mean (daily)", f"{desc['mean']:.4%}")
c[1].metric("Std", f"{desc['std']:.4%}")
c[2].metric("Skew", f"{desc['skew']:.3f}")
c[3].metric("Excess kurt", f"{desc['excess_kurtosis']:.3f}")
c[4].metric("Min", f"{desc['min']:.2%}")
c[5].metric("Max", f"{desc['max']:.2%}")

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tabs = st.tabs(["Normality", "Stationarity", "Autocorrelation",
                 "Heteroscedasticity", "Random walk", "Two-sample"])

with tabs[0]:
    st.subheader("Normality tests")
    st.dataframe(ST.normality_tests(series), width="stretch", hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(series, nbins=80, marginal="box", title="Return histogram")
        # Overlay normal
        x = np.linspace(series.min(), series.max(), 200)
        pdf = sps.norm.pdf(x, series.mean(), series.std())
        # Normalize to histogram count
        bins = np.histogram(series.dropna(), bins=80)[0]
        scale = bins.max() / pdf.max() if pdf.max() > 0 else 1
        fig.add_scatter(x=x, y=pdf * scale, mode="lines", name="Normal fit",
                         line=dict(color="red"))
        fig.update_xaxes(tickformat=".1%")
        st.plotly_chart(fig, width="stretch")
    with c2:
        qq = sps.probplot(series.dropna(), dist="norm")
        fig = go.Figure()
        fig.add_scatter(x=qq[0][0], y=qq[0][1], mode="markers", name="Empirical")
        fig.add_scatter(x=qq[0][0], y=qq[1][0] * qq[0][0] + qq[1][1],
                         mode="lines", name="Normal fit", line=dict(color="red"))
        fig.update_layout(title="Q-Q plot vs normal",
                            xaxis_title="Theoretical quantiles",
                            yaxis_title="Sample quantiles")
        st.plotly_chart(fig, width="stretch")

with tabs[1]:
    st.subheader("Stationarity tests")
    st.caption("ADF: H₀ = unit root (non-stationary). KPSS: H₀ = stationary.")
    res = ST.stationarity_tests(series)
    if res.empty:
        st.info("statsmodels not available — install it to enable these tests.")
    else:
        st.dataframe(res, width="stretch", hide_index=True)

    # Also test the price level / equity curve (typically non-stationary)
    st.markdown("**Compare: price level vs returns** (price levels typically fail ADF)")
    if target == "Portfolio":
        price_test = (1 + port_ret).cumprod()
    elif target == "Individual asset":
        price_test = prices[sym]
    else:
        price_test = bench_prices
    st.dataframe(ST.stationarity_tests(price_test), width="stretch", hide_index=True)

with tabs[2]:
    lags = st.slider("Lag order", 1, 30, 10)
    st.subheader("Autocorrelation & serial dependence")
    st.dataframe(ST.autocorrelation_tests(series, lags=lags),
                  width="stretch", hide_index=True)

    # ACF/PACF plots via statsmodels
    try:
        from statsmodels.tsa.stattools import acf, pacf
        n_lags = min(40, len(series) // 4)
        acf_vals = acf(series.dropna(), nlags=n_lags, fft=True)
        pacf_vals = pacf(series.dropna(), nlags=n_lags)
        ci = 1.96 / np.sqrt(len(series))

        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            fig.add_bar(x=list(range(len(acf_vals))), y=acf_vals, name="ACF")
            fig.add_hline(y=ci, line_dash="dash", line_color="red")
            fig.add_hline(y=-ci, line_dash="dash", line_color="red")
            fig.update_layout(title="Autocorrelation function")
            st.plotly_chart(fig, width="stretch")
        with c2:
            fig = go.Figure()
            fig.add_bar(x=list(range(len(pacf_vals))), y=pacf_vals, name="PACF")
            fig.add_hline(y=ci, line_dash="dash", line_color="red")
            fig.add_hline(y=-ci, line_dash="dash", line_color="red")
            fig.update_layout(title="Partial autocorrelation function")
            st.plotly_chart(fig, width="stretch")
    except ImportError:
        pass

with tabs[3]:
    arch_lags = st.slider("ARCH lag order", 1, 20, 5)
    st.subheader("Heteroscedasticity / Volatility clustering")
    st.dataframe(ST.heteroscedasticity_tests(series, lags=arch_lags),
                  width="stretch", hide_index=True)

    # Volatility series
    vol = series.rolling(21).std() * np.sqrt(252)
    fig = px.line(vol, title="Realized rolling 21-day volatility (annualized)")
    fig.update_layout(showlegend=False)
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, width="stretch")

with tabs[4]:
    st.subheader("Random walk / efficient market tests")
    st.dataframe(ST.runs_test(series), width="stretch", hide_index=True)

    k_options = [2, 4, 8, 16]
    rows = [ST.variance_ratio_test(series, k=k) for k in k_options]
    rows = [r for r in rows if not r.empty]
    if rows:
        st.dataframe(pd.concat(rows, ignore_index=True),
                      width="stretch", hide_index=True)

    st.markdown("**Mean tests**")
    st.dataframe(ST.t_test_mean(series, mu0=0.0), width="stretch", hide_index=True)
    st.dataframe(ST.sign_test(series), width="stretch", hide_index=True)

with tabs[5]:
    st.subheader("Two-sample comparison")
    other = st.selectbox("Compare against",
                          ["Benchmark"] + [t for t in tickers if t != label])
    other_series = bench_returns if other == "Benchmark" else returns[other]
    res = ST.two_sample_tests(series, other_series.dropna(), label, other)
    st.dataframe(res, width="stretch", hide_index=True)

    df = pd.DataFrame({label: series, other: other_series.reindex(series.index)})
    fig = px.histogram(df.melt(var_name="series", value_name="ret"), x="ret", color="series",
                        nbins=80, barmode="overlay", opacity=0.6,
                        title="Distribution comparison")
    fig.update_xaxes(tickformat=".1%")
    st.plotly_chart(fig, width="stretch")
