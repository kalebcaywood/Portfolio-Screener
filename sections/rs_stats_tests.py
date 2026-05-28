"""Return Stream Analyzer — Statistical hypothesis tests + rolling diagnostics.

Mirrors the Portfolio Analyzer's Statistical Tests page but works on
frequency-aware return streams uploaded via RSA Home. Includes rolling
versions of every key test so you can see how distribution shape, normality,
autocorrelation, and volatility clustering evolve over time — useful for
style-drift and regime-change detection.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats as sps

import rsa as RS
import stats_tests as ST
from data import benchmark_picker_and_data
from theme import inject_css

inject_css()

st.title("Statistical Tests")
st.caption(
    "Distribution, stationarity, autocorrelation, heteroscedasticity, and "
    "rolling-window diagnostics for a single return stream."
)

# ─── Guard ───────────────────────────────────────────────────────────────────
if "rsa_returns" not in st.session_state:
    st.warning("No return streams loaded. Go to **Return Stream Analyzer → Home** first.")
    st.stop()

returns_df: pd.DataFrame = st.session_state["rsa_returns"]
freq: str = st.session_state.get("rsa_frequency", "M")
streams: list[str] = st.session_state["rsa_streams"]

st.sidebar.header("Test target")
target = st.sidebar.radio("Series to test", ["Stream", "Benchmark"], index=0)
stream = st.sidebar.selectbox("Stream", streams, key="rsa_stats_stream")

bench_name, _, bench_returns_daily = benchmark_picker_and_data()
_, bench_aligned = RS.align_to_period(returns_df[[stream]], bench_returns_daily, freq)

if target == "Benchmark":
    if bench_aligned.empty:
        st.warning(f"Benchmark {bench_name} could not be aligned to your stream's frequency.")
        st.stop()
    series = bench_aligned.dropna()
    label = bench_name
else:
    series = returns_df[stream].dropna()
    label = stream

st.markdown(f"### Testing **{label}** — n = {len(series)} {RS.FREQ_LABEL.get(freq, freq).lower()} observations")

# ─── Descriptive ─────────────────────────────────────────────────────────────
desc = ST.describe_distribution(series)
m = st.columns(6)
m[0].metric("Mean", f"{desc['mean']:+.4%}")
m[1].metric("Std", f"{desc['std']:.4%}")
m[2].metric("Skew", f"{desc['skew']:.3f}")
m[3].metric("Excess kurtosis", f"{desc['excess_kurtosis']:.3f}")
m[4].metric("Min", f"{desc['min']:.2%}")
m[5].metric("Max", f"{desc['max']:.2%}")

# ─── Tabs ────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "Normality", "Stationarity", "Autocorrelation", "Heteroscedasticity",
    "Random walk", "Two-sample", "Rolling diagnostics",
])

# Tab 1: Normality
with tabs[0]:
    st.subheader("Normality tests")
    st.caption(
        "Tests whether the return distribution is consistent with a normal "
        "distribution. Most fund returns are not — typically negatively skewed "
        "with fat tails (excess kurtosis > 0)."
    )
    st.dataframe(ST.normality_tests(series), hide_index=True, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(series, nbins=40, marginal="box",
                            title="Return histogram with normal overlay")
        x = np.linspace(series.min(), series.max(), 200)
        pdf = sps.norm.pdf(x, series.mean(), series.std())
        bin_counts = np.histogram(series.dropna(), bins=40)[0]
        scale = bin_counts.max() / pdf.max() if pdf.max() > 0 else 1
        fig.add_scatter(x=x, y=pdf * scale, mode="lines", name="Normal fit",
                          line=dict(color="#FF8200"))
        fig.update_xaxes(tickformat=".1%")
        st.plotly_chart(fig, width="stretch")
    with c2:
        qq = sps.probplot(series.dropna(), dist="norm")
        fig = go.Figure()
        fig.add_scatter(x=qq[0][0], y=qq[0][1], mode="markers", name="Empirical",
                          marker=dict(opacity=0.6))
        fig.add_scatter(x=qq[0][0], y=qq[1][0] * qq[0][0] + qq[1][1],
                          mode="lines", name="Normal fit", line=dict(color="#FF8200"))
        fig.update_layout(title="Q-Q plot vs normal", xaxis_title="Theoretical quantiles",
                            yaxis_title="Sample quantiles")
        st.plotly_chart(fig, width="stretch")

# Tab 2: Stationarity
with tabs[1]:
    st.subheader("Stationarity tests")
    st.caption("ADF: H₀ = unit root (non-stationary). KPSS: H₀ = stationary.")
    res = ST.stationarity_tests(series)
    if res.empty:
        st.info("statsmodels not available — install it to enable these tests.")
    else:
        st.dataframe(res, hide_index=True, width="stretch")

    st.markdown("**Compare: return level vs cumulative return** "
                  "(cumulative return is typically non-stationary)")
    cum = (1 + series).cumprod()
    st.dataframe(ST.stationarity_tests(cum), hide_index=True, width="stretch")

# Tab 3: Autocorrelation
with tabs[2]:
    lags = st.slider("Lag order", 1, 30, 10)
    st.subheader("Autocorrelation tests")
    st.dataframe(ST.autocorrelation_tests(series, lags=lags),
                   hide_index=True, width="stretch")

    try:
        from statsmodels.tsa.stattools import acf, pacf
        n_lags = min(40, len(series) // 4)
        acf_vals = acf(series.dropna(), nlags=n_lags, fft=True)
        pacf_vals = pacf(series.dropna(), nlags=n_lags)
        ci = 1.96 / np.sqrt(len(series))

        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            fig.add_bar(x=list(range(len(acf_vals))), y=acf_vals, name="ACF",
                          marker_color="#1e40af")
            fig.add_hline(y=ci, line_dash="dash", line_color="#b91c1c")
            fig.add_hline(y=-ci, line_dash="dash", line_color="#b91c1c")
            fig.update_layout(title="Autocorrelation function (ACF)")
            st.plotly_chart(fig, width="stretch")
        with c2:
            fig = go.Figure()
            fig.add_bar(x=list(range(len(pacf_vals))), y=pacf_vals, name="PACF",
                          marker_color="#0891b2")
            fig.add_hline(y=ci, line_dash="dash", line_color="#b91c1c")
            fig.add_hline(y=-ci, line_dash="dash", line_color="#b91c1c")
            fig.update_layout(title="Partial autocorrelation function (PACF)")
            st.plotly_chart(fig, width="stretch")
    except ImportError:
        pass

# Tab 4: Heteroscedasticity
with tabs[3]:
    arch_lags = st.slider("ARCH lag order", 1, 20, 5)
    st.subheader("Heteroscedasticity / volatility clustering")
    st.dataframe(ST.heteroscedasticity_tests(series, lags=arch_lags),
                   hide_index=True, width="stretch")

    vol_win = max(3, RS.periods_per_year(freq) // 2)
    vol_series = series.rolling(vol_win).std() * np.sqrt(RS.periods_per_year(freq))
    fig = px.line(vol_series,
                    title=f"Realized rolling {vol_win}-period vol (annualized)")
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width="stretch")

# Tab 5: Random walk
with tabs[4]:
    st.subheader("Random-walk / efficient-market tests")
    st.dataframe(ST.runs_test(series), hide_index=True, width="stretch")
    k_options = [2, 4, 8, 16]
    rows = [ST.variance_ratio_test(series, k=k) for k in k_options]
    rows = [r for r in rows if not r.empty]
    if rows:
        st.dataframe(pd.concat(rows, ignore_index=True),
                       hide_index=True, width="stretch")
    st.dataframe(ST.t_test_mean(series, mu0=0.0), hide_index=True, width="stretch")
    st.dataframe(ST.sign_test(series), hide_index=True, width="stretch")

# Tab 6: Two-sample
with tabs[5]:
    other_options = ["Benchmark"] + [s for s in streams if s != label]
    other = st.selectbox("Compare against", other_options)
    if other == "Benchmark":
        other_series = bench_aligned.dropna()
        other_label = bench_name
    else:
        other_series = returns_df[other].dropna()
        other_label = other

    if other_series.empty:
        st.warning(f"Comparison series '{other_label}' is empty or unavailable.")
    else:
        res = ST.two_sample_tests(series, other_series, label, other_label)
        st.dataframe(res, hide_index=True, width="stretch")
        df_plot = pd.DataFrame({label: series, other_label: other_series.reindex(series.index)})
        fig = px.histogram(df_plot.melt(var_name="series", value_name="ret"),
                            x="ret", color="series", nbins=40, barmode="overlay",
                            opacity=0.6, title="Distribution comparison")
        fig.update_xaxes(tickformat=".1%")
        st.plotly_chart(fig, width="stretch")

# Tab 7: Rolling diagnostics
with tabs[6]:
    st.subheader("Rolling test statistics")
    st.caption(
        "Watch distribution shape, normality, and autocorrelation evolve over "
        "time. Useful for detecting style drift, regime changes, and breaks in "
        "manager behavior."
    )
    ppy = RS.periods_per_year(freq)
    default_w = min(max(ppy, 12), len(series) // 2) if ppy > 1 else 12
    window = st.slider(
        "Rolling window (periods)", 6, max(12, len(series) // 2),
        default_w, step=1, key="rsa_rolling_window",
    )

    if len(series) < window + 6:
        st.warning(f"Need at least {window + 6} observations for window={window}. Series has {len(series)}.")
        st.stop()

    # Rolling moments
    roll_skew = series.rolling(window).skew()
    roll_kurt = series.rolling(window).kurt()

    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(roll_skew, title=f"Rolling {window}-period skewness")
        fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
        fig.update_layout(showlegend=False, height=320)
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig = px.line(roll_kurt, title=f"Rolling {window}-period excess kurtosis")
        fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
        fig.add_hline(y=3, line_dash="dot", line_color="#b91c1c",
                        annotation_text="kurt > 3 = fat tails")
        fig.update_layout(showlegend=False, height=320)
        st.plotly_chart(fig, width="stretch")

    # Rolling Jarque-Bera p-value (normality)
    def _rolling_jb_pval(s: pd.Series, w: int) -> pd.Series:
        out = pd.Series(index=s.index, dtype=float)
        vals = s.values
        for i in range(w, len(vals) + 1):
            window_data = vals[i - w:i]
            window_data = window_data[~np.isnan(window_data)]
            if len(window_data) >= 8:
                try:
                    _, p = sps.jarque_bera(window_data)
                    out.iloc[i - 1] = p
                except Exception:
                    out.iloc[i - 1] = np.nan
        return out

    # Rolling Ljung-Box p-value (autocorrelation regime)
    def _rolling_ljungbox_pval(s: pd.Series, w: int, lags: int = 5) -> pd.Series:
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
        except ImportError:
            return pd.Series(dtype=float)
        out = pd.Series(index=s.index, dtype=float)
        vals = s.values
        for i in range(w, len(vals) + 1):
            window_data = vals[i - w:i]
            window_data = window_data[~np.isnan(window_data)]
            if len(window_data) >= max(2 * lags, 8):
                try:
                    lb = acorr_ljungbox(window_data, lags=[lags], return_df=True)
                    out.iloc[i - 1] = float(lb["lb_pvalue"].iloc[0])
                except Exception:
                    out.iloc[i - 1] = np.nan
        return out

    with st.spinner("Computing rolling tests..."):
        jb_p = _rolling_jb_pval(series, window)
        lb_p = _rolling_ljungbox_pval(series, window, lags=5)
        roll_sharpe = (series.rolling(window).mean() * ppy) / (series.rolling(window).std() * np.sqrt(ppy))

    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(jb_p,
                        title=f"Rolling Jarque-Bera p-value (normality)",
                        labels={"value": "p-value", "index": "Date"})
        fig.add_hline(y=0.05, line_dash="dash", line_color="#b91c1c",
                        annotation_text="α = 0.05")
        fig.update_layout(showlegend=False, height=320,
                            yaxis=dict(range=[0, 1]))
        st.plotly_chart(fig, width="stretch")
    with c2:
        if not lb_p.empty:
            fig = px.line(lb_p,
                            title=f"Rolling Ljung-Box p-value (lag 5, autocorrelation)",
                            labels={"value": "p-value", "index": "Date"})
            fig.add_hline(y=0.05, line_dash="dash", line_color="#b91c1c",
                            annotation_text="α = 0.05")
            fig.update_layout(showlegend=False, height=320,
                                yaxis=dict(range=[0, 1]))
            st.plotly_chart(fig, width="stretch")

    # Rolling Sharpe + rolling beta (if benchmark available)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(roll_sharpe.dropna(),
                        title=f"Rolling {window}-period Sharpe")
        fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
        fig.update_layout(showlegend=False, height=320)
        st.plotly_chart(fig, width="stretch")
    with c2:
        if target == "Stream" and not bench_aligned.empty:
            common = series.index.intersection(bench_aligned.dropna().index)
            r_c = series.loc[common]
            b_c = bench_aligned.loc[common]
            cov = r_c.rolling(window).cov(b_c)
            var = b_c.rolling(window).var()
            roll_beta = (cov / var).dropna()
            fig = px.line(roll_beta,
                            title=f"Rolling {window}-period β vs {bench_name}")
            fig.add_hline(y=1, line_dash="dash", line_color="#94a3b8")
            fig.update_layout(showlegend=False, height=320)
            st.plotly_chart(fig, width="stretch")

    # Summary of rolling-test regime
    st.markdown("##### Rolling-test regime summary")
    if not jb_p.dropna().empty:
        jb_normal_pct = float((jb_p.dropna() > 0.05).mean())
        st.metric("Fraction of rolling windows where normality is NOT rejected",
                    f"{jb_normal_pct:.1%}",
                    help="If consistently below ~50%, returns are persistently non-normal — use Cornish-Fisher VaR.")
    if not lb_p.empty and not lb_p.dropna().empty:
        lb_no_autocorr_pct = float((lb_p.dropna() > 0.05).mean())
        st.metric("Fraction of rolling windows with NO significant autocorrelation",
                    f"{lb_no_autocorr_pct:.1%}",
                    help="Low values suggest persistent serial dependence — return-smoothing or stale pricing.")
