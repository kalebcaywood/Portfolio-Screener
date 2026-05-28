"""Return Stream Analyzer — CAPM and multi-factor regression.

Single-factor CAPM against the picked benchmark, plus multi-factor regression
against the FACTOR_PROXIES ETF catalog (market, value, growth, momentum,
min-vol, quality, intl, EM, bonds, gold, USD). Frequency-aware: daily
benchmark/factor returns are compounded to the stream's frequency.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import factor_models as FM
import rsa as RS
from data import FACTOR_PROXIES, benchmark_picker_and_data, fetch_prices
from theme import inject_css

inject_css()

st.title("Factor Models")
st.caption("CAPM and custom multi-factor OLS for return streams. Frequency-aligned to your data.")

if "rsa_returns" not in st.session_state:
    st.warning("No return streams loaded. Go to **Return Stream Analyzer → Home** first.")
    st.stop()

returns_df: pd.DataFrame = st.session_state["rsa_returns"]
freq: str = st.session_state.get("rsa_frequency", "M")
streams: list[str] = st.session_state["rsa_streams"]
rf: float = float(st.session_state.get("rsa_rf", 0.04))

st.sidebar.header("Regression target")
stream = st.sidebar.selectbox("Stream", streams, key="rsa_factor_stream")
bench_name, _, bench_returns_daily = benchmark_picker_and_data()

r = returns_df[stream].dropna()
_, bench_aligned = RS.align_to_period(returns_df[[stream]], bench_returns_daily, freq)
bench_aligned = bench_aligned.dropna()

if r.empty or bench_aligned.empty:
    st.warning("Stream or benchmark series is empty / un-alignable.")
    st.stop()

st.markdown(f"### {stream}  vs  {bench_name}")
st.caption(f"{len(r):,} observations  ·  {RS.FREQ_LABEL.get(freq, freq).lower()} frequency")

tabs = st.tabs(["CAPM", "Multi-factor", "Rolling alpha / beta"])

# Tab 1: CAPM
with tabs[0]:
    # Use the same regression mechanics as the Portfolio Analyzer's factor_models.capm
    # but with frequency-aware annualization.
    common = r.index.intersection(bench_aligned.index)
    if len(common) < 6:
        st.error("Need at least 6 overlapping observations.")
        st.stop()
    df_reg = pd.DataFrame({"r": r.loc[common], "m": bench_aligned.loc[common]})

    ppy = RS.periods_per_year(freq)
    rf_per_period = rf / ppy
    y = df_reg["r"] - rf_per_period
    x = df_reg["m"] - rf_per_period

    # OLS with full statistics
    try:
        import statsmodels.api as sm
        Xc = sm.add_constant(x)
        model = sm.OLS(y, Xc).fit()
        alpha_period = float(model.params["const"])
        alpha_ann = alpha_period * ppy
        alpha_tstat = float(model.tvalues["const"])
        alpha_p = float(model.pvalues["const"])
        beta = float(model.params["m"])
        beta_tstat = float(model.tvalues["m"])
        beta_p = float(model.pvalues["m"])
        r_sq = float(model.rsquared)
        r_sq_adj = float(model.rsquared_adj)
        n_obs = int(model.nobs)
        resid_std = float(model.resid.std())
        dw = float(sm.stats.stattools.durbin_watson(model.resid))
        ci = model.conf_int()
        beta_ci = (float(ci.loc["m", 0]), float(ci.loc["m", 1]))
        residuals = model.resid
        fitted = model.fittedvalues
    except ImportError:
        from scipy import stats as sps
        slope, intercept, r_val, p_val, std_err = sps.linregress(x, y)
        alpha_period = float(intercept); alpha_ann = alpha_period * ppy
        beta = float(slope); r_sq = float(r_val ** 2)
        alpha_tstat = alpha_p = beta_tstat = beta_p = np.nan
        r_sq_adj = resid_std = dw = np.nan
        n_obs = len(x); beta_ci = (np.nan, np.nan)
        residuals = pd.Series(y.values - (intercept + slope * x.values), index=x.index)
        fitted = intercept + slope * x

    c = st.columns(5)
    c[0].metric("Alpha (annual)", f"{alpha_ann:+.2%}",
                  help=f"t = {alpha_tstat:.2f}, p = {alpha_p:.3f}")
    c[1].metric("Beta", f"{beta:.3f}",
                  help=f"t = {beta_tstat:.2f}, 95% CI [{beta_ci[0]:.2f}, {beta_ci[1]:.2f}]")
    c[2].metric("R²", f"{r_sq:.3f}")
    c[3].metric("Adj R²", f"{r_sq_adj:.3f}" if not np.isnan(r_sq_adj) else "—")
    c[4].metric("N obs", f"{n_obs}")

    c2 = st.columns(4)
    c2[0].metric("Residual σ (per period)",
                   f"{resid_std:.4f}" if not np.isnan(resid_std) else "—")
    c2[1].metric("Durbin-Watson",
                   f"{dw:.2f}" if not np.isnan(dw) else "—",
                   help="≈2 = no residual autocorrelation. <1.5 or >2.5 = check serial dependence.")
    c2[2].metric("Alpha t-stat",
                   f"{alpha_tstat:.2f}" if not np.isnan(alpha_tstat) else "—",
                   help="> 2 ≈ statistically significant outperformance")
    c2[3].metric("Alpha p-value",
                   f"{alpha_p:.4f}" if not np.isnan(alpha_p) else "—")

    # Scatter
    fit_x = np.linspace(x.min(), x.max(), 100)
    fit_y = alpha_period + beta * fit_x
    fig = go.Figure()
    fig.add_scatter(x=x, y=y, mode="markers",
                      marker=dict(size=7, opacity=0.6),
                      name=f"{stream} excess returns")
    fig.add_scatter(x=fit_x, y=fit_y, mode="lines",
                      line=dict(color="#FF8200", width=2),
                      name=f"OLS: y = {alpha_period:+.4f} + {beta:.3f}·x")
    fig.update_layout(
        title=f"CAPM regression — {stream} vs {bench_name}",
        xaxis_title=f"{bench_name} excess return",
        yaxis_title=f"{stream} excess return",
        xaxis_tickformat=".1%", yaxis_tickformat=".1%",
        height=480,
    )
    st.plotly_chart(fig, width="stretch")

    # Residual diagnostics
    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(residuals, title="Residuals over time")
        fig.update_layout(showlegend=False, height=320)
        fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig = px.histogram(residuals, nbins=30, title="Residual distribution")
        fig.update_layout(showlegend=False, height=320)
        st.plotly_chart(fig, width="stretch")

# Tab 2: Multi-factor
with tabs[1]:
    st.subheader("Custom multi-factor regression")
    st.caption("Pick ETF proxies. Each factor's daily returns are compounded to your stream's frequency before regression.")
    chosen = st.multiselect(
        "Factor proxies",
        list(FACTOR_PROXIES.keys()),
        default=["Market (SPY)", "Value (IWD large value)", "Growth (IWF large growth)",
                  "Momentum (MTUM)", "Min-Vol / Low-Risk (USMV)", "Quality (QUAL)"],
    )
    if chosen and st.button("Run multi-factor regression", type="primary"):
        seen, factor_tickers = set(), {}
        for name in chosen:
            t = FACTOR_PROXIES[name]
            if t not in seen:
                seen.add(t)
                factor_tickers[name] = t
        with st.spinner("Fetching factor returns..."):
            fp = fetch_prices(tuple(factor_tickers.values()), period="10y")
        if fp.empty:
            st.error("Could not fetch factor data.")
        else:
            # Align factors to the stream's frequency
            fp_daily_returns = fp.pct_change().dropna(how="all")
            # Compound to the stream's frequency
            period_rule = {"D": None, "W": "W", "M": "ME", "Q": "QE", "A": "YE"}.get(freq)
            if period_rule:
                factor_returns_aligned = (1 + fp_daily_returns).resample(period_rule).prod() - 1
            else:
                factor_returns_aligned = fp_daily_returns
            # Match indices to the stream
            factor_returns_aligned = factor_returns_aligned.reindex(
                r.index, method="nearest", tolerance=pd.Timedelta(days=15)
            ).dropna(how="all")
            # Rename to friendly labels
            label_map = {t: name for name, t in factor_tickers.items()}
            factor_returns_aligned = factor_returns_aligned.rename(columns=label_map)
            available = [c for c in factor_returns_aligned.columns if c in label_map.values()]

            res = FM.multi_factor_regression(r, factor_returns_aligned[available], rf=rf)
            if "error" in res:
                st.error(res["error"])
            else:
                c = st.columns(4)
                c[0].metric("R²", f"{res['r_squared']:.3f}")
                c[1].metric("Adj R²", f"{res['r_squared_adj']:.3f}")
                c[2].metric("F-stat", f"{res['f_stat']:.1f}",
                              help=f"p = {res['f_pvalue']:.4f}")
                c[3].metric("N obs", f"{res['n_obs']}")

                ft = res["factor_table"]
                disp = ft.copy()
                disp["coefficient"] = disp["coefficient"].apply(lambda x: f"{x:.4f}")
                disp["std_error"] = disp["std_error"].apply(lambda x: f"{x:.4f}")
                disp["t_stat"] = disp["t_stat"].apply(lambda x: f"{x:.2f}")
                disp["p_value"] = disp["p_value"].apply(lambda x: f"{x:.4f}")
                disp["ci_low"] = disp["ci_low"].apply(lambda x: f"{x:.3f}")
                disp["ci_high"] = disp["ci_high"].apply(lambda x: f"{x:.3f}")
                st.dataframe(disp, hide_index=True, width="stretch")

                bar = ft[ft["factor"] != "const"].copy()
                fig = px.bar(bar, x="factor", y="coefficient", color="t_stat",
                              color_continuous_scale="RdBu_r", color_continuous_midpoint=0,
                              title="Factor exposures (coefficients) — color = t-stat")
                fig.update_xaxes(tickangle=-30)
                st.plotly_chart(fig, width="stretch")

# Tab 3: Rolling alpha/beta
with tabs[2]:
    ppy = RS.periods_per_year(freq)
    default_w = min(max(ppy, 12), len(r) // 2) if ppy > 1 else 12
    window = st.slider("Rolling window (periods)", 6, max(12, len(r) // 2),
                         default_w, step=1)
    common = r.index.intersection(bench_aligned.index)
    rc = r.loc[common]; bc = bench_aligned.loc[common]
    rf_period = rf / ppy
    ex_r = rc - rf_period; ex_m = bc - rf_period
    cov = ex_r.rolling(window).cov(ex_m)
    var = ex_m.rolling(window).var()
    roll_beta = (cov / var).dropna()
    roll_alpha_period = (ex_r.rolling(window).mean() - roll_beta * ex_m.rolling(window).mean()).dropna()
    roll_alpha_ann = roll_alpha_period * ppy

    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(roll_alpha_ann,
                        title=f"Rolling {window}-period alpha (annualized) vs {bench_name}")
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(showlegend=False, height=360)
        fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig = px.line(roll_beta, title=f"Rolling {window}-period β vs {bench_name}")
        fig.add_hline(y=1, line_dash="dash", line_color="#94a3b8")
        fig.update_layout(showlegend=False, height=360)
        st.plotly_chart(fig, width="stretch")
