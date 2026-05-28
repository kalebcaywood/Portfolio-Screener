"""CAPM and multi-factor regression analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import factor_models as FM
from data import benchmark_picker_and_data, FACTOR_PROXIES, fetch_prices, portfolio_returns, require_portfolio
from theme import inject_css

inject_css()
st.title("Factor Models")
st.caption("CAPM and custom multi-factor OLS with full regression statistics.")

tickers, weights, prices, returns, _, _, rf = require_portfolio()
bench_name, bench_prices, bench_returns = benchmark_picker_and_data()
port_ret = portfolio_returns(returns, weights)

st.sidebar.header("Regression target")
target = st.sidebar.radio("Target series", ["Portfolio", "Individual asset"], index=0)
if target == "Individual asset":
    sym = st.sidebar.selectbox("Asset", tickers)
    y_series = returns[sym]
    label = sym
else:
    y_series = port_ret
    label = "Portfolio"

tab_capm, tab_multi, tab_rolling = st.tabs(["CAPM", "Multi-factor", "Rolling alpha/beta"])

# ─── CAPM ─────────────────────────────────────────────────────────────────────
with tab_capm:
    st.subheader(f"CAPM regression — {label} vs {bench_name}")
    res = FM.capm(y_series, bench_returns, rf=rf)
    if "error" in res:
        st.error(res["error"])
    else:
        c = st.columns(5)
        c[0].metric("Alpha (annual)", f"{res['alpha_annual']:.2%}",
                     help=f"t = {res.get('alpha_tstat', float('nan')):.2f}, p = {res.get('alpha_pvalue', float('nan')):.3f}")
        c[1].metric("Beta", f"{res['beta']:.3f}",
                     help=f"t = {res.get('beta_tstat', float('nan')):.2f}, 95% CI [{res.get('beta_ci_low', float('nan')):.2f}, {res.get('beta_ci_high', float('nan')):.2f}]")
        c[2].metric("R²", f"{res['r_squared']:.3f}")
        c[3].metric("Adj R²", f"{res.get('r_squared_adj', float('nan')):.3f}")
        c[4].metric("N obs", f"{res['n_obs']}")

        c2 = st.columns(4)
        c2[0].metric("Residual σ", f"{res.get('residual_std', float('nan')):.4f}")
        c2[1].metric("Durbin-Watson", f"{res.get('durbin_watson', float('nan')):.2f}",
                       help="≈2 indicates no residual autocorrelation")
        c2[2].metric("F-statistic", f"{res.get('f_stat', float('nan')):.2f}")
        c2[3].metric("F p-value", f"{res.get('f_pvalue', float('nan')):.4f}")

        # Regression scatter
        df = pd.concat([y_series, bench_returns], axis=1, join="inner").dropna()
        df.columns = ["y", "m"]
        df["y_ex"] = df["y"] - rf / 252
        df["m_ex"] = df["m"] - rf / 252
        fit_x = np.linspace(df["m_ex"].min(), df["m_ex"].max(), 100)
        fit_y = res["alpha_daily"] + res["beta"] * fit_x

        fig = go.Figure()
        fig.add_scatter(x=df["m_ex"], y=df["y_ex"], mode="markers",
                          name="Daily excess returns", marker=dict(opacity=0.5, size=4))
        fig.add_scatter(x=fit_x, y=fit_y, mode="lines",
                          name=f"OLS fit: y = {res['alpha_daily']:.4f} + {res['beta']:.3f}·x",
                          line=dict(color="red"))
        fig.update_layout(title="CAPM excess-return regression",
                            xaxis_title="Market excess return",
                            yaxis_title=f"{label} excess return",
                            xaxis_tickformat=".1%", yaxis_tickformat=".1%")
        st.plotly_chart(fig, width="stretch")

        # Residual diagnostic
        if "_residuals" in res:
            resids = res["_residuals"]
            c1, c2 = st.columns(2)
            with c1:
                fig = px.line(resids, title="Residuals over time")
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, width="stretch")
            with c2:
                fig = px.histogram(resids, nbins=60, title="Residual distribution")
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, width="stretch")

# ─── Multi-factor ─────────────────────────────────────────────────────────────
with tab_multi:
    st.subheader("Custom multi-factor regression")
    st.caption("Pick proxy ETFs as factor returns. The model regresses your target on the chosen factors.")

    chosen_factors = st.multiselect(
        "Factor proxies",
        list(FACTOR_PROXIES.keys()),
        default=["Market (SPY)", "Value (IWD large value)", "Growth (IWF large growth)",
                  "Momentum (MTUM)", "Min-Vol / Low-Risk (USMV)", "Quality (QUAL)"],
    )
    if chosen_factors and st.button("Run regression", type="primary"):
        # Deduplicate by underlying ticker (same ETF can't appear twice as a factor)
        seen_tickers: set[str] = set()
        factor_tickers: dict[str, str] = {}
        for name in chosen_factors:
            t = FACTOR_PROXIES[name]
            if t not in seen_tickers:
                seen_tickers.add(t)
                factor_tickers[name] = t

        fetch_list = tuple(factor_tickers.values())
        with st.spinner("Fetching factor data..."):
            factor_prices = fetch_prices(fetch_list, period="5y")

        if factor_prices.empty:
            st.error("Could not fetch factor data.")
        else:
            available = [t for t in factor_tickers.values() if t in factor_prices.columns]
            missing = [name for name, t in factor_tickers.items() if t not in factor_prices.columns]
            if missing:
                st.warning(f"No data for: {', '.join(missing)}")
            if not available:
                st.error("None of the requested factor tickers returned data.")
                st.stop()

            factor_returns = factor_prices[available].pct_change().dropna(how="all")
            label_map = {t: name for name, t in factor_tickers.items() if t in available}
            factor_returns = factor_returns.rename(columns=label_map)

            res = FM.multi_factor_regression(y_series, factor_returns, rf=rf)
            if "error" in res:
                st.error(res["error"])
            else:
                c = st.columns(4)
                c[0].metric("R²", f"{res['r_squared']:.3f}")
                c[1].metric("Adj R²", f"{res['r_squared_adj']:.3f}")
                c[2].metric("F-stat (p)", f"{res['f_stat']:.1f} ({res['f_pvalue']:.4f})")
                c[3].metric("AIC / BIC", f"{res['aic']:.0f} / {res['bic']:.0f}")

                st.subheader("Factor loadings")
                ft = res["factor_table"]
                disp = ft.copy()
                disp["coefficient"] = disp["coefficient"].apply(lambda x: f"{x:.4f}")
                disp["std_error"] = disp["std_error"].apply(lambda x: f"{x:.4f}")
                disp["t_stat"] = disp["t_stat"].apply(lambda x: f"{x:.2f}")
                disp["p_value"] = disp["p_value"].apply(lambda x: f"{x:.4f}")
                disp["ci_low"] = disp["ci_low"].apply(lambda x: f"{x:.3f}")
                disp["ci_high"] = disp["ci_high"].apply(lambda x: f"{x:.3f}")
                st.dataframe(disp, width="stretch", hide_index=True)

                # Factor exposure chart
                bar = ft[ft["factor"] != "const"].copy()
                fig = px.bar(bar, x="factor", y="coefficient", color="t_stat",
                               color_continuous_scale="RdBu_r", color_continuous_midpoint=0,
                               title="Factor loadings (coefficients)")
                fig.update_xaxes(tickangle=-30)
                st.plotly_chart(fig, width="stretch")

                # Return attribution
                att = FM.factor_attribution(y_series, factor_returns, rf=rf)
                if not att.empty:
                    st.subheader("Annualized return attribution")
                    disp = att.copy()
                    for col in ["exposure", "factor_return", "contribution"]:
                        disp[col] = disp[col].apply(lambda x: f"{x:.2%}" if col != "exposure" else f"{x:.3f}")
                    st.dataframe(disp, width="stretch", hide_index=True)

                    fig = px.bar(att, x="factor", y="contribution",
                                   title="Return contribution by factor (annual)")
                    fig.update_yaxes(tickformat=".1%")
                    st.plotly_chart(fig, width="stretch")

# ─── Rolling alpha/beta ───────────────────────────────────────────────────────
with tab_rolling:
    window = st.slider("Rolling window (days)", 21, 252, 63, step=21)
    rdf = FM.rolling_capm(y_series, bench_returns, window=window, rf=rf)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(rdf["alpha"], title=f"Rolling CAPM alpha (annualized, {window}d)")
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig = px.line(rdf["beta"], title=f"Rolling CAPM beta ({window}d)")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch")
