"""Risk decomposition: systematic vs idiosyncratic + concentration, tail, correlation, suggestions."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import analytics as A
import factor_models as FM
import risk as R
from data import (FACTOR_PROXIES, fetch_currency_map, fetch_prices,
                   portfolio_returns, require_portfolio)
from theme import badge, inject_css

st.set_page_config(page_title="Risk Decomposition", layout="wide")
inject_css()
st.title("Risk Decomposition & Improvement Suggestions")
st.caption(
    "Splits portfolio risk into **systematic** (market-driven) vs **idiosyncratic** "
    "(stock-specific) components, surfaces concentration / tail / correlation issues, "
    "and offers rule-based suggestions for how to improve."
)

tickers, weights, prices, returns, bench_prices, bench_returns, rf = require_portfolio()
port_ret = portfolio_returns(returns, weights)

# ─── Portfolio-level CAPM decomposition ───────────────────────────────────────
capm_res = FM.capm(port_ret, bench_returns, rf=rf)
beta = capm_res.get("beta", np.nan)
r_squared = capm_res.get("r_squared", np.nan)

total_var = float(port_ret.var() * 252)
total_vol = float(np.sqrt(total_var))
bench_var = float(bench_returns.var() * 252) if len(bench_returns) > 1 else np.nan
bench_vol = float(np.sqrt(bench_var)) if not np.isnan(bench_var) else np.nan

if not np.isnan(beta) and not np.isnan(bench_var):
    systematic_var = beta ** 2 * bench_var
    systematic_vol = float(np.sqrt(systematic_var))
    idio_var = max(total_var - systematic_var, 0.0)
    idio_vol = float(np.sqrt(idio_var))
    systematic_share = systematic_var / total_var if total_var > 0 else np.nan
    idio_share = 1.0 - systematic_share if not np.isnan(systematic_share) else np.nan
else:
    systematic_var = systematic_vol = idio_var = idio_vol = np.nan
    systematic_share = idio_share = np.nan

# Cross-tab variables: assigned in the VaR tab, consumed in the Suggestions tab.
# Initialize here so the suggestion engine never crashes on a missing name.
var_95_realized_breach = np.nan
var_99_realized_breach = np.nan
cvar_95 = np.nan
var_95 = np.nan

tabs = st.tabs([
    "Systematic vs Idiosyncratic",
    "Active risk vs benchmark",
    "Tail risk & VaR",
    "Concentration risk",
    "Correlation & diversification",
    "Improvement suggestions",
])

# ══════════════════════════════════════════════════════════════════════════════
# Tab 1: Systematic vs Idiosyncratic
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.subheader("Portfolio-level single-factor decomposition")
    st.markdown(
        r"$\sigma^2_{total} = \beta^2 \cdot \sigma^2_{market} + \sigma^2_{idiosyncratic}$"
    )

    c = st.columns(5)
    c[0].metric("Total ann. vol", f"{total_vol:.2%}")
    c[1].metric("Beta to SPX",
                 f"{beta:.3f}" if not np.isnan(beta) else "—")
    c[2].metric("R² (market)",
                 f"{r_squared:.3f}" if not np.isnan(r_squared) else "—")
    c[3].metric("Systematic share",
                 f"{systematic_share:.1%}" if not np.isnan(systematic_share) else "—",
                 help="Variance explained by market beta")
    c[4].metric("Idiosyncratic share",
                 f"{idio_share:.1%}" if not np.isnan(idio_share) else "—",
                 help="Stock-specific variance — diversifiable")

    c2 = st.columns(4)
    c2[0].metric("Systematic vol",
                  f"{systematic_vol:.2%}" if not np.isnan(systematic_vol) else "—")
    c2[1].metric("Idiosyncratic vol",
                  f"{idio_vol:.2%}" if not np.isnan(idio_vol) else "—")
    c2[2].metric("Benchmark vol", f"{bench_vol:.2%}" if not np.isnan(bench_vol) else "—")
    c2[3].metric("Observations", f"{capm_res.get('n_obs', '—')}")

    # Stacked horizontal bar
    if not np.isnan(systematic_share):
        fig = go.Figure()
        fig.add_bar(x=[systematic_share], y=["Variance share"], orientation="h",
                      name=f"Systematic ({systematic_share:.1%})",
                      marker_color="#3498db",
                      text=[f"{systematic_share:.1%}"], textposition="inside")
        fig.add_bar(x=[idio_share], y=["Variance share"], orientation="h",
                      name=f"Idiosyncratic ({idio_share:.1%})",
                      marker_color="#e67e22",
                      text=[f"{idio_share:.1%}"], textposition="inside")
        fig.update_layout(barmode="stack", height=180,
                            title="Portfolio variance composition",
                            xaxis_tickformat=".0%",
                            yaxis=dict(showticklabels=False),
                            margin=dict(t=40, b=20, l=20, r=20))
        st.plotly_chart(fig, width="stretch")

    st.markdown("---")
    st.subheader("Per-asset decomposition")
    st.caption(
        "For each holding: σ²_total = β² × σ²_market + σ²_idiosyncratic. "
        "Names with **high idiosyncratic share** carry stock-specific risk that "
        "diversification can reduce."
    )

    per_asset_rows: list[dict] = []
    bench_ann_vol = float(bench_returns.std() * np.sqrt(252))

    for t in tickers:
        if t not in returns.columns:
            continue
        ar = returns[t].dropna()
        if len(ar) < 30:
            continue
        ar_capm = FM.capm(ar, bench_returns, rf=rf)
        b_i = ar_capm.get("beta", np.nan)
        r2_i = ar_capm.get("r_squared", np.nan)
        asset_vol = float(ar.std() * np.sqrt(252))
        sys_vol_i = abs(b_i) * bench_ann_vol if not np.isnan(b_i) else np.nan
        idio_vol_i = float(np.sqrt(max(asset_vol ** 2 - sys_vol_i ** 2, 0.0))) \
            if not np.isnan(sys_vol_i) else np.nan
        idio_share_i = (idio_vol_i ** 2 / asset_vol ** 2) \
            if asset_vol > 0 and not np.isnan(idio_vol_i) else np.nan
        per_asset_rows.append({
            "ticker": t,
            "weight": float(weights.get(t, 0)),
            "ann_vol": asset_vol,
            "beta": b_i,
            "r_squared": r2_i,
            "systematic_vol": sys_vol_i,
            "idiosyncratic_vol": idio_vol_i,
            "idio_share": idio_share_i,
        })

    per_asset = pd.DataFrame(per_asset_rows)
    if not per_asset.empty:
        disp = per_asset.copy()
        for col in ["weight", "ann_vol", "systematic_vol", "idiosyncratic_vol", "idio_share"]:
            disp[col] = disp[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "—")
        for col in ["beta", "r_squared"]:
            disp[col] = disp[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
        st.dataframe(disp, hide_index=True, width="stretch")

        plot_df = per_asset.melt(
            id_vars=["ticker"],
            value_vars=["systematic_vol", "idiosyncratic_vol"],
            var_name="component", value_name="vol",
        ).dropna()
        fig = px.bar(
            plot_df, x="ticker", y="vol", color="component", barmode="stack",
            color_discrete_map={"systematic_vol": "#3498db",
                                  "idiosyncratic_vol": "#e67e22"},
            title="Per-asset annualized vol decomposition",
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, width="stretch")

        # Highlight high-idio names
        high_idio = per_asset.dropna(subset=["idio_share"]).sort_values("idio_share", ascending=False).head(5)
        if not high_idio.empty:
            st.markdown("**Top 5 holdings by idiosyncratic share** (most stock-specific risk):")
            hi_disp = high_idio[["ticker", "weight", "idio_share", "r_squared"]].copy()
            hi_disp["weight"] = hi_disp["weight"].apply(lambda x: f"{x:.2%}")
            hi_disp["idio_share"] = hi_disp["idio_share"].apply(lambda x: f"{x:.1%}")
            hi_disp["r_squared"] = hi_disp["r_squared"].apply(lambda x: f"{x:.3f}")
            st.dataframe(hi_disp, hide_index=True, width="stretch")

    # ─── Multi-factor decomposition ──────────────────────────────────────────
    st.markdown("---")
    st.subheader("Multi-factor R² (broader systematic measure)")
    st.caption(
        "A multi-factor model typically captures more variance than CAPM alone. "
        "The residual after a richer factor model is a tighter measure of truly "
        "idiosyncratic / alpha-bearing risk."
    )

    chosen_factors = st.multiselect(
        "Factors to include",
        list(FACTOR_PROXIES.keys()),
        default=["Market (SPY)", "Value (IWD large value)", "Growth (IWF large growth)",
                  "Momentum (MTUM)", "Min-Vol / Low-Risk (USMV)", "Quality (QUAL)"],
        key="mf_factors_decomp",
    )

    if chosen_factors and st.button("Run multi-factor regression", key="mf_run_decomp"):
        seen: set[str] = set()
        factor_tickers: dict[str, str] = {}
        for name in chosen_factors:
            t = FACTOR_PROXIES[name]
            if t not in seen:
                seen.add(t)
                factor_tickers[name] = t
        with st.spinner("Fetching factor data..."):
            fp = fetch_prices(tuple(factor_tickers.values()), period="5y")
        if fp.empty:
            st.error("Could not fetch factor prices.")
        else:
            available = [t for t in factor_tickers.values() if t in fp.columns]
            fr = fp[available].pct_change().dropna(how="all")
            fr = fr.rename(columns={t: name for name, t in factor_tickers.items() if t in available})
            mf = FM.multi_factor_regression(port_ret, fr, rf=rf)
            if "error" in mf:
                st.error(mf["error"])
            else:
                mf_r2 = mf["r_squared"]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Multi-factor R²", f"{mf_r2:.3f}")
                c2.metric("Systematic share", f"{mf_r2:.1%}")
                c3.metric("Idiosyncratic share", f"{1 - mf_r2:.1%}")
                c4.metric("Δ vs CAPM",
                           f"{(mf_r2 - r_squared):+.3f}" if not np.isnan(r_squared) else "—",
                           help="How much more variance the multi-factor model explains vs CAPM alone")

                fig = go.Figure()
                fig.add_bar(x=[r_squared], y=["CAPM"], orientation="h",
                              marker_color="#95a5a6", name="Market only",
                              text=[f"{r_squared:.1%}"], textposition="inside")
                fig.add_bar(x=[mf_r2], y=["Multi-factor"], orientation="h",
                              marker_color="#16a085", name="Multi-factor",
                              text=[f"{mf_r2:.1%}"], textposition="inside")
                fig.update_layout(title="Systematic variance explained: CAPM vs multi-factor",
                                    xaxis_tickformat=".0%", height=220,
                                    showlegend=False,
                                    margin=dict(t=40, b=20, l=20, r=20))
                st.plotly_chart(fig, width="stretch")

                # Significant factor exposures
                ft = mf["factor_table"]
                non_const = ft[ft["factor"] != "const"].copy()
                non_const["abs_t"] = non_const["t_stat"].abs()
                fig = px.bar(
                    non_const.sort_values("abs_t", ascending=False),
                    x="factor", y="coefficient", color="t_stat",
                    color_continuous_scale="RdBu_r", color_continuous_midpoint=0,
                    title="Factor exposures (β) — color = t-statistic significance",
                )
                fig.update_xaxes(tickangle=-30)
                st.plotly_chart(fig, width="stretch")

# ══════════════════════════════════════════════════════════════════════════════
# Tab 2: Active Risk vs Benchmark (tracking error, info ratio, capture)
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.subheader("Active risk vs benchmark (S&P 500)")
    st.markdown(
        "Active return = portfolio − benchmark. **Tracking error** is the annualized "
        "standard deviation of active returns — how much your portfolio swings *around* "
        "the benchmark. **Information ratio** = active return ÷ tracking error."
    )

    if bench_returns.empty or len(bench_returns) < 30:
        st.info("Benchmark data is unavailable or too short — active risk metrics need a benchmark.")
    else:
        ab = A.alpha_beta(port_ret, bench_returns, rf=rf)
        active = (port_ret - bench_returns.reindex(port_ret.index)).dropna()
        ann_active_return = float(active.mean() * 252)
        te = float(active.std() * np.sqrt(252))
        info_ratio = ann_active_return / te if te > 0 else np.nan

        c = st.columns(5)
        c[0].metric("Tracking error (annual)", f"{te:.2%}",
                     help="σ of (portfolio − benchmark) returns, annualized")
        c[1].metric("Active return (annual)", f"{ann_active_return:+.2%}",
                     help="Mean of daily active returns × 252")
        c[2].metric("Information ratio", f"{info_ratio:.2f}" if not np.isnan(info_ratio) else "—",
                     help="Active return ÷ tracking error. > 0.5 is good.")
        c[3].metric("Alpha (CAPM, annual)",
                     f"{ab.get('alpha', float('nan')):+.2%}" if not np.isnan(ab.get("alpha", np.nan)) else "—",
                     help="Excess return after adjusting for beta")
        c[4].metric("R² (market)",
                     f"{r_squared:.3f}" if not np.isnan(r_squared) else "—")

        c2 = st.columns(5)
        c2[0].metric("Beta",
                       f"{beta:.3f}" if not np.isnan(beta) else "—")
        c2[1].metric("Up capture",
                       f"{ab.get('up_capture', float('nan')):.2f}" if not np.isnan(ab.get("up_capture", np.nan)) else "—",
                       help="Portfolio's avg return in up-market days ÷ benchmark's. > 1 = outperforms in rallies.")
        c2[2].metric("Down capture",
                       f"{ab.get('down_capture', float('nan')):.2f}" if not np.isnan(ab.get("down_capture", np.nan)) else "—",
                       help="Portfolio's avg return in down-market days ÷ benchmark's. < 1 = loses less in sell-offs.")
        c2[3].metric("Treynor",
                       f"{ab.get('treynor', float('nan')):.4f}" if not np.isnan(ab.get("treynor", np.nan)) else "—",
                       help="(Excess return) / beta")
        c2[4].metric("M²",
                       f"{A.m_squared(port_ret, bench_returns, rf):.2%}",
                       help="Modigliani-Modigliani — return at the benchmark's risk level")

        # Cumulative active return
        cum_active = (1 + active).cumprod() - 1
        fig = px.line(cum_active, title="Cumulative active return (portfolio − benchmark)",
                        labels={"value": "Cumulative excess return"})
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch")

        # Rolling tracking error
        st.markdown("---")
        st.subheader("Rolling tracking error & information ratio")
        window = st.slider("Window (days)", 21, 252, 63, step=21, key="te_window")
        rolling_te = active.rolling(window).std() * np.sqrt(252)
        rolling_active = active.rolling(window).mean() * 252
        rolling_ir = rolling_active / rolling_te.replace(0, np.nan)

        c1, c2 = st.columns(2)
        with c1:
            fig = px.line(rolling_te,
                            title=f"Rolling tracking error ({window}d, annualized)")
            fig.update_yaxes(tickformat=".0%")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width="stretch")
        with c2:
            fig = px.line(rolling_ir,
                            title=f"Rolling information ratio ({window}d)")
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width="stretch")

        # Up / down capture visual
        st.markdown("---")
        st.subheader("Capture in up vs down markets")
        up_mask = bench_returns.reindex(port_ret.index) > 0
        down_mask = bench_returns.reindex(port_ret.index) < 0
        up_port = port_ret[up_mask].mean() * 252
        up_bench = bench_returns.reindex(port_ret.index)[up_mask].mean() * 252
        down_port = port_ret[down_mask].mean() * 252
        down_bench = bench_returns.reindex(port_ret.index)[down_mask].mean() * 252

        cap_df = pd.DataFrame({
            "regime": ["Up days", "Down days"],
            "Portfolio": [up_port, down_port],
            "Benchmark": [up_bench, down_bench],
        })
        long_cap = cap_df.melt(id_vars="regime", var_name="series", value_name="ann_return")
        fig = px.bar(long_cap, x="regime", y="ann_return", color="series", barmode="group",
                       title="Annualized return conditional on market regime",
                       color_discrete_map={"Portfolio": "#3498db", "Benchmark": "#95a5a6"})
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, width="stretch")

        # Store for suggestions
        te_for_suggestions = te
        ir_for_suggestions = info_ratio
        up_cap_for_suggestions = ab.get("up_capture", np.nan)
        down_cap_for_suggestions = ab.get("down_capture", np.nan)


# ══════════════════════════════════════════════════════════════════════════════
# Tab 4: Concentration risk (was tab 2)
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    abs_w = weights.abs()
    total_abs = float(abs_w.sum())
    norm_abs = abs_w / total_abs if total_abs > 0 else abs_w

    hhi = float((norm_abs ** 2).sum()) if total_abs > 0 else np.nan
    effective_n = (1 / hhi) if hhi and not np.isnan(hhi) and hhi > 0 else np.nan
    top1 = float(norm_abs.max()) if total_abs > 0 else np.nan
    top1_name = norm_abs.idxmax() if total_abs > 0 else ""
    top3 = float(norm_abs.nlargest(3).sum()) if total_abs > 0 else np.nan
    top5 = float(norm_abs.nlargest(5).sum()) if total_abs > 0 else np.nan
    top10 = float(norm_abs.nlargest(10).sum()) if total_abs > 0 else np.nan

    st.subheader("Position concentration")
    c = st.columns(6)
    c[0].metric(f"Largest ({top1_name})", f"{top1:.1%}" if not np.isnan(top1) else "—")
    c[1].metric("Top 3", f"{top3:.1%}" if not np.isnan(top3) else "—")
    c[2].metric("Top 5", f"{top5:.1%}" if not np.isnan(top5) else "—")
    c[3].metric("Top 10", f"{top10:.1%}" if not np.isnan(top10) else "—")
    c[4].metric("HHI", f"{hhi:.3f}" if not np.isnan(hhi) else "—",
                 help="Herfindahl: sum of squared weights. Lower = more diversified.")
    c[5].metric("Eff. # assets", f"{effective_n:.1f}" if not np.isnan(effective_n) else "—",
                 help="1/HHI. Compare to nominal count.")

    # Position size distribution
    plot_w = weights.sort_values()
    fig = px.bar(
        plot_w, orientation="h",
        color=plot_w > 0,
        color_discrete_map={True: "#2ecc71", False: "#e74c3c"},
        title="Position sizes (signed weights)",
    )
    fig.update_xaxes(tickformat=".0%")
    fig.update_layout(height=max(300, 22 * len(plot_w)), showlegend=False,
                       yaxis=dict(title=None))
    st.plotly_chart(fig, width="stretch")

    # Cumulative concentration curve
    sorted_abs = norm_abs.sort_values(ascending=False).reset_index(drop=True)
    cumulative = sorted_abs.cumsum()
    fig = px.line(cumulative, title="Cumulative gross weight (sorted from largest)",
                    labels={"index": "Position rank", "value": "Cumulative |w|"})
    fig.add_hline(y=0.50, line_dash="dash", line_color="orange",
                    annotation_text="50% mark")
    fig.add_hline(y=0.80, line_dash="dash", line_color="red",
                    annotation_text="80% mark")
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width="stretch")

    # Sector & country concentration via lazy currency fetch
    st.markdown("---")
    st.subheader("Sector & country concentration")
    with st.spinner("Fetching ticker metadata..."):
        meta = fetch_currency_map(tickers)
    meta_df = pd.DataFrame([
        {"ticker": t, "country": meta[t]["country"], "currency": meta[t]["currency"],
          "weight": float(weights.get(t, 0))} for t in tickers
    ])
    cur_conc = meta_df.groupby("currency")["weight"].apply(lambda x: x.abs().sum()).sort_values(ascending=False)
    co_conc = meta_df[meta_df["country"] != ""].groupby("country")["weight"].apply(lambda x: x.abs().sum()).sort_values(ascending=False)

    c1, c2 = st.columns(2)
    with c1:
        if not cur_conc.empty:
            top_cur = cur_conc.iloc[0]
            top_cur_name = cur_conc.index[0]
            st.metric(f"Largest currency: {top_cur_name}", f"{top_cur:.1%}")
            fig = px.pie(values=cur_conc.values, names=cur_conc.index, hole=0.4,
                           title="Gross exposure by currency")
            st.plotly_chart(fig, width="stretch")
    with c2:
        if not co_conc.empty:
            top_co = co_conc.iloc[0]
            top_co_name = co_conc.index[0]
            st.metric(f"Largest country: {top_co_name}", f"{top_co:.1%}")
            fig = px.pie(values=co_conc.values, names=co_conc.index, hole=0.4,
                           title="Gross exposure by country")
            st.plotly_chart(fig, width="stretch")

# ══════════════════════════════════════════════════════════════════════════════
# Tab 3: Tail risk & VaR
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    daily = port_ret.dropna()
    skew_val = float(daily.skew())
    kurt_val = float(daily.kurtosis())
    var_95 = R.var_historical(daily, 0.05)
    var_99 = R.var_historical(daily, 0.01)
    cvar_95 = R.cvar_historical(daily, 0.05)
    cvar_99 = R.cvar_historical(daily, 0.01)
    max_dd = A.max_drawdown(daily)
    ulcer = A.ulcer_index(daily)

    st.subheader("Distribution & tail risk")
    c = st.columns(4)
    c[0].metric("Skewness", f"{skew_val:.3f}",
                 help="Negative = more frequent large losses than gains")
    c[1].metric("Excess kurtosis", f"{kurt_val:.3f}",
                 help="> 0 = fatter tails than a normal distribution")
    c[2].metric("Max drawdown", f"{max_dd:.2%}")
    c[3].metric("Ulcer Index", f"{ulcer:.4f}",
                 help="RMS of drawdowns — measures depth and duration of losses")

    # ─── Full VaR table: methods × confidence levels with $ amounts ─────────
    st.markdown("---")
    st.subheader("Value at Risk (VaR) — full method comparison")
    st.caption(
        "Negative numbers are loss thresholds. **Historical** uses the empirical "
        "distribution. **Parametric** assumes normal returns. **Cornish-Fisher** "
        "adjusts the normal VaR for observed skew & kurtosis. **Monte Carlo** "
        "draws from a fitted normal distribution."
    )

    notional = st.number_input(
        "Portfolio notional ($)", 1000.0, 1e12,
        float(st.session_state.get("aum", 100000.0)),
        step=1000.0, key="var_notional",
        help="Defaults to fund AUM from the Home page",
    )
    horizon = st.select_slider("Holding horizon (days)",
                                  options=[1, 2, 5, 10, 20, 60],
                                  value=1,
                                  help="VaR scales as σ·√t under iid assumption")

    var_summary = R.var_summary(daily, alphas=(0.10, 0.05, 0.01))
    # Scale to horizon and notional
    scale = float(np.sqrt(horizon))
    var_disp = var_summary.copy()
    for col in var_disp.columns:
        if col == "confidence":
            continue
        var_disp[col] = var_disp[col].apply(
            lambda v: f"{v * scale:.3%}  ({v * scale * notional:+,.0f})"
            if pd.notna(v) else "—"
        )
    var_disp.columns = [
        "Confidence", "Historical", "Parametric", "Cornish-Fisher",
        "Monte Carlo", "CVaR Hist.", "CVaR Param.",
    ]
    st.dataframe(var_disp, hide_index=True, width="stretch")
    st.caption(f"Values shown for **{horizon}-day** horizon with **${notional:,.0f}** notional.")

    # ─── VaR back-test: how often did actual losses exceed the parametric VaR? ─
    st.markdown("---")
    st.subheader("VaR back-test (rolling 252-day)")
    st.caption(
        "Compares realized losses against a rolling-window parametric VaR. "
        "If the model is well-calibrated, breach rate ≈ (1 − confidence). "
        "**Far above** = model under-estimates risk (fatter tails than normal). "
        "**Far below** = over-cautious."
    )

    breach_rows: list[dict] = []
    for alpha in (0.10, 0.05, 0.01):
        rolling_mu = daily.rolling(252).mean()
        rolling_sigma = daily.rolling(252).std()
        from scipy.stats import norm as _norm
        z = _norm.ppf(alpha)
        rolling_var = rolling_mu + z * rolling_sigma
        breaches = (daily < rolling_var) & rolling_var.notna()
        valid = rolling_var.notna()
        breach_rate = float(breaches.sum() / valid.sum()) if valid.sum() > 0 else np.nan
        expected = alpha
        breach_rows.append({
            "Confidence": f"{1 - alpha:.0%}",
            "Expected breach rate": f"{expected:.2%}",
            "Realized breach rate": f"{breach_rate:.2%}" if not np.isnan(breach_rate) else "—",
            "Excess (realized − expected)": f"{(breach_rate - expected):+.2%}" if not np.isnan(breach_rate) else "—",
            "Breach count": f"{int(breaches.sum())}",
            "N obs": f"{int(valid.sum())}",
        })
    breach_df = pd.DataFrame(breach_rows)
    st.dataframe(breach_df, hide_index=True, width="stretch")

    # Save for suggestion engine
    var_99_realized_breach = (daily < (daily.rolling(252).mean()
                                          + _norm.ppf(0.01) * daily.rolling(252).std())).mean()
    var_95_realized_breach = (daily < (daily.rolling(252).mean()
                                          + _norm.ppf(0.05) * daily.rolling(252).std())).mean()

    # ─── Distribution chart with VaR markers ────────────────────────────────
    st.markdown("---")
    st.subheader("Return distribution & drawdown")
    fig = px.histogram(daily, nbins=80, title="Daily return distribution",
                         marginal="box")
    fig.add_vline(x=var_95, line_dash="dash", line_color="orange",
                    annotation_text=f"VaR 95% = {var_95:.2%}")
    fig.add_vline(x=var_99, line_dash="dash", line_color="red",
                    annotation_text=f"VaR 99% = {var_99:.2%}")
    fig.add_vline(x=cvar_95, line_dash="dot", line_color="orange",
                    annotation_text=f"CVaR 95% = {cvar_95:.2%}",
                    annotation_position="top")
    fig.update_xaxes(tickformat=".1%")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width="stretch")

    dd_series = A.drawdown_series(daily)
    fig = px.area(dd_series, title="Drawdown over time")
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width="stretch")

    # Stress VaR — worst rolling 5-day window
    st.markdown("---")
    st.subheader("Stress VaR (worst historical windows)")
    rolling_5d = daily.rolling(5).sum()
    rolling_20d = daily.rolling(20).sum()
    worst_5d = float(rolling_5d.min()) if not rolling_5d.empty else np.nan
    worst_20d = float(rolling_20d.min()) if not rolling_20d.empty else np.nan
    worst_5d_date = rolling_5d.idxmin() if not rolling_5d.empty else None
    worst_20d_date = rolling_20d.idxmin() if not rolling_20d.empty else None
    c = st.columns(2)
    c[0].metric("Worst 5-day stretch",
                 f"{worst_5d:.2%}" if not np.isnan(worst_5d) else "—",
                 help=f"Ending {worst_5d_date.strftime('%Y-%m-%d')}" if worst_5d_date is not None else "")
    c[1].metric("Worst 20-day stretch",
                 f"{worst_20d:.2%}" if not np.isnan(worst_20d) else "—",
                 help=f"Ending {worst_20d_date.strftime('%Y-%m-%d')}" if worst_20d_date is not None else "")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 5: Correlation & diversification (was tab 3)
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    corr = returns.corr()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    upper_stacked = upper.stack()

    if not upper_stacked.empty:
        avg_corr = float(upper_stacked.mean())
        median_corr = float(upper_stacked.median())
        max_pair = float(upper_stacked.max())
        min_pair = float(upper_stacked.min())
    else:
        avg_corr = median_corr = max_pair = min_pair = np.nan

    dr = R.diversification_ratio(returns, weights)

    st.subheader("Correlation & diversification metrics")
    c = st.columns(5)
    c[0].metric("Avg pairwise ρ",
                 f"{avg_corr:.3f}" if not np.isnan(avg_corr) else "—",
                 help="Lower is better diversified")
    c[1].metric("Median pairwise ρ",
                 f"{median_corr:.3f}" if not np.isnan(median_corr) else "—")
    c[2].metric("Max pair ρ",
                 f"{max_pair:.3f}" if not np.isnan(max_pair) else "—")
    c[3].metric("Min pair ρ",
                 f"{min_pair:.3f}" if not np.isnan(min_pair) else "—")
    c[4].metric("Diversification ratio",
                 f"{dr:.3f}" if not np.isnan(dr) else "—",
                 help="(|w|-weighted avg vol) / (portfolio vol). > 1 = diversification benefit.")

    if not upper_stacked.empty:
        # MultiIndex level names may collide (both "Ticker"); rename explicitly
        pairs = upper_stacked.copy()
        pairs.index.names = ["asset_a", "asset_b"]
        pairs.name = "correlation"
        top_pairs = pairs.reset_index().sort_values("correlation", ascending=False)
        st.markdown("**Most correlated pairs (consider trimming for diversification):**")
        st.dataframe(top_pairs.head(10), hide_index=True, width="stretch")

    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                     zmin=-1, zmax=1, aspect="auto",
                     title="Asset return correlation matrix")
    fig.update_layout(height=500)
    st.plotly_chart(fig, width="stretch")

# ══════════════════════════════════════════════════════════════════════════════
# Tab 6: Suggestions (was tab 4)
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.subheader("Actionable improvement suggestions")
    st.caption(
        "Heuristic rules over the metrics above. These are **diagnostics, not financial advice**. "
        "Always combine with your own conviction and constraints."
    )

    suggestions: list[dict] = []

    # Concentration
    if not np.isnan(top1):
        if top1 > 0.25:
            suggestions.append({
                "severity": "high",
                "category": "Concentration",
                "issue": "Single-name concentration",
                "current": f"Largest position {top1_name} = {top1:.0%} of gross",
                "action": "Trim below 15-20% to reduce single-name risk. Idiosyncratic blow-ups become survivable when no single name dominates.",
            })
        elif top1 > 0.15:
            suggestions.append({
                "severity": "medium",
                "category": "Concentration",
                "issue": "Single-name concentration",
                "current": f"Largest position {top1_name} = {top1:.0%} of gross",
                "action": "Consider trimming below 15% for better diversification.",
            })

    if not np.isnan(top5) and top5 > 0.70:
        suggestions.append({
            "severity": "high",
            "category": "Concentration",
            "issue": "Top-5 concentration",
            "current": f"Top 5 positions = {top5:.0%} of gross",
            "action": "Add more positions or trim top names — a healthy 10-20 stock portfolio usually has top-5 below 60%.",
        })

    if not np.isnan(effective_n) and effective_n < len(weights) * 0.5:
        suggestions.append({
            "severity": "medium",
            "category": "Concentration",
            "issue": "Diversification inefficiency",
            "current": f"Effective # assets = {effective_n:.1f} (nominal = {len(weights)})",
            "action": "Weights are heavily skewed. Equal- or risk-parity weighting would improve diversification efficiency.",
        })

    # Idiosyncratic risk
    if not np.isnan(idio_share):
        if idio_share > 0.65:
            suggestions.append({
                "severity": "high",
                "category": "Idiosyncratic risk",
                "issue": "High stock-specific risk",
                "current": f"{idio_share:.0%} of variance is idiosyncratic (R² = {r_squared:.2f})",
                "action": (
                    "Your portfolio is dominated by stock-specific risk. Reduce by: "
                    "(a) adding more uncorrelated names, "
                    "(b) substituting broad-market ETFs for some single-stock positions, "
                    "(c) increasing names per sector to dilute single-name impact."
                ),
            })
        elif idio_share > 0.45:
            suggestions.append({
                "severity": "medium",
                "category": "Idiosyncratic risk",
                "issue": "Moderate stock-specific risk",
                "current": f"{idio_share:.0%} of variance is idiosyncratic",
                "action": "Some diversification headroom remains. Adding 5-10 uncorrelated positions could pull this below 40%.",
            })
        elif idio_share < 0.15:
            suggestions.append({
                "severity": "info",
                "category": "Idiosyncratic risk",
                "issue": "Very low idiosyncratic risk",
                "current": f"{idio_share:.0%} idiosyncratic — portfolio behaves nearly like the market",
                "action": "You're essentially indexed. Consider whether the active risk you're paying for is worth it — a low-cost index fund may suffice.",
            })

    # Systematic risk / beta
    if not np.isnan(beta):
        if beta > 1.3:
            suggestions.append({
                "severity": "medium",
                "category": "Systematic risk",
                "issue": "Elevated market beta",
                "current": f"β = {beta:.2f} — portfolio moves ~{beta:.1f}× the market",
                "action": (
                    "Reduce systematic risk by: "
                    "(a) adding low-beta / defensive names (utilities, staples, healthcare), "
                    "(b) overlay with USMV (min-vol ETF), "
                    "(c) allocate to bonds / gold (negatively correlated with equities in stress)."
                ),
            })
        elif beta < 0.5:
            suggestions.append({
                "severity": "info",
                "category": "Systematic risk",
                "issue": "Low market beta",
                "current": f"β = {beta:.2f} — limited equity-market exposure",
                "action": "Good for downside protection but expect underperformance in strong bull markets.",
            })
        if not np.isnan(systematic_share) and systematic_share > 0.80:
            suggestions.append({
                "severity": "info",
                "category": "Systematic risk",
                "issue": "Dominant systematic risk",
                "current": f"{systematic_share:.0%} of variance is market-driven",
                "action": "Most of your risk is the market itself. To take genuinely differentiated risk, raise active share or include lower-correlation asset classes.",
            })

    # Correlation
    if not np.isnan(avg_corr):
        if avg_corr > 0.65:
            suggestions.append({
                "severity": "high",
                "category": "Diversification",
                "issue": "High pairwise correlation",
                "current": f"Avg pairwise ρ = {avg_corr:.2f}",
                "action": (
                    "Names move in lockstep. Diversify by adding: "
                    "international equities (EFA, EEM), "
                    "bonds (TLT, AGG), "
                    "real assets (gold GLD, REITs VNQ), "
                    "or low-correlation sectors (utilities, healthcare)."
                ),
            })
        elif avg_corr > 0.50:
            suggestions.append({
                "severity": "medium",
                "category": "Diversification",
                "issue": "Moderate pairwise correlation",
                "current": f"Avg pairwise ρ = {avg_corr:.2f}",
                "action": "Could improve diversification by adding less-correlated sectors or international exposure.",
            })

    # Diversification ratio
    if not np.isnan(dr):
        if dr < 1.2:
            suggestions.append({
                "severity": "medium",
                "category": "Diversification",
                "issue": "Low diversification ratio",
                "current": f"DR = {dr:.2f}",
                "action": "Portfolio vol is close to gross-weighted asset vol — there's little diversification benefit. Add uncorrelated assets.",
            })

    # Tail risk
    if skew_val < -0.5:
        suggestions.append({
            "severity": "medium",
            "category": "Tail risk",
            "issue": "Left-skewed returns",
            "current": f"Skewness = {skew_val:.2f}",
            "action": "Distribution has heavy left tail — losses are more frequent / severe than gains. Consider tail hedges (long-vol, put spreads) or trim high-vol momentum names.",
        })
    if kurt_val > 5:
        suggestions.append({
            "severity": "medium",
            "category": "Tail risk",
            "issue": "Fat-tailed returns",
            "current": f"Excess kurtosis = {kurt_val:.2f}",
            "action": "Extreme events more frequent than a normal distribution implies. Parametric VaR will underestimate tail risk — rely on historical or Cornish-Fisher VaR.",
        })

    # Drawdown
    if not np.isnan(max_dd):
        if max_dd < -0.40:
            suggestions.append({
                "severity": "high",
                "category": "Drawdown",
                "issue": "Severe historical drawdown",
                "current": f"Max drawdown = {max_dd:.0%}",
                "action": "Portfolio has experienced very large losses. Make sure position sizing and your time horizon can tolerate this. Consider defensive overlay or smaller risk budget.",
            })
        elif max_dd < -0.25:
            suggestions.append({
                "severity": "medium",
                "category": "Drawdown",
                "issue": "Notable drawdown history",
                "current": f"Max drawdown = {max_dd:.0%}",
                "action": "25%+ drawdown is meaningful. Verify your risk tolerance matches.",
            })

    # Sharpe
    sharpe_val = A.sharpe(daily, rf)
    if sharpe_val < 0.3 and not np.isnan(sharpe_val):
        suggestions.append({
            "severity": "medium",
            "category": "Risk-adjusted return",
            "issue": "Low Sharpe ratio",
            "current": f"Sharpe = {sharpe_val:.2f}",
            "action": "Risk-adjusted return is weak. Review whether each position is earning its risk budget; the Optimization page can suggest reweightings with higher Sharpe.",
        })

    # Tracking error & information ratio (computed in tab 2 if benchmark exists)
    if not bench_returns.empty and len(bench_returns) >= 30:
        ab_for_sug = A.alpha_beta(port_ret, bench_returns, rf=rf)
        active_for_sug = (port_ret - bench_returns.reindex(port_ret.index)).dropna()
        te_val = float(active_for_sug.std() * np.sqrt(252)) if len(active_for_sug) > 1 else np.nan
        active_ann = float(active_for_sug.mean() * 252) if len(active_for_sug) > 0 else np.nan
        ir_val = active_ann / te_val if te_val and te_val > 0 else np.nan
        up_cap = ab_for_sug.get("up_capture", np.nan)
        down_cap = ab_for_sug.get("down_capture", np.nan)

        if not np.isnan(te_val):
            if te_val > 0.10:
                suggestions.append({
                    "severity": "high",
                    "category": "Active risk",
                    "issue": "Very high tracking error",
                    "current": f"TE = {te_val:.1%} (you swing {te_val:.0%} around the benchmark each year)",
                    "action": (
                        "Portfolio behaves very differently from the benchmark. If you intended "
                        "index-like behavior, reduce single-name bets and add broader exposure. "
                        "If active risk is intentional, make sure your information ratio justifies it."
                    ),
                })
            elif te_val > 0.06:
                suggestions.append({
                    "severity": "medium",
                    "category": "Active risk",
                    "issue": "Elevated tracking error",
                    "current": f"TE = {te_val:.1%}",
                    "action": (
                        "Active risk is meaningful. Track whether your information ratio is "
                        "compensating you — at TE = 6%, you need ≥ 3-4% active return to "
                        "achieve IR ≈ 0.5-0.7."
                    ),
                })
            elif te_val < 0.015:
                suggestions.append({
                    "severity": "info",
                    "category": "Active risk",
                    "issue": "Very low tracking error (closet indexing)",
                    "current": f"TE = {te_val:.1%}",
                    "action": (
                        "Portfolio is hugging the benchmark. If you're paying active-management "
                        "fees, you may not be getting differentiated exposure — a low-cost index "
                        "fund could deliver similar behavior cheaper."
                    ),
                })

        if not np.isnan(ir_val):
            if ir_val > 0.75:
                suggestions.append({
                    "severity": "info",
                    "category": "Active risk",
                    "issue": "Strong information ratio",
                    "current": f"IR = {ir_val:.2f}",
                    "action": "Active bets are being well-compensated. Maintain discipline on position sizing.",
                })
            elif ir_val < 0.0 and abs(active_ann) > 0.02:
                suggestions.append({
                    "severity": "high",
                    "category": "Active risk",
                    "issue": "Negative information ratio",
                    "current": f"IR = {ir_val:.2f}, active return = {active_ann:+.2%}",
                    "action": (
                        "You're underperforming the benchmark while taking active risk. "
                        "Either rebalance toward the benchmark (lower TE) or reconsider your "
                        "active positions."
                    ),
                })
            elif 0 < ir_val < 0.3 and te_val > 0.04:
                suggestions.append({
                    "severity": "medium",
                    "category": "Active risk",
                    "issue": "Low information ratio for the active risk taken",
                    "current": f"IR = {ir_val:.2f}, TE = {te_val:.1%}",
                    "action": (
                        "Active risk isn't earning its keep. Either tighten positions toward "
                        "benchmark weights or sharpen conviction in your active bets."
                    ),
                })

        if not np.isnan(up_cap) and not np.isnan(down_cap):
            if down_cap > up_cap and down_cap > 1.0:
                suggestions.append({
                    "severity": "high",
                    "category": "Active risk",
                    "issue": "Asymmetric down-market behavior",
                    "current": f"Up capture = {up_cap:.2f}, Down capture = {down_cap:.2f}",
                    "action": (
                        "You're losing more in sell-offs than you're gaining in rallies — bad "
                        "asymmetry. Add defensive / low-vol names (USMV, utilities, staples) "
                        "or reduce beta."
                    ),
                })
            elif down_cap < 0.85 and up_cap > 0.85:
                suggestions.append({
                    "severity": "info",
                    "category": "Active risk",
                    "issue": "Defensive return profile",
                    "current": f"Up capture = {up_cap:.2f}, Down capture = {down_cap:.2f}",
                    "action": "Captures most of the upside but cushions drawdowns — a desirable asymmetry.",
                })

    # VaR back-test breaches — does parametric VaR work for this portfolio?
    if not np.isnan(var_95_realized_breach):
        breach_excess_95 = float(var_95_realized_breach) - 0.05
        breach_excess_99 = float(var_99_realized_breach) - 0.01 if not np.isnan(var_99_realized_breach) else np.nan

        if breach_excess_95 > 0.03:  # > 8% breach rate when expected 5%
            suggestions.append({
                "severity": "high",
                "category": "VaR model risk",
                "issue": "Parametric VaR under-estimates losses",
                "current": (f"Realized 95% breach rate = {var_95_realized_breach:.1%} "
                              f"(expected 5%); excess = {breach_excess_95:+.1%}"),
                "action": (
                    "Your returns have fatter tails than the normal distribution assumed by "
                    "parametric VaR. Use **historical VaR** or **Cornish-Fisher VaR** for "
                    "decision-making, and budget for losses larger than the normal-VaR suggests."
                ),
            })
        elif breach_excess_95 < -0.02:
            suggestions.append({
                "severity": "info",
                "category": "VaR model risk",
                "issue": "Parametric VaR is conservative",
                "current": f"Realized 95% breach rate = {var_95_realized_breach:.1%} (expected 5%)",
                "action": "Normal-VaR over-states risk for this portfolio. Capital allocation may be unnecessarily conservative.",
            })

        if not np.isnan(breach_excess_99) and breach_excess_99 > 0.015:  # > 2.5% breach when 1% expected
            suggestions.append({
                "severity": "high",
                "category": "VaR model risk",
                "issue": "Deep-tail breaches more frequent than expected",
                "current": f"Realized 99% breach rate = {var_99_realized_breach:.1%} (expected 1%)",
                "action": (
                    "Extreme losses happen more often than normal-VaR predicts. Consider "
                    "stress-testing (Stress Tests page) and ensuring position sizing assumes "
                    "tail events 2-3× more frequent than naive VaR implies."
                ),
            })

    # CVaR vs VaR gap — fat tail indicator
    if not np.isnan(cvar_95) and not np.isnan(var_95) and var_95 != 0:
        cvar_ratio = abs(cvar_95 / var_95)
        if cvar_ratio > 1.5:
            suggestions.append({
                "severity": "medium",
                "category": "VaR model risk",
                "issue": "Large CVaR/VaR gap",
                "current": f"CVaR 95% / VaR 95% = {cvar_ratio:.2f}",
                "action": (
                    "Average loss beyond the VaR threshold is much worse than the VaR itself — "
                    "tail losses are severe when they happen. Size positions assuming you'll "
                    "see CVaR-level losses, not just VaR-level."
                ),
            })

    # Display
    if not suggestions:
        st.success(
            "No significant issues flagged — your portfolio looks well-balanced "
            "across concentration, idiosyncratic/systematic decomposition, correlation, and tail-risk heuristics."
        )
    else:
        # Sort by severity (high first)
        sev_order = {"high": 0, "medium": 1, "info": 2}
        suggestions.sort(key=lambda x: sev_order.get(x["severity"], 99))

        n_high = sum(1 for s in suggestions if s["severity"] == "high")
        n_med = sum(1 for s in suggestions if s["severity"] == "medium")
        n_info = sum(1 for s in suggestions if s["severity"] == "info")

        c1, c2, c3 = st.columns(3)
        c1.metric("High priority", n_high)
        c2.metric("Medium priority", n_med)
        c3.metric("Informational", n_info)

        st.markdown("")  # spacer

        for s in suggestions:
            sev_label = {"high": "HIGH", "medium": "MEDIUM", "info": "INFO"}.get(s["severity"], "INFO")
            with st.container(border=True):
                cols = st.columns([1.4, 5])
                with cols[0]:
                    st.markdown(
                        f"{badge(sev_label, kind=s['severity'])} "
                        f"<span style='font-size:11px;color:#64748b;"
                        f"letter-spacing:0.05em;text-transform:uppercase;margin-left:6px;'>"
                        f"{s['category']}</span>",
                        unsafe_allow_html=True,
                    )
                with cols[1]:
                    st.markdown(f"**{s['issue']}**")
                    st.markdown(
                        f"<span style='color:#64748b;font-size:13px;'>Current state:</span> "
                        f"<span style='color:#0f172a;font-size:13px;'>{s['current']}</span>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<span style='color:#1e40af;font-weight:600;'>Action: </span>"
                        f"<span style='color:#0f172a;'>{s['action']}</span>",
                        unsafe_allow_html=True,
                    )
