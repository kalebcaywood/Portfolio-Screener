"""Return Stream Analyzer — Performance & Risk for a single selected stream.

Reads the parsed returns from session_state['rsa_returns'] (populated on the
RSA Home page). Picks one stream at a time for deep analysis, with the
existing benchmark picker (from the Portfolio Analyzer catalog) usable for
relative metrics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats as sps

import rsa as RS
from data import benchmark_picker_and_data
from theme import inject_css

inject_css()

st.title("Performance & Risk")
st.caption("Per-stream analytics for a single return series, with optional benchmark comparison.")

# ─── Guard: must have uploaded returns ───────────────────────────────────────
if "rsa_returns" not in st.session_state:
    st.warning("No return streams loaded. Go to **Return Stream Analyzer → Home** and upload a CSV first.")
    st.stop()

returns_df: pd.DataFrame = st.session_state["rsa_returns"]
freq: str = st.session_state.get("rsa_frequency", "M")
streams: list[str] = st.session_state["rsa_streams"]
rf: float = float(st.session_state.get("rsa_rf", 0.04))

# ─── Stream + benchmark selection ───────────────────────────────────────────
st.sidebar.header("Selection")
stream = st.sidebar.selectbox("Return stream to analyze", streams,
                                key="rsa_selected_stream")
bench_name, bench_prices, bench_returns_daily = benchmark_picker_and_data()

# Align benchmark to the stream's frequency
returns_df_subset, bench_aligned = RS.align_to_period(
    returns_df[[stream]], bench_returns_daily, freq
)

r = returns_df_subset[stream].dropna()
if r.empty:
    st.error(f"Selected stream '{stream}' has no usable data.")
    st.stop()

st.markdown(f"### {stream}")
st.caption(
    f"{len(r):,} observations · {RS.FREQ_LABEL.get(freq, freq).lower()} frequency · "
    f"{r.index.min():%Y-%m-%d} to {r.index.max():%Y-%m-%d} · benchmark: **{bench_name}**"
)

# ─── Tabs ────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "Summary", "Equity curve & drawdown", "Distribution", "Rolling stats",
    "Benchmark comparison", "Crisis windows",
])

# ─── Tab 1: Summary ──────────────────────────────────────────────────────────
with tabs[0]:
    stats = RS.summary_stats(r, freq=freq, rf=rf,
                                bench=bench_aligned if not bench_aligned.empty else None)

    def card(col, label, val, fmt="{:.2%}"):
        if val is None or pd.isna(val):
            col.metric(label, "—")
        else:
            col.metric(label, fmt.format(val))

    st.markdown("##### Return")
    row = st.columns(5)
    card(row[0], "Total return", stats["total_return"])
    card(row[1], "CAGR", stats["cagr"])
    card(row[2], "Annualized return", stats["ann_return"])
    card(row[3], "Best period", stats["best_period"])
    card(row[4], "Worst period", stats["worst_period"])

    st.markdown("##### Risk")
    row = st.columns(5)
    card(row[0], "Annualized vol", stats["ann_vol"])
    card(row[1], "Max drawdown", stats["max_drawdown"])
    card(row[2], "Skew", stats["skew"], fmt="{:.3f}")
    card(row[3], "Excess kurtosis", stats["kurtosis"], fmt="{:.3f}")
    card(row[4], "Hit rate", stats["hit_rate"])

    st.markdown("##### Risk-adjusted")
    row = st.columns(4)
    card(row[0], "Sharpe", stats["sharpe"], fmt="{:.2f}")
    card(row[1], "Sortino", stats["sortino"], fmt="{:.2f}")
    card(row[2], "Calmar", stats["calmar"], fmt="{:.2f}")
    card(row[3], "Periods", stats["n_obs"], fmt="{:.0f}")

    if "alpha" in stats and not bench_aligned.empty:
        st.markdown(f"##### Relative to {bench_name}")
        row = st.columns(5)
        card(row[0], "Alpha (annual)", stats["alpha"])
        card(row[1], "Beta", stats["beta"], fmt="{:.3f}")
        card(row[2], "R²", stats["r_squared"], fmt="{:.3f}")
        card(row[3], "Tracking error", stats["tracking_error"])
        card(row[4], "Info ratio", stats["info_ratio"], fmt="{:.2f}")

        row = st.columns(3)
        card(row[0], "Active return", stats["active_return"])
        card(row[1], "Up capture", stats["up_capture"], fmt="{:.2f}")
        card(row[2], "Down capture", stats["down_capture"], fmt="{:.2f}")

# ─── Tab 2: Equity curve & drawdown ──────────────────────────────────────────
with tabs[1]:
    cum = (1 + r).cumprod()
    if not bench_aligned.empty:
        bench_aligned_clean = bench_aligned.dropna().reindex(r.index, method="nearest",
                                                                 tolerance=pd.Timedelta(days=15))
        bench_cum = (1 + bench_aligned_clean.fillna(0)).cumprod()
        df_plot = pd.DataFrame({stream: cum, bench_name: bench_cum})
    else:
        df_plot = pd.DataFrame({stream: cum})

    fig = px.line(df_plot, title="Cumulative growth of $1")
    fig.update_layout(height=460, hovermode="x unified")
    st.plotly_chart(fig, width="stretch")

    # Drawdown
    dd = RS.drawdown_series(r)
    fig = px.area(dd, title=f"{stream} — drawdown")
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(showlegend=False, height=320)
    st.plotly_chart(fig, width="stretch")

    # Excess return vs benchmark
    if not bench_aligned.empty:
        active = (r - bench_aligned_clean).dropna()
        active_cum = (1 + active).cumprod() - 1
        fig = px.line(active_cum, title=f"Cumulative active return vs {bench_name}")
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(showlegend=False, height=320)
        fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
        st.plotly_chart(fig, width="stretch")

# ─── Tab 3: Distribution ─────────────────────────────────────────────────────
with tabs[2]:
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(r, nbins=40, marginal="box",
                            title="Return distribution",
                            labels={"value": "Period return"})
        fig.update_xaxes(tickformat=".1%")
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, width="stretch")
    with c2:
        qq = sps.probplot(r.dropna(), dist="norm")
        fig = go.Figure()
        fig.add_scatter(x=qq[0][0], y=qq[0][1], mode="markers", name="Empirical",
                          marker=dict(size=6, opacity=0.7))
        fig.add_scatter(x=qq[0][0], y=qq[1][0] * qq[0][0] + qq[1][1],
                          mode="lines", name="Normal fit", line=dict(color="#FF8200"))
        fig.update_layout(title="Q-Q plot vs normal distribution",
                            xaxis_title="Theoretical quantiles",
                            yaxis_title="Sample quantiles", height=400)
        st.plotly_chart(fig, width="stretch")

    # Tail metrics
    st.markdown("##### Percentile profile")
    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    pct_data = {f"P{p}": [float(np.percentile(r, p))] for p in pcts}
    pct_df = pd.DataFrame(pct_data)
    for c in pct_df.columns:
        pct_df[c] = pct_df[c].apply(lambda x: f"{x:+.2%}")
    st.dataframe(pct_df, hide_index=True, width="stretch")

# ─── Tab 4: Rolling stats ────────────────────────────────────────────────────
with tabs[3]:
    ppy = RS.periods_per_year(freq)
    default_window = min(max(ppy, 6), len(r) // 3) if ppy > 1 else 6
    window = st.slider(
        "Rolling window (periods)",
        min_value=3, max_value=max(6, len(r) // 2),
        value=default_window, step=1,
        help=f"Number of {RS.FREQ_LABEL.get(freq, freq).lower()} periods in the rolling window",
    )
    roll = RS.rolling_stats(r, window, freq=freq, rf=rf)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(roll["return"], title=f"Rolling {window}-period annualized return")
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(showlegend=False, height=340)
        st.plotly_chart(fig, width="stretch")
        fig = px.line(roll["sharpe"], title=f"Rolling {window}-period Sharpe")
        fig.update_layout(showlegend=False, height=340)
        fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig = px.line(roll["vol"], title=f"Rolling {window}-period annualized vol")
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(showlegend=False, height=340)
        st.plotly_chart(fig, width="stretch")

        if not bench_aligned.empty:
            bench_clean = bench_aligned.dropna()
            common = r.index.intersection(bench_clean.index)
            if len(common) >= window:
                roll_corr = r.loc[common].rolling(window).corr(bench_clean.loc[common])
                fig = px.line(roll_corr, title=f"Rolling {window}-period correlation with {bench_name}")
                fig.update_layout(showlegend=False, height=340)
                fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
                st.plotly_chart(fig, width="stretch")

# ─── Tab 5: Benchmark comparison ─────────────────────────────────────────────
with tabs[4]:
    if bench_aligned.empty:
        st.info(f"Benchmark data for {bench_name} couldn't be aligned to your stream's dates.")
    else:
        bench_clean = bench_aligned.dropna()
        common = r.index.intersection(bench_clean.index)
        r_aligned = r.loc[common]
        b_aligned = bench_clean.loc[common]

        # Scatter plot of returns
        scatter_df = pd.DataFrame({stream: r_aligned, bench_name: b_aligned})
        slope, intercept, r_val, p_val, _ = sps.linregress(b_aligned, r_aligned)
        fit_x = np.linspace(b_aligned.min(), b_aligned.max(), 100)
        fit_y = intercept + slope * fit_x
        fig = go.Figure()
        fig.add_scatter(x=b_aligned, y=r_aligned, mode="markers",
                          marker=dict(size=8, opacity=0.6),
                          name=f"{stream} returns")
        fig.add_scatter(x=fit_x, y=fit_y, mode="lines",
                          line=dict(color="#FF8200", width=2),
                          name=f"y = {intercept:.4f} + {slope:.3f}·x")
        fig.update_layout(
            title=f"{stream} vs {bench_name} — period returns",
            xaxis_title=f"{bench_name} return", yaxis_title=f"{stream} return",
            xaxis_tickformat=".1%", yaxis_tickformat=".1%", height=480,
        )
        st.plotly_chart(fig, width="stretch")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Beta", f"{slope:.3f}")
        c2.metric("Alpha (per period)", f"{intercept:+.4%}")
        c3.metric("R²", f"{r_val**2:.3f}")
        c4.metric("Periods", f"{len(common)}")

        # Up vs down market table
        up = scatter_df[scatter_df[bench_name] > 0]
        down = scatter_df[scatter_df[bench_name] < 0]
        regime_df = pd.DataFrame({
            "Regime": ["Up periods", "Down periods", "All periods"],
            "Count": [len(up), len(down), len(scatter_df)],
            f"{stream} avg": [up[stream].mean(), down[stream].mean(), scatter_df[stream].mean()],
            f"{bench_name} avg": [up[bench_name].mean(), down[bench_name].mean(), scatter_df[bench_name].mean()],
        })
        regime_df[f"{stream} avg"] = regime_df[f"{stream} avg"].apply(lambda x: f"{x:+.2%}")
        regime_df[f"{bench_name} avg"] = regime_df[f"{bench_name} avg"].apply(lambda x: f"{x:+.2%}")
        st.dataframe(regime_df, hide_index=True, width="stretch")

# ─── Tab 6: Crisis windows ───────────────────────────────────────────────────
with tabs[5]:
    st.caption("Performance during historical crisis windows. Rows omitted if no overlap with the stream's date range.")
    CRISES = {
        "Global Financial Crisis": ("2008-09-01", "2009-03-31"),
        "Eurozone Crisis (Aug 2011)": ("2011-07-22", "2011-10-04"),
        "Aug 2015 China Selloff": ("2015-08-17", "2015-08-25"),
        "Q4 2018 Correction": ("2018-10-01", "2018-12-24"),
        "Feb 2018 Volmageddon": ("2018-01-26", "2018-02-08"),
        "COVID Crash": ("2020-02-19", "2020-03-23"),
        "2022 Bear Market": ("2022-01-03", "2022-10-12"),
        "March 2023 Banking Crisis": ("2023-03-08", "2023-03-15"),
    }
    rows = []
    for name, (start, end) in CRISES.items():
        try:
            window = r.loc[start:end]
        except Exception:
            window = pd.Series(dtype=float)
        if window.empty:
            continue
        period_ret = float((1 + window).prod() - 1)
        n = len(window)
        worst = float(window.min()) if n > 0 else np.nan
        rows.append({
            "Crisis": name,
            "Window": f"{start} → {end}",
            "Periods covered": n,
            f"{stream} return": period_ret,
            "Worst period": worst,
        })
    if rows:
        crisis_df = pd.DataFrame(rows)
        for col in [f"{stream} return", "Worst period"]:
            crisis_df[col] = crisis_df[col].apply(lambda x: f"{x:+.2%}" if pd.notna(x) else "—")
        st.dataframe(crisis_df, hide_index=True, width="stretch")
    else:
        st.info("Your stream's date range doesn't overlap any catalogued crisis windows.")
