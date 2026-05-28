"""Return Stream Analyzer — multi-stream comparison.

Side-by-side stats, overlaid equity curves, drawdown comparison, correlation
matrix, and risk/return scatter for every (or a subset of) uploaded stream.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import rsa as RS
from data import benchmark_picker_and_data
from theme import inject_css

inject_css()

st.title("Stream Comparison")
st.caption("Compare every uploaded return stream side by side.")

if "rsa_returns" not in st.session_state:
    st.warning("No return streams loaded. Go to **Return Stream Analyzer → Home** and upload a CSV first.")
    st.stop()

returns_df: pd.DataFrame = st.session_state["rsa_returns"]
freq: str = st.session_state.get("rsa_frequency", "M")
all_streams: list[str] = st.session_state["rsa_streams"]
rf: float = float(st.session_state.get("rsa_rf", 0.04))

# ─── Selection ───────────────────────────────────────────────────────────────
st.sidebar.header("Selection")
selected = st.sidebar.multiselect(
    "Streams to compare", all_streams, default=all_streams,
)
include_benchmark = st.sidebar.checkbox("Overlay benchmark", value=True)
bench_name, _, bench_returns_daily = benchmark_picker_and_data()

if not selected:
    st.info("Pick at least one stream in the sidebar.")
    st.stop()

# ─── Align benchmark to stream frequency ────────────────────────────────────
returns_subset = returns_df[selected].copy()
_, bench_aligned = RS.align_to_period(returns_subset, bench_returns_daily, freq)
bench_aligned = bench_aligned.dropna() if include_benchmark else pd.Series(dtype=float)

# ─── Cumulative growth overlay ──────────────────────────────────────────────
st.markdown("##### Cumulative growth of $1")
cum = (1 + returns_subset.fillna(0)).cumprod()
if include_benchmark and not bench_aligned.empty:
    # Align benchmark to the same index
    bench_cum = (1 + bench_aligned.reindex(cum.index, method="nearest",
                                             tolerance=pd.Timedelta(days=15)).fillna(0)).cumprod()
    cum[bench_name] = bench_cum
fig = px.line(cum, title=f"Indexed growth — {RS.FREQ_LABEL.get(freq, freq)} periods")
fig.update_layout(height=440, hovermode="x unified")
st.plotly_chart(fig, width="stretch")

# ─── Drawdown overlay ───────────────────────────────────────────────────────
st.markdown("##### Drawdown comparison")
dd_data = {}
for s in selected:
    dd_data[s] = RS.drawdown_series(returns_subset[s].dropna())
dd_df = pd.DataFrame(dd_data)
fig = px.line(dd_df, title="Drawdown by stream")
fig.update_yaxes(tickformat=".0%")
fig.update_layout(height=380, hovermode="x unified")
st.plotly_chart(fig, width="stretch")

# ─── Stats table — every stream + benchmark ─────────────────────────────────
st.markdown("##### Summary statistics")
rows = []
streams_to_show = list(selected)
if include_benchmark and not bench_aligned.empty:
    streams_to_show = streams_to_show + [bench_name]

bench_for_alpha = bench_aligned if not bench_aligned.empty else None
for s in streams_to_show:
    if s == bench_name:
        series = bench_aligned.dropna()
        b = None  # don't compute alpha-against-self
    else:
        series = returns_subset[s].dropna()
        b = bench_for_alpha
    stats = RS.summary_stats(series, freq=freq, rf=rf, bench=b)
    rows.append({
        "Stream": s,
        "Total return": stats["total_return"],
        "CAGR": stats["cagr"],
        "Ann. vol": stats["ann_vol"],
        "Sharpe": stats["sharpe"],
        "Sortino": stats["sortino"],
        "Calmar": stats["calmar"],
        "Max DD": stats["max_drawdown"],
        "Hit rate": stats["hit_rate"],
        "Alpha (vs " + bench_name + ")": stats.get("alpha", np.nan),
        "Beta": stats.get("beta", np.nan),
        "Info ratio": stats.get("info_ratio", np.nan),
    })

stats_df = pd.DataFrame(rows)
disp = stats_df.copy()
for col in ["Total return", "CAGR", "Ann. vol", "Max DD", "Hit rate",
              f"Alpha (vs {bench_name})"]:
    disp[col] = disp[col].apply(lambda x: f"{x:+.2%}" if pd.notna(x) else "—")
for col in ["Sharpe", "Sortino", "Calmar", "Beta", "Info ratio"]:
    disp[col] = disp[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
st.dataframe(disp, hide_index=True, width="stretch")

# ─── Risk/return scatter ────────────────────────────────────────────────────
st.markdown("##### Risk / return scatter")
scatter_df = stats_df[["Stream", "Ann. vol", "CAGR", "Sharpe"]].dropna()
if len(scatter_df) >= 2:
    fig = px.scatter(
        scatter_df, x="Ann. vol", y="CAGR", text="Stream",
        size="Sharpe".__class__.__name__ if False else None,
        title="Annualized return vs annualized volatility",
        labels={"Ann. vol": "Annualized volatility", "CAGR": "CAGR"},
    )
    fig.update_traces(textposition="top center", marker=dict(size=14))
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(height=480, showlegend=False)
    st.plotly_chart(fig, width="stretch")

# ─── Correlation matrix ──────────────────────────────────────────────────────
if len(selected) >= 2 or (include_benchmark and len(selected) >= 1):
    st.markdown("##### Correlation matrix (period returns)")
    corr_data = returns_subset.copy()
    if include_benchmark and not bench_aligned.empty:
        corr_data[bench_name] = bench_aligned.reindex(corr_data.index, method="nearest",
                                                          tolerance=pd.Timedelta(days=15))
    corr = corr_data.corr()
    fig = px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1,
                     color_continuous_scale="RdBu_r", aspect="auto")
    fig.update_layout(height=max(360, 50 * len(corr)))
    st.plotly_chart(fig, width="stretch")

# ─── Rolling rank chart ─────────────────────────────────────────────────────
if len(selected) >= 2:
    st.markdown("##### Rolling 12-period CAGR rank")
    ppy = RS.periods_per_year(freq)
    window = min(max(ppy, 6), len(returns_subset) // 2)
    rolling_ann = (returns_subset.rolling(window).mean() * ppy).dropna()
    if not rolling_ann.empty:
        ranks = rolling_ann.rank(axis=1, ascending=False)
        fig = px.line(ranks, title=f"Stream ranking by rolling {window}-period annualized return")
        fig.update_yaxes(autorange="reversed", title="Rank (1 = best)")
        fig.update_layout(height=380, hovermode="x unified")
        st.plotly_chart(fig, width="stretch")
