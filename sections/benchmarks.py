"""Major Benchmarks — live index/ETF tracker with overlay comparison.

Pulls live data from Yahoo Finance for the catalog of major equity, fixed-income,
and real-asset benchmarks (MSCI ACWI, S&P 500, MSCI EM, MSCI ACWI ex-US, etc.).
Cached for 30 minutes; refreshes automatically when stale.

Pick any subset and the page renders an overlay chart (rebased to 100), a stats
table (CAGR, vol, Sharpe, max DD, YTD), and a correlation matrix. Download the
underlying return series for use elsewhere.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from data import MAJOR_BENCHMARKS, fetch_benchmark_prices
from theme import inject_css

inject_css()

st.title("Major Benchmarks")
st.caption(
    "Live equity, fixed-income, and real-asset indices — pulled from Yahoo Finance "
    "every 30 minutes. Pick any combination to overlay and compare. iShares ETFs are "
    "used as the investable proxy for MSCI indices."
)

# ─── Catalog table with categories ───────────────────────────────────────────
catalog_df = pd.DataFrame([
    {"Name": name, "Ticker": meta["ticker"], "Category": meta["category"],
      "Asset class": meta["asset_class"], "Description": meta["description"]}
    for name, meta in MAJOR_BENCHMARKS.items()
])

with st.expander(f"Benchmark catalog — {len(catalog_df)} indices available",
                  expanded=False):
    st.dataframe(catalog_df, hide_index=True, width="stretch")

st.markdown("---")

# ─── Selection ────────────────────────────────────────────────────────────────
DEFAULT_SELECTION = ["MSCI ACWI", "S&P 500", "MSCI ACWI ex-US", "MSCI EM"]

c1, c2 = st.columns([3, 1])
with c1:
    selected = st.multiselect(
        "Benchmarks to compare",
        options=list(MAJOR_BENCHMARKS.keys()),
        default=DEFAULT_SELECTION,
        help="Pick any combination. The chart and stats below update live.",
    )
with c2:
    period = st.selectbox(
        "Lookback", ["1mo", "3mo", "6mo", "ytd", "1y", "2y", "5y", "10y", "max"],
        index=6,
    )

if not selected:
    st.info("Pick at least one benchmark above to see the overlay and stats.")
    st.stop()

# ─── Fetch ───────────────────────────────────────────────────────────────────
with st.spinner(f"Pulling live data for {len(selected)} benchmarks..."):
    prices = fetch_benchmark_prices(tuple(selected), period=period)

if prices.empty:
    st.error("No data returned. Check your internet connection or try a different lookback.")
    st.stop()

prices = prices.dropna(how="all")
returns = prices.pct_change().dropna(how="all")

# ─── Headline tiles per benchmark (latest level + key changes) ───────────────
st.markdown("##### Live levels")
cols = st.columns(max(1, min(len(selected), 5)))
for i, name in enumerate(selected):
    if name not in prices.columns:
        continue
    series = prices[name].dropna()
    if series.empty:
        continue
    latest = float(series.iloc[-1])
    # Compute YTD return
    ytd_start = series[series.index.year == series.index[-1].year]
    ytd_ret = float(series.iloc[-1] / ytd_start.iloc[0] - 1) if len(ytd_start) > 0 else np.nan
    # 1-day change
    one_day = float(series.iloc[-1] / series.iloc[-2] - 1) if len(series) >= 2 else np.nan
    col = cols[i % len(cols)]
    col.metric(
        name,
        f"{latest:,.2f}",
        f"{one_day:+.2%} (1D)" if not np.isnan(one_day) else None,
    )

st.caption(f"As of **{prices.index[-1]:%Y-%m-%d}** — data refreshes every 30 minutes.")

# ─── Overlay chart (rebased to 100) ─────────────────────────────────────────
st.markdown("---")
st.markdown("##### Cumulative growth, rebased to 100")
first = prices.bfill().iloc[0]
rebased = prices.div(first).mul(100)

fig = px.line(
    rebased, title=f"{period.upper()} performance — rebased to 100 at start",
    labels={"value": "Indexed level (start = 100)", "index": "Date", "variable": "Benchmark"},
)
fig.update_layout(height=480, hovermode="x unified")
fig.add_hline(y=100, line_dash="dash", line_color="#94a3b8")
st.plotly_chart(fig, width="stretch")

# Cumulative return percentage view
cum_ret = rebased.div(100) - 1
fig = px.line(
    cum_ret, title=f"{period.upper()} cumulative return (%)",
    labels={"value": "Cumulative return", "variable": "Benchmark"},
)
fig.update_yaxes(tickformat=".0%")
fig.update_layout(height=380, hovermode="x unified", showlegend=True)
fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
st.plotly_chart(fig, width="stretch")

# ─── Summary statistics ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("##### Summary statistics")

stats_rows = []
for name in selected:
    if name not in returns.columns:
        continue
    r = returns[name].dropna()
    p = prices[name].dropna()
    if len(r) < 5 or len(p) < 5:
        continue
    n_years = len(r) / 252
    cum = float((1 + r).prod() - 1)
    cagr = float((1 + cum) ** (1 / n_years) - 1) if n_years > 0 else np.nan
    ann_vol = float(r.std() * np.sqrt(252))
    sharpe = float((r.mean() * 252) / ann_vol) if ann_vol > 0 else np.nan
    cum_curve = (1 + r).cumprod()
    max_dd = float((cum_curve / cum_curve.cummax() - 1).min())
    ytd_start = p[p.index.year == p.index[-1].year]
    ytd_ret = float(p.iloc[-1] / ytd_start.iloc[0] - 1) if len(ytd_start) > 0 else np.nan
    best = float(r.max())
    worst = float(r.min())
    stats_rows.append({
        "Benchmark": name,
        "Ticker": MAJOR_BENCHMARKS[name]["ticker"],
        "Total return": cum,
        "CAGR": cagr,
        "Ann. vol": ann_vol,
        "Sharpe": sharpe,
        "Max drawdown": max_dd,
        "YTD return": ytd_ret,
        "Best day": best,
        "Worst day": worst,
    })

stats_df = pd.DataFrame(stats_rows)
if not stats_df.empty:
    disp = stats_df.copy()
    for col in ["Total return", "CAGR", "Ann. vol", "Max drawdown",
                  "YTD return", "Best day", "Worst day"]:
        disp[col] = disp[col].apply(lambda x: f"{x:+.2%}" if pd.notna(x) else "—")
    disp["Sharpe"] = disp["Sharpe"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
    st.dataframe(disp, hide_index=True, width="stretch")

# ─── Correlation matrix ────────────────────────────────────────────────────
if len(selected) >= 2:
    st.markdown("---")
    st.markdown("##### Return correlation matrix")
    corr = returns.corr()
    fig = px.imshow(
        corr, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r",
        aspect="auto",
    )
    fig.update_layout(height=max(360, 50 * len(selected)),
                       title=f"{period.upper()} daily return correlations")
    st.plotly_chart(fig, width="stretch")

# ─── Downloads ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("##### Export")
c1, c2 = st.columns(2)
with c1:
    st.download_button(
        "Download prices (CSV)",
        data=prices.to_csv().encode("utf-8"),
        file_name=f"benchmark_prices_{period}.csv",
        mime="text/csv",
        width="stretch",
    )
with c2:
    st.download_button(
        "Download daily returns (CSV)",
        data=returns.to_csv().encode("utf-8"),
        file_name=f"benchmark_returns_{period}.csv",
        mime="text/csv",
        width="stretch",
    )

st.caption(
    "Data via Yahoo Finance (yfinance). MSCI indices use the corresponding "
    "iShares ETF as the investable proxy (ACWI, ACWX, EEM, EFA, etc. — all "
    "track their MSCI parent within a few basis points after fees)."
)
