"""Correlation analysis: matrices, rolling correlations, clustering."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st

from data import benchmark_picker_and_data, require_portfolio
from theme import inject_css
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

inject_css()
st.title("Correlation & Co-movement Analysis")
st.caption("Pairwise correlations, rolling co-movement, and hierarchical clustering of return patterns.")

tickers, weights, prices, returns, _, _, rf = require_portfolio()
bench_name, bench_prices, bench_returns = benchmark_picker_and_data()

method = st.sidebar.radio("Correlation method", ["pearson", "spearman", "kendall"], index=0)
corr = returns.corr(method=method)

tab1, tab2, tab3 = st.tabs(["Correlation matrix", "Rolling pair correlations", "Hierarchical clustering"])

# ─── Static correlation matrix ────────────────────────────────────────────────
with tab1:
    st.subheader(f"{method.title()} correlation matrix")
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                     zmin=-1, zmax=1, aspect="auto")
    fig.update_layout(height=600)
    st.plotly_chart(fig, width="stretch")

    # Summary stats
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean correlation", f"{upper.stack().mean():.3f}")
    c2.metric("Median correlation", f"{upper.stack().median():.3f}")
    c3.metric("Max pair", f"{upper.stack().max():.3f}")
    c4.metric("Min pair", f"{upper.stack().min():.3f}")

    # Top correlated pairs (rename MultiIndex levels explicitly to avoid collision)
    st.subheader("Highest correlated pairs")
    stacked = upper.stack()
    stacked.index.names = ["a", "b"]
    stacked.name = "correlation"
    pairs = stacked.reset_index().sort_values("correlation", ascending=False).head(15)
    st.dataframe(pairs, hide_index=True, width="stretch")

# ─── Rolling correlations ─────────────────────────────────────────────────────
with tab2:
    st.subheader("Rolling correlation between pairs")
    window = st.slider("Rolling window (days)", 21, 252, 63, step=21)
    pair1 = st.selectbox("Asset A", tickers, index=0)
    pair2 = st.selectbox("Asset B", tickers, index=min(1, len(tickers) - 1))

    if pair1 != pair2:
        rolling_corr = returns[pair1].rolling(window).corr(returns[pair2])
        fig = px.line(rolling_corr,
                        title=f"{window}-day rolling correlation: {pair1} vs {pair2}")
        fig.update_layout(showlegend=False)
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, width="stretch")

    # Average rolling correlation across all pairs (sub-sampled for performance)
    st.subheader("Average pairwise correlation over time")
    n_assets = returns.shape[1]
    if n_assets >= 2 and len(returns) >= window:
        n_pairs = n_assets * (n_assets - 1) / 2
        idx_set = list(range(window, len(returns) + 1,
                              max(1, (len(returns) - window) // 200)))
        records = []
        for i in idx_set:
            c = returns.iloc[i - window:i].corr().values
            # Sum upper triangle: (total - diagonal) / 2
            upper_sum = (c.sum() - np.trace(c)) / 2
            records.append({"date": returns.index[i - 1],
                              "mean_corr": upper_sum / n_pairs})
        if records:
            df = pd.DataFrame(records)
            fig = px.line(df, x="date", y="mean_corr",
                            title=f"Rolling mean of all pairwise correlations ({window}d)")
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig, width="stretch")

# ─── Hierarchical clustering ──────────────────────────────────────────────────
with tab3:
    st.subheader("Hierarchical clustering by return co-movement")
    st.caption("Distance = 1 - |correlation|. Tickers in the same branch behave similarly.")
    link_method = st.selectbox("Linkage", ["average", "single", "complete", "ward"], index=0)
    n_clusters = st.slider("Number of clusters to highlight", 2, max(2, len(tickers) - 1), min(4, len(tickers) - 1))

    # Build distance matrix. .values can be a read-only view in newer
    # pandas — copy() guarantees a writable buffer for fill_diagonal.
    dist_arr = (1 - corr.abs()).to_numpy(copy=True)
    np.fill_diagonal(dist_arr, 0)
    condensed = squareform(dist_arr, checks=False)
    Z = linkage(condensed, method=link_method)

    # Dendrogram
    fig = ff.create_dendrogram(corr.values, labels=corr.columns.tolist(),
                                  linkagefun=lambda x: linkage(x, method=link_method))
    fig.update_layout(title=f"Dendrogram ({link_method} linkage on 1-|ρ| distance)",
                        height=500)
    st.plotly_chart(fig, width="stretch")

    # Cluster assignments
    clusters = fcluster(Z, t=n_clusters, criterion="maxclust")
    cluster_df = pd.DataFrame({"ticker": corr.columns, "cluster": clusters}).sort_values("cluster")
    st.subheader(f"Cluster assignments (k = {n_clusters})")
    st.dataframe(cluster_df, hide_index=True, width="stretch")

    # Reordered correlation matrix by cluster
    order = cluster_df["ticker"].tolist()
    reord = corr.loc[order, order]
    fig = px.imshow(reord, text_auto=".2f", color_continuous_scale="RdBu_r",
                     zmin=-1, zmax=1, aspect="auto",
                     title="Correlation matrix reordered by cluster")
    fig.update_layout(height=600)
    st.plotly_chart(fig, width="stretch")
