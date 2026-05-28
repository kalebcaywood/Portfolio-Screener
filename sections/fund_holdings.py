"""Fund Holdings Analyzer — ingests large Bloomberg-format multi-fund books.

Built for institutional holdings files (thousands of positions across multiple
funds) where Market Value is already provided. Geographic and concentration
breakdowns run on the full file instantly (no price fetching). Sector breakdown
fetches GICS sectors on demand, scoped to the selected fund.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import bloomberg as BB
from data import fetch_sector
from theme import COLORWAY, inject_css

inject_css()
st.title("Fund Holdings Analyzer")
st.caption(
    "Upload a Bloomberg-format multi-fund holdings file (Fund, Ticker, Quantity, "
    "Market Value). Geographic and concentration breakdowns run on the full file; "
    "sector lookups are fetched on demand per fund."
)

# ─── Upload ──────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload holdings CSV", type=["csv"],
    help="Bloomberg tickers (e.g. 'AAPL US', '2670 JP') are auto-converted. "
         "Recognized columns: Fund, Ticker/Security, Quantity, Market Value.",
)

if uploaded is None:
    st.info(
        "Upload a holdings file to begin. The analyzer handles Bloomberg ticker "
        "formats and tags every position with country / region automatically."
    )
    st.stop()

with st.spinner("Parsing holdings file..."):
    parsed = BB.load_bloomberg_csv(uploaded)

for w in parsed["warnings"]:
    st.warning(w)
for e in parsed["errors"]:
    st.error(e)
if parsed["df"] is None:
    st.stop()

df_all = parsed["df"]
funds = sorted(df_all["fund"].unique())

# ─── Fund selector ───────────────────────────────────────────────────────────
st.sidebar.header("View")
fund_choice = st.sidebar.selectbox(
    "Fund", ["All funds (aggregate)"] + funds,
    help="Pick a single fund or view the combined book.",
)
if fund_choice == "All funds (aggregate)":
    df = df_all.copy()
    scope_label = "All funds"
else:
    df = df_all[df_all["fund"] == fund_choice].copy()
    scope_label = fund_choice

total_mv = float(df["market_value"].sum())

# ─── Headline metrics ────────────────────────────────────────────────────────
c = st.columns(5)
c[0].metric("Total market value", f"${total_mv:,.0f}")
c[1].metric("Positions", f"{len(df):,}")
c[2].metric("Unique tickers", f"{df['yahoo'].nunique():,}")
c[3].metric("Countries", df[df["country"] != "Unknown"]["country"].nunique())
c[4].metric("Funds in view", df["fund"].nunique())

st.markdown("---")

(tab_geo, tab_conc, tab_sector, tab_holdings,
 tab_vs_fund, tab_vs_bench) = st.tabs([
    "Geographic breakdown", "Concentration heatmap", "Sector breakdown",
    "Top holdings", "Fund vs Fund", "Fund vs Benchmark",
])

# ══════════════════════════════════════════════════════════════════════════════
# Tab 1: Geographic breakdown
# ══════════════════════════════════════════════════════════════════════════════
with tab_geo:
    st.subheader(f"Geographic exposure — {scope_label}")

    by_country = (df.groupby("country")["market_value"].sum()
                    .sort_values(ascending=False).reset_index())
    by_country["pct"] = by_country["market_value"] / total_mv
    by_region = (df.groupby("region")["market_value"].sum()
                   .sort_values(ascending=False).reset_index())
    by_region["pct"] = by_region["market_value"] / total_mv

    c1, c2 = st.columns([3, 2])
    with c1:
        fig = px.bar(by_country.head(20).sort_values("market_value"),
                      x="market_value", y="country", orientation="h",
                      title="Market value by country (top 20)",
                      labels={"market_value": "Market value ($)", "country": ""})
        fig.update_layout(height=max(400, 24 * min(len(by_country), 20)))
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig = px.pie(by_region, names="region", values="market_value", hole=0.45,
                      title="By region")
        st.plotly_chart(fig, width="stretch")

    # Treemap: region → country
    st.subheader("Region → country treemap")
    tm = df.groupby(["region", "country"])["market_value"].sum().reset_index()
    tm = tm[tm["market_value"] > 0]
    fig = px.treemap(tm, path=["region", "country"], values="market_value",
                      color="market_value", color_continuous_scale="Oranges",
                      title="Exposure hierarchy (size = market value)")
    fig.update_layout(height=520, margin=dict(t=50, b=20, l=20, r=20))
    st.plotly_chart(fig, width="stretch")

    # Tables
    st.subheader("Detail")
    c1, c2 = st.columns(2)
    with c1:
        disp = by_country.copy()
        disp["market_value"] = disp["market_value"].apply(lambda x: f"${x:,.0f}")
        disp["pct"] = disp["pct"].apply(lambda x: f"{x:.2%}")
        disp.columns = ["Country", "Market value", "% of book"]
        st.dataframe(disp, hide_index=True, width="stretch")
    with c2:
        disp = by_region.copy()
        disp["market_value"] = disp["market_value"].apply(lambda x: f"${x:,.0f}")
        disp["pct"] = disp["pct"].apply(lambda x: f"{x:.2%}")
        disp.columns = ["Region", "Market value", "% of book"]
        st.dataframe(disp, hide_index=True, width="stretch")

# ══════════════════════════════════════════════════════════════════════════════
# Tab 2: Concentration heatmap
# ══════════════════════════════════════════════════════════════════════════════
with tab_conc:
    st.subheader("Concentration heatmap")
    st.caption(
        "Each cell shows the percentage of a fund's market value allocated to a "
        "geography. Rows are funds, columns are regions or countries."
    )

    granularity = st.radio("Column granularity", ["Region", "Country"],
                            horizontal=True)
    geo_col = "region" if granularity == "Region" else "country"

    # Build fund × geography matrix of % allocation (row-normalized)
    pivot = df_all.pivot_table(index="fund", columns=geo_col,
                                values="market_value", aggfunc="sum", fill_value=0)
    # Row-normalize to % of each fund
    row_tot = pivot.sum(axis=1).replace(0, np.nan)
    pct_pivot = pivot.div(row_tot, axis=0)

    if granularity == "Country":
        # Keep only the most material columns to stay readable
        col_tot = pct_pivot.sum(axis=0).sort_values(ascending=False)
        keep = col_tot.head(20).index.tolist()
        pct_pivot = pct_pivot[keep]

    # Order columns by total exposure
    pct_pivot = pct_pivot[pct_pivot.sum().sort_values(ascending=False).index]

    fig = px.imshow(
        pct_pivot,
        labels=dict(x=granularity, y="Fund", color="% of fund"),
        color_continuous_scale="Oranges",
        aspect="auto",
        text_auto=".0%",
    )
    fig.update_layout(height=max(400, 38 * len(pct_pivot)),
                       title=f"Fund × {granularity} concentration (% of each fund)")
    fig.update_xaxes(tickangle=-40)
    st.plotly_chart(fig, width="stretch")

    # Position-level concentration within the selected scope
    st.markdown("---")
    st.subheader(f"Position concentration — {scope_label}")

    pos = (df.groupby(["yahoo", "country"])["market_value"].sum()
             .sort_values(ascending=False).reset_index())
    pos["pct"] = pos["market_value"] / total_mv
    pos_sorted = pos["pct"].sort_values(ascending=False).reset_index(drop=True)

    hhi = float((pos["pct"] ** 2).sum())
    eff_n = (1 / hhi) if hhi > 0 else np.nan
    top10 = float(pos["pct"].head(10).sum())
    top25 = float(pos["pct"].head(25).sum())

    m = st.columns(4)
    m[0].metric("Largest position", f"{pos['pct'].iloc[0]:.2%}" if len(pos) else "—")
    m[1].metric("Top 10 weight", f"{top10:.1%}")
    m[2].metric("Top 25 weight", f"{top25:.1%}")
    m[3].metric("Effective # names", f"{eff_n:,.0f}" if not np.isnan(eff_n) else "—",
                 help="1 / Herfindahl index — diversification-adjusted position count")

    # Top-position treemap colored by country
    top_n = min(50, len(pos))
    tm2 = pos.head(top_n).copy()
    fig = px.treemap(tm2, path=["country", "yahoo"], values="market_value",
                      color="country", color_discrete_sequence=COLORWAY,
                      title=f"Top {top_n} positions (size = market value, color = country)")
    fig.update_layout(height=520, margin=dict(t=50, b=20, l=20, r=20))
    st.plotly_chart(fig, width="stretch")

# ══════════════════════════════════════════════════════════════════════════════
# Tab 3: Sector breakdown (on-demand fetch)
# ══════════════════════════════════════════════════════════════════════════════
with tab_sector:
    st.subheader(f"Sector breakdown — {scope_label}")
    st.caption(
        "Sectors are looked up per ticker from Yahoo Finance and cached for 24h. "
        "This is scoped to your selected fund to keep it fast — pick a single fund "
        "in the sidebar before running. Foreign tickers may return blank sectors."
    )

    n_unique = df["yahoo"].nunique()
    if fund_choice == "All funds (aggregate)" and n_unique > 800:
        st.warning(
            f"This view has {n_unique:,} unique tickers — too many to fetch sectors "
            "interactively. Select a single fund in the sidebar first."
        )
    else:
        max_fetch = st.slider("Max tickers to look up (by market value)",
                               25, min(800, n_unique), min(200, n_unique), step=25)
        if st.button("Fetch sectors", type="primary"):
            # Rank unique tickers by market value, take top N
            ranked = (df.groupby("yahoo")["market_value"].sum()
                        .sort_values(ascending=False).head(max_fetch))
            tickers_to_fetch = ranked.index.tolist()

            progress = st.progress(0.0)
            status = st.empty()
            sector_map: dict[str, str] = {}
            for i, t in enumerate(tickers_to_fetch):
                sector_map[t] = fetch_sector(t) or "Unknown"
                if i % 5 == 0 or i == len(tickers_to_fetch) - 1:
                    progress.progress((i + 1) / len(tickers_to_fetch))
                    status.text(f"Fetched {i + 1}/{len(tickers_to_fetch)} — {t}")
            progress.empty()
            status.empty()

            scoped = df[df["yahoo"].isin(tickers_to_fetch)].copy()
            scoped["sector"] = scoped["yahoo"].map(sector_map).fillna("Unknown")
            covered_mv = float(scoped["market_value"].sum())

            st.session_state["fh_sectors"] = scoped
            st.session_state["fh_sector_coverage"] = covered_mv / total_mv

        if "fh_sectors" in st.session_state:
            scoped = st.session_state["fh_sectors"]
            coverage = st.session_state.get("fh_sector_coverage", 0)
            st.caption(f"Sector data covers **{coverage:.1%}** of this view's market value.")

            by_sector = (scoped.groupby("sector")["market_value"].sum()
                           .sort_values(ascending=False).reset_index())
            by_sector["pct"] = by_sector["market_value"] / scoped["market_value"].sum()

            c1, c2 = st.columns([2, 3])
            with c1:
                fig = px.pie(by_sector, names="sector", values="market_value",
                              hole=0.45, title="By sector")
                st.plotly_chart(fig, width="stretch")
            with c2:
                fig = px.bar(by_sector.sort_values("market_value"),
                              x="market_value", y="sector", orientation="h",
                              title="Sector exposure",
                              labels={"market_value": "Market value ($)", "sector": ""})
                st.plotly_chart(fig, width="stretch")

            # Sector × region cross-tab heatmap
            if "region" in scoped.columns:
                cross = scoped.pivot_table(index="sector", columns="region",
                                            values="market_value", aggfunc="sum",
                                            fill_value=0)
                cross_pct = cross.div(cross.sum().sum())
                fig = px.imshow(cross_pct, color_continuous_scale="Oranges",
                                 aspect="auto", text_auto=".1%",
                                 labels=dict(x="Region", y="Sector", color="% of book"),
                                 title="Sector × region concentration")
                fig.update_layout(height=max(350, 34 * len(cross_pct)))
                st.plotly_chart(fig, width="stretch")

            disp = by_sector.copy()
            disp["market_value"] = disp["market_value"].apply(lambda x: f"${x:,.0f}")
            disp["pct"] = disp["pct"].apply(lambda x: f"{x:.2%}")
            disp.columns = ["Sector", "Market value", "% of covered"]
            st.dataframe(disp, hide_index=True, width="stretch")

# ══════════════════════════════════════════════════════════════════════════════
# Tab 4: Top holdings
# ══════════════════════════════════════════════════════════════════════════════
with tab_holdings:
    st.subheader(f"Top holdings — {scope_label}")
    st.caption(
        "Each row is one **unique Yahoo ticker** — duplicates from different "
        "Bloomberg long/short forms (e.g. `NVDA`, `NVDA US`, `NVDA US Equity`) "
        "and across multiple funds are combined into a single line with summed value."
    )

    # Group ONLY by the canonical Yahoo ticker. Aggregate everything else:
    # take the first country/region (deterministic since yahoo → country),
    # sum market value, count distinct source funds and Bloomberg forms.
    holdings = (df.groupby("yahoo", as_index=False)
                  .agg(country=("country", "first"),
                        region=("region", "first"),
                        market_value=("market_value", "sum"),
                        n_funds=("fund", "nunique"),
                        n_forms=("raw_ticker", "nunique"),
                        n_positions=("fund", "size"))
                  .sort_values("market_value", ascending=False)
                  .reset_index(drop=True))
    holdings["pct"] = holdings["market_value"] / total_mv
    holdings["cumulative_pct"] = holdings["pct"].cumsum()

    n_show = st.slider("Show top N", 10, min(200, len(holdings)),
                        min(50, len(holdings)), step=10)

    disp = holdings.head(n_show).copy()
    disp["market_value"] = disp["market_value"].apply(lambda x: f"${x:,.0f}")
    disp["pct"] = disp["pct"].apply(lambda x: f"{x:.2%}")
    disp["cumulative_pct"] = disp["cumulative_pct"].apply(lambda x: f"{x:.1%}")
    disp.columns = ["Ticker", "Country", "Region", "Market value",
                     "# funds", "# Bloomberg forms", "# rows", "% of book", "Cumulative %"]
    st.dataframe(disp, hide_index=True, width="stretch")

    # Cumulative concentration curve
    fig = px.line(holdings.reset_index(), x=holdings.reset_index().index + 1,
                   y="cumulative_pct",
                   title="Cumulative concentration curve",
                   labels={"x": "Number of positions (ranked)", "cumulative_pct": "Cumulative % of book"})
    fig.add_hline(y=0.50, line_dash="dash", line_color="#FF8200",
                    annotation_text="50%")
    fig.add_hline(y=0.80, line_dash="dot", line_color="#b45309",
                    annotation_text="80%")
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, width="stretch")

    # Download cleaned + geo-tagged holdings
    csv_bytes = holdings.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download geo-tagged holdings (CSV)",
        data=csv_bytes,
        file_name=f"holdings_{scope_label.replace(' ', '_')}.csv",
        mime="text/csv",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Tab 5: Fund vs Fund comparison
# ══════════════════════════════════════════════════════════════════════════════
with tab_vs_fund:
    st.subheader("Compare two funds in this book")
    st.caption(
        "Side-by-side exposures, position overlap, and concentration. "
        "Both funds come from the file you uploaded."
    )

    if len(funds) < 2:
        st.info("This book has fewer than two funds — nothing to compare.")
    else:
        cs1, cs2 = st.columns(2)
        with cs1:
            fund_a = st.selectbox("Fund A", funds, index=0, key="fh_compare_a")
        with cs2:
            fund_b = st.selectbox("Fund B", funds,
                                    index=min(1, len(funds) - 1),
                                    key="fh_compare_b")

        if fund_a == fund_b:
            st.warning("Pick two different funds.")
        else:
            df_a = df_all[df_all["fund"] == fund_a].copy()
            df_b = df_all[df_all["fund"] == fund_b].copy()
            mv_a = float(df_a["market_value"].sum())
            mv_b = float(df_b["market_value"].sum())

            # ── Headline metrics side by side ────────────────────────────
            st.markdown("---")
            m = st.columns(4)
            m[0].metric("Market value",
                          f"${mv_a:,.0f}", f"vs ${mv_b:,.0f}")
            m[1].metric("Positions",
                          f"{len(df_a):,}",
                          f"vs {len(df_b):,} ({len(df_b) - len(df_a):+,})")
            m[2].metric("Unique tickers",
                          f"{df_a['yahoo'].nunique():,}",
                          f"vs {df_b['yahoo'].nunique():,}")
            m[3].metric("Countries",
                          df_a[df_a['country'] != 'Unknown']['country'].nunique(),
                          f"vs {df_b[df_b['country'] != 'Unknown']['country'].nunique()}")

            # ── Region comparison (grouped bar) ──────────────────────────
            st.markdown("---")
            st.markdown("##### Region exposure")
            ra = df_a.groupby("region")["market_value"].sum() / mv_a if mv_a else df_a.groupby("region")["market_value"].sum()
            rb = df_b.groupby("region")["market_value"].sum() / mv_b if mv_b else df_b.groupby("region")["market_value"].sum()
            all_regions = sorted(set(ra.index) | set(rb.index))
            region_df = pd.DataFrame({
                "Region": all_regions,
                fund_a: [float(ra.get(r, 0)) for r in all_regions],
                fund_b: [float(rb.get(r, 0)) for r in all_regions],
            })
            region_df["Active (A − B)"] = region_df[fund_a] - region_df[fund_b]

            long_df = region_df.melt(id_vars="Region", value_vars=[fund_a, fund_b],
                                       var_name="Fund", value_name="Weight")
            fig = px.bar(long_df, x="Region", y="Weight", color="Fund",
                          barmode="group",
                          title=f"Region exposure: {fund_a} vs {fund_b}")
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, width="stretch")

            disp = region_df.copy()
            for c in [fund_a, fund_b, "Active (A − B)"]:
                disp[c] = disp[c].apply(lambda v: f"{v:+.2%}" if v != 0 else "0.00%")
            st.dataframe(disp, hide_index=True, width="stretch")

            # ── Country comparison (top 15) ──────────────────────────────
            st.markdown("##### Country exposure (top 15)")
            ca = df_a.groupby("country")["market_value"].sum() / mv_a if mv_a else df_a.groupby("country")["market_value"].sum()
            cb = df_b.groupby("country")["market_value"].sum() / mv_b if mv_b else df_b.groupby("country")["market_value"].sum()
            all_countries = sorted(set(ca.index) | set(cb.index),
                                     key=lambda x: -(ca.get(x, 0) + cb.get(x, 0)))[:15]
            country_df = pd.DataFrame({
                "Country": all_countries,
                fund_a: [float(ca.get(c, 0)) for c in all_countries],
                fund_b: [float(cb.get(c, 0)) for c in all_countries],
            })
            country_df["Active (A − B)"] = country_df[fund_a] - country_df[fund_b]
            long_df = country_df.melt(id_vars="Country", value_vars=[fund_a, fund_b],
                                        var_name="Fund", value_name="Weight")
            fig = px.bar(long_df, x="Country", y="Weight", color="Fund",
                          barmode="group",
                          title=f"Top-15 country exposure: {fund_a} vs {fund_b}")
            fig.update_yaxes(tickformat=".0%")
            fig.update_xaxes(tickangle=-30)
            st.plotly_chart(fig, width="stretch")

            # ── Position overlap (Jaccard + value overlap) ───────────────
            st.markdown("---")
            st.markdown("##### Position overlap")

            agg_a = df_a.groupby("yahoo", as_index=False).agg(
                mv_a=("market_value", "sum"),
                country=("country", "first"),
            )
            agg_b = df_b.groupby("yahoo", as_index=False).agg(
                mv_b=("market_value", "sum"),
                country=("country", "first"),
            )
            merged = agg_a.merge(agg_b, on="yahoo", how="outer", suffixes=("_a", "_b"))
            merged["country"] = merged["country_a"].fillna(merged["country_b"])
            merged["mv_a"] = merged["mv_a"].fillna(0)
            merged["mv_b"] = merged["mv_b"].fillna(0)

            shared = merged[(merged["mv_a"] > 0) & (merged["mv_b"] > 0)]
            only_a = merged[(merged["mv_a"] > 0) & (merged["mv_b"] == 0)]
            only_b = merged[(merged["mv_a"] == 0) & (merged["mv_b"] > 0)]
            n_union = len(merged)
            jaccard = len(shared) / n_union if n_union > 0 else 0
            shared_val_a = float(shared["mv_a"].sum())
            shared_val_b = float(shared["mv_b"].sum())
            shared_pct_a = shared_val_a / mv_a if mv_a else 0
            shared_pct_b = shared_val_b / mv_b if mv_b else 0

            ms = st.columns(5)
            ms[0].metric("Tickers in both", f"{len(shared):,}")
            ms[1].metric(f"Only in {fund_a}", f"{len(only_a):,}")
            ms[2].metric(f"Only in {fund_b}", f"{len(only_b):,}")
            ms[3].metric("Jaccard overlap", f"{jaccard:.1%}",
                           help="# shared tickers / # unique tickers across both funds")
            ms[4].metric("Shared value",
                           f"{shared_pct_a:.1%} of A · {shared_pct_b:.1%} of B")

            # Common holdings table — shared by combined value
            if not shared.empty:
                shared_disp = shared.copy()
                shared_disp["pct_a"] = shared_disp["mv_a"] / mv_a if mv_a else 0
                shared_disp["pct_b"] = shared_disp["mv_b"] / mv_b if mv_b else 0
                shared_disp["combined"] = shared_disp["mv_a"] + shared_disp["mv_b"]
                shared_disp = shared_disp.sort_values("combined", ascending=False).head(25)

                tbl = shared_disp[["yahoo", "country", "mv_a", "pct_a",
                                     "mv_b", "pct_b"]].copy()
                tbl.columns = ["Ticker", "Country",
                                f"{fund_a} $", f"{fund_a} %",
                                f"{fund_b} $", f"{fund_b} %"]
                tbl[f"{fund_a} $"] = tbl[f"{fund_a} $"].apply(lambda x: f"${x:,.0f}")
                tbl[f"{fund_b} $"] = tbl[f"{fund_b} $"].apply(lambda x: f"${x:,.0f}")
                tbl[f"{fund_a} %"] = tbl[f"{fund_a} %"].apply(lambda x: f"{x:.2%}")
                tbl[f"{fund_b} %"] = tbl[f"{fund_b} %"].apply(lambda x: f"{x:.2%}")
                st.markdown(f"**Top 25 shared positions (by combined value)**")
                st.dataframe(tbl, hide_index=True, width="stretch")

            # ── Concentration metrics side by side ──────────────────────
            st.markdown("---")
            st.markdown("##### Concentration metrics")
            def _conc(d):
                abs_w = d.groupby("yahoo")["market_value"].sum().abs()
                if abs_w.sum() == 0:
                    return dict(hhi=np.nan, eff_n=np.nan, top1=np.nan, top10=np.nan)
                n = abs_w / abs_w.sum()
                hhi = float((n ** 2).sum())
                return dict(
                    hhi=hhi,
                    eff_n=(1 / hhi) if hhi > 0 else np.nan,
                    top1=float(n.max()),
                    top10=float(n.nlargest(10).sum()),
                )
            ca_conc = _conc(df_a)
            cb_conc = _conc(df_b)
            conc_df = pd.DataFrame({
                "Metric": ["HHI", "Effective # names", "Largest position", "Top 10 weight"],
                fund_a: [
                    f"{ca_conc['hhi']:.3f}", f"{ca_conc['eff_n']:.1f}",
                    f"{ca_conc['top1']:.2%}", f"{ca_conc['top10']:.2%}",
                ],
                fund_b: [
                    f"{cb_conc['hhi']:.3f}", f"{cb_conc['eff_n']:.1f}",
                    f"{cb_conc['top1']:.2%}", f"{cb_conc['top10']:.2%}",
                ],
            })
            st.dataframe(conc_df, hide_index=True, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 6: Fund vs Benchmark
# ══════════════════════════════════════════════════════════════════════════════
with tab_vs_bench:
    st.subheader("Compare a fund's regional exposure to a benchmark")
    st.caption(
        "Compares the fund's actual country/region breakdown (from the uploaded "
        "holdings file) to the benchmark's approximate regional composition. "
        "Benchmark weights are static reference values updated ~annually; useful "
        "for over/underweight directional analysis, not exact tracking."
    )

    # Selectors
    cs1, cs2 = st.columns(2)
    with cs1:
        bench_fund = st.selectbox(
            "Fund to compare", ["All funds (aggregate)"] + funds,
            key="fh_bench_fund",
        )
    with cs2:
        from data import BENCHMARK_REGION_WEIGHTS
        bench_options = list(BENCHMARK_REGION_WEIGHTS.keys())
        bench_pick = st.selectbox(
            "Benchmark", bench_options,
            index=bench_options.index("S&P 500"),
            key="fh_bench_pick",
        )

    if bench_fund == "All funds (aggregate)":
        cmp_df = df_all.copy()
        cmp_label = "All funds"
    else:
        cmp_df = df_all[df_all["fund"] == bench_fund].copy()
        cmp_label = bench_fund

    cmp_mv = float(cmp_df["market_value"].sum())
    if cmp_mv == 0:
        st.warning("Selected fund has no value.")
        st.stop()

    # Fund's region weights
    fund_regions = (cmp_df.groupby("region")["market_value"].sum() / cmp_mv).to_dict()
    bench_regions = BENCHMARK_REGION_WEIGHTS[bench_pick]

    all_regions = sorted(set(fund_regions.keys()) | set(bench_regions.keys()))
    rows = []
    for r in all_regions:
        if r == "Unknown":
            continue
        f_w = float(fund_regions.get(r, 0))
        b_w = float(bench_regions.get(r, 0))
        rows.append({
            "Region": r,
            f"{cmp_label}": f_w,
            f"{bench_pick}": b_w,
            "Active (Fund − Bench)": f_w - b_w,
        })
    cmp_table = pd.DataFrame(rows)

    # Grouped bar chart
    long_df = cmp_table.melt(id_vars=["Region"],
                              value_vars=[cmp_label, bench_pick],
                              var_name="Source", value_name="Weight")
    fig = px.bar(long_df, x="Region", y="Weight", color="Source",
                  barmode="group",
                  title=f"Region exposure: {cmp_label} vs {bench_pick}")
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, width="stretch")

    # Active-weight bar (signed)
    fig = px.bar(
        cmp_table.sort_values("Active (Fund − Bench)"),
        x="Region", y="Active (Fund − Bench)",
        color="Active (Fund − Bench)",
        color_continuous_scale="RdBu_r", color_continuous_midpoint=0,
        title=f"Active region weight: {cmp_label} − {bench_pick} (positive = overweight)",
    )
    fig.update_yaxes(tickformat=".0%")
    fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
    st.plotly_chart(fig, width="stretch")

    # Table
    disp = cmp_table.copy()
    for c in [cmp_label, bench_pick, "Active (Fund − Bench)"]:
        disp[c] = disp[c].apply(lambda v: f"{v:+.2%}" if v != 0 else "0.00%")
    st.dataframe(disp, hide_index=True, width="stretch")

    # ── Summary diff metrics ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("##### Summary")
    active_abs = sum(abs(f - b) for f, b in zip(
        [fund_regions.get(r, 0) for r in all_regions if r != "Unknown"],
        [bench_regions.get(r, 0) for r in all_regions if r != "Unknown"],
    ))
    active_share_region = active_abs / 2  # half-sum-abs convention

    ms = st.columns(4)
    ms[0].metric("Fund market value", f"${cmp_mv:,.0f}")
    ms[1].metric("Region active share", f"{active_share_region:.1%}",
                   help="Half the sum of absolute over/under-weights — 0% = identical regions, 100% = totally disjoint")

    # Find the biggest over/underweight region
    if not cmp_table.empty:
        biggest_over = cmp_table.loc[cmp_table["Active (Fund − Bench)"].idxmax()]
        biggest_under = cmp_table.loc[cmp_table["Active (Fund − Bench)"].idxmin()]
        ms[2].metric(f"Largest overweight: {biggest_over['Region']}",
                       f"{biggest_over['Active (Fund − Bench)']:+.1%}")
        ms[3].metric(f"Largest underweight: {biggest_under['Region']}",
                       f"{biggest_under['Active (Fund − Bench)']:+.1%}")

    st.caption(
        "**Note**: benchmark region weights are static reference figures "
        "(updated approximately annually). For exact, live benchmark holdings, "
        "the iShares ETF holdings file (downloadable from BlackRock) would be "
        "the right source — that's a future enhancement."
    )
