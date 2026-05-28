"""Currency exposure, FX risk decomposition, and global interest rate dashboards."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data import (BOND_ETFS, CURRENCY_TO_FX, FX_PAIRS, RATE_INDICATORS,
                   fetch_currency_map, fetch_prices, require_portfolio)
from theme import inject_css, ut_sidebar_brand

inject_css()
ut_sidebar_brand()
st.title("Currency Risk & Interest Rates")
st.caption("Currency exposure of your holdings, FX-impact attribution on returns, and global rate / FX dashboards.")

tickers, weights, prices, returns, bench_prices, bench_returns, rf = require_portfolio()

# ─── Fetch currency metadata for every holding ───────────────────────────────
with st.spinner("Fetching currency metadata for holdings..."):
    cur_map = fetch_currency_map(tickers)

cur_df = pd.DataFrame(
    [
        {
            "ticker": t,
            "currency": cur_map[t]["currency"],
            "country": cur_map[t]["country"],
            "exchange": cur_map[t]["exchange"],
            "long_name": cur_map[t]["long_name"],
            "weight": float(weights.get(t, 0)),
        }
        for t in tickers
    ]
)

tab_exposure, tab_fx, tab_rates, tab_pairs = st.tabs(
    ["Currency exposure", "FX risk decomposition",
     "Interest rates", "FX pair explorer"]
)

# ──────────────────────────────────────────────────────────────────────────────
# Tab 1: Currency exposure
# ──────────────────────────────────────────────────────────────────────────────
with tab_exposure:
    st.subheader("Holdings by listing currency")
    disp = cur_df.copy()
    disp["weight"] = disp["weight"].apply(lambda x: f"{x:.2%}")
    st.dataframe(disp[["ticker", "long_name", "country", "currency", "exchange", "weight"]],
                  hide_index=True, width="stretch")

    # Signed metrics (for the headline numbers) — meaningful even with shorts
    signed_by_cur = cur_df.groupby("currency", as_index=False)["weight"].sum()
    usd_weight = float(signed_by_cur.loc[signed_by_cur["currency"] == "USD", "weight"].sum())
    net_total = float(cur_df["weight"].sum())
    foreign_weight = net_total - usd_weight

    # Gross-exposure metrics (for pie charts) — required so shorts don't break the viz
    cur_df_abs = cur_df.copy()
    cur_df_abs["abs_weight"] = cur_df_abs["weight"].abs()
    gross_by_cur = cur_df_abs.groupby("currency", as_index=False)["abs_weight"].sum().sort_values("abs_weight", ascending=False)
    gross_by_country = cur_df_abs[cur_df_abs["country"] != ""].groupby("country", as_index=False)["abs_weight"].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Net USD exposure", f"{usd_weight:.1%}")
    c2.metric("Net foreign FX exposure", f"{foreign_weight:.1%}")
    c3.metric("# distinct currencies", cur_df["currency"].nunique())
    c4.metric("# distinct countries",
                cur_df[cur_df["country"] != ""]["country"].nunique())

    c1, c2 = st.columns(2)
    with c1:
        if not gross_by_cur.empty:
            fig = px.pie(gross_by_cur, names="currency", values="abs_weight",
                           hole=0.4, title="Gross exposure by listing currency")
            st.plotly_chart(fig, width="stretch")
    with c2:
        if not gross_by_country.empty:
            fig = px.pie(gross_by_country, names="country", values="abs_weight",
                           hole=0.4, title="Gross exposure by country of issue")
            st.plotly_chart(fig, width="stretch")

# ──────────────────────────────────────────────────────────────────────────────
# Tab 2: FX risk decomposition
# ──────────────────────────────────────────────────────────────────────────────
with tab_fx:
    st.subheader("Decompose USD-equivalent return into local + FX components")
    st.caption(
        "USD-equivalent return ≈ (1 + local_return) × (1 + fx_change) − 1. "
        "If your home currency is not USD, this still works directionally but the "
        "specific numbers are USD-denominated."
    )

    foreign_tickers = [t for t in tickers if cur_map[t]["currency"] != "USD"]
    if not foreign_tickers:
        st.info("All holdings are USD-denominated — there is no FX risk to decompose.")
    else:
        currencies_needed = sorted({cur_map[t]["currency"] for t in foreign_tickers})
        fx_tickers_needed = [CURRENCY_TO_FX[c] for c in currencies_needed if CURRENCY_TO_FX.get(c)]
        unsupported = [c for c in currencies_needed if CURRENCY_TO_FX.get(c) is None and c != "USD"]
        if unsupported:
            st.warning(f"FX pair not catalogued for: {', '.join(unsupported)}")

        with st.spinner("Fetching FX rates..."):
            fx_prices = fetch_prices(tuple(fx_tickers_needed), period="5y") if fx_tickers_needed else pd.DataFrame()

        window_days = st.select_slider("Decomposition window",
                                          options=[21, 63, 126, 252, 504, 756],
                                          value=252,
                                          help="Trading-day lookback for the decomposition (252 ≈ 1 year)")

        rows = []
        for t in foreign_tickers:
            cur = cur_map[t]["currency"]
            fx_t = CURRENCY_TO_FX.get(cur)
            if fx_t is None or fx_t not in fx_prices.columns:
                rows.append({"ticker": t, "currency": cur,
                              "weight": float(weights.get(t, 0)),
                              "local_return": np.nan, "fx_impact": np.nan,
                              "usd_return": np.nan, "weighted_fx_impact": np.nan})
                continue
            if len(prices[t]) < window_days + 1 or len(fx_prices[fx_t]) < window_days + 1:
                continue
            local_ret = float(prices[t].iloc[-1] / prices[t].iloc[-window_days] - 1)
            fx_ret = float(fx_prices[fx_t].iloc[-1] / fx_prices[fx_t].iloc[-window_days] - 1)
            # USD-base pairs (USDJPY=X) → rising = foreign weakening = NEGATIVE FX impact for the holder
            # Foreign-base pairs (EURUSD=X) → rising = foreign strengthening = POSITIVE FX impact
            fx_impact = -fx_ret if fx_t.startswith("USD") else fx_ret
            usd_ret = (1 + local_ret) * (1 + fx_impact) - 1
            w = float(weights.get(t, 0))
            rows.append({"ticker": t, "currency": cur, "weight": w,
                          "local_return": local_ret, "fx_impact": fx_impact,
                          "usd_return": usd_ret, "weighted_fx_impact": w * fx_impact})

        if rows:
            decomp = pd.DataFrame(rows)
            disp = decomp.copy()
            for col in ["weight", "local_return", "fx_impact", "usd_return", "weighted_fx_impact"]:
                disp[col] = disp[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "—")
            st.dataframe(disp, hide_index=True, width="stretch")

            # Aggregate portfolio-level FX drag/boost
            total_fx_impact = decomp["weighted_fx_impact"].sum(skipna=True)
            c1, c2 = st.columns(2)
            c1.metric(
                f"Aggregate FX impact on portfolio ({window_days}d)",
                f"{total_fx_impact:.2%}",
                help="Sum of (weight × FX impact) across foreign holdings"
            )
            c2.metric(
                "Foreign-asset weight covered",
                f"{decomp['weight'].sum():.2%}"
            )

            # Stacked bar of local + FX
            long_df = decomp.melt(id_vars=["ticker"],
                                    value_vars=["local_return", "fx_impact"],
                                    var_name="component", value_name="return")
            fig = px.bar(long_df.dropna(), x="ticker", y="return", color="component",
                          barmode="stack",
                          title=f"{window_days}-day return decomposition: local-currency vs FX")
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, width="stretch")

        # Rolling FX exposure: portfolio vs DXY
        st.markdown("---")
        st.subheader("Portfolio sensitivity to USD index")
        if "DX-Y.NYB" in fx_prices.columns:
            dxy_ret = fx_prices["DX-Y.NYB"].pct_change().dropna()
            port_ret_series = returns.reindex(dxy_ret.index).fillna(0).dot(
                weights.reindex(returns.columns).fillna(0)
            )
            aligned = pd.concat([port_ret_series, dxy_ret], axis=1, join="inner").dropna()
            aligned.columns = ["portfolio", "dxy"]
            if len(aligned) >= 30:
                window = st.slider("Rolling window (days)", 21, 252, 63, step=21,
                                    key="dxy_window")
                cov = aligned["portfolio"].rolling(window).cov(aligned["dxy"])
                var = aligned["dxy"].rolling(window).var()
                roll_beta = cov / var
                roll_corr = aligned["portfolio"].rolling(window).corr(aligned["dxy"])

                full_beta = aligned["portfolio"].cov(aligned["dxy"]) / aligned["dxy"].var()
                full_corr = aligned["portfolio"].corr(aligned["dxy"])
                c1, c2 = st.columns(2)
                c1.metric("Beta to DXY (full sample)", f"{full_beta:.3f}",
                           help="Sensitivity of portfolio daily return to USD index moves")
                c2.metric("Correlation with DXY", f"{full_corr:.3f}")

                fig = px.line(pd.DataFrame({"Rolling β to DXY": roll_beta,
                                              "Rolling ρ with DXY": roll_corr}),
                                title=f"Rolling DXY exposure ({window}d)")
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                st.plotly_chart(fig, width="stretch")
        else:
            st.info("DXY data unavailable — skipping USD-index sensitivity analysis.")

# ──────────────────────────────────────────────────────────────────────────────
# Tab 3: Interest rates
# ──────────────────────────────────────────────────────────────────────────────
with tab_rates:
    st.subheader("US Treasury yield indicators")
    st.caption("Yields are reported by Yahoo as percent (e.g., 4.5 = 4.5%).")

    with st.spinner("Fetching US Treasury yield data..."):
        rate_prices = fetch_prices(tuple(RATE_INDICATORS.values()), period="5y")

    if rate_prices.empty:
        st.warning("No yield data returned.")
    else:
        # Rename to friendly labels
        rev = {v: k for k, v in RATE_INDICATORS.items()}
        rate_display = rate_prices.rename(columns=rev)

        latest = rate_display.iloc[-1].dropna()
        c = st.columns(max(1, len(latest)))
        for i, (name, val) in enumerate(latest.items()):
            ch = float(rate_display[name].iloc[-1] - rate_display[name].iloc[-21]) if len(rate_display) > 21 else 0
            c[i].metric(name, f"{val:.2f}%", f"{ch:+.2f}% (1M)",
                         delta_color="inverse")

        fig = px.line(rate_display, title="US Treasury yields over time (%)")
        fig.update_yaxes(ticksuffix="%")
        st.plotly_chart(fig, width="stretch")

        # Yield curve snapshot
        st.subheader("Current yield curve")
        tenor_order = ["US 13-Week (^IRX)", "US 5-Year (^FVX)",
                        "US 10-Year (^TNX)", "US 30-Year (^TYX)"]
        curve = pd.DataFrame({
            "tenor": [t for t in tenor_order if t in latest.index],
            "yield": [float(latest[t]) for t in tenor_order if t in latest.index],
        })
        fig = px.line(curve, x="tenor", y="yield", markers=True,
                       title="Current Treasury yield curve")
        fig.update_yaxes(ticksuffix="%")
        st.plotly_chart(fig, width="stretch")

        # 10Y-3M spread (recession bellwether)
        if "US 10-Year (^TNX)" in rate_display.columns and "US 13-Week (^IRX)" in rate_display.columns:
            spread = rate_display["US 10-Year (^TNX)"] - rate_display["US 13-Week (^IRX)"]
            fig = px.line(spread,
                           title="10-Year minus 13-Week yield spread (negative = inversion / recession signal)")
            fig.update_layout(showlegend=False)
            fig.update_yaxes(ticksuffix="%")
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, width="stretch")

    st.markdown("---")
    st.subheader("Bond ETF proxies for international rates")
    st.caption("Yahoo Finance does not list direct foreign-government yield series, so we use "
                "fixed-income ETFs as proxies. These are *total return* prices, not yields.")

    bond_pick = st.multiselect(
        "Show bond ETF total returns",
        list(BOND_ETFS.keys()),
        default=["US 20+ Treasury (TLT)", "US 7-10Y Treasury (IEF)",
                  "Intl Treasury USD-Hedged (BWX)", "Emerging Market USD Bonds (EMB)"],
    )
    if bond_pick:
        bond_tickers = tuple(BOND_ETFS[k] for k in bond_pick)
        with st.spinner("Fetching bond ETF data..."):
            bond_prices = fetch_prices(bond_tickers, period="5y")
        if bond_prices.empty:
            st.warning("Bond ETF fetch returned no data.")
        else:
            available = {k: v for k, v in BOND_ETFS.items() if v in bond_prices.columns}
            bp = bond_prices[list(available.values())].dropna(how="all")
            if bp.empty:
                st.warning("Bond ETFs fetched but contain no usable data.")
            else:
                first = bp.bfill().iloc[0]
                norm = (bp.div(first) * 100).rename(columns={v: k for k, v in available.items()})
                fig = px.line(norm, title="Bond ETF total return (rebased to 100)")
                st.plotly_chart(fig, width="stretch")

                # Annualized vol of each
                rets = bp.pct_change().dropna()
                ann_vol = rets.std() * np.sqrt(252)
                vol_df = pd.DataFrame({
                    "etf": [k for k, v in available.items() if v in ann_vol.index],
                    "ann_vol": [ann_vol[v] for k, v in available.items() if v in ann_vol.index],
                })
                vol_df["ann_vol"] = vol_df["ann_vol"].apply(lambda x: f"{x:.2%}")
                st.dataframe(vol_df, hide_index=True, width="stretch")

# ──────────────────────────────────────────────────────────────────────────────
# Tab 4: FX pair explorer
# ──────────────────────────────────────────────────────────────────────────────
with tab_pairs:
    st.subheader("FX pair explorer")
    pair_pick = st.multiselect(
        "FX pairs to compare",
        list(FX_PAIRS.keys()),
        default=["USD Index (DXY)", "EUR/USD", "USD/JPY", "GBP/USD"],
    )
    if not pair_pick:
        st.info("Pick at least one FX pair above.")
    else:
        fx_t = tuple(FX_PAIRS[k] for k in pair_pick)
        with st.spinner("Fetching FX data..."):
            fx_prices = fetch_prices(fx_t, period="5y")
        if fx_prices.empty:
            st.warning("FX fetch returned no data.")
        else:
            available = {k: v for k, v in FX_PAIRS.items() if v in fx_prices.columns}
            fp = fx_prices[list(available.values())].dropna(how="all")
            first = fp.bfill().iloc[0]
            norm = (fp.div(first) * 100).rename(columns={v: k for k, v in available.items()})
            fig = px.line(norm, title="FX rates (rebased to 100)")
            st.plotly_chart(fig, width="stretch")

            fx_returns = fp.pct_change().dropna()
            if not fx_returns.empty:
                ann_vol = fx_returns.std() * np.sqrt(252)
                corr_matrix = fx_returns.corr()
                corr_matrix.index = [
                    next((k for k, v in available.items() if v == idx), idx)
                    for idx in corr_matrix.index
                ]
                corr_matrix.columns = corr_matrix.index

                c1, c2 = st.columns([1, 1])
                with c1:
                    vol_df = pd.DataFrame({
                        "pair": [k for k, v in available.items() if v in ann_vol.index],
                        "annualized_vol": [ann_vol[v] for k, v in available.items() if v in ann_vol.index],
                    })
                    vol_df["annualized_vol"] = vol_df["annualized_vol"].apply(lambda x: f"{x:.2%}")
                    st.subheader("Annualized FX volatility")
                    st.dataframe(vol_df, hide_index=True, width="stretch")
                with c2:
                    if len(corr_matrix) > 1:
                        st.subheader("FX correlation matrix")
                        fig = px.imshow(corr_matrix, text_auto=".2f",
                                          color_continuous_scale="RdBu_r",
                                          zmin=-1, zmax=1, aspect="auto")
                        st.plotly_chart(fig, width="stretch")
