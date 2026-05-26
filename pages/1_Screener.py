"""Per-ticker fundamental and technical screener."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data import format_market_cap
from theme import inject_css, ut_sidebar_brand
from screener import compute_portfolio, fetch_history, returns_from_prices
from scoring import composite_score

st.set_page_config(page_title="Screener", layout="wide")
inject_css()
ut_sidebar_brand()

PCT_COLS = {"roe", "roa", "gross_margin", "op_margin", "net_margin", "rev_growth",
            "earnings_growth", "eps_growth_q", "div_yield", "payout_ratio",
            "ret_1m", "ret_3m", "ret_6m", "ret_1y", "ret_ytd", "volatility",
            "max_dd", "momentum_12_1", "pct_from_52w_high"}


def fmt(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    has_currency = "currency" in out.columns
    for c in out.columns:
        if c in PCT_COLS and pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].apply(lambda v: f"{v:.2%}" if pd.notna(v) else "—")
        elif c == "market_cap" and pd.api.types.is_numeric_dtype(out[c]):
            if has_currency:
                out[c] = [format_market_cap(v, c_) for v, c_ in zip(out[c], out["currency"])]
            else:
                out[c] = out[c].apply(
                    lambda v: format_market_cap(v, "USD") if pd.notna(v) else "—"
                )
        elif c == "price" and pd.api.types.is_numeric_dtype(out[c]) and has_currency:
            out[c] = [
                f"{v:,.2f} {c_}" if pd.notna(v) else "—"
                for v, c_ in zip(out[c], out["currency"])
            ]
        elif pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].apply(lambda v: f"{v:,.2f}" if pd.notna(v) else "—")
    return out


st.title("Equity Screener")
st.caption("Fundamental, technical, and composite-factor screening. Data via yfinance.")

# Use tickers from session state if available, else allow manual input
default_tickers = st.session_state.get("tickers")
if default_tickers:
    st.info(f"Using {len(default_tickers)} tickers from your loaded portfolio. "
            "Override below if needed.")
    default_str = ", ".join(default_tickers)
else:
    default_str = "AAPL, MSFT, GOOGL, NVDA, META, JPM, JNJ, XOM, PG, KO, V, UNH, HD, CAT, BRK-B"

st.sidebar.header("Tickers")
txt = st.sidebar.text_area("Tickers", default_str, height=110)
tickers = sorted({t.strip().upper() for t in txt.replace(",", " ").split() if t.strip()})

st.sidebar.markdown("---")
st.sidebar.header("Screen filters")
min_mcap_b = st.sidebar.number_input("Min market cap ($B)", 0.0, 5000.0, 0.0, step=10.0)
max_pe = st.sidebar.number_input("Max trailing P/E (0 = no limit)", 0.0, 1000.0, 0.0)
min_roe = st.sidebar.number_input("Min ROE", -1.0, 2.0, -1.0, step=0.05, format="%.2f")
min_piotroski = st.sidebar.slider("Min Piotroski F-Score", 0, 9, 0)
min_div = st.sidebar.number_input("Min dividend yield", 0.0, 0.20, 0.0, step=0.005, format="%.3f")
max_de = st.sidebar.number_input("Max debt/equity (0 = no limit)", 0.0, 50.0, 0.0, step=0.5)

if st.sidebar.button("Run screener", type="primary", width="stretch") and tickers:
    progress = st.progress(0.0)
    status = st.empty()

    def cb(i, n, sym):
        progress.progress((i + 1) / n)
        status.text(f"Fetching {sym} ({i + 1}/{n})")

    df = compute_portfolio(tickers, progress_callback=cb)
    df = composite_score(df)
    progress.empty()
    status.empty()
    st.session_state["screener_data"] = df

if "screener_data" not in st.session_state:
    st.info("Configure tickers and filters in the sidebar, then click **Run screener**.")
    st.stop()

df: pd.DataFrame = st.session_state["screener_data"]

sectors = sorted([s for s in df["sector"].dropna().unique() if s and s != "Unknown"])
sector_pick = st.sidebar.multiselect("Sectors (empty = all)", sectors)

countries = sorted([c for c in df.get("country", pd.Series([], dtype=object)).dropna().unique() if c]) if "country" in df.columns else []
country_pick = st.sidebar.multiselect("Countries (empty = all)", countries) if countries else []

currencies = sorted([c for c in df.get("currency", pd.Series([], dtype=object)).dropna().unique() if c]) if "currency" in df.columns else []
currency_pick = st.sidebar.multiselect("Currencies (empty = all)", currencies) if currencies else []

fdf = df.copy()
if min_mcap_b > 0:
    fdf = fdf[fdf["market_cap"].fillna(0) >= min_mcap_b * 1e9]
if max_pe > 0:
    fdf = fdf[fdf["pe_trailing"].fillna(np.inf) <= max_pe]
if min_roe > -1:
    fdf = fdf[fdf["roe"].fillna(-np.inf) >= min_roe]
if min_piotroski > 0:
    fdf = fdf[fdf["piotroski_f"].fillna(-1) >= min_piotroski]
if min_div > 0:
    fdf = fdf[fdf["div_yield"].fillna(0) >= min_div]
if max_de > 0:
    fdf = fdf[fdf["debt_equity"].fillna(np.inf) <= max_de]
if sector_pick:
    fdf = fdf[fdf["sector"].isin(sector_pick)]
if country_pick and "country" in fdf.columns:
    fdf = fdf[fdf["country"].isin(country_pick)]
if currency_pick and "currency" in fdf.columns:
    fdf = fdf[fdf["currency"].isin(currency_pick)]

st.caption(f"**{len(fdf)}** of {len(df)} tickers passed filters")

tabs = st.tabs(["Summary", "Valuation", "Quality", "Risk & Momentum",
                "Composite ranking", "Charts", "Raw data"])

with tabs[0]:
    cols = ["ticker", "name", "country", "currency", "exchange", "sector",
            "market_cap", "price", "pe_trailing", "pb", "roe", "div_yield",
            "ret_1y", "piotroski_f", "score_composite"]
    cols = [c for c in cols if c in fdf.columns]
    st.dataframe(fmt(fdf[cols]), width="stretch", hide_index=True)

    # Quick currency / country breakdown when present
    if "currency" in fdf.columns and fdf["currency"].nunique() > 1:
        st.markdown("**Listing breakdown**")
        c1, c2 = st.columns(2)
        with c1:
            cur_counts = fdf["currency"].value_counts().reset_index()
            cur_counts.columns = ["currency", "n_tickers"]
            st.dataframe(cur_counts, hide_index=True, width="stretch")
        with c2:
            if "country" in fdf.columns:
                co_counts = fdf[fdf["country"] != ""]["country"].value_counts().reset_index()
                co_counts.columns = ["country", "n_tickers"]
                st.dataframe(co_counts, hide_index=True, width="stretch")

with tabs[1]:
    cols = ["ticker", "name", "pe_trailing", "pe_forward", "peg", "pb", "ps",
            "ev_ebitda", "ev_rev", "rev_growth", "earnings_growth", "score_value"]
    cols = [c for c in cols if c in fdf.columns]
    st.dataframe(fmt(fdf[cols].sort_values("score_value", ascending=False)),
                 width="stretch", hide_index=True)
    if {"pe_trailing", "pb"}.issubset(fdf.columns):
        plot_df = fdf.dropna(subset=["pe_trailing", "pb"])
        if len(plot_df):
            fig = px.scatter(plot_df, x="pe_trailing", y="pb", text="ticker", color="sector",
                             size=plot_df["market_cap"].fillna(plot_df["market_cap"].median()),
                             title="Valuation map: P/E vs P/B")
            fig.update_traces(textposition="top center")
            st.plotly_chart(fig, width="stretch")

with tabs[2]:
    cols = ["ticker", "name", "roe", "roa", "gross_margin", "op_margin", "net_margin",
            "current_ratio", "quick_ratio", "debt_equity", "piotroski_f",
            "div_yield", "payout_ratio", "score_quality"]
    cols = [c for c in cols if c in fdf.columns]
    st.dataframe(fmt(fdf[cols].sort_values("score_quality", ascending=False)),
                 width="stretch", hide_index=True)

with tabs[3]:
    cols = ["ticker", "ret_1m", "ret_3m", "ret_6m", "ret_1y", "ret_ytd",
            "momentum_12_1", "pct_from_52w_high", "rsi_14", "volatility",
            "max_dd", "sharpe", "sortino", "beta_1y", "score_momentum", "score_low_risk"]
    cols = [c for c in cols if c in fdf.columns]
    st.dataframe(fmt(fdf[cols].sort_values("score_momentum", ascending=False)),
                 width="stretch", hide_index=True)
    if {"volatility", "ret_1y"}.issubset(fdf.columns):
        plot_df = fdf.dropna(subset=["volatility", "ret_1y"])
        if len(plot_df):
            fig = px.scatter(plot_df, x="volatility", y="ret_1y", text="ticker", color="sector",
                             title="Risk/return: 1Y return vs annualized volatility")
            fig.update_traces(textposition="top center")
            fig.update_xaxes(tickformat=".0%")
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, width="stretch")

with tabs[4]:
    score_cols = ["ticker", "name", "sector", "score_value", "score_quality",
                  "score_momentum", "score_low_risk", "score_composite", "rank"]
    score_cols = [c for c in score_cols if c in fdf.columns]
    ranked = fdf[score_cols].sort_values("score_composite", ascending=False)
    st.dataframe(ranked.round(3), width="stretch", hide_index=True)

    factor_cols = [c for c in ["score_value", "score_quality", "score_momentum", "score_low_risk"]
                    if c in ranked.columns]
    if factor_cols:
        long = ranked.melt(id_vars=["ticker"], value_vars=factor_cols,
                            var_name="factor", value_name="z")
        fig = px.bar(long, x="ticker", y="z", color="factor", barmode="group",
                      title="Factor exposure by ticker (z-score)")
        st.plotly_chart(fig, width="stretch")

with tabs[5]:
    if len(fdf):
        sel = st.selectbox("Detail ticker", fdf["ticker"].tolist())
        h = fetch_history(sel, "2y")
        if not h.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=h.index, y=h["Close"], name="Close"))
            fig.add_trace(go.Scatter(x=h.index, y=h["Close"].rolling(50).mean(),
                                       name="50-day MA", line=dict(dash="dash")))
            fig.add_trace(go.Scatter(x=h.index, y=h["Close"].rolling(200).mean(),
                                       name="200-day MA", line=dict(dash="dash")))
            fig.update_layout(title=f"{sel} — 2-year price", height=480)
            st.plotly_chart(fig, width="stretch")

with tabs[6]:
    st.dataframe(fdf, width="stretch", hide_index=True)
    st.download_button("Download as CSV", fdf.to_csv(index=False).encode("utf-8"),
                        "screener_results.csv", "text/csv")
