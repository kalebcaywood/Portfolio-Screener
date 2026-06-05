"""Per-ticker fundamental and technical screener."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import universes as U
from data import format_market_cap
from theme import badge, inject_css, page_header
from screener import compute_portfolio, fetch_history, returns_from_prices
from scoring import REC_ORDER, composite_score, recommend

inject_css()
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


page_header(
    "Universal Equity Screener",
    "Fundamental, technical, and composite-factor screening — US and foreign "
    "equities. Data via yfinance.",
)

# ─── Universe selection ──────────────────────────────────────────────────────
st.sidebar.header("Universe")
src = st.sidebar.radio(
    "Source", ["Preset", "Custom list", "My portfolio"], index=0,
    help="Browse a curated preset, paste your own tickers, or pull from a "
         "portfolio you've loaded on the Home page.",
)

if src == "Preset":
    preset = st.sidebar.selectbox("Preset universe", U.all_preset_names())
    tickers = U.resolve(preset)
    st.sidebar.caption(f"{len(tickers)} names · {preset}")
elif src == "My portfolio":
    tickers = sorted({str(t).strip().upper()
                      for t in (st.session_state.get("tickers") or [])
                      if str(t).strip()})
    if tickers:
        st.sidebar.caption(f"{len(tickers)} names from your loaded portfolio")
    else:
        st.sidebar.caption("No portfolio loaded — build one on the Home page first.")
else:  # Custom list
    default_str = (
        "AAPL, MSFT, GOOGL, NVDA, META, JPM, JNJ, XOM, PG, BRK-B, "
        "2330.TW, ASML.AS, NESN.SW, 7203.T, 005930.KS"
    )
    txt = st.sidebar.text_area(
        "Tickers", default_str, height=120,
        help="Comma- or space-separated Yahoo symbols. Foreign listings use "
             "exchange suffixes: London BARC.L · Tokyo 7203.T · HK 0700.HK · "
             "Taiwan 2330.TW · Switzerland NESN.SW · Korea 005930.KS",
    )
    tickers = sorted({t.strip().upper() for t in txt.replace(",", " ").split() if t.strip()})

if tickers and src != "Preset":
    st.sidebar.caption(f"{len(tickers)} unique tickers queued")

st.sidebar.markdown("---")
st.sidebar.header("Screen filters")
min_mcap_b = st.sidebar.number_input("Min market cap ($B)", 0.0, 5000.0, 0.0, step=10.0)
max_pe = st.sidebar.number_input("Max trailing P/E (0 = no limit)", 0.0, 1000.0, 0.0)
min_roe = st.sidebar.number_input("Min ROE", -1.0, 2.0, -1.0, step=0.05, format="%.2f")
min_piotroski = st.sidebar.slider("Min Piotroski F-Score", 0, 9, 0)
min_div = st.sidebar.number_input("Min dividend yield", 0.0, 0.20, 0.0, step=0.005, format="%.3f")
max_de = st.sidebar.number_input("Max debt/equity (0 = no limit)", 0.0, 50.0, 0.0, step=0.5)

st.sidebar.markdown("---")
want_piotroski = st.sidebar.checkbox(
    "Compute Piotroski F-Score", value=False,
    help="Adds 3 financial-statement calls per name (slower). Auto-enabled if "
         "you set a Min Piotroski filter.",
)
include_fin = want_piotroski or (min_piotroski > 0)

if st.sidebar.button("Run screener", type="primary", width="stretch") and tickers:
    progress = st.progress(0.0)
    status = st.empty()

    def cb(i, n, sym):
        progress.progress((i + 1) / n)
        status.text(f"Fetched {i + 1}/{n} — {sym}")

    label = "with Piotroski" if include_fin else "fast"
    with st.spinner(f"Fetching {len(tickers)} names ({label})…"):
        df = compute_portfolio(tickers, progress_callback=cb, include_financials=include_fin)
        df = composite_score(df)
        df = recommend(df)
    progress.empty()
    status.empty()
    st.session_state["screener_data"] = df

if "screener_data" not in st.session_state:
    st.info("Configure tickers and filters in the sidebar, then click **Run screener**.")
    st.stop()

df: pd.DataFrame = st.session_state["screener_data"]

# Surface any tickers that failed to fetch (errored rows have an 'error' field)
if "error" in df.columns:
    failed = df[df["error"].notna()][["ticker", "error"]]
    if not failed.empty:
        with st.expander(f"{len(failed)} ticker(s) failed to fetch — click to see why", expanded=False):
            st.dataframe(failed, hide_index=True, width="stretch")
        # Drop failed rows from downstream analysis so they don't poison plots/tables
        df = df[df["error"].isna()].drop(columns=["error"]) if "error" in df.columns else df

# If Yahoo rate-limited us, every row will have NaN for market cap AND price.
# Detect that case and surface a clear diagnostic instead of showing a sea of dashes.
if not df.empty:
    fundamental_cols = [c for c in ("market_cap", "price", "pe_trailing") if c in df.columns]
    if fundamental_cols:
        all_nan = df[fundamental_cols].isna().all(axis=None)
        if all_nan:
            st.error(
                "Yahoo Finance returned no fundamental data for any ticker. "
                "This almost always means the cloud host has been rate-limited "
                "by Yahoo. Wait 60 seconds and click **Run screener** again, "
                "or refresh the page to retry."
            )
            st.stop()

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

(tab_summary, tab_rec, tab_analysis, tab_val, tab_qual,
 tab_risk, tab_rank, tab_raw) = st.tabs([
    "Summary", "Recommendations", "Analysis", "Valuation", "Quality",
    "Risk & Momentum", "Composite ranking", "Raw data",
])

with tab_summary:
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

with tab_rec:
    if "recommendation" not in fdf.columns or not len(fdf):
        st.info("Run a screen to generate recommendations.")
    else:
        # Tier counts across the filtered set
        counts = (fdf["recommendation"].value_counts()
                  .reindex(REC_ORDER).fillna(0).astype(int))
        cc = st.columns(len(REC_ORDER))
        for i, tier in enumerate(REC_ORDER):
            cc[i].metric(tier.title(), int(counts[tier]))
        st.caption(
            "Transparent, rules-based tiers: the composite z-score sets the base "
            "rank, then explicit quality / value / momentum / risk flags promote "
            "or demote each name. The **why** is in the Reasons column "
            "(✓ strengths, ✗ cautions)."
        )

        TIER_COLORS = {
            "STRONG BUY": ("#11321F", "#4ADE80"), "BUY": ("#14352A", "#5EE08A"),
            "WATCH": ("#332A0D", "#FBBF24"), "HOLD": ("#1E2638", "#AEB8CC"),
            "AVOID": ("#331A1A", "#FB7185"),
        }
        order_map = {t: i for i, t in enumerate(REC_ORDER)}
        rdf = fdf.copy()
        rdf["_tr"] = rdf["recommendation"].map(order_map).fillna(99)
        sort_key = "rec_score" if "rec_score" in rdf.columns else "score_composite"
        rdf = rdf.sort_values(["_tr", sort_key], ascending=[True, False])

        rec_cols = ["ticker", "name", "sector", "recommendation", "rec_reasons",
                    "score_composite", "pe_trailing", "roe", "ret_1y",
                    "div_yield", "market_cap"]
        rec_cols = [c for c in rec_cols if c in rdf.columns]
        disp = fmt(rdf[rec_cols])

        def _style_rec(v):
            bg, fg = TIER_COLORS.get(v, ("", ""))
            return f"background-color:{bg};color:{fg};font-weight:700;" if bg else ""

        styled = disp.style.map(_style_rec, subset=["recommendation"])
        st.dataframe(styled, width="stretch", hide_index=True)

with tab_val:
    cols = ["ticker", "name", "pe_trailing", "pe_forward", "peg", "pb", "ps",
            "ev_ebitda", "ev_rev", "rev_growth", "earnings_growth", "score_value"]
    cols = [c for c in cols if c in fdf.columns]
    st.dataframe(fmt(fdf[cols].sort_values("score_value", ascending=False)),
                 width="stretch", hide_index=True)
    if {"pe_trailing", "pb"}.issubset(fdf.columns):
        plot_df = fdf.dropna(subset=["pe_trailing", "pb"]).copy()
        if len(plot_df):
            # Build a size column that never has NaN (would crash plotly)
            mc = plot_df["market_cap"] if "market_cap" in plot_df.columns else pd.Series(dtype=float)
            mc_med = mc.median() if mc.notna().any() else 1.0
            size_series = mc.fillna(mc_med).clip(lower=1.0)
            color_col = "sector" if "sector" in plot_df.columns else None
            fig = px.scatter(plot_df, x="pe_trailing", y="pb", text="ticker",
                             color=color_col, size=size_series,
                             title="Valuation map: P/E vs P/B")
            fig.update_traces(textposition="top center")
            st.plotly_chart(fig, width="stretch")

with tab_qual:
    cols = ["ticker", "name", "roe", "roa", "gross_margin", "op_margin", "net_margin",
            "current_ratio", "quick_ratio", "debt_equity", "piotroski_f",
            "div_yield", "payout_ratio", "score_quality"]
    cols = [c for c in cols if c in fdf.columns]
    st.dataframe(fmt(fdf[cols].sort_values("score_quality", ascending=False)),
                 width="stretch", hide_index=True)

with tab_risk:
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

with tab_rank:
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

with tab_analysis:
    if not len(fdf):
        st.info("No tickers passed the current filters — adjust the sidebar to analyze a name.")
    else:
        sel = st.selectbox("Analyze ticker", fdf["ticker"].tolist())
        row = fdf[fdf["ticker"] == sel].iloc[0] if len(fdf[fdf["ticker"] == sel]) else None

        if row is not None:
            name = row.get("name", sel)
            sector = row.get("sector", "—")
            country = row.get("country", "—")
            currency = row.get("currency", "USD")
            st.markdown(f"#### {name}")
            st.caption(f"{sel}  ·  {sector}  ·  {country}  ·  listed in {currency}")

            # ── Recommendation banner ──────────────────────────────────────
            rec = row.get("recommendation", "HOLD")
            reasons = row.get("rec_reasons", "")
            rank = row.get("rank")
            badge_kind = {"STRONG BUY": "success", "BUY": "success", "WATCH": "warning",
                          "HOLD": "neutral", "AVOID": "danger"}.get(rec, "neutral")
            rank_txt = f"  ·  composite rank #{int(rank)} of {len(df)}" if pd.notna(rank) else ""
            st.markdown(
                f"{badge(rec, kind=badge_kind)} "
                f"<span style='color:#8A95AB;font-size:13px;margin-left:8px;'>{reasons}{rank_txt}</span>",
                unsafe_allow_html=True,
            )
            st.markdown("")

            def _show(col, label, val, fmt_str="{:.2f}", pct=False, suffix=""):
                if pd.notna(val):
                    s = (f"{val:.2%}" if pct else fmt_str.format(val)) + suffix
                    col.metric(label, s)
                else:
                    col.metric(label, "—")

            # ── Fundamental analysis, grouped ──────────────────────────────
            st.markdown("##### Valuation")
            c = st.columns(6)
            _show(c[0], "Price", row.get("price"))
            _show(c[1], "P/E (trail)", row.get("pe_trailing"))
            _show(c[2], "P/E (fwd)", row.get("pe_forward"))
            _show(c[3], "PEG", row.get("peg"))
            _show(c[4], "P/B", row.get("pb"))
            _show(c[5], "EV/EBITDA", row.get("ev_ebitda"))

            st.markdown("##### Profitability & growth")
            c = st.columns(6)
            _show(c[0], "ROE", row.get("roe"), pct=True)
            _show(c[1], "ROA", row.get("roa"), pct=True)
            _show(c[2], "Net margin", row.get("net_margin"), pct=True)
            _show(c[3], "Gross margin", row.get("gross_margin"), pct=True)
            _show(c[4], "Rev growth", row.get("rev_growth"), pct=True)
            _show(c[5], "EPS growth", row.get("earnings_growth"), pct=True)

            st.markdown("##### Health, dividend & risk")
            c = st.columns(6)
            _show(c[0], "Debt/Equity", row.get("debt_equity"))
            _show(c[1], "Current ratio", row.get("current_ratio"))
            _show(c[2], "Piotroski", row.get("piotroski_f"), fmt_str="{:.0f}")
            _show(c[3], "Div yield", row.get("div_yield"), pct=True)
            _show(c[4], "Beta (1Y)", row.get("beta_1y"))
            _show(c[5], "Max DD", row.get("max_dd"), pct=True)

        st.markdown("---")
        h = fetch_history(sel, "2y")
        if h.empty or "Close" not in h.columns:
            st.warning(f"No price history available for {sel} — Yahoo may not have data for this listing.")
        else:
            c1, c2 = st.columns([2, 1])
            with c1:
                close = h["Close"]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=close.index, y=close, name="Close",
                                         line=dict(color="#5BA8E8", width=2)))
                if len(close) >= 50:
                    fig.add_trace(go.Scatter(x=close.index, y=close.rolling(50).mean(),
                                             name="50-day MA",
                                             line=dict(color="#FF8200", dash="dash", width=1.4)))
                if len(close) >= 200:
                    fig.add_trace(go.Scatter(x=close.index, y=close.rolling(200).mean(),
                                             name="200-day MA",
                                             line=dict(color="#FB7185", dash="dash", width=1.4)))
                fig.update_layout(title=f"{sel} — 2-year price with moving averages",
                                  height=430, hovermode="x unified")
                st.plotly_chart(fig, width="stretch")
            with c2:
                rets = h["Close"].pct_change().dropna()
                if len(rets) > 10:
                    fig = px.histogram(rets, nbins=40, title="Daily return distribution")
                    fig.update_xaxes(tickformat=".1%")
                    fig.update_layout(showlegend=False, height=430)
                    st.plotly_chart(fig, width="stretch")

with tab_raw:
    st.dataframe(fdf, width="stretch", hide_index=True)
    st.download_button("Download as CSV", fdf.to_csv(index=False).encode("utf-8"),
                        "screener_results.csv", "text/csv")
