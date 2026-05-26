"""Quantitative Portfolio Analytics — landing page + portfolio builder."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from data import BENCHMARK, RISK_FREE_RATE, compute_returns, fetch_prices
from portfolio_input import (clean_ticker, df_to_weight_series, is_valid_ticker,
                              normalize_weights, parse_csv, validate_price_data)
from theme import inject_css, status_line, ut_header, ut_sidebar_brand


SAMPLE_CSV = """ticker,description,weight
AAPL,Apple Inc.,0.15
MSFT,Microsoft Corporation,0.15
GOOGL,Alphabet Inc.,0.12
NVDA,NVIDIA Corporation,0.12
JPM,JPMorgan Chase,0.10
JNJ,Johnson & Johnson,0.10
XOM,Exxon Mobil,0.08
PG,Procter & Gamble,0.08
KO,Coca-Cola,0.05
V,Visa Inc.,0.05
"""


def live_exposure(df: pd.DataFrame) -> dict | None:
    """Compute live exposure metrics from a draft portfolio dataframe (no normalization)."""
    if df is None or df.empty or "weight" not in df.columns:
        return None
    w = pd.to_numeric(df["weight"], errors="coerce").fillna(0)
    if (w == 0).all():
        return None
    abs_sum = float(w.abs().sum())
    # Mirror normalize_weights percent-detection logic for the preview
    interpreted_as_pct = 5 <= abs_sum <= 500
    if interpreted_as_pct:
        w = w / 100
    signed = float(w.sum())
    gross = float(w.abs().sum())
    n_long = int((w > 0).sum())
    n_short = int((w < 0).sum())
    leverage = (gross / abs(signed)) if abs(signed) > 1e-9 else float("inf")
    return {
        "n": int((w != 0).sum()),
        "n_long": n_long,
        "n_short": n_short,
        "net": signed,
        "gross": gross,
        "leverage": leverage,
        "interpreted_as_pct": interpreted_as_pct,
    }


def render_exposure_preview(metrics: dict, title: str = "Live exposure preview") -> None:
    """Render exposure metrics as a row of st.metric cards."""
    st.markdown(f"**{title}**")
    c = st.columns(6)
    c[0].metric("Positions (L / S)",
                 f"{metrics['n']}", f"{metrics['n_long']}L / {metrics['n_short']}S")
    c[1].metric("Net exposure", f"{metrics['net']:.1%}")
    c[2].metric("Gross exposure", f"{metrics['gross']:.1%}")
    lev = metrics["leverage"]
    c[3].metric("Leverage", "∞ (market-neutral)" if lev > 100 else f"{lev:.2f}×")
    if metrics["interpreted_as_pct"]:
        c[4].metric("Input format", "Percent → decimal")
    else:
        c[4].metric("Input format", "Decimal")
    # Categorize the strategy
    if abs(metrics["net"] - 1.0) < 0.01 and abs(metrics["gross"] - 1.0) < 0.01:
        label = "Long-only 100%"
    elif abs(metrics["net"]) < 0.01:
        label = "Market-neutral"
    elif metrics["leverage"] > 1.05 and metrics["n_short"] > 0:
        label = f"{int(round(metrics['gross']*100 - (100 - metrics['net']*100)))}/{int(round((metrics['gross'] - metrics['net'])*100))}-style"
    elif metrics["gross"] > 1.05:
        label = "Leveraged long"
    elif metrics["n_short"] > 0:
        label = "Long-short"
    else:
        label = "Custom"
    c[5].metric("Strategy", label)

st.set_page_config(page_title="UT Portfolio Analytics", layout="wide", page_icon=":material/show_chart:")
inject_css()
ut_sidebar_brand()
ut_header("Quantitative Portfolio Analytics", "University of Tennessee")

with st.expander("Feature map", expanded=False):
    st.markdown(
        """
| Page | Capabilities |
|---|---|
| **Screener** | Per-ticker fundamental + technical screen, Piotroski F-Score, multi-factor composite |
| **Performance** | CAGR, Sharpe, Sortino, Calmar, Omega, alpha/beta, capture ratios, drawdown episodes, rolling metrics, monthly heatmap |
| **Risk Metrics** | Historical / parametric / Cornish-Fisher / Monte Carlo VaR & CVaR, component & marginal risk, diversification ratio |
| **Statistical Tests** | Normality (Shapiro / D'Agostino / Anderson / JB / KS), stationarity (ADF, KPSS), Ljung-Box, ARCH-LM, variance-ratio, runs test |
| **Factor Models** | CAPM with full OLS output, custom multi-factor regression, return attribution, rolling alpha/beta |
| **Optimization** | Max-Sharpe, min-variance, Equal Risk Contribution, efficient frontier, target-return solver |
| **Monte Carlo** | Multivariate-normal & historical / block bootstrap simulation with fan chart and risk summary |
| **Stress Tests** | Replay 8 historical crises, custom shocks, beta-propagated market shock |
| **Correlation** | Pearson / Spearman / Kendall, rolling correlations, hierarchical clustering |
| **Currency & Rates** | FX exposure by currency / country, FX-impact attribution, US Treasury yield curve, bond-ETF proxies for foreign rates, FX pair explorer |
| **Risk Decomposition** | Systematic vs idiosyncratic split (CAPM + multi-factor), concentration / sector / country, tail & drawdown, correlation diagnostics, **rule-based suggestions for improvement** |
| **Pacing & Reup** | Per-position vintage tracker, composite reup / pullback score, suggested rebalance, bootstrap-based forward probability of success & risk |
"""
    )

with st.expander("Long-short and leveraged portfolios (130/30, market-neutral, ...)", expanded=False):
    st.markdown(
        """
The app supports portfolios where **gross exposure ≠ 100%** — turn on both
**Allow shorts** and **Allow leverage** in the sidebar.

| Strategy | Net | Gross | Leverage | How to enter |
|---|---|---|---|---|
| Long-only | 100% | 100% | 1.0× | Default settings |
| Long-short (no leverage) | 100% | varies | varies | Allow shorts on, leverage off (auto-rescales net to 100%) |
| **130/30** | 100% | 160% | 1.6× | Both on; enter `130, -30` (or `1.30, -0.30`) — totals respected |
| 150/50 | 100% | 200% | 2.0× | Both on; enter weights with longs summing 150, shorts -50 |
| Market-neutral | ≈ 0% | varies | ∞ | Both on; longs and shorts cancel |
| Leveraged long | > 100% | > 100% | > 1.0× | Leverage on, shorts off; enter weights summing >1 |

**How the math works:**
- Portfolio return = `Σ wᵢ × rᵢ` (signed weights, no rescaling)
- For a 130/30 where longs average +10% and shorts average +5%:
  `return = 1.30 × 10% + (-0.30 × 5%) = 11.5%`
- Vol, VaR, Sharpe, drawdown, etc. all flow from the leveraged return series
- Risk contributions (component VaR / vol) can be **negative** for hedge-like shorts

**Caveat:** the optimizer (Optimization page) assumes net = 100%; running it
on a leveraged portfolio produces unleveraged comparison strategies — useful
context, not a leveraged optimization.
"""
    )

with st.expander("Foreign equity formats", expanded=False):
    st.markdown(
        """
Yahoo Finance uses an **exchange suffix** for non-US listings. The screener accepts all of these:

| Region | Format | Examples |
|---|---|---|
| Japan (Tokyo) | `NNNN.T` | `7203.T` (Toyota), `9984.T` (SoftBank), `6758.T` (Sony) |
| Hong Kong | `NNNN.HK` | `0700.HK` (Tencent), `9988.HK` (Alibaba), `0005.HK` (HSBC) |
| China Shanghai | `NNNNNN.SS` | `600519.SS` (Moutai), `601398.SS` (ICBC) |
| China Shenzhen | `NNNNNN.SZ` | `000858.SZ` (Wuliangye), `300750.SZ` (CATL) |
| Korea | `NNNNNN.KS` / `.KQ` | `005930.KS` (Samsung), `035420.KS` (Naver) |
| Taiwan | `NNNN.TW` | `2330.TW` (TSMC), `2317.TW` (Foxconn) |
| India | `XXXX.NS` / `.BO` | `RELIANCE.NS`, `INFY.NS`, `TCS.NS` |
| London | `XXXX.L` | `BARC.L` (Barclays), `ULVR.L` (Unilever), `AZN.L` (AstraZeneca) |
| Germany | `XXXX.DE` | `SAP.DE`, `BMW.DE`, `SIE.DE` |
| France | `XXXX.PA` | `MC.PA` (LVMH), `BNP.PA`, `SAN.PA` (Sanofi) |
| Netherlands | `XXXX.AS` | `ASML.AS`, `HEIA.AS` (Heineken) |
| Switzerland | `XXXX.SW` | `NESN.SW` (Nestlé), `NOVN.SW` (Novartis) |
| Italy | `XXXX.MI` | `ENI.MI`, `UCG.MI` (UniCredit) |
| Sweden | `XXXX-X.ST` | `VOLV-B.ST` (Volvo B-shares) |
| Brazil | `XXXXN.SA` | `PETR4.SA` (Petrobras), `VALE3.SA` |
| Canada | `XXXX.TO` | `SHOP.TO`, `RY.TO` (Royal Bank) |
| Australia | `XXX.AX` | `BHP.AX`, `CBA.AX` |

Use the **Currency & Rates** page to see your portfolio's FX exposure, USD-equivalent return decomposition, and global rate context.
"""
    )

st.markdown("---")
st.header("Build your portfolio")

mode = st.radio("Input method", ["Manual table", "Upload CSV"], horizontal=True)

# Seed the editable table
if "manual_df" not in st.session_state:
    st.session_state["manual_df"] = pd.DataFrame(
        {
            "ticker": ["AAPL", "MSFT", "GOOGL", "NVDA", "JPM", "JNJ", "XOM", "PG"],
            "description": [
                "Apple", "Microsoft", "Alphabet", "Nvidia",
                "JPMorgan Chase", "Johnson & Johnson", "Exxon Mobil", "Procter & Gamble",
            ],
            "weight": [0.15, 0.15, 0.12, 0.13, 0.12, 0.11, 0.12, 0.10],
        }
    )

captured_df: pd.DataFrame | None = None
errors: list[str] = []
warnings_list: list[str] = []
input_mode: str | None = None

if mode == "Manual table":
    btn_col1, btn_col2, _ = st.columns([1, 1, 4])
    with btn_col1:
        if st.button("Reset to sample", help="Restore the default sample portfolio"):
            st.session_state["manual_df"] = pd.DataFrame(
                {
                    "ticker": ["AAPL", "MSFT", "GOOGL", "NVDA", "JPM", "JNJ", "XOM", "PG"],
                    "description": ["Apple", "Microsoft", "Alphabet", "Nvidia",
                                      "JPMorgan Chase", "Johnson & Johnson",
                                      "Exxon Mobil", "Procter & Gamble"],
                    "weight": [0.15, 0.15, 0.12, 0.13, 0.12, 0.11, 0.12, 0.10],
                }
            )
            st.rerun()
    with btn_col2:
        if "weights" in st.session_state and st.button(
            "Load current portfolio", help="Copy your loaded portfolio into the editor for tweaking"
        ):
            cur_w = st.session_state["weights"]
            cur_desc = st.session_state.get("descriptions", {})
            st.session_state["manual_df"] = pd.DataFrame(
                {
                    "ticker": cur_w.index.tolist(),
                    "description": [cur_desc.get(t, "") for t in cur_w.index],
                    "weight": cur_w.values,
                }
            )
            st.rerun()

    st.caption(
        "Edit rows below — use the **+** button to add tickers, or click a cell to edit. "
        "Weights can be decimals (`0.15`) **or** percentages (`15`); the format is auto-detected. "
        "Negative weights represent shorts."
    )
    edited = st.data_editor(
        st.session_state["manual_df"],
        num_rows="dynamic",
        width="stretch",
        key="portfolio_editor",
        column_config={
            "ticker": st.column_config.TextColumn(
                "Ticker", required=True, max_chars=15,
                help="Yahoo Finance symbol (AAPL, BRK-B, 7203.T, BARC.L, …)",
            ),
            "description": st.column_config.TextColumn(
                "Description", required=False, max_chars=80,
                help="Optional label — not used in analytics",
            ),
            "weight": st.column_config.NumberColumn(
                "Weight", required=True, format="%.4f", min_value=-5.0, max_value=5.0,
                help="Decimal (0.15) or percent (15); negative = short",
            ),
        },
    )
    st.session_state["manual_df"] = edited.copy()

    # Live exposure preview before submit
    live = live_exposure(edited)
    if live is not None:
        render_exposure_preview(live)

    # Clean & validate manual input
    tmp = edited.copy()
    tmp["ticker"] = tmp["ticker"].apply(clean_ticker)
    tmp = tmp[tmp["ticker"] != ""]
    bad = tmp[~tmp["ticker"].apply(is_valid_ticker)]
    if not bad.empty:
        warnings_list.append(
            f"Removed {len(bad)} invalid ticker(s): {', '.join(bad['ticker'].tolist())}"
        )
        tmp = tmp[tmp["ticker"].apply(is_valid_ticker)]

    if tmp.empty:
        errors.append("No valid tickers in table.")
    else:
        if tmp["ticker"].duplicated().any():
            dups = tmp.loc[tmp["ticker"].duplicated(keep=False), "ticker"].unique().tolist()
            warnings_list.append(f"Consolidated duplicate ticker(s): {', '.join(dups)}")
            tmp = tmp.groupby("ticker", as_index=False).agg(
                {"description": "first", "weight": "sum"}
            )
        captured_df = tmp.reset_index(drop=True)
        input_mode = "weight"

else:  # CSV upload
    csv_col1, csv_col2 = st.columns([3, 1])
    with csv_col1:
        uploaded = st.file_uploader(
            "Upload CSV",
            type=["csv"],
            help="Required: ticker column. Optional: description, weight (or shares + cost_basis).",
        )
    with csv_col2:
        st.markdown("&nbsp;")  # spacer
        st.download_button(
            "Download sample CSV",
            data=SAMPLE_CSV,
            file_name="sample_portfolio.csv",
            mime="text/csv",
            help="Get a template you can edit and re-upload",
            width="stretch",
        )

    st.caption(
        "Recognized column names — "
        "Ticker: `ticker`, `symbol`, `stock`, `asset`. "
        "Description: `description`, `name`, `company`. "
        "Weight: `weight`, `allocation`, `%`, `pct`. "
        "Shares: `shares`, `qty`, `quantity`."
    )
    with st.expander("Example CSV formats"):
        st.code(
            "# Format 1: ticker + description + weight (decimal)\n"
            "ticker,description,weight\n"
            "AAPL,Apple Inc.,0.20\n"
            "MSFT,Microsoft Corp.,0.20\n"
            "GOOGL,Alphabet Inc.,0.15\n"
            "\n"
            "# Format 2: percentages (auto-detected)\n"
            "Symbol,Name,Allocation\n"
            "AAPL,Apple,20\n"
            "MSFT,Microsoft,20\n"
            "GOOGL,Alphabet,15\n"
            "\n"
            "# Format 3: shares-based\n"
            "ticker,description,shares,cost_basis\n"
            "AAPL,Apple,10,150.00\n"
            "MSFT,Microsoft,5,300.00\n"
            "\n"
            "# Format 4: long-short with negative weights (130/30)\n"
            "ticker,description,weight\n"
            "AAPL,Apple,30\n"
            "MSFT,Microsoft,30\n"
            "GOOGL,Alphabet,25\n"
            "NVDA,Nvidia,25\n"
            "JPM,JPMorgan,20\n"
            "XOM,Exxon (short),-15\n"
            "KO,Coca-Cola (short),-15\n",
            language="csv",
        )

    if uploaded is not None:
        parsed = parse_csv(uploaded)
        errors.extend(parsed["errors"])
        warnings_list.extend(parsed["warnings"])
        captured_df = parsed["df"]
        input_mode = parsed["mode"]
        if captured_df is not None and not captured_df.empty:
            st.subheader("Parsed CSV preview")

            # Per-row validation indicators
            preview = captured_df.copy()
            preview.insert(0, "valid",
                            preview["ticker"].apply(lambda t: "Yes" if is_valid_ticker(t) else "No"))
            if "weight" in preview.columns:
                preview["weight"] = preview["weight"].apply(
                    lambda v: f"{v:+.4f}" if pd.notna(v) and v != 0 else (f"{v:.4f}" if pd.notna(v) else "—")
                )
            st.dataframe(preview, hide_index=True, width="stretch")

            # Live exposure preview
            live = live_exposure(captured_df)
            if live is not None:
                render_exposure_preview(live, title="Live exposure preview (before normalization)")

# ─── Settings sidebar ────────────────────────────────────────────────────────
st.sidebar.header("Settings")
period = st.sidebar.selectbox(
    "Lookback period", ["1y", "2y", "3y", "5y", "10y", "max"], index=3
)
rf = st.sidebar.number_input(
    "Risk-free rate (annual)", 0.0, 0.20, RISK_FREE_RATE, step=0.005, format="%.3f"
)
aum = st.sidebar.number_input(
    "Fund AUM ($)", min_value=1000.0, max_value=1e12,
    value=float(st.session_state.get("aum", 1_000_000.0)),
    step=10000.0, format="%.2f",
    help=(
        "Total assets under management. Per-position dollar amounts and share counts "
        "are computed as weight × AUM. Other pages (Risk, Monte Carlo, Stress, Risk "
        "Decomposition) use this as their default notional."
    ),
)
allow_short = st.sidebar.checkbox(
    "Allow short / negative weights", value=False,
    help="Permits negative weights (short positions).",
)
allow_leverage = st.sidebar.checkbox(
    "Allow leverage (gross > 100%)", value=False,
    help=(
        "For 130/30, market-neutral, or other strategies where gross exposure "
        "exceeds 100%. When enabled, weights are NOT rescaled — your input is "
        "taken as-is. Net exposure may also differ from 100%."
    ),
)
if allow_leverage and not allow_short:
    st.sidebar.caption("Tip: leverage with long-only ⇒ leveraged-long (e.g. 150% long, 0% short).")

# Show validation messages
for msg in warnings_list:
    st.warning(msg)
for msg in errors:
    st.error(msg)

submit_disabled = (
    captured_df is None or captured_df.empty or len(errors) > 0
)

submit_col1, submit_col2 = st.columns([3, 1])
with submit_col1:
    submit = st.button(
        "Fetch & validate portfolio",
        type="primary",
        disabled=submit_disabled,
        width="stretch",
    )
with submit_col2:
    if captured_df is not None and not captured_df.empty and "weight" in captured_df.columns:
        export_csv = captured_df[["ticker", "description", "weight"]].copy() \
            if "description" in captured_df.columns else captured_df[["ticker", "weight"]].copy()
        st.download_button(
            "Save current as CSV",
            data=export_csv.to_csv(index=False),
            file_name="my_portfolio.csv",
            mime="text/csv",
            width="stretch",
        )

if submit_disabled:
    if captured_df is None or captured_df.empty:
        st.caption("Enter tickers and weights above to enable the fetch button.")
    elif len(errors) > 0:
        st.caption("Resolve the errors above before submitting.")

if submit and captured_df is not None and not captured_df.empty:
    tickers_in: list[str] = captured_df["ticker"].tolist()
    fetch_set = tuple(sorted(set(tickers_in) | {BENCHMARK}))

    with st.spinner(f"Fetching {len(tickers_in)} tickers + benchmark..."):
        prices = fetch_prices(fetch_set, period=period)

    if prices.empty:
        st.error("Price fetch returned no data. Check ticker symbols.")
        st.stop()

    valid, missing, low_history = validate_price_data(prices, tickers_in, min_obs=30)
    if missing:
        st.warning(f"No price data returned for: {', '.join(missing)}")
    if low_history:
        st.warning(f"Insufficient history (< 30 days), excluded: {', '.join(low_history)}")
    if not valid:
        st.error("No tickers with sufficient price history after validation.")
        st.stop()

    # Benchmark fallback
    if BENCHMARK not in prices.columns:
        st.warning(f"Benchmark {BENCHMARK} unavailable; benchmark-relative metrics will be limited.")
        bench_prices = pd.Series(dtype=float)
        bench_returns = pd.Series(dtype=float)
    else:
        bench_prices = prices[BENCHMARK].dropna()
        bench_returns = bench_prices.pct_change().dropna()

    returns = compute_returns(prices[valid])

    # Build raw weights from the validated captured_df, restricted to valid tickers
    valid_df = captured_df[captured_df["ticker"].isin(valid)].copy()
    if valid_df.empty:
        st.error("No overlap between user tickers and successfully-fetched prices.")
        st.stop()

    raw_w = df_to_weight_series(
        valid_df, mode=input_mode or "equal",
        latest_prices=prices.iloc[-1] if "shares" in valid_df.columns else None,
    )

    try:
        weights, info_msgs = normalize_weights(
            raw_w, allow_short=allow_short, allow_leverage=allow_leverage
        )
    except ValueError as e:
        st.error(f"Weight normalization failed: {e}")
        st.stop()

    for msg in info_msgs:
        st.info(msg)

    # Final guard
    if weights.empty or weights.abs().sum() == 0:
        st.error("All weights are zero after validation.")
        st.stop()

    desc_map: dict[str, str] = {}
    if "description" in valid_df.columns:
        desc_map = (
            valid_df.dropna(subset=["ticker"])
                     .drop_duplicates(subset=["ticker"])
                     .set_index("ticker")["description"]
                     .to_dict()
        )

    final_tickers = list(weights.index)
    st.session_state.update(
        {
            "tickers": final_tickers,
            "weights": weights,
            "descriptions": desc_map,
            "prices": prices[final_tickers],
            "returns": returns[final_tickers],
            "benchmark_prices": bench_prices,
            "benchmark_returns": bench_returns,
            "portfolio_df": valid_df,
            "rf": rf,
            "period": period,
            "aum": aum,
            "allow_short": allow_short,
            "allow_leverage": allow_leverage,
        }
    )
    st.success(
        f"Loaded **{len(final_tickers)}** tickers with **{len(prices)}** days of data. "
        "Use the **left sidebar** to navigate analysis pages."
    )

# ─── Display loaded portfolio ────────────────────────────────────────────────
if "weights" in st.session_state:
    st.markdown("---")
    st.subheader("Current portfolio")
    w = st.session_state["weights"]
    prices = st.session_state["prices"]
    desc = st.session_state.get("descriptions", {})

    info_df = pd.DataFrame(
        {
            "ticker": w.index,
            "description": [desc.get(t, "") for t in w.index],
            "weight": w.values,
            "latest_price": prices.iloc[-1].reindex(w.index).values,
        }
    )
    display_df = info_df.copy()
    display_df["weight"] = display_df["weight"].apply(lambda x: f"{x:.2%}")
    display_df["latest_price"] = display_df["latest_price"].apply(
        lambda x: f"${x:,.2f}" if pd.notna(x) else "—"
    )

    # Exposure metrics — meaningful for long-only AND leveraged/long-short
    net = float(w.sum())
    gross = float(w.abs().sum())
    n_long = int((w > 0).sum())
    n_short = int((w < 0).sum())
    leverage = (gross / abs(net)) if abs(net) > 1e-9 else float("nan")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Assets (long / short)", f"{len(w)}", f"{n_long}L / {n_short}S")
    c2.metric("Net exposure", f"{net:.1%}")
    c3.metric("Gross exposure", f"{gross:.1%}")
    c4.metric(
        "Leverage", "—" if pd.isna(leverage) else f"{leverage:.2f}×",
        help="Gross ÷ |Net|. 1.0× = long-only fully invested; 1.6× = 130/30; ∞ = market-neutral."
    )

    c5, c6, c7 = st.columns(3)
    c5.metric("Lookback", st.session_state.get("period", "—"))
    c6.metric("Risk-free rate", f"{st.session_state.get('rf', 0):.2%}")
    c7.metric("Days of data", len(prices))

    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.dataframe(display_df, hide_index=True, width="stretch")
    with col_b:
        plot_df = info_df[info_df["weight"].abs() > 1e-6].copy()
        if not plot_df.empty:
            if (plot_df["weight"] < 0).any():
                # Long-short: use a sorted horizontal bar chart
                plot_df = plot_df.sort_values("weight")
                plot_df["side"] = plot_df["weight"].apply(lambda x: "Long" if x > 0 else "Short")
                fig = px.bar(
                    plot_df, x="weight", y="ticker", orientation="h", color="side",
                    color_discrete_map={"Long": "#2ecc71", "Short": "#e74c3c"},
                    title="Portfolio weights (long vs short)",
                )
                fig.update_xaxes(tickformat=".0%")
                fig.update_layout(height=max(300, 25 * len(plot_df)),
                                    margin=dict(t=40, b=20, l=20, r=20),
                                    yaxis=dict(title=None))
            else:
                fig = px.pie(plot_df, names="ticker", values="weight", hole=0.4)
                fig.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=400)
            st.plotly_chart(fig, width="stretch")

    # ─── Position breakdown by dollar amount and share count ──────────────────
    st.markdown("---")
    st.subheader("Position breakdown — by AUM")
    aum_loaded = float(st.session_state.get("aum", 100000.0))
    st.caption(
        f"Dollar amounts and share counts derived from **${aum_loaded:,.2f}** AUM "
        "× signed weight. Adjust AUM in the sidebar and re-fetch to update."
    )

    latest_prices = prices.iloc[-1].reindex(w.index)
    breakdown = pd.DataFrame({
        "ticker": w.index,
        "description": [desc.get(t, "") for t in w.index],
        "weight": w.values,
        "dollar_amount": w.values * aum_loaded,
        "latest_price": latest_prices.values,
    })
    # Shares = dollar / price. For foreign listings (price in local currency)
    # this is a USD-denominated unit count; flag in the column header.
    breakdown["shares"] = breakdown["dollar_amount"] / breakdown["latest_price"]

    # Portfolio-level dollar metrics
    long_mask = breakdown["weight"] > 0
    short_mask = breakdown["weight"] < 0
    total_long_dollars = float(breakdown.loc[long_mask, "dollar_amount"].sum())
    total_short_dollars = float(breakdown.loc[short_mask, "dollar_amount"].sum())
    net_dollars = float(breakdown["dollar_amount"].sum())
    gross_dollars = float(breakdown["dollar_amount"].abs().sum())

    m = st.columns(5)
    m[0].metric("Total AUM", f"${aum_loaded:,.0f}")
    m[1].metric("Long deployed", f"${total_long_dollars:,.0f}",
                  f"{(total_long_dollars / aum_loaded):.1%}" if aum_loaded else None)
    m[2].metric("Short proceeds", f"${total_short_dollars:,.0f}",
                  f"{(total_short_dollars / aum_loaded):.1%}" if aum_loaded else None)
    m[3].metric("Net invested", f"${net_dollars:,.0f}",
                  f"{(net_dollars / aum_loaded):.1%}" if aum_loaded else None)
    m[4].metric("Gross deployed", f"${gross_dollars:,.0f}",
                  f"{(gross_dollars / aum_loaded):.1%}" if aum_loaded else None)

    # Sanity-check: net invested should equal AUM × net exposure
    expected_net = aum_loaded * float(w.sum())
    if abs(net_dollars - expected_net) > 0.01:
        st.warning(
            f"Sanity check: net invested ${net_dollars:,.2f} ≠ AUM × net "
            f"weight ${expected_net:,.2f}. This shouldn't happen."
        )

    # Display formatted table
    disp_bd = breakdown.copy()
    disp_bd["weight"] = disp_bd["weight"].apply(lambda x: f"{x:+.2%}" if x != 0 else "0.00%")
    disp_bd["dollar_amount"] = disp_bd["dollar_amount"].apply(
        lambda x: f"${x:+,.2f}" if pd.notna(x) and x != 0 else "$0.00"
    )
    disp_bd["latest_price"] = disp_bd["latest_price"].apply(
        lambda x: f"${x:,.2f}" if pd.notna(x) else "—"
    )
    disp_bd["shares"] = disp_bd["shares"].apply(
        lambda x: f"{x:+,.2f}" if pd.notna(x) and x != 0 else "—"
    )
    disp_bd.columns = ["Ticker", "Description", "Weight",
                        "$ Position (long/short)", "Latest price",
                        "Shares (long/short)"]
    st.dataframe(disp_bd, hide_index=True, width="stretch")

    # Visual: dollar amounts per position
    plot_bd = breakdown.copy()
    plot_bd["side"] = plot_bd["weight"].apply(lambda x: "Long" if x > 0 else ("Short" if x < 0 else "Flat"))
    plot_bd = plot_bd.sort_values("dollar_amount")
    fig = px.bar(
        plot_bd, x="dollar_amount", y="ticker", orientation="h",
        color="side",
        color_discrete_map={"Long": "#2ecc71", "Short": "#e74c3c", "Flat": "#95a5a6"},
        title="Dollar position size per ticker",
        text=plot_bd["dollar_amount"].apply(lambda x: f"${x:+,.0f}"),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=max(350, 28 * len(plot_bd)),
                        margin=dict(t=50, b=20, l=20, r=80),
                        xaxis_title="USD position size",
                        yaxis=dict(title=None))
    st.plotly_chart(fig, width="stretch")

    # Note about foreign tickers
    foreign_tickers = [t for t in w.index
                        if t in prices.columns and st.session_state.get("descriptions", {})]
    # Detect foreign via currency: we don't have currency info loaded by default on Home.
    # Use a simple heuristic — tickers with a '.' or all-numeric main are likely foreign.
    likely_foreign = [t for t in w.index if "." in t or t.replace("-", "").isdigit()]
    if likely_foreign:
        st.caption(
            f"Note: foreign tickers detected ({', '.join(likely_foreign)}). "
            "Share counts shown are USD-amount ÷ local-currency price — the "
            "**Currency & Rates** page provides a proper FX-adjusted view."
        )

    # Download breakdown as CSV
    csv_bytes = breakdown.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download position breakdown as CSV",
        data=csv_bytes,
        file_name="position_breakdown.csv",
        mime="text/csv",
    )

    st.info("**Next:** open any analysis page from the left sidebar.")
else:
    st.info("Configure tickers + weights above and click **Fetch & validate portfolio**.")
