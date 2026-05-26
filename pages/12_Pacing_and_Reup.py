"""Commitment pacing model and reup/pullback decision support.

For each position in the portfolio, this page surfaces:
  - Vintage-to-date performance metrics (return, vol, Sharpe, alpha)
  - Current drawdown state and momentum
  - A composite reup score combining alpha significance, momentum,
    drawdown depth, Sharpe, and trend slope
  - Bootstrap-based forward probability of success / loss over a
    chosen horizon
  - Suggested target weights based on the reup categorization

Caveats: bootstrap assumes iid returns; alpha t-stat treats the residual
as iid normal; this is a diagnostic tool, not investment advice.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import analytics as A
import factor_models as FM
from data import portfolio_returns, require_portfolio
from theme import REC_PALETTE, badge, inject_css, style_recommendation_column

st.set_page_config(page_title="Pacing & Reup", layout="wide")
inject_css()
st.title("Commitment Pacing & Reup / Pullback Decisions")
st.caption(
    "Track each position's vintage-to-date performance, get composite reup signals "
    "with statistical backing, and estimate forward probabilities via bootstrap."
)

tickers, weights, prices, returns, bench_prices, bench_returns, rf = require_portfolio()
aum = float(st.session_state.get("aum", 100000.0))
port_ret = portfolio_returns(returns, weights)


# ─── Pacing computation per position ─────────────────────────────────────────

def compute_position_pacing(ticker: str, ret_series: pd.Series,
                              bench: pd.Series, rf: float,
                              weight: float, aum: float) -> dict | None:
    """Return a dict of vintage-to-date health/pacing metrics."""
    pr = ret_series.dropna()
    if len(pr) < 30:
        return None

    entry_date = pr.index[0]
    last_date = pr.index[-1]
    days_held = int((last_date - entry_date).days)

    cum_ret = float((1 + pr).prod() - 1)
    ann_ret = float(A.annualized_return(pr))
    ann_vol = float(A.annualized_vol(pr))
    sharpe = float(A.sharpe(pr, rf)) if ann_vol > 0 else np.nan

    capm = FM.capm(pr, bench, rf=rf) if not bench.empty else {}
    beta = float(capm.get("beta", np.nan)) if capm else np.nan
    alpha = float(capm.get("alpha_annual", np.nan)) if capm else np.nan
    alpha_tstat = float(capm.get("alpha_tstat", np.nan)) if capm else np.nan
    alpha_pvalue = float(capm.get("alpha_pvalue", np.nan)) if capm else np.nan
    r_squared = float(capm.get("r_squared", np.nan)) if capm else np.nan

    cum = (1 + pr).cumprod()
    peak = cum.cummax()
    dd = cum / peak - 1
    current_dd = float(dd.iloc[-1])
    max_dd = float(dd.min())
    days_since_peak = int((last_date - peak.idxmax()).days)

    # Recent vs full vol → regime indicator
    recent_vol = float(pr.iloc[-63:].std() * np.sqrt(252)) if len(pr) >= 63 else np.nan
    vol_regime = (recent_vol / ann_vol - 1) if ann_vol > 0 and not np.isnan(recent_vol) else np.nan

    # 12-1 momentum
    if len(pr) >= 252:
        mom_12_1 = float((1 + pr.iloc[-252:-21]).prod() - 1)
    elif len(pr) >= 63:
        mom_12_1 = float((1 + pr.iloc[:-21]).prod() - 1)
    else:
        mom_12_1 = np.nan

    # Trend slope: OLS slope of log cumulative returns over last 63 days
    if len(pr) >= 63:
        recent_log = np.log(1 + pr.iloc[-63:]).cumsum()
        x = np.arange(len(recent_log))
        slope = float(np.polyfit(x, recent_log.values, 1)[0])
    else:
        slope = np.nan

    return {
        "ticker": ticker,
        "weight": float(weight),
        "dollar_position": float(weight) * aum,
        "entry_date": entry_date.strftime("%Y-%m-%d"),
        "days_held": days_held,
        "n_obs": len(pr),
        "total_return": cum_ret,
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "beta": beta,
        "alpha": alpha,
        "alpha_tstat": alpha_tstat,
        "alpha_pvalue": alpha_pvalue,
        "r_squared": r_squared,
        "current_dd": current_dd,
        "max_dd": max_dd,
        "days_since_peak": days_since_peak,
        "recent_vol": recent_vol,
        "vol_regime": vol_regime,
        "mom_12_1": mom_12_1,
        "trend_slope": slope,
    }


def compute_reup_score(p: dict) -> dict:
    """Composite reup score in [-1, 1] from pacing metrics."""
    def safe_tanh(x, scale):
        if pd.isna(x):
            return 0.0
        return float(np.tanh(x * scale))

    alpha_s = safe_tanh(p["alpha_tstat"], 0.5)         # t-stat of ±3 → ±0.91
    mom_s = safe_tanh(p["mom_12_1"], 1.5)              # 30% mom → ~0.42
    dd_s = safe_tanh(p["current_dd"], 3.0)             # -20% DD → -0.76
    sharpe_s = safe_tanh(p["sharpe"], 0.7)             # Sharpe 1.0 → 0.60
    trend_s = safe_tanh(p["trend_slope"], 500.0)       # heuristic scaling

    composite = (alpha_s + mom_s + dd_s + sharpe_s + trend_s) / 5
    return {
        "alpha_score": alpha_s,
        "mom_score": mom_s,
        "dd_score": dd_s,
        "sharpe_score": sharpe_s,
        "trend_score": trend_s,
        "composite": float(composite),
    }


def categorize_reup(score: float) -> tuple[str, str]:
    """Map composite score to (recommendation, rationale)."""
    if score > 0.4:
        return ("REUP", "Strong positive signals — consider materially increasing the position")
    if score > 0.1:
        return ("ADD", "Modest positive signals — small add justified")
    if score > -0.1:
        return ("HOLD", "Mixed / neutral signals — maintain current size")
    if score > -0.4:
        return ("TRIM", "Modest negative signals — reduce position size")
    return ("EXIT", "Strong negative signals — consider full exit or major cut")


def forward_probabilities(ret_series: pd.Series, n_sims: int = 5000,
                            horizon: int = 252, seed: int = 42) -> dict | None:
    """Bootstrap-resample historical returns to estimate forward probabilities."""
    rets = ret_series.dropna().values
    if len(rets) < 30:
        return None
    rng = np.random.default_rng(seed)
    sampled = rng.choice(rets, size=(n_sims, horizon), replace=True)
    cum_paths = np.cumprod(1 + sampled, axis=1)
    terminal = cum_paths[:, -1]
    peaks = np.maximum.accumulate(cum_paths, axis=1)
    dds = (cum_paths / peaks - 1).min(axis=1)
    terminal_ret = terminal - 1
    return {
        "p_positive": float((terminal_ret > 0).mean()),
        "p_above_5": float((terminal_ret > 0.05).mean()),
        "p_above_10": float((terminal_ret > 0.10).mean()),
        "p_above_25": float((terminal_ret > 0.25).mean()),
        "p_below_neg10": float((terminal_ret < -0.10).mean()),
        "p_below_neg25": float((terminal_ret < -0.25).mean()),
        "p_dd_15": float((dds < -0.15).mean()),
        "p_dd_30": float((dds < -0.30).mean()),
        "p_dd_50": float((dds < -0.50).mean()),
        "median_return": float(np.median(terminal_ret)),
        "mean_return": float(terminal_ret.mean()),
        "p5": float(np.percentile(terminal_ret, 5)),
        "p10": float(np.percentile(terminal_ret, 10)),
        "p25": float(np.percentile(terminal_ret, 25)),
        "p50": float(np.percentile(terminal_ret, 50)),
        "p75": float(np.percentile(terminal_ret, 75)),
        "p90": float(np.percentile(terminal_ret, 90)),
        "p95": float(np.percentile(terminal_ret, 95)),
    }


# Compute pacing for all positions
with st.spinner("Computing pacing metrics..."):
    pacing_results: list[dict] = []
    for t in tickers:
        if t in returns.columns:
            p = compute_position_pacing(t, returns[t], bench_returns, rf,
                                          float(weights.get(t, 0)), aum)
            if p:
                p["scores"] = compute_reup_score(p)
                rec, msg = categorize_reup(p["scores"]["composite"])
                p["recommendation"] = rec
                p["rec_msg"] = msg
                pacing_results.append(p)

if not pacing_results:
    st.error("No positions have sufficient data (≥ 30 observations) to compute pacing metrics.")
    st.stop()


tabs = st.tabs([
    "Pacing tracker",
    "Reup / Pullback signals",
    "Forward probability of success",
    "Vintage cumulative returns",
    "Methodology",
])

# ══════════════════════════════════════════════════════════════════════════════
# Tab 1: Pacing tracker
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.subheader("Per-position vintage-to-date metrics")
    st.caption(
        "'Entry' = first available data point for that ticker within the lookback "
        "window. 'Days held' is your effective track record length for the position."
    )

    pacing_df = pd.DataFrame(pacing_results)
    display_df = pacing_df.copy()
    display_df["weight"] = display_df["weight"].apply(lambda x: f"{x:+.2%}")
    display_df["dollar_position"] = display_df["dollar_position"].apply(lambda x: f"${x:+,.0f}")

    pct_cols = ["total_return", "ann_return", "ann_vol", "alpha",
                 "current_dd", "max_dd", "mom_12_1", "vol_regime", "recent_vol"]
    for col in pct_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:+.2%}" if pd.notna(x) else "—"
            )
    for col in ["sharpe", "beta", "alpha_tstat", "alpha_pvalue", "r_squared", "trend_slope"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:.3f}" if pd.notna(x) else "—"
            )

    show_cols = ["ticker", "weight", "dollar_position", "entry_date", "days_held",
                  "total_return", "ann_return", "ann_vol", "sharpe", "beta", "alpha",
                  "alpha_tstat", "alpha_pvalue", "current_dd", "days_since_peak",
                  "max_dd", "mom_12_1", "vol_regime"]
    show_cols = [c for c in show_cols if c in display_df.columns]
    st.dataframe(display_df[show_cols], hide_index=True, width="stretch")

    # Visual: total return per position colored by alpha t-stat
    fig = px.bar(
        pacing_df.sort_values("total_return"),
        x="ticker", y="total_return", color="alpha_tstat",
        color_continuous_scale="RdBu_r", color_continuous_midpoint=0,
        hover_data=["weight", "ann_vol", "sharpe", "current_dd", "days_held"],
        title="Vintage-to-date return per position (color = alpha t-stat vs SPX)",
    )
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, width="stretch")

    # Current drawdown state
    fig = px.bar(
        pacing_df.sort_values("current_dd"),
        x="ticker", y="current_dd", color="days_since_peak",
        color_continuous_scale="Reds",
        hover_data=["weight", "max_dd"],
        title="Current drawdown by position (darker red = longer time since peak)",
    )
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, width="stretch")

# ══════════════════════════════════════════════════════════════════════════════
# Tab 2: Reup / Pullback signals
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.subheader("Composite reup / pullback signals")
    st.caption(
        "Composite score combines five normalized sub-scores (alpha t-stat, "
        "12-1 momentum, current drawdown, Sharpe, recent trend). "
        "**Higher = more reup-worthy.** See the **Methodology** tab for formulas."
    )

    sig_rows = []
    for p in pacing_results:
        s = p["scores"]
        sig_rows.append({
            "ticker": p["ticker"],
            "current_weight": p["weight"],
            "current_$": p["dollar_position"],
            "alpha_score": s["alpha_score"],
            "mom_score": s["mom_score"],
            "dd_score": s["dd_score"],
            "sharpe_score": s["sharpe_score"],
            "trend_score": s["trend_score"],
            "composite": s["composite"],
            "recommendation": p["recommendation"],
            "msg": p["rec_msg"],
            "alpha_tstat": p["alpha_tstat"],
            "mom_12_1": p["mom_12_1"],
            "current_dd": p["current_dd"],
            "sharpe": p["sharpe"],
        })

    sig_df = pd.DataFrame(sig_rows).sort_values("composite", ascending=False)

    # Counts of each recommendation
    rec_counts = sig_df["recommendation"].value_counts()
    rec_dollars = sig_df.groupby("recommendation")["current_$"].apply(lambda x: x.abs().sum())
    c = st.columns(5)
    for i, label in enumerate(["REUP", "ADD", "HOLD", "TRIM", "EXIT"]):
        n = int(rec_counts.get(label, 0))
        d = float(rec_dollars.get(label, 0))
        c[i].metric(label, n,
                     f"${d:,.0f} ({d / aum:.1%})" if aum > 0 else f"${d:,.0f}")

    # Table — color the Action column via Pandas Styler
    st.markdown("---")
    disp = sig_df[["ticker", "current_weight", "alpha_tstat", "mom_12_1",
                    "current_dd", "sharpe", "composite", "recommendation", "msg"]].copy()
    disp["current_weight"] = disp["current_weight"].apply(lambda x: f"{x:+.2%}")
    disp["alpha_tstat"] = disp["alpha_tstat"].apply(
        lambda x: f"{x:+.2f}" if pd.notna(x) else "—"
    )
    disp["mom_12_1"] = disp["mom_12_1"].apply(
        lambda x: f"{x:+.1%}" if pd.notna(x) else "—"
    )
    disp["current_dd"] = disp["current_dd"].apply(
        lambda x: f"{x:.1%}" if pd.notna(x) else "—"
    )
    disp["sharpe"] = disp["sharpe"].apply(
        lambda x: f"{x:.2f}" if pd.notna(x) else "—"
    )
    disp["composite"] = disp["composite"].apply(lambda x: f"{x:+.3f}")
    disp.columns = ["Ticker", "Weight", "α t-stat", "Mom 12-1",
                     "Curr DD", "Sharpe", "Score", "Action", "Rationale"]
    st.dataframe(style_recommendation_column(disp, "Action"),
                  hide_index=True, width="stretch")

    # Score waterfall
    fig = px.bar(
        sig_df, x="ticker", y="composite",
        color="composite", color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0,
        title="Composite reup score per position (higher = more reup-worthy)",
    )
    fig.add_hline(y=0.4, line_dash="dash", line_color="green",
                    annotation_text="REUP threshold")
    fig.add_hline(y=0.1, line_dash="dot", line_color="blue",
                    annotation_text="ADD")
    fig.add_hline(y=-0.1, line_dash="dot", line_color="orange",
                    annotation_text="TRIM")
    fig.add_hline(y=-0.4, line_dash="dash", line_color="red",
                    annotation_text="EXIT threshold")
    st.plotly_chart(fig, width="stretch")

    # Component score heatmap
    st.markdown("---")
    st.subheader("Component score breakdown")
    st.caption("Where is each signal coming from? Red = negative contribution, green = positive.")
    comp_cols = ["alpha_score", "mom_score", "dd_score", "sharpe_score", "trend_score"]
    comp_df = sig_df.set_index("ticker")[comp_cols]
    fig = px.imshow(comp_df, color_continuous_scale="RdBu_r",
                     color_continuous_midpoint=0, aspect="auto",
                     zmin=-1, zmax=1, text_auto=".2f",
                     labels=dict(x="Component", y="Ticker", color="Score"))
    fig.update_layout(title="Sub-scores by ticker", height=max(300, 25 * len(comp_df)))
    st.plotly_chart(fig, width="stretch")

    # Suggested rebalance
    st.markdown("---")
    st.subheader("Suggested rebalance")
    st.caption(
        "Heuristic target weights — **REUP**: +25%, **ADD**: +10%, **HOLD**: 0%, "
        "**TRIM**: -20%, **EXIT**: full cut. Suggested weights are renormalized to "
        "preserve your current **net exposure**. Use this as a sanity check, not a "
        "mechanical rule."
    )

    adj_map = {"REUP": 1.25, "ADD": 1.10, "HOLD": 1.0, "TRIM": 0.80, "EXIT": 0.0}

    rebal_rows = []
    for p in pacing_results:
        cur = p["weight"]
        adj = adj_map.get(p["recommendation"], 1.0)
        suggested_raw = cur * adj
        rebal_rows.append({
            "ticker": p["ticker"],
            "current_weight": cur,
            "current_$": cur * aum,
            "recommendation": p["recommendation"],
            "suggested_weight_raw": suggested_raw,
        })

    rebal_df = pd.DataFrame(rebal_rows)
    current_net = float(rebal_df["current_weight"].sum())
    suggested_net = float(rebal_df["suggested_weight_raw"].sum())
    if abs(suggested_net) > 1e-6:
        scale = current_net / suggested_net
        rebal_df["suggested_weight"] = rebal_df["suggested_weight_raw"] * scale
    else:
        rebal_df["suggested_weight"] = rebal_df["suggested_weight_raw"]

    rebal_df["weight_change"] = rebal_df["suggested_weight"] - rebal_df["current_weight"]
    rebal_df["suggested_$"] = rebal_df["suggested_weight"] * aum
    rebal_df["dollar_change"] = rebal_df["weight_change"] * aum

    disp_r = rebal_df[["ticker", "recommendation", "current_weight", "suggested_weight",
                        "weight_change", "current_$", "suggested_$", "dollar_change"]].copy()
    disp_r["current_weight"] = disp_r["current_weight"].apply(lambda x: f"{x:+.2%}")
    disp_r["suggested_weight"] = disp_r["suggested_weight"].apply(lambda x: f"{x:+.2%}")
    disp_r["weight_change"] = disp_r["weight_change"].apply(lambda x: f"{x:+.2%}")
    disp_r["current_$"] = disp_r["current_$"].apply(lambda x: f"${x:+,.0f}")
    disp_r["suggested_$"] = disp_r["suggested_$"].apply(lambda x: f"${x:+,.0f}")
    disp_r["dollar_change"] = disp_r["dollar_change"].apply(lambda x: f"${x:+,.0f}")
    disp_r.columns = ["Ticker", "Action", "Current Wt", "Suggested Wt", "Δ Weight",
                       "Current $", "Suggested $", "Δ Dollars"]
    st.dataframe(disp_r, hide_index=True, width="stretch")

    # Current vs suggested comparison
    rebal_long = rebal_df.melt(
        id_vars=["ticker"], value_vars=["current_weight", "suggested_weight"],
        var_name="basis", value_name="weight",
    )
    rebal_long["basis"] = rebal_long["basis"].map({
        "current_weight": "Current", "suggested_weight": "Suggested"
    })
    fig = px.bar(rebal_long, x="ticker", y="weight", color="basis", barmode="group",
                  color_discrete_map={"Current": "#3498db", "Suggested": "#16a085"},
                  title="Current vs suggested weights")
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, width="stretch")

    # Sanity checks
    new_net = float(rebal_df["suggested_weight"].sum())
    new_gross = float(rebal_df["suggested_weight"].abs().sum())
    c = st.columns(4)
    c[0].metric("Current net", f"{current_net:.1%}")
    c[1].metric("Suggested net", f"{new_net:.1%}",
                 f"{(new_net - current_net):+.1%}")
    cur_gross = float(rebal_df["current_weight"].abs().sum())
    c[2].metric("Current gross", f"{cur_gross:.1%}")
    c[3].metric("Suggested gross", f"{new_gross:.1%}",
                 f"{(new_gross - cur_gross):+.1%}")

# ══════════════════════════════════════════════════════════════════════════════
# Tab 3: Forward probability of success
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.subheader("Statistical probability of success — bootstrap simulation")
    st.caption(
        "Each position's historical daily returns are resampled with replacement "
        "to generate **n** forward paths over the chosen horizon. Probabilities are "
        "the fraction of paths satisfying each event. **Caveat:** assumes future "
        "returns are iid samples from the past distribution — fragile in regime shifts."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        horizon = st.slider("Forward horizon (trading days)",
                              21, 504, 252, step=21,
                              help="252 = 1 year, 504 = 2 years")
    with c2:
        n_sims = st.slider("Bootstrap simulations",
                             1000, 20000, 5000, step=1000)
    with c3:
        seed_val = st.number_input("Random seed (for reproducibility)",
                                       value=42, step=1)

    if st.button("Run forward probability analysis", type="primary"):
        prob_rows = []
        with st.spinner(f"Simulating {n_sims:,} paths × {horizon}d for each position..."):
            for p in pacing_results:
                t = p["ticker"]
                if t not in returns.columns:
                    continue
                probs = forward_probabilities(returns[t], n_sims=n_sims,
                                                 horizon=horizon, seed=int(seed_val))
                if probs:
                    row = {
                        "ticker": t,
                        "weight": p["weight"],
                        "current_$": p["dollar_position"],
                        "recommendation": p["recommendation"],
                    }
                    row.update(probs)
                    prob_rows.append(row)

        if prob_rows:
            prob_df = pd.DataFrame(prob_rows)
            st.session_state["pacing_probs"] = prob_df

    if "pacing_probs" in st.session_state:
        prob_df = st.session_state["pacing_probs"]

        # Portfolio level first
        with st.spinner("Computing portfolio-level forward probabilities..."):
            port_probs = forward_probabilities(port_ret, n_sims=n_sims,
                                                  horizon=horizon, seed=int(seed_val))

        if port_probs:
            st.subheader(f"Portfolio-level forward probabilities ({horizon}d)")
            c = st.columns(5)
            c[0].metric("P(positive)", f"{port_probs['p_positive']:.1%}")
            c[1].metric("P(> +10%)", f"{port_probs['p_above_10']:.1%}")
            c[2].metric("P(> +25%)", f"{port_probs['p_above_25']:.1%}")
            c[3].metric("P(< -10%)", f"{port_probs['p_below_neg10']:.1%}")
            c[4].metric("P(DD > 30%)", f"{port_probs['p_dd_30']:.1%}")

            c2 = st.columns(5)
            c2[0].metric("Median return", f"{port_probs['median_return']:+.1%}")
            c2[1].metric("5th pctile", f"{port_probs['p5']:+.1%}",
                            help="VaR-style worst-case outcome at 95% confidence")
            c2[2].metric("25th pctile", f"{port_probs['p25']:+.1%}")
            c2[3].metric("75th pctile", f"{port_probs['p75']:+.1%}")
            c2[4].metric("95th pctile", f"{port_probs['p95']:+.1%}",
                            help="95th-percentile (upside) outcome")

        st.markdown("---")
        st.subheader("Per-position forward probabilities")

        disp = prob_df.copy()
        for col in ["p_positive", "p_above_5", "p_above_10", "p_above_25",
                      "p_below_neg10", "p_below_neg25", "p_dd_15", "p_dd_30", "p_dd_50"]:
            if col in disp.columns:
                disp[col] = disp[col].apply(lambda x: f"{x:.1%}")
        for col in ["median_return", "mean_return", "p5", "p10", "p25", "p50", "p75", "p90", "p95"]:
            if col in disp.columns:
                disp[col] = disp[col].apply(lambda x: f"{x:+.1%}")
        disp["weight"] = disp["weight"].apply(lambda x: f"{x:+.2%}")
        disp["current_$"] = disp["current_$"].apply(lambda x: f"${x:+,.0f}")

        show_cols = ["ticker", "weight", "recommendation",
                      "p_positive", "p_above_10", "p_above_25",
                      "p_below_neg10", "p_below_neg25",
                      "p_dd_15", "p_dd_30",
                      "median_return", "p5", "p25", "p75", "p95"]
        show_cols = [c for c in show_cols if c in disp.columns]
        st.dataframe(disp[show_cols], hide_index=True, width="stretch")

        # Upside probability chart
        upside_df = prob_df.sort_values("p_positive")
        fig = px.bar(
            upside_df, x="ticker",
            y=["p_positive", "p_above_10", "p_above_25"],
            barmode="group",
            title=f"Upside probability over {horizon}d horizon",
            labels={"value": "Probability", "variable": "Threshold"},
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, width="stretch")

        # Downside probability chart
        downside_df = prob_df.sort_values("p_dd_30", ascending=False)
        fig = px.bar(
            downside_df, x="ticker",
            y=["p_below_neg10", "p_below_neg25", "p_dd_15", "p_dd_30"],
            barmode="group",
            title=f"Downside risk probability over {horizon}d horizon",
            labels={"value": "Probability", "variable": "Event"},
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, width="stretch")

        # Reup/exit overlay: scatter of P(positive) vs current weight, colored by rec
        st.markdown("---")
        st.subheader("Reup signal overlay on probability of success")
        st.caption(
            "Top-right (high P(positive), low weight) suggests room to add. "
            "Top-left (high P, high weight) suggests positions to maintain. "
            "Bottom (low P) regardless of weight suggests reduce/exit."
        )
        rec_color_map = {"REUP": "#27ae60", "ADD": "#3498db",
                          "HOLD": "#95a5a6", "TRIM": "#f39c12", "EXIT": "#c0392b"}
        fig = px.scatter(
            prob_df, x="weight", y="p_positive",
            color="recommendation", color_discrete_map=rec_color_map,
            size="p_above_25", text="ticker",
            hover_data=["p_above_10", "p_below_neg25", "p_dd_30"],
            title=f"P(positive over {horizon}d) vs current weight",
            labels={"weight": "Current weight", "p_positive": f"P(positive over {horizon}d)"},
        )
        fig.update_traces(textposition="top center")
        fig.update_xaxes(tickformat=".0%")
        fig.update_yaxes(tickformat=".0%")
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray",
                        annotation_text="50/50 line")
        st.plotly_chart(fig, width="stretch")
    else:
        st.info(
            "Configure horizon / sims / seed above and click "
            "**Run forward probability analysis** to populate this section."
        )

# ══════════════════════════════════════════════════════════════════════════════
# Tab 4: Vintage cumulative returns
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.subheader("Cumulative return since inception, per position")
    st.caption(
        "'Inception' = earliest available data point per ticker within the loaded "
        "lookback window. Useful for spotting vintage outperformers and laggards."
    )

    cum_records = []
    for p in pacing_results:
        t = p["ticker"]
        if t not in returns.columns:
            continue
        pr = returns[t].dropna()
        if len(pr) < 30:
            continue
        cum = (1 + pr).cumprod() - 1
        for date, val in cum.items():
            cum_records.append({
                "ticker": t, "date": date, "cum_return": float(val),
                "recommendation": p["recommendation"],
            })

    if cum_records:
        cum_df = pd.DataFrame(cum_records)
        fig = px.line(
            cum_df, x="date", y="cum_return", color="ticker",
            title="Cumulative return per position since first data point",
        )
        fig.update_yaxes(tickformat=".0%")
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, width="stretch")

        # Final return colored by recommendation
        final_rows = []
        for p in pacing_results:
            t = p["ticker"]
            if t not in returns.columns:
                continue
            cum = (1 + returns[t].dropna()).cumprod() - 1
            if len(cum) > 0:
                final_rows.append({
                    "ticker": t,
                    "since_inception_return": float(cum.iloc[-1]),
                    "weight": p["weight"],
                    "recommendation": p["recommendation"],
                })
        final_df = pd.DataFrame(final_rows).sort_values("since_inception_return")
        rec_color_map = {"REUP": "#27ae60", "ADD": "#3498db",
                          "HOLD": "#95a5a6", "TRIM": "#f39c12", "EXIT": "#c0392b"}
        fig = px.bar(
            final_df, x="ticker", y="since_inception_return",
            color="recommendation", color_discrete_map=rec_color_map,
            hover_data=["weight"],
            title="Since-inception return per position, colored by reup recommendation",
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, width="stretch")

        # Days held vs total return scatter
        scatter_rows = []
        for p in pacing_results:
            scatter_rows.append({
                "ticker": p["ticker"],
                "days_held": p["days_held"],
                "total_return": p["total_return"],
                "ann_return": p["ann_return"],
                "weight": p["weight"],
                "recommendation": p["recommendation"],
            })
        scat_df = pd.DataFrame(scatter_rows)
        fig = px.scatter(
            scat_df, x="days_held", y="total_return", text="ticker",
            color="recommendation", color_discrete_map=rec_color_map,
            size=np.abs(scat_df["weight"]) * 100 + 5,
            hover_data=["ann_return", "weight"],
            title="Days held vs total return (size = |weight|)",
        )
        fig.update_traces(textposition="top center")
        fig.update_yaxes(tickformat=".0%")
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, width="stretch")

# ══════════════════════════════════════════════════════════════════════════════
# Tab 5: Methodology
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.subheader("Methodology")
    st.markdown(r"""
#### Pacing tracker
For each position we compute:
- **Days held** — calendar days between earliest data point and most recent
- **Total return** — $\prod(1 + r_t) - 1$ over the holding window
- **Annualized return / vol** — daily mean × 252, daily std × √252
- **Sharpe** — $(r_{\text{ann}} - r_f) / \sigma_{\text{ann}}$
- **Alpha & t-stat** — OLS regression of position excess returns on market excess returns:
  $$r_p - r_f = \alpha + \beta (r_m - r_f) + \epsilon$$
  Alpha t-stat > 2 (or p-value < 0.05) suggests statistically significant outperformance.
- **Current drawdown** — $(V_t / \max(V_{0..t})) - 1$
- **Days since peak** — duration of the current drawdown episode
- **12-1 momentum** — return from day -252 to day -21
- **Trend slope** — OLS slope of `log(cumulative return)` over the last 63 trading days
- **Vol regime** — recent 63-day vol vs full-period vol (positive = elevated vol now)

#### Reup score
Composite of 5 normalized sub-scores in [-1, 1]:

| Sub-score | Formula | Interpretation |
|---|---|---|
| Alpha score | $\tanh(\text{α t-stat} \times 0.5)$ | Statistical significance of outperformance |
| Momentum | $\tanh(\text{12-1 return} \times 1.5)$ | Recent price strength |
| Drawdown | $\tanh(\text{current DD} \times 3)$ | How deep below peak |
| Sharpe | $\tanh(\text{Sharpe} \times 0.7)$ | Risk-adjusted return |
| Trend | $\tanh(\text{slope} \times 500)$ | Recent direction |

**Composite** = mean of the five. **Categorization**:

| Score | Action |
|---|---|
| > +0.4 | REUP — strong positive signals |
| +0.1 to +0.4 | ADD — modest positive |
| -0.1 to +0.1 | HOLD — mixed / neutral |
| -0.4 to -0.1 | TRIM — modest negative |
| < -0.4 | EXIT — strong negative |

#### Suggested rebalance
Per-action multiplier on current weight, then renormalized so the suggested weights sum to the same **net exposure** as today:

| Action | Multiplier |
|---|---|
| REUP | × 1.25 |
| ADD | × 1.10 |
| HOLD | × 1.00 |
| TRIM | × 0.80 |
| EXIT | × 0.00 |

Note: this preserves net exposure but may shift **gross** exposure (relevant for 130/30).

#### Forward probability analysis
For each position, bootstrap-resample historical daily returns (with replacement) over the chosen horizon (e.g. 252 = 1 year), running n_sims independent paths.

For each simulated path, compute:
- Terminal cumulative return
- Maximum drawdown along the path

Then count the fraction of paths satisfying each event:
- **P(positive)** — terminal return > 0
- **P(> +10% / +25%)** — outperformance thresholds
- **P(< -10% / -25%)** — loss thresholds
- **P(DD > 15% / 30% / 50%)** — drawdown thresholds

Empirical percentiles (5th, 25th, 50th, 75th, 95th) of terminal returns are also reported — the **5th percentile** is the bootstrap analog of historical VaR at 95% confidence over the chosen horizon.

#### Caveats
- **iid assumption** — Bootstrap treats returns as independently identically distributed. Real markets exhibit regime changes, volatility clustering, and serial correlation that this method ignores. Use the **Statistical Tests** page to check whether your portfolio's returns plausibly satisfy iid.
- **Backward-looking only** — The strongest historical alpha doesn't guarantee continued alpha. Use this as one input alongside fundamental research.
- **Heuristic thresholds** — The 0.4 / 0.1 / -0.1 / -0.4 boundaries are pragmatic, not optimal. Adjust to match your risk tolerance.
- **Not financial advice** — This is a diagnostic tool. Combine with your conviction, fundamental research, and risk management framework.
""")
