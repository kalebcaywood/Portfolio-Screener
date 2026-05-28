"""Return Stream Analyzer — Risk metrics: VaR, CVaR, distribution, drawdown.

Frequency-aware VaR/CVaR computation across four methods (historical,
parametric, Cornish-Fisher, Monte Carlo) at multiple confidence levels.
Per-stream view with the option to compare to the picked benchmark.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats as sps

import risk as R
import rsa as RS
from data import benchmark_picker_and_data
from theme import inject_css

inject_css()

st.title("Risk Metrics")
st.caption("VaR, CVaR, tail-risk diagnostics, and drawdown analytics for the selected return stream.")

# ─── Guard + selection ──────────────────────────────────────────────────────
if "rsa_returns" not in st.session_state:
    st.warning("No return streams loaded. Go to **Return Stream Analyzer → Home** first.")
    st.stop()

returns_df: pd.DataFrame = st.session_state["rsa_returns"]
freq: str = st.session_state.get("rsa_frequency", "M")
streams: list[str] = st.session_state["rsa_streams"]
rf: float = float(st.session_state.get("rsa_rf", 0.04))

st.sidebar.header("Selection")
stream = st.sidebar.selectbox("Return stream", streams, key="rsa_risk_stream")
notional = st.sidebar.number_input(
    "Notional ($)", 1000.0, 1e12,
    float(st.session_state.get("aum", 1_000_000.0)),
    step=1000.0,
)
alphas_str = st.sidebar.text_input("Confidence alphas (comma-separated)", "0.01, 0.05, 0.10")
try:
    alphas = tuple(float(x) for x in alphas_str.split(","))
except Exception:
    alphas = (0.01, 0.05, 0.10)

bench_name, _, bench_returns_daily = benchmark_picker_and_data()

r = returns_df[stream].dropna()
if r.empty:
    st.error(f"Stream {stream} is empty.")
    st.stop()

st.markdown(f"### {stream} — {RS.FREQ_LABEL.get(freq, freq).lower()} risk profile")
st.caption(f"{len(r):,} observations · benchmark: **{bench_name}** · notional: **${notional:,.0f}**")

# ─── Headline tail-risk metrics ─────────────────────────────────────────────
skew_v = float(r.skew())
kurt_v = float(r.kurtosis())
max_dd = RS.max_drawdown(r)
worst_day = float(r.min())
best_day = float(r.max())
ulcer = float(np.sqrt((RS.drawdown_series(r) ** 2).mean()))

m = st.columns(6)
m[0].metric("Skewness", f"{skew_v:.3f}")
m[1].metric("Excess kurtosis", f"{kurt_v:.3f}")
m[2].metric("Max drawdown", f"{max_dd:.2%}")
m[3].metric("Ulcer Index", f"{ulcer:.4f}")
m[4].metric("Best period", f"{best_day:+.2%}")
m[5].metric("Worst period", f"{worst_day:+.2%}")

st.markdown("---")
st.subheader("Value at Risk")
st.caption(
    "Period-level VaR (one period = one observation in your data's frequency). "
    "Multiply by notional for $ amounts; for multi-period horizons use sqrt-time "
    "scaling under iid assumption."
)

# Build VaR table — methods × confidence levels
var_df = R.var_summary(r, alphas=alphas)
dollar_df = var_df.copy()
for col in dollar_df.columns:
    if col != "confidence":
        dollar_df[col] = dollar_df[col].apply(
            lambda v: f"{v:.3%}  (${v * notional:+,.0f})" if pd.notna(v) else "—"
        )
st.dataframe(dollar_df, hide_index=True, width="stretch")

# Distribution histogram with VaR markers
fig = go.Figure()
fig.add_histogram(x=r, nbinsx=40, name="Period returns",
                    marker_color="#cbd5e1", opacity=0.85)
colors = ["#FF8200", "#b45309", "#b91c1c"]
for i, a in enumerate(alphas):
    v = R.var_historical(r, a)
    fig.add_vline(x=v, line_dash="dash", line_color=colors[i % len(colors)],
                    annotation_text=f"VaR {1 - a:.0%}",
                    annotation_position="top")
fig.update_layout(title="Period-return distribution with VaR thresholds",
                    xaxis_tickformat=".1%", showlegend=False, height=420)
st.plotly_chart(fig, width="stretch")

# ─── Distribution percentile table ──────────────────────────────────────────
st.markdown("---")
st.subheader("Percentile profile")
pcts = [1, 2.5, 5, 10, 25, 50, 75, 90, 95, 97.5, 99]
prow = {f"P{p}": float(np.percentile(r, p)) for p in pcts}
prow_df = pd.DataFrame([{k: f"{v:+.2%}" for k, v in prow.items()}])
st.dataframe(prow_df, hide_index=True, width="stretch")

# ─── Drawdown analytics ────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Drawdown analysis")

dd = RS.drawdown_series(r)
fig = px.area(dd, title=f"{stream} — drawdown")
fig.update_yaxes(tickformat=".0%")
fig.update_layout(showlegend=False, height=360)
st.plotly_chart(fig, width="stretch")

# Top drawdown episodes
def _drawdown_episodes(returns: pd.Series, top_n: int = 8) -> pd.DataFrame:
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd_s = cum / peak - 1
    if dd_s.empty:
        return pd.DataFrame()
    episodes = []
    in_dd = False
    peak_date = trough_date = None
    trough_val = 0.0
    for date, val in dd_s.items():
        if val < -1e-8 and not in_dd:
            in_dd = True
            peak_date = trough_date = date
            trough_val = val
        elif val < -1e-8 and in_dd:
            if val < trough_val:
                trough_val = val
                trough_date = date
        elif val >= -1e-8 and in_dd:
            episodes.append({
                "Peak": peak_date.strftime("%Y-%m-%d"),
                "Trough": trough_date.strftime("%Y-%m-%d"),
                "Recovery": date.strftime("%Y-%m-%d"),
                "Depth": trough_val,
                "Drawdown periods": (trough_date - peak_date).days,
                "Recovery periods": (date - trough_date).days,
            })
            in_dd = False
    if in_dd:
        episodes.append({
            "Peak": peak_date.strftime("%Y-%m-%d"),
            "Trough": trough_date.strftime("%Y-%m-%d"),
            "Recovery": "ongoing",
            "Depth": trough_val,
            "Drawdown periods": (trough_date - peak_date).days,
            "Recovery periods": None,
        })
    df = pd.DataFrame(episodes)
    if df.empty:
        return df
    return df.sort_values("Depth").head(top_n).reset_index(drop=True)

eps = _drawdown_episodes(r)
if not eps.empty:
    st.markdown("##### Top drawdown episodes")
    disp = eps.copy()
    disp["Depth"] = disp["Depth"].apply(lambda x: f"{x:.2%}")
    st.dataframe(disp, hide_index=True, width="stretch")

# ─── Benchmark relative VaR ─────────────────────────────────────────────────
_, bench_aligned = RS.align_to_period(returns_df[[stream]], bench_returns_daily, freq)
if not bench_aligned.empty:
    st.markdown("---")
    st.subheader(f"Relative-risk vs {bench_name}")
    common = r.index.intersection(bench_aligned.dropna().index)
    if len(common) >= 6:
        active = (r.loc[common] - bench_aligned.dropna().loc[common])
        rows = []
        for a in alphas:
            v_self = R.var_historical(r.loc[common], a)
            v_bench = R.var_historical(bench_aligned.dropna().loc[common], a)
            v_active = R.var_historical(active, a)
            rows.append({
                "Confidence": f"{1 - a:.0%}",
                f"{stream} VaR": f"{v_self:.3%}",
                f"{bench_name} VaR": f"{v_bench:.3%}",
                "Active VaR (stream − bench)": f"{v_active:+.3%}",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(
            "Active VaR is the tail of the **active return** (stream return minus "
            "benchmark return). Magnitudes reflect tracking error of the worst periods."
        )
