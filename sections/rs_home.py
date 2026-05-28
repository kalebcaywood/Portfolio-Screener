"""Return Stream Analyzer — Home page (upload + parser + stream management).

Drop a CSV of fund/manager returns and this page parses it, auto-detects the
frequency (daily/weekly/monthly/quarterly), shows a clean preview, and stores
the result in session_state so every other RSA page can use it without
re-uploading.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import rsa as RS
from theme import inject_css

inject_css()

st.title("Return Stream Analyzer")
st.caption(
    "Upload pure return time series (no holdings required). Auto-detects format "
    "and frequency. Once loaded, every RSA tab reads from this same data."
)

# ─── Sample CSV for quick testing ───────────────────────────────────────────
SAMPLE_CSV = """Date,GrowthFund,ValueFund,SmallCap,MarketNeutral
2021-01-31,0.0142,0.0089,0.0167,0.0034
2021-02-28,-0.0273,0.0142,0.0021,-0.0012
2021-03-31,0.0410,0.0521,0.0386,0.0089
2021-04-30,0.0512,0.0274,0.0301,0.0048
2021-05-31,0.0089,0.0186,-0.0042,0.0017
2021-06-30,0.0231,-0.0042,0.0152,0.0058
2021-07-31,0.0301,0.0118,0.0094,0.0028
2021-08-31,0.0257,0.0204,0.0173,0.0041
2021-09-30,-0.0428,-0.0521,-0.0612,-0.0028
2021-10-31,0.0712,0.0532,0.0489,0.0078
2021-11-30,-0.0042,-0.0163,-0.0289,0.0019
2021-12-31,0.0421,0.0589,0.0312,0.0067
2022-01-31,-0.0712,-0.0231,-0.0921,-0.0034
2022-02-28,-0.0312,-0.0089,-0.0218,0.0012
2022-03-31,0.0418,0.0367,0.0489,0.0058
2022-04-30,-0.0921,-0.0412,-0.0612,-0.0089
2022-05-31,-0.0231,0.0089,-0.0312,0.0021
2022-06-30,-0.0812,-0.0621,-0.0921,-0.0058
2022-07-31,0.0921,0.0521,0.0721,0.0089
2022-08-31,-0.0421,-0.0312,-0.0512,-0.0023
2022-09-30,-0.0921,-0.0712,-0.0821,-0.0078
2022-10-31,0.0721,0.0812,0.0612,0.0089
2022-11-30,0.0512,0.0612,0.0312,0.0067
2022-12-31,-0.0612,-0.0312,-0.0512,-0.0019
2023-01-31,0.0721,0.0512,0.0612,0.0089
2023-02-28,-0.0212,-0.0089,-0.0312,0.0023
2023-03-31,0.0312,0.0421,0.0218,0.0058
2023-04-30,0.0212,0.0312,0.0089,0.0034
2023-05-31,0.0089,-0.0089,-0.0212,0.0021
2023-06-30,0.0521,0.0412,0.0612,0.0078
2023-07-31,0.0312,0.0412,0.0521,0.0058
2023-08-31,-0.0312,-0.0212,-0.0421,-0.0017
2023-09-30,-0.0512,-0.0412,-0.0612,-0.0034
2023-10-31,-0.0312,-0.0312,-0.0421,-0.0029
2023-11-30,0.0921,0.0812,0.0921,0.0089
2023-12-31,0.0521,0.0412,0.0612,0.0067
2024-01-31,0.0312,0.0089,-0.0089,0.0048
2024-02-29,0.0612,0.0421,0.0512,0.0058
2024-03-31,0.0412,0.0312,0.0312,0.0048
2024-04-30,-0.0312,-0.0212,-0.0512,-0.0019
2024-05-31,0.0521,0.0412,0.0312,0.0058
2024-06-30,0.0421,0.0089,-0.0089,0.0034
2024-07-31,0.0212,0.0521,0.0721,0.0048
2024-08-31,0.0312,0.0212,0.0089,0.0028
2024-09-30,0.0212,0.0312,0.0212,0.0048
2024-10-31,-0.0089,-0.0212,-0.0312,-0.0011
2024-11-30,0.0712,0.0612,0.0821,0.0089
2024-12-31,-0.0212,-0.0089,-0.0312,-0.0019
2025-01-31,0.0312,0.0212,0.0089,0.0048
2025-02-28,-0.0089,0.0089,-0.0212,0.0023
2025-03-31,-0.0312,-0.0089,-0.0412,-0.0017
2025-04-30,-0.0421,-0.0212,-0.0621,-0.0034
2025-05-31,0.0612,0.0421,0.0512,0.0067
2025-06-30,0.0521,0.0312,0.0421,0.0048
2025-07-31,0.0312,0.0421,0.0612,0.0058
2025-08-31,0.0312,0.0212,0.0312,0.0048
2025-09-30,0.0089,0.0212,0.0089,0.0028
2025-10-31,0.0212,0.0312,0.0512,0.0048
2025-11-30,0.0421,0.0312,0.0312,0.0058
2025-12-31,-0.0212,-0.0089,-0.0312,-0.0019
"""

# ─── Upload UI ───────────────────────────────────────────────────────────────
c1, c2 = st.columns([3, 1])
with c1:
    uploaded = st.file_uploader(
        "Upload return-stream CSV",
        type=["csv"],
        help=(
            "Wide format: Date column + one return column per fund.\n"
            "Long format: Date, Stream/Fund, Return columns.\n"
            "Returns can be decimals (0.0142) or percent strings ('1.42%')."
        ),
    )
with c2:
    st.markdown("&nbsp;")
    st.download_button(
        "Download sample CSV",
        data=SAMPLE_CSV,
        file_name="sample_returns.csv",
        mime="text/csv",
        help="60 months of synthetic returns for 4 funds — edit and re-upload",
        width="stretch",
    )

with st.expander("Accepted formats", expanded=False):
    st.code(
        "# Format 1: WIDE — one column per stream\n"
        "Date,GrowthFund,ValueFund,SmallCap\n"
        "2024-01-31,0.0142,0.0089,0.0167\n"
        "2024-02-29,-0.0273,0.0142,0.0021\n"
        "\n"
        "# Format 2: LONG — one row per stream-period\n"
        "Date,Stream,Return\n"
        "2024-01-31,GrowthFund,0.0142\n"
        "2024-01-31,ValueFund,0.0089\n"
        "2024-02-29,GrowthFund,-0.0273\n"
        "\n"
        "# Format 3: PERCENT STRINGS (auto-detected)\n"
        "Date,FundA\n"
        "2024-01-31,1.42%\n"
        "2024-02-29,-2.73%\n"
        "\n"
        "Recognized date columns: date, period, month, quarter, asofdate, monthend\n"
        "Recognized stream columns: stream, fund, manager, strategy, name\n"
        "Recognized return columns: return, ret, performance, monthlyreturn, tr",
        language="csv",
    )

# ─── Sidebar settings ───────────────────────────────────────────────────────
st.sidebar.header("RSA settings")
rf_input = st.sidebar.number_input(
    "Risk-free rate (annual)", 0.0, 0.20, float(st.session_state.get("rsa_rf", 0.04)),
    step=0.005, format="%.3f",
)
st.session_state["rsa_rf"] = rf_input

if uploaded is None and "rsa_returns" not in st.session_state:
    st.info("Upload a return CSV or download the sample above to get started.")
    st.stop()

# ─── Parse ──────────────────────────────────────────────────────────────────
if uploaded is not None:
    with st.spinner("Parsing CSV..."):
        parsed = RS.parse_return_csv(uploaded)

    for w in parsed["warnings"]:
        st.warning(w)
    for e in parsed["errors"]:
        st.error(e)
    if parsed["df"] is None:
        st.stop()

    st.session_state["rsa_returns"] = parsed["df"]
    st.session_state["rsa_frequency"] = parsed["frequency"]
    st.session_state["rsa_format"] = parsed["format"]
    st.session_state["rsa_streams"] = parsed["streams"]

returns_df: pd.DataFrame = st.session_state["rsa_returns"]
freq: str = st.session_state["rsa_frequency"]
streams: list[str] = st.session_state["rsa_streams"]

# ─── Detected summary ───────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Detected")

mcol = st.columns(5)
mcol[0].metric("Return streams", len(streams))
mcol[1].metric("Frequency", RS.FREQ_LABEL.get(freq, freq))
mcol[2].metric("Periods per year", RS.periods_per_year(freq))
mcol[3].metric("Observations", len(returns_df))
date_span = f"{returns_df.index.min():%Y-%m-%d} → {returns_df.index.max():%Y-%m-%d}"
mcol[4].metric("Date range", date_span.split(" → ")[1], delta=f"from {date_span.split(' → ')[0]}")

# Allow user to override detected frequency
new_freq = st.selectbox(
    "Override frequency (if auto-detection is wrong)",
    options=["D", "W", "M", "Q", "A"],
    index=["D", "W", "M", "Q", "A"].index(freq),
    format_func=lambda x: f"{RS.FREQ_LABEL.get(x, x)} ({RS.periods_per_year(x)} per year)",
)
if new_freq != freq:
    st.session_state["rsa_frequency"] = new_freq
    freq = new_freq

st.markdown("---")
st.subheader("Cumulative growth — all streams")

cum = (1 + returns_df.fillna(0)).cumprod()
fig = px.line(
    cum, title=f"Growth of $1 (rebased) — {RS.FREQ_LABEL.get(freq, freq)}",
    labels={"value": "Indexed level (start = 1)", "index": "Date", "variable": "Stream"},
)
fig.update_layout(height=440, hovermode="x unified")
st.plotly_chart(fig, width="stretch")

# ─── Quick stats per stream ─────────────────────────────────────────────────
st.markdown("---")
st.subheader("Quick stats")

rows = []
for s in streams:
    stats = RS.summary_stats(returns_df[s].dropna(), freq=freq, rf=rf_input)
    rows.append({
        "Stream": s,
        "Obs": stats["n_obs"],
        "Total return": stats["total_return"],
        "CAGR": stats["cagr"],
        "Ann. vol": stats["ann_vol"],
        "Sharpe": stats["sharpe"],
        "Sortino": stats["sortino"],
        "Calmar": stats["calmar"],
        "Max DD": stats["max_drawdown"],
        "Hit rate": stats["hit_rate"],
        "Skew": stats["skew"],
    })

stats_df = pd.DataFrame(rows)
disp = stats_df.copy()
for col in ["Total return", "CAGR", "Ann. vol", "Max DD", "Hit rate"]:
    disp[col] = disp[col].apply(lambda x: f"{x:+.2%}" if pd.notna(x) and col != "Hit rate" else (f"{x:.1%}" if pd.notna(x) else "—"))
for col in ["Sharpe", "Sortino", "Calmar", "Skew"]:
    disp[col] = disp[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
st.dataframe(disp, hide_index=True, width="stretch")

# ─── Raw return preview ─────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Raw returns (first 12 + last 5 periods)")
combined = pd.concat([returns_df.head(12), returns_df.tail(5)])
disp = combined.copy()
for c in disp.columns:
    disp[c] = disp[c].apply(lambda x: f"{x:+.2%}" if pd.notna(x) else "—")
st.dataframe(disp, width="stretch")

# Download cleaned data
st.download_button(
    "Download cleaned wide-format CSV",
    data=returns_df.to_csv().encode("utf-8"),
    file_name="rsa_cleaned_returns.csv",
    mime="text/csv",
)

st.markdown("---")
st.info(
    "**Next:** navigate to **Performance & Risk** or **Comparison** in this section "
    "for deeper analysis. The data you uploaded here is automatically used."
)
