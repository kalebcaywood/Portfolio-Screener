"""Quantitative Portfolio Analytics — entry point with sectioned top-bar navigation.

The five top-level sections are:
  1. App                   — landing page / portfolio builder
  2. Fund Holdings         — Bloomberg multi-fund book analyzer
  3. Equity Screener       — per-ticker fundamental + technical screener
  4. Return Stream Analyzer — performance, risk, factor, optimization, MC, stress, etc.
  5. Credit Analyzer       — fixed-income / credit work (under construction)

Each section's pages live under sections/ and are wired into st.navigation
with `position="top"` so the headers render as a horizontal bar at the top.
"""
from __future__ import annotations

import streamlit as st

from theme import inject_css

# ─── Page config — owned by the entry point only ────────────────────────────
st.set_page_config(
    page_title="UT Portfolio Analytics",
    layout="wide",
    initial_sidebar_state="auto",
)
inject_css()


# ─── Section structure ──────────────────────────────────────────────────────
sections = {
    "App": [
        st.Page("sections/home.py", title="Home", default=True),
    ],
    "Fund Holdings": [
        st.Page("sections/fund_holdings.py", title="Holdings Analyzer"),
    ],
    "Equity Screener": [
        st.Page("sections/screener.py", title="Screener"),
    ],
    "Return Stream Analyzer": [
        st.Page("sections/rsa_performance.py",    title="Performance"),
        st.Page("sections/rsa_risk_metrics.py",   title="Risk Metrics"),
        st.Page("sections/rsa_stats_tests.py",    title="Statistical Tests"),
        st.Page("sections/rsa_factor_models.py",  title="Factor Models"),
        st.Page("sections/rsa_optimization.py",   title="Optimization"),
        st.Page("sections/rsa_monte_carlo.py",    title="Monte Carlo"),
        st.Page("sections/rsa_stress_tests.py",   title="Stress Tests"),
        st.Page("sections/rsa_correlation.py",    title="Correlation"),
        st.Page("sections/rsa_currency_rates.py", title="Currency & Rates"),
        st.Page("sections/rsa_risk_decomp.py",    title="Risk Decomposition"),
        st.Page("sections/rsa_pacing_reup.py",    title="Pacing & Reup"),
    ],
    "Credit Analyzer": [
        st.Page("sections/credit_placeholder.py", title="Coming Soon"),
    ],
}

# Render top-bar navigation
pg = st.navigation(sections, position="top")
pg.run()
