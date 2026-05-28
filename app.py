"""Quantitative Portfolio Analytics — entry point with UT brand header + top-bar nav.

Top-level sections:
  1. Portfolio Analyzer       — holdings-based workbench (Home + 11 analytics pages)
  2. Fund Holdings            — Bloomberg multi-fund book analyzer
  3. Equity Screener          — per-ticker fundamental + technical screener
  4. Return Stream Analyzer   — pure-returns workbench (to be built)
  5. Credit Analyzer          — fixed-income / credit analytics (to be built)
"""
from __future__ import annotations

import streamlit as st

from theme import inject_css, ut_header

# ─── Page config — single source of truth ────────────────────────────────────
st.set_page_config(
    page_title="UT Portfolio Analytics",
    layout="wide",
    initial_sidebar_state="auto",  # user-collapsible
)
inject_css()

# ─── UT brand header at the top of every page ───────────────────────────────
ut_header("Quantitative Portfolio Analytics", "University of Tennessee")

# ─── Sections ────────────────────────────────────────────────────────────────
sections = {
    "Portfolio Analyzer": [
        st.Page("sections/home.py",               title="Home", default=True),
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
    "Fund Holdings": [
        st.Page("sections/fund_holdings.py", title="Holdings Analyzer"),
    ],
    "Equity Screener": [
        st.Page("sections/screener.py", title="Screener"),
    ],
    "Return Stream Analyzer": [
        st.Page("sections/rsa_placeholder.py", title="Coming Soon"),
    ],
    "Credit Analyzer": [
        st.Page("sections/credit_placeholder.py", title="Coming Soon"),
    ],
}

pg = st.navigation(sections, position="top")
pg.run()
