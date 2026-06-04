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
import streamlit.components.v1 as components

from news import render_news_ticker
from theme import (
    inject_css,
    render_logout_button,
    render_theme_toggle,
    require_password,
    ut_header,
)
from jarvis_assistant import render_jarvis

# ─── Page config — single source of truth ────────────────────────────────────
st.set_page_config(
    page_title="UT Portfolio Analytics",
    layout="wide",
    initial_sidebar_state="auto",  # user-collapsible
)

# ─── Password gate — runs BEFORE any other UI rendering ────────────────────
# If the user isn't authenticated, this halts the script and shows the gate.
require_password()

inject_css()

# Optional sign-out link in the sidebar (always shown post-auth)
render_logout_button()

# Dark / Light appearance toggle (sidebar, top)
render_theme_toggle()

# ─── UT brand header at the top of every page ───────────────────────────────
ut_header("Quantitative Portfolio Analytics", "University of Tennessee")

# ─── Circulating WSJ headline ticker (cached 15 min; no-op if offline) ───────
render_news_ticker()

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
        st.Page("sections/screener.py",     title="Universal Equity Screener"),
        st.Page("sections/eq_tearsheet.py", title="Equity Tearsheet"),
    ],
    "Return Stream Analyzer": [
        st.Page("sections/rs_home.py",        title="Home",                url_path="rs_home"),
        st.Page("sections/rs_performance.py", title="Performance & Risk"),
        st.Page("sections/rs_risk.py",        title="Risk Metrics",        url_path="rs_risk"),
        st.Page("sections/rs_stats_tests.py", title="Statistical Tests",   url_path="rs_stats_tests"),
        st.Page("sections/rs_factor.py",      title="Factor Models",       url_path="rs_factor"),
        st.Page("sections/rs_comparison.py",  title="Stream Comparison"),
    ],
    "Credit Analyzer": [
        st.Page("sections/credit_placeholder.py", title="Coming Soon"),
    ],
}

pg = st.navigation(sections, position="top")

# ─── Header layout fixups via JS (CSS-only often misses Streamlit's internal
#     selectors). Runs in an invisible iframe and styles window.parent.document
#     directly: (1) center the top nav, and (2) relocate the WSJ ticker to be a
#     direct child of <body> so its position:fixed anchors to the viewport even
#     on Streamlit Cloud, where viewer chrome can introduce a transformed
#     ancestor that would otherwise trap a fixed element back into the flow. ──
components.html(
    """
<script>
(function () {
  const PDOC = () => window.parent.document;

  // Move the WSJ ticker to be a direct child of <body> so its position:fixed
  // anchors to the viewport. (Streamlit Cloud's viewer chrome can put a
  // transform on an ancestor, which would otherwise trap the fixed bar back
  // into the page flow.) We deliberately do NOT touch the top nav: Streamlit's
  // rc-overflow widget collapses the 5 section tabs into a bogus "N more"
  // overflow button if its width/position is overridden.
  function relocateTicker() {
    const doc = PDOC();
    const all = doc.querySelectorAll('.news-tkr');
    if (!all.length) return;
    const keep = all[all.length - 1];
    all.forEach((t) => { if (t !== keep && t.parentElement) t.parentElement.removeChild(t); });
    if (keep.parentElement !== doc.body) doc.body.appendChild(keep);
  }

  let obs = null;
  function onMutation() {
    // Cheap early-out: only touch the DOM when the ticker is actually missing
    // from <body> or duplicated, so we never thrash the surrounding layout
    // (an over-eager observer was a prime suspect for nav glitches).
    const doc = PDOC();
    const all = doc.querySelectorAll('.news-tkr');
    if (!all.length) return;
    if (all.length === 1 && all[0].parentElement === doc.body) return;
    obs.disconnect();
    relocateTicker();
    obs.observe(doc.body, { childList: true, subtree: true });
  }

  relocateTicker();
  obs = new MutationObserver(onMutation);
  obs.observe(PDOC().body, { childList: true, subtree: true });
})();
</script>
""",
    height=0,
)

pg.run()

# ─── Floating Jarvis assistant (circular button, bottom-right of every page) ──
render_jarvis()
