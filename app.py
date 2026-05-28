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

# ─── Force-center the top nav via JS (CSS-only often misses Streamlit's
#     internal selectors; this runs in an invisible iframe and uses
#     window.parent.document to style the nav directly). ────────────────
components.html(
    """
<script>
(function () {
  function centerNav() {
    try {
      const doc = window.parent.document;
      const candidates = [
        '[data-testid="stTopNav"]',
        '[data-testid="stHeader"] nav',
        '[data-testid="stHeader"] [role="navigation"]',
        '[data-testid="stHeader"] [role="menubar"]',
        '[data-testid="stHeader"] [role="tablist"]',
        '[data-testid="stHeader"] [data-baseweb="tab-list"]',
        'header nav',
      ];
      let nav = null;
      for (const sel of candidates) {
        nav = doc.querySelector(sel);
        if (nav) break;
      }
      if (!nav) return false;
      // Style the nav itself
      nav.style.setProperty('display', 'flex', 'important');
      nav.style.setProperty('justify-content', 'center', 'important');
      nav.style.setProperty('margin', '0 auto', 'important');
      nav.style.setProperty('flex', '1 1 auto', 'important');
      // Center its parent flex container too (the header)
      let p = nav.parentElement;
      while (p && p !== doc.body) {
        if (window.parent.getComputedStyle(p).display === 'flex') {
          p.style.setProperty('justify-content', 'center', 'important');
        }
        p = p.parentElement;
      }
      // Recursively center any flex children inside the nav
      nav.querySelectorAll('*').forEach((el) => {
        const cs = window.parent.getComputedStyle(el);
        if (cs.display && cs.display.includes('flex')) {
          el.style.setProperty('justify-content', 'center', 'important');
        }
      });
      return true;
    } catch (e) { return false; }
  }

  // Try immediately, then watch for DOM mutations (Streamlit re-renders often)
  let done = false;
  const tryIt = () => { if (!done && centerNav()) { done = true; } };
  tryIt();
  if (!done) {
    const obs = new MutationObserver(tryIt);
    obs.observe(window.parent.document.body, { childList: true, subtree: true });
    setTimeout(() => obs.disconnect(), 8000);
  }
  // Re-apply periodically in case Streamlit re-renders the nav
  setInterval(centerNav, 1500);
})();
</script>
""",
    height=0,
)

pg.run()
