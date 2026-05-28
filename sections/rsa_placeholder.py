"""Return Stream Analyzer — placeholder until the return-stream workbench is built."""
from __future__ import annotations

import streamlit as st

from theme import inject_css

inject_css()

st.title("Return Stream Analyzer")
st.caption("Pure-returns analytics — coming soon.")

st.markdown(
    """
This section will be a separate workbench for cases where you have **return
streams without underlying holdings** — fund tear-sheets, manager-supplied
monthly returns, hedge-fund performance time series, custom strategy backtests.

### How this differs from Portfolio Analyzer

| | Portfolio Analyzer | Return Stream Analyzer |
|---|---|---|
| **Input** | Holdings (tickers + weights) | Pure return time series (daily, weekly, monthly) |
| **Data source** | Live prices via yfinance | User-uploaded returns CSV |
| **What it analyzes** | Position-level and aggregate risk, factor exposures, optimization, etc. | Risk-adjusted returns, drawdown profile, factor regressions, peer comparison |
| **When to use** | Managing your own book or modeling a hypothetical portfolio | Evaluating an external manager / strategy you don't have holdings for |

### Planned modules

| Tab | Purpose |
|---|---|
| **Performance summary** | CAGR, Sharpe, Sortino, Calmar, Omega, drawdown stats, win rate |
| **Drawdown analysis** | Episode table (peak / trough / recovery), Ulcer Index, time underwater |
| **Distribution & tails** | Skew, kurtosis, VaR, CVaR, Q-Q plot, regime decomposition |
| **Factor attribution** | Regress returns on Fama-French / custom factor proxies, alpha t-stat |
| **Peer / benchmark comparison** | Side-by-side stats vs benchmark or peer group |
| **Rolling diagnostics** | Rolling Sharpe / beta / vol / corr to track style drift |
| **Style analysis** | Sharpe-style returns-based style attribution |

### Input format

When this section is built, it will accept a CSV with columns like:

```
Date, ReturnStreamName, Return
2020-01-31, MyFund, 0.0142
2020-02-29, MyFund, -0.0731
...
```

Or wide-format:

```
Date, FundA, FundB, FundC, Benchmark
2020-01-31, 0.0142, 0.0089, 0.0210, 0.0167
...
```

### Status

Not started. The next major build after the **Portfolio Analyzer** is fully
polished. For now, use the **Portfolio Analyzer** section for holdings-based
analytics.
"""
)
