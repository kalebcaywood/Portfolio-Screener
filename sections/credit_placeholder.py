"""Credit Analyzer — placeholder until the fixed-income module is built."""
from __future__ import annotations

import streamlit as st

from theme import inject_css, ut_sidebar_brand

inject_css()
ut_sidebar_brand()

st.title("Credit Analyzer")
st.caption("Fixed-income and credit analytics — coming soon.")

st.markdown(
    """
This section will house the credit analytics workbench. Planned modules:

| Tab | Purpose |
|---|---|
| **Credit profile** | Ratings distribution (IG / HY / unrated), duration & convexity buckets, sector mix |
| **Spreads & yields** | OAS / Z-spread vs history, yield-to-worst, breakeven analysis |
| **Curve & rate sensitivity** | Treasury curve, DV01 / key-rate duration, parallel & curve shocks |
| **Default risk** | Issuer-level default probabilities (Merton or rating-implied), expected loss, recovery |
| **Issuer concentration** | Top issuers, sector exposure, IG / HY mix, concentration heatmap |
| **Stress scenarios** | Spread widening, rate shocks, downgrade migration |

### What's needed to build this out

The Credit Analyzer needs **bond-level holdings data** that the existing equity
analytics tooling can't supply. The expected CSV columns are something like:

```
CUSIP, Issuer, Sector, Rating, Coupon, Maturity, Duration, OAS,
Notional, Market Value
```

Upload a sample once available and the parser + analytics will be built around
that exact schema — same approach as the Bloomberg equity file already wired
into the **Fund Holdings** section.

### What works without a holdings file

Even without bond-level data, some credit context is already available on the
**Return Stream Analyzer → Currency & Rates** page:
- US Treasury yield curve (`^IRX`, `^FVX`, `^TNX`, `^TYX`)
- Bond ETF total-return proxies (TLT, IEF, LQD, HYG, EMB, BWX, BNDX)
- 10Y – 13W spread (recession bellwether)

Once you have a credit holdings export, this section will become a real tool.
"""
)
