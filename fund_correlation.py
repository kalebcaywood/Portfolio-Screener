"""Group-level return-correlation ('pivot correlation') for a fund's holdings.

Given a fund's holdings, this builds a compact matrix of AVERAGE pairwise return
correlation *between* groups (grouped by sector, region, or a sector×region
composite). The off-diagonal cells answer "how correlated is, say, US-Tech to
Europe-Financials?"; the diagonal is each group's internal cohesion.

Return correlation is the only true quantitative correlation here — sector and
region are categorical labels and serve as the pivot/grouping axes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from screener import fetch_prices_bulk

WINDOW_DAYS = {"1y": 252, "2y": 504, "5y": None}  # None ⇒ full (up to 5y)


def _norm_returns(close: pd.Series, window_days: int | None) -> pd.Series:
    """Daily returns on a tz-naive calendar-date index (so foreign listings on
    different exchange calendars align to the same dates)."""
    s = close.dropna()
    if window_days and len(s) > window_days:
        s = s.iloc[-window_days:]
    idx = pd.DatetimeIndex(s.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    s = pd.Series(np.asarray(s.values, dtype=float), index=idx.normalize())
    s = s[~s.index.duplicated(keep="last")]
    return s.pct_change()


def returns_matrix(tickers: list[str], window: str = "2y") -> pd.DataFrame:
    """Wide daily-returns DataFrame (dates × tickers) for the window.

    Prices come from one bulk 5y download (cached); each ticker is sliced to the
    requested window and aligned on the union of calendar dates.
    """
    wd = WINDOW_DAYS.get(window)
    price_map = fetch_prices_bulk(tuple(dict.fromkeys(tickers)), "5y")
    cols: dict[str, pd.Series] = {}
    for t in tickers:
        close = price_map.get(t)
        if close is not None and len(close) > 5:
            cols[t] = _norm_returns(close, wd)
    if not cols:
        return pd.DataFrame()
    return pd.DataFrame(cols).dropna(how="all")


def correlation_matrix(rdf: pd.DataFrame, min_overlap: int = 30) -> pd.DataFrame:
    """Pairwise Pearson correlation, dropping tickers with too few observations."""
    if rdf.empty:
        return pd.DataFrame()
    good = [c for c in rdf.columns if int(rdf[c].notna().sum()) >= min_overlap]
    if not good:
        return pd.DataFrame()
    return rdf[good].corr(min_periods=min_overlap)


def group_correlation(corr: pd.DataFrame, group_of: dict[str, str],
                      weights: dict[str, float] | None = None,
                      ) -> tuple[pd.DataFrame, dict[str, int]]:
    """Average pairwise correlation between groups.

    Diagonal[g] = mean correlation over DISTINCT within-group pairs (internal
    cohesion), NaN if the group has < 2 holdings. Off-diagonal[g,h] = mean
    correlation over all (g-holding, h-holding) pairs. When ``weights`` is
    given each pair is weighted by wᵢ·wⱼ (so large positions dominate),
    otherwise it's a simple average.

    Returns (matrix, group_sizes).
    """
    if corr.empty:
        return pd.DataFrame(), {}
    tickers = [t for t in corr.columns if t in group_of]
    groups = sorted({group_of[t] for t in tickers})
    w = {t: (float(weights.get(t, 0.0)) if weights else 1.0) for t in tickers}
    members = {g: [t for t in tickers if group_of[t] == g] for g in groups}

    M = pd.DataFrame(index=groups, columns=groups, dtype=float)
    for gi in groups:
        A = members[gi]
        for gj in groups:
            B = members[gj]
            same = gi == gj
            num = den = 0.0
            for a in A:
                for b in B:
                    if same and a == b:
                        continue  # exclude self-correlation (=1) on the diagonal
                    c = corr.loc[a, b]
                    if pd.isna(c):
                        continue
                    ww = w[a] * w[b]
                    if ww <= 0:
                        ww = 1.0 if weights is None else ww
                    num += ww * float(c)
                    den += ww
            M.loc[gi, gj] = (num / den) if den > 0 else np.nan
    sizes = {g: len(members[g]) for g in groups}
    return M, sizes


def overall_avg_correlation(corr: pd.DataFrame,
                            weights: dict[str, float] | None = None) -> float:
    """Fund-wide average of all distinct pairwise correlations (optionally
    position-weighted) — a single 'how correlated is this book' number."""
    if corr.empty or len(corr) < 2:
        return float("nan")
    cols = list(corr.columns)
    num = den = 0.0
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            c = corr.loc[a, b]
            if pd.isna(c):
                continue
            ww = (float(weights.get(a, 0.0)) * float(weights.get(b, 0.0))) if weights else 1.0
            if ww <= 0 and weights is None:
                ww = 1.0
            num += ww * float(c)
            den += ww
    return float(num / den) if den > 0 else float("nan")


def extremes(M: pd.DataFrame) -> dict:
    """Most- and least-correlated off-diagonal group pairs from a group matrix."""
    out = {"max_pair": None, "max_val": np.nan, "min_pair": None, "min_val": np.nan}
    if M.empty or len(M) < 2:
        return out
    best_v, worst_v = -np.inf, np.inf
    for i, gi in enumerate(M.index):
        for gj in M.columns[i + 1:]:
            v = M.loc[gi, gj]
            if pd.isna(v):
                continue
            if v > best_v:
                best_v, out["max_pair"] = v, (gi, gj)
            if v < worst_v:
                worst_v, out["min_pair"] = v, (gi, gj)
    out["max_val"] = best_v if np.isfinite(best_v) else np.nan
    out["min_val"] = worst_v if np.isfinite(worst_v) else np.nan
    return out
