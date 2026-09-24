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


def rho_to_book(corr: pd.DataFrame,
                weights: dict[str, float] | None = None) -> dict[str, float]:
    """For each holding, its average correlation to the REST of the book.

    This is the 'return' exposure dimension: high ⇒ the name just re-expresses
    the portfolio, low ⇒ it is genuinely carrying its own risk.
    """
    out: dict[str, float] = {}
    cols = list(corr.columns)
    for a in cols:
        num = den = 0.0
        for b in cols:
            if a == b:
                continue
            c = corr.loc[a, b]
            if pd.isna(c):
                continue
            w = float(weights.get(b, 0.0)) if weights else 1.0
            if w <= 0:
                continue
            num += w * float(c)
            den += w
        out[a] = (num / den) if den > 0 else float("nan")
    return out


def _avg_within(corr: pd.DataFrame, tickers: list[str], t: str,
                group_of: dict[str, str]) -> float:
    """Average correlation of ``t`` to its OWN group's other members."""
    g = group_of.get(t)
    vals = [corr.loc[t, o] for o in tickers
            if o != t and group_of.get(o) == g and pd.notna(corr.loc[t, o])]
    return float(np.mean(vals)) if vals else float("nan")


def _unit(d: dict[str, float], lo: float = 0.08,
          sharpen: float = 3.0) -> dict[str, float]:
    """Min-max a score dict into [lo, 1], with a power curve.

    ``sharpen`` matters visually: these become barycentric pull weights, and
    three middling weights blend to a point near the centre of the pole
    triangle. Cubing lets a genuinely dominant force win, so a holding that is
    really a sector story travels to the sector corner instead of hovering in
    the undifferentiated middle. NaN (a group of one) gets ``lo``.
    """
    vals = [v for v in d.values() if pd.notna(v)]
    if not vals:
        return {k: lo for k in d}
    mn, mx = min(vals), max(vals)
    span = (mx - mn) or 1.0
    return {k: (lo + (1 - lo) * (((v - mn) / span) ** sharpen)) if pd.notna(v) else lo
            for k, v in d.items()}


DRIVERS = ("SECTOR", "REGION", "MARKET")


def _pct3(a: float, b: float, c: float) -> list[int]:
    """Three shares as whole percents that sum to exactly 100 (largest
    remainder), so the tooltip never reads 99% or 101%."""
    tot = a + b + c
    if tot <= 0:
        return [34, 33, 33]
    raw = [100 * a / tot, 100 * b / tot, 100 * c / tot]
    out = [int(v) for v in raw]
    for _ in range(100 - sum(out)):
        i = max(range(3), key=lambda k: raw[k] - out[k])
        out[i] += 1
    return out


def _barycentre(ks: float, kr: float, kb: float,
                sa: tuple[float, float, float],
                ra: tuple[float, float, float],
                rp: tuple[float, float, float]) -> tuple[float, float, float]:
    """Where a holding rests: its sector anchor, region anchor and the return
    pole, mixed in proportion to how much each explains its co-movement.

    This is a ternary (barycentric) position, so a dot sitting at a corner is a
    pure story and one in the middle genuinely is a three-way blend — unlike a
    spring equilibrium, where competing pulls cancel and everything piles up in
    the centre regardless of what it means.
    """
    tot = ks + kr + kb
    if tot <= 0:
        return (0.0, 0.0, 0.0)
    ws, wr, wb = ks / tot, kr / tot, kb / tot
    p = [ws * sa[i] + wr * ra[i] + wb * rp[i] for i in range(3)]
    p = [v * SPREAD_GAIN for v in p]
    n = float(np.sqrt(sum(v * v for v in p)))
    if n > SPREAD_MAX:                       # keep the gain from flinging outliers
        p = [v * SPREAD_MAX / n for v in p]
    return tuple(p)


def _fan(pole: tuple[float, float, float], n: int, radius: float
         ) -> list[tuple[float, float, float]]:
    """Spread n sub-anchors in a ring around a pole so each category pulls from
    its own direction rather than all of them from one point."""
    px, py, pz = pole
    pts = []
    for k in range(max(n, 1)):
        a = 2 * np.pi * k / max(n, 1)
        pts.append((px + radius * np.cos(a),
                    py + radius * np.sin(a) * 0.6,
                    pz + radius * np.sin(2 * a) * 0.80))
    return pts


# Pull-mode poles. Wide triangle summing to ~origin so the settled cloud stays
# centred, and genuinely separated in z as well — poles sharing a z plane make
# the whole field read as flat no matter how you orbit it.
POLES = {
    "sector": (0.00, 2.00, 0.90),
    "region": (-1.90, -1.00, -0.95),
    "return": (1.90, -1.00, 0.30),
}

# Barycentric weights rarely reach a pure corner, so rest points cluster near
# the middle of the triangle. Expand about the origin to use the space; this is
# presentation gain only — relative geometry (which corner a dot is nearest) is
# untouched.
SPREAD_GAIN = 1.75
SPREAD_MAX = 2.30


def embedding_payload(corr: pd.DataFrame, holdings: pd.DataFrame,
                      sector_of: dict[str, str],
                      link_top_k: int = 6, min_rho: float = 0.20) -> dict:
    """Everything the 3D exposure field needs, as plain JSON-able types.

    ``holdings`` needs columns yahoo / fund / market_value / region. Each node
    carries both its live-simulation identity (sector, region, return pull) and
    its snap-mode axis coordinates, so the view can morph between the two.
    """
    tickers = [t for t in corr.columns if t in set(holdings["yahoo"])]
    if len(tickers) < 2:
        return {}

    mv = holdings.groupby("yahoo")["market_value"].sum()
    mv = mv.reindex(tickers).fillna(0.0)
    tot = float(mv.sum()) or 1.0
    wmap = {t: float(mv[t]) / tot for t in tickers}

    rho = rho_to_book(corr.loc[tickers, tickers], wmap)
    rvals = [v for v in rho.values() if pd.notna(v)]
    rlo, rhi = (min(rvals), max(rvals)) if rvals else (0.0, 1.0)
    span = (rhi - rlo) or 1.0

    region_of = dict(zip(holdings["yahoo"], holdings["region"]))
    name_of = (dict(zip(holdings["yahoo"], holdings["security_name"]))
               if "security_name" in holdings.columns else {})

    sectors = sorted({sector_of.get(t, "Unknown") for t in tickers})
    regions = sorted({region_of.get(t, "Unknown") for t in tickers})
    # Fans must be wide relative to pole spacing, or every node lands on the
    # same barycentre and the cloud collapses into an undifferentiated ball.
    s_anch = dict(zip(sectors, _fan(POLES["sector"], len(sectors), 1.15)))
    r_anch = dict(zip(regions, _fan(POLES["region"], len(regions), 1.00)))

    # How strongly each dimension EXPLAINS this holding's co-movement. These
    # become the three per-node spring constants — without them every node has
    # identical pulls, the symmetric pole triangle cancels, and the whole cloud
    # collapses onto one point.
    csub = corr.loc[tickers, tickers]
    sec_of = {t: sector_of.get(t, "Unknown") for t in tickers}
    reg_of = {t: region_of.get(t, "Unknown") for t in tickers}
    ks = _unit({t: _avg_within(csub, tickers, t, sec_of) for t in tickers})
    kr = _unit({t: _avg_within(csub, tickers, t, reg_of) for t in tickers})
    kb = _unit({t: rho.get(t, np.nan) for t in tickers})

    wmax = max(wmap.values()) or 1.0
    nodes = []
    for i, t in enumerate(tickers):
        s, rg = sector_of.get(t, "Unknown"), region_of.get(t, "Unknown")
        r = rho.get(t, np.nan)
        r01 = float((r - rlo) / span) if pd.notna(r) else 0.5
        si, ri = sectors.index(s), regions.index(rg)
        # deterministic in-cell jitter so co-located names stay distinguishable
        j = (hash(t) % 1000) / 1000.0 - 0.5
        nodes.append({
            "id": t, "name": name_of.get(t, t), "sector": s, "region": rg,
            "si": si, "ri": ri, "w": wmap[t], "wr": float(wmap[t] / wmax),
            "rho": None if pd.isna(r) else round(float(r), 4), "r01": round(r01, 4),
            # per-node pull strengths: sector / region / whole-book
            "ks": round(ks[t], 4), "kr": round(kr[t], 4), "kb": round(kb[t], 4),
            # the same mix as readable whole percents, plus the verdict — this
            # is what the position encodes, so the view can state it outright
            "mix": _pct3(ks[t], kr[t], kb[t]),
            "drv": DRIVERS[max(range(3), key=[ks[t], kr[t], kb[t]].__getitem__)],
            # barycentric rest position — the three-way mix as an actual point
            **{k: round(v, 4) for k, v in
               zip(("bx", "by", "bz"), _barycentre(ks[t], kr[t], kb[t],
                                                   s_anch[s], r_anch[rg],
                                                   POLES["return"]))},
            # snap-mode target: X = region, Y = sector, Z = correlation to book
            "ax": round(-1 + 2 * (ri + 0.5) / len(regions) + j * 0.10, 4),
            "ay": round(-1 + 2 * (si + 0.5) / len(sectors) + j * 0.06, 4),
            "az": round(-1 + 2 * r01, 4),
            "sa": [round(v, 4) for v in s_anch[s]],
            "ra": [round(v, 4) for v in r_anch[rg]],
        })

    idx = {t: i for i, t in enumerate(tickers)}
    seen, links = set(), []
    for t in tickers:
        row = corr.loc[t, tickers].drop(labels=[t], errors="ignore")
        row = row.dropna().sort_values(ascending=False).head(link_top_k)
        for other, v in row.items():
            if float(v) < min_rho:
                continue
            key = tuple(sorted((t, other)))
            if key in seen:
                continue
            seen.add(key)
            links.append({"s": idx[t], "t": idx[other], "r": round(float(v), 3)})

    fundmap: dict[str, list] = {}
    for f, g in holdings.groupby("fund"):
        sub = g[g["yahoo"].isin(idx)].groupby("yahoo")["market_value"].sum()
        s = float(sub.sum())
        if s <= 0 or sub.empty:
            continue
        fundmap[str(f)] = [[idx[t], round(float(v) / s, 5)] for t, v in sub.items()]

    counts = {d: sum(1 for n in nodes if n["drv"] == d) for d in DRIVERS}
    return {
        "nodes": nodes, "links": links, "driverCounts": counts,
        "sectors": sectors, "regions": regions,
        "sectorAnchors": {k: [round(v, 4) for v in p] for k, p in s_anch.items()},
        "regionAnchors": {k: [round(v, 4) for v in p] for k, p in r_anch.items()},
        "poles": {k: list(v) for k, v in POLES.items()},
        "funds": fundmap,
        "rhoRange": [round(rlo, 3), round(rhi, 3)],
    }


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
