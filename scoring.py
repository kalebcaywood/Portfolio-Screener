"""Multi-factor composite scoring via cross-sectional z-scores."""
from __future__ import annotations

import numpy as np
import pandas as pd


def z_score(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    mu, sd = s.mean(skipna=True), s.std(skipna=True)
    if not sd or np.isnan(sd) or sd == 0:
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sd


def _mean_z(df: pd.DataFrame, cols: list[str], invert: list[str] | None = None) -> pd.Series:
    invert = invert or []
    parts = []
    for c in cols:
        if c in df.columns:
            z = z_score(df[c])
            if c in invert:
                z = -z
            # winsorize at +/-3 to limit outlier dominance
            z = z.clip(lower=-3, upper=3)
            parts.append(z)
    if not parts:
        return pd.Series(np.nan, index=df.index)
    return pd.concat(parts, axis=1).mean(axis=1, skipna=True)


def composite_score(df: pd.DataFrame) -> pd.DataFrame:
    """Compute value/quality/momentum/low-risk factor z-scores and an equal-weight composite."""
    out = df.copy()

    # Value: lower ratios are better → invert
    out["score_value"] = _mean_z(
        out,
        ["pe_trailing", "pe_forward", "pb", "ps", "ev_ebitda", "ev_rev", "peg"],
        invert=["pe_trailing", "pe_forward", "pb", "ps", "ev_ebitda", "ev_rev", "peg"],
    )

    # Quality: higher is better, plus Piotroski
    out["score_quality"] = _mean_z(
        out,
        ["roe", "roa", "gross_margin", "op_margin", "net_margin", "current_ratio", "piotroski_f"],
        invert=[],
    )

    # Momentum: higher returns are better
    out["score_momentum"] = _mean_z(
        out,
        ["ret_3m", "ret_6m", "ret_1y", "momentum_12_1"],
        invert=[],
    )

    # Low-risk: lower vol better; max_dd is negative so higher (less negative) is better
    out["score_low_risk"] = _mean_z(
        out,
        ["volatility", "max_dd", "beta_1y"],
        invert=["volatility", "beta_1y"],
    )

    components = ["score_value", "score_quality", "score_momentum", "score_low_risk"]
    out["score_composite"] = out[components].mean(axis=1, skipna=True)
    out["rank"] = out["score_composite"].rank(ascending=False, method="min")

    return out


# ─── Rules-based recommendation engine ───────────────────────────────────────
REC_ORDER = ["STRONG BUY", "BUY", "WATCH", "HOLD", "AVOID"]


def _g(row: pd.Series, key: str):
    v = row.get(key, np.nan)
    return v if pd.notna(v) else np.nan


def recommend(df: pd.DataFrame) -> pd.DataFrame:
    """Assign a transparent BUY/WATCH/HOLD/AVOID tier + plain-English reasons.

    The composite z-score sets the base (cross-sectional percentile within the
    screened set); explicit green/red flags on quality, value, momentum, and
    risk then promote or demote each name. Everything is rule-based and shown
    to the user as the ``why`` — no black box.
    """
    out = df.copy()
    if "score_composite" not in out.columns or out.empty:
        out["recommendation"] = "HOLD"
        out["rec_reasons"] = "insufficient data"
        out["rec_score"] = np.nan
        return out

    pct = out["score_composite"].rank(pct=True)
    tiers, reasons, rscore = [], [], []

    for idx, r in out.iterrows():
        p = float(pct.get(idx, 0.5)) if pd.notna(pct.get(idx, np.nan)) else 0.5
        green, red = [], []

        roe, nm, pf = _g(r, "roe"), _g(r, "net_margin"), _g(r, "piotroski_f")
        if pd.notna(roe) and roe >= 0.15: green.append(f"ROE {roe:.0%}")
        elif pd.notna(roe) and roe < 0.05: red.append(f"thin ROE {roe:.0%}")
        if pd.notna(nm) and nm >= 0.15: green.append(f"margin {nm:.0%}")
        if pd.notna(pf) and pf >= 7: green.append(f"F-Score {pf:.0f}")
        elif pd.notna(pf) and pf <= 3: red.append(f"weak F-Score {pf:.0f}")

        pe, peg = _g(r, "pe_trailing"), _g(r, "peg")
        if pd.notna(pe) and 0 < pe <= 18: green.append(f"P/E {pe:.0f}")
        elif pd.notna(pe) and pe > 45: red.append(f"rich P/E {pe:.0f}")
        if pd.notna(peg) and 0 < peg <= 1.2: green.append(f"PEG {peg:.1f}")

        r1, m121 = _g(r, "ret_1y"), _g(r, "momentum_12_1")
        if pd.notna(r1) and r1 > 0.12: green.append(f"+{r1:.0%} 1Y")
        elif pd.notna(r1) and r1 < -0.20: red.append(f"{r1:.0%} 1Y")

        de, dd = _g(r, "debt_equity"), _g(r, "max_dd")
        if pd.notna(de) and de > 2.5: red.append(f"leverage {de:.1f}x")
        if pd.notna(dd) and dd < -0.45: red.append(f"drawdown {dd:.0%}")

        ng, nr = len(green), len(red)
        rscore.append(round(p + 0.05 * ng - 0.08 * nr, 3))

        # Green/red flags lead; percentile is a tie-breaker. A clean sweep of
        # fundamentals earns the top tier even if the composite's low-risk leg
        # docked the name for volatility.
        if nr == 0 and ng >= 4:
            tier = "STRONG BUY"                 # clean, broadly strong
        elif nr == 0 and ng >= 2 and p >= 0.40:
            tier = "BUY"                        # clean, decent rank
        elif ng >= 2 and nr == 1:
            tier = "WATCH"                      # strong but one caveat
        elif nr >= 2:
            tier = "AVOID"                      # multiple red flags
        elif p <= 0.25 and ng < 2:
            tier = "AVOID"                      # bottom rank, nothing going for it
        else:
            tier = "HOLD"
        tiers.append(tier)

        parts = []
        if green:
            parts.append("✓ " + ", ".join(green))
        if red:
            parts.append("✗ " + ", ".join(red))
        reasons.append("   ".join(parts) if parts else "balanced / mid-pack")

    out["recommendation"] = tiers
    out["rec_reasons"] = reasons
    out["rec_score"] = rscore
    return out
