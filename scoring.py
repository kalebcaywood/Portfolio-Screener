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
