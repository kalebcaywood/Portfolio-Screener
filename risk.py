"""Risk metrics: VaR, CVaR, component risk, stress testing."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252


# ─── VaR family (returns are NEGATIVE for losses) ────────────────────────────

def var_historical(returns: pd.Series, alpha: float = 0.05) -> float:
    if len(returns) < 20:
        return np.nan
    return float(np.percentile(returns.dropna(), alpha * 100))


def cvar_historical(returns: pd.Series, alpha: float = 0.05) -> float:
    v = var_historical(returns, alpha)
    if np.isnan(v):
        return np.nan
    tail = returns[returns <= v]
    return float(tail.mean()) if len(tail) else np.nan


def var_parametric(returns: pd.Series, alpha: float = 0.05) -> float:
    mu = returns.mean()
    sigma = returns.std()
    z = stats.norm.ppf(alpha)
    return float(mu + z * sigma)


def cvar_parametric(returns: pd.Series, alpha: float = 0.05) -> float:
    mu = returns.mean()
    sigma = returns.std()
    z = stats.norm.ppf(alpha)
    # E[X | X < VaR] for normal = mu - sigma * phi(z)/Phi(z) where Phi(z)=alpha
    return float(mu - sigma * stats.norm.pdf(z) / alpha)


def var_cornish_fisher(returns: pd.Series, alpha: float = 0.05) -> float:
    """Adjusts normal VaR for skew/excess kurtosis."""
    mu = returns.mean()
    sigma = returns.std()
    s = returns.skew()
    k = returns.kurtosis()  # excess
    z = stats.norm.ppf(alpha)
    z_adj = (z
             + (z ** 2 - 1) * s / 6
             + (z ** 3 - 3 * z) * k / 24
             - (2 * z ** 3 - 5 * z) * s ** 2 / 36)
    return float(mu + z_adj * sigma)


def var_monte_carlo(returns: pd.Series, alpha: float = 0.05,
                     n_sims: int = 10000, horizon: int = 1,
                     seed: int | None = 42) -> float:
    if seed is not None:
        np.random.seed(seed)
    mu = returns.mean()
    sigma = returns.std()
    sims = np.random.normal(mu, sigma, (n_sims, horizon)).sum(axis=1)
    return float(np.percentile(sims, alpha * 100))


def var_summary(returns: pd.Series, alphas: tuple = (0.01, 0.05, 0.10)) -> pd.DataFrame:
    rows = []
    for a in alphas:
        rows.append({
            "confidence": f"{1 - a:.0%}",
            "var_historical": var_historical(returns, a),
            "var_parametric": var_parametric(returns, a),
            "var_cornish_fisher": var_cornish_fisher(returns, a),
            "var_monte_carlo": var_monte_carlo(returns, a),
            "cvar_historical": cvar_historical(returns, a),
            "cvar_parametric": cvar_parametric(returns, a),
        })
    return pd.DataFrame(rows)


# ─── Component / marginal risk ───────────────────────────────────────────────

def risk_contribution(returns: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    """Each asset's marginal and component contribution to portfolio volatility.

    Respects signed and leveraged weights — does NOT rescale them. For a 130/30,
    a short position with negative weight will have a negative or positive risk
    contribution depending on its covariance with the rest of the portfolio.
    """
    common = returns.columns.intersection(weights.index)
    R = returns[common].dropna()
    w = weights.reindex(common).fillna(0).values
    cov = R.cov().values * TRADING_DAYS  # annualize cov
    port_var = float(w @ cov @ w)
    port_vol = float(np.sqrt(port_var))
    if port_vol == 0:
        return pd.DataFrame()
    marginal = (cov @ w) / port_vol
    contribution = w * marginal
    total = contribution.sum()
    pct = contribution / total if abs(total) > 1e-12 else contribution * 0
    return pd.DataFrame({
        "ticker": list(common),
        "weight": w,
        "marginal_vol_contribution": marginal,
        "component_vol_contribution": contribution,
        "pct_of_total_risk": pct,
    }).set_index("ticker")


def component_var(returns: pd.DataFrame, weights: pd.Series,
                   alpha: float = 0.05) -> pd.DataFrame:
    """Per-asset contribution to parametric portfolio VaR.

    Respects signed and leveraged weights as-given.
    """
    common = returns.columns.intersection(weights.index)
    R = returns[common].dropna()
    w = weights.reindex(common).fillna(0).values
    cov = R.cov().values
    port_vol = float(np.sqrt(w @ cov @ w))
    z = stats.norm.ppf(alpha)
    marginal_var_vec = z * (cov @ w) / port_vol if port_vol > 0 else np.zeros(len(w))
    component_var_vec = w * marginal_var_vec
    total = component_var_vec.sum()
    pct = component_var_vec / total if abs(total) > 1e-12 else np.zeros_like(component_var_vec)
    return pd.DataFrame({
        "ticker": list(common),
        "weight": w,
        "marginal_var": marginal_var_vec,
        "component_var": component_var_vec,
        "pct_of_var": pct,
    }).set_index("ticker")


def diversification_ratio(returns: pd.DataFrame, weights: pd.Series) -> float:
    """Gross-weighted asset vol / portfolio vol. Higher = better diversification.

    Uses |weight| in the numerator so short positions still count as risk taken,
    and so the ratio remains positive and interpretable for 130/30 and similar
    long-short books.
    """
    common = returns.columns.intersection(weights.index)
    R = returns[common].dropna()
    w = weights.reindex(common).fillna(0).values
    asset_vol = R.std().values
    weighted_avg_vol = float(np.abs(w) @ asset_vol)
    port_vol = float(np.sqrt(w @ R.cov().values @ w))
    return weighted_avg_vol / port_vol if port_vol > 0 else np.nan


def effective_n_assets(weights: pd.Series) -> float:
    """1 / Herfindahl index — effective number of independent positions."""
    w = (weights / weights.sum()).values
    h = (w ** 2).sum()
    return float(1 / h) if h > 0 else np.nan


# ─── Stress testing ──────────────────────────────────────────────────────────

HISTORICAL_SCENARIOS = {
    "Global Financial Crisis (2008)": ("2008-09-01", "2009-03-31"),
    "Eurozone Crisis (Aug 2011)": ("2011-07-22", "2011-10-04"),
    "Aug 2015 China Selloff": ("2015-08-17", "2015-08-25"),
    "Q4 2018 Correction": ("2018-10-01", "2018-12-24"),
    "Feb 2018 Volmageddon": ("2018-01-26", "2018-02-08"),
    "COVID Crash (Feb-Mar 2020)": ("2020-02-19", "2020-03-23"),
    "2022 Bear Market": ("2022-01-03", "2022-10-12"),
    "March 2023 Banking Crisis": ("2023-03-08", "2023-03-15"),
}


def stress_test_historical(prices: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    """Apply portfolio weights to historical crisis return windows.

    Skips scenarios that fall outside the available price history. For each in-range
    scenario, computes peak-to-trough return per asset over the window and the
    weighted-sum portfolio return.
    """
    common = prices.columns.intersection(weights.index)
    if len(common) == 0:
        return pd.DataFrame()

    w = weights.reindex(common).fillna(0)
    if w.abs().sum() == 0:
        return pd.DataFrame()
    # Do NOT rescale — caller's weights already represent intended exposure

    rows = []
    px_min = prices.index.min()
    px_max = prices.index.max()

    for name, (start, end) in HISTORICAL_SCENARIOS.items():
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        # Skip scenarios with no overlap with our price history
        if end_ts < px_min or start_ts > px_max:
            rows.append({"scenario": name, "start": start, "end": end,
                          "n_days": 0, "portfolio_return": np.nan,
                          "worst_asset": "", "worst_asset_return": np.nan,
                          "best_asset": "", "best_asset_return": np.nan})
            continue
        try:
            window = prices[common].loc[start:end].dropna(how="all")
        except Exception:
            window = pd.DataFrame()

        if window.empty or len(window) < 2:
            rows.append({"scenario": name, "start": start, "end": end,
                          "n_days": 0, "portfolio_return": np.nan,
                          "worst_asset": "", "worst_asset_return": np.nan,
                          "best_asset": "", "best_asset_return": np.nan})
            continue

        first = window.bfill().iloc[0]
        last = window.ffill().iloc[-1]
        period_returns = (last / first - 1).dropna()
        if period_returns.empty:
            continue
        aligned_w = w.reindex(period_returns.index).fillna(0)
        port = float((period_returns * aligned_w).sum())
        worst = period_returns.idxmin()
        best = period_returns.idxmax()
        rows.append({
            "scenario": name, "start": start, "end": end, "n_days": len(window),
            "portfolio_return": port,
            "worst_asset": worst, "worst_asset_return": float(period_returns[worst]),
            "best_asset": best, "best_asset_return": float(period_returns[best]),
        })
    return pd.DataFrame(rows)


def custom_shock_test(returns: pd.DataFrame, weights: pd.Series,
                       shocks: dict[str, float]) -> dict:
    """Apply per-asset return shocks (e.g. {'AAPL': -0.20, 'MSFT': -0.15}).

    Respects leveraged/signed weights as-given.
    """
    common = returns.columns.intersection(weights.index)
    w = weights.reindex(common).fillna(0)
    port_shock = float(sum(w.get(t, 0) * s for t, s in shocks.items() if t in common))
    contributions = {t: float(w.get(t, 0) * shocks.get(t, 0)) for t in common}
    return {"portfolio_shock": port_shock, "contributions": contributions}


def factor_shock_test(returns: pd.DataFrame, weights: pd.Series,
                       bench_returns: pd.Series, bench_shock: float) -> dict:
    """Shock benchmark by X%, apply implied per-asset moves via individual betas."""
    common = returns.columns.intersection(weights.index)
    R = returns[common].dropna()
    aligned_bench = bench_returns.reindex(R.index).dropna()
    R = R.loc[aligned_bench.index]

    betas = {}
    for t in R.columns:
        cov = np.cov(R[t], aligned_bench)[0, 1]
        var = aligned_bench.var()
        betas[t] = cov / var if var > 0 else 0.0

    shocks = {t: bench_shock * betas[t] for t in R.columns}
    return custom_shock_test(returns, weights, shocks) | {"betas": betas, "bench_shock": bench_shock}
