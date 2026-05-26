"""Monte Carlo simulation for portfolio future paths."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _ensure_psd(cov: np.ndarray, jitter: float = 1e-10) -> np.ndarray:
    """Add a tiny diagonal jitter so multivariate_normal accepts the cov matrix."""
    cov = (cov + cov.T) / 2  # enforce symmetry
    return cov + np.eye(cov.shape[0]) * jitter


def simulate_normal(returns: pd.DataFrame, weights: pd.Series,
                     horizon_days: int = 252, n_sims: int = 5000,
                     initial_value: float = 100000.0,
                     seed: int | None = 42) -> np.ndarray:
    """Multivariate normal Monte Carlo of portfolio value paths."""
    if seed is not None:
        np.random.seed(seed)
    common = returns.columns.intersection(weights.index)
    R = returns[common].dropna(how="all")
    if R.empty or len(common) == 0:
        return np.full((n_sims, horizon_days + 1), initial_value)
    mu = R.mean().values
    cov = _ensure_psd(R.cov().values)
    w_aligned = weights.reindex(common).fillna(0)
    if w_aligned.abs().sum() == 0:
        return np.full((n_sims, horizon_days + 1), initial_value)
    # Respect leveraged / signed weights as-given (no rescaling)
    w = w_aligned.values

    # multivariate_normal handles 1x1 cov correctly; check_valid="ignore" silences
    # warnings about non-PSD which we've already softened via _ensure_psd
    sims = np.random.multivariate_normal(mu, cov, size=(n_sims, horizon_days),
                                            check_valid="ignore")
    port_returns = sims @ w
    paths = initial_value * np.cumprod(1 + port_returns, axis=1)
    return np.concatenate([np.full((n_sims, 1), initial_value), paths], axis=1)


def simulate_bootstrap(returns: pd.DataFrame, weights: pd.Series,
                        horizon_days: int = 252, n_sims: int = 5000,
                        initial_value: float = 100000.0,
                        block_size: int = 1,
                        seed: int | None = 42) -> np.ndarray:
    """Historical bootstrap (optionally block-bootstrap to preserve serial dependence)."""
    if seed is not None:
        np.random.seed(seed)
    common = returns.columns.intersection(weights.index)
    R = returns[common].dropna(how="all").fillna(0)
    w_aligned = weights.reindex(common).fillna(0)
    if R.empty or w_aligned.abs().sum() == 0:
        return np.full((n_sims, horizon_days + 1), initial_value)
    # Respect leveraged / signed weights as-given
    w = w_aligned.values
    port_hist = R.values @ w
    n_obs = len(port_hist)
    if n_obs == 0:
        return np.full((n_sims, horizon_days + 1), initial_value)

    paths = np.full((n_sims, horizon_days + 1), initial_value)
    if block_size <= 1:
        for i in range(n_sims):
            sampled = np.random.choice(port_hist, size=horizon_days, replace=True)
            paths[i, 1:] = initial_value * np.cumprod(1 + sampled)
    else:
        for i in range(n_sims):
            seq = []
            while len(seq) < horizon_days:
                start = np.random.randint(0, n_obs - block_size + 1)
                seq.extend(port_hist[start:start + block_size])
            seq = np.array(seq[:horizon_days])
            paths[i, 1:] = initial_value * np.cumprod(1 + seq)
    return paths


def simulate_gbm(returns: pd.Series, horizon_days: int = 252, n_sims: int = 5000,
                  initial_value: float = 100000.0, seed: int | None = 42) -> np.ndarray:
    """Geometric Brownian Motion on portfolio returns (univariate)."""
    if seed is not None:
        np.random.seed(seed)
    mu = returns.mean()
    sigma = returns.std()
    dt = 1
    shocks = np.random.normal((mu - 0.5 * sigma ** 2) * dt, sigma * np.sqrt(dt),
                                size=(n_sims, horizon_days))
    paths = initial_value * np.exp(np.cumsum(shocks, axis=1))
    return np.concatenate([np.full((n_sims, 1), initial_value), paths], axis=1)


def summarize_paths(paths: np.ndarray, initial_value: float = 100000.0) -> dict:
    terminal = paths[:, -1]
    rets = terminal / initial_value - 1
    return {
        "n_simulations": paths.shape[0],
        "horizon_days": paths.shape[1] - 1,
        "expected_terminal": float(terminal.mean()),
        "median_terminal": float(np.median(terminal)),
        "std_terminal": float(terminal.std()),
        "min_terminal": float(terminal.min()),
        "max_terminal": float(terminal.max()),
        "expected_return": float(rets.mean()),
        "median_return": float(np.median(rets)),
        "prob_loss": float((rets < 0).mean()),
        "prob_10pct_loss": float((rets < -0.10).mean()),
        "prob_25pct_loss": float((rets < -0.25).mean()),
        "prob_50pct_loss": float((rets < -0.50).mean()),
        "prob_double": float((rets > 1.0).mean()),
        "var_95": float(np.percentile(rets, 5)),
        "var_99": float(np.percentile(rets, 1)),
        "cvar_95": float(rets[rets <= np.percentile(rets, 5)].mean()),
        "cvar_99": float(rets[rets <= np.percentile(rets, 1)].mean()),
        "p5": float(np.percentile(terminal, 5)),
        "p25": float(np.percentile(terminal, 25)),
        "p50": float(np.percentile(terminal, 50)),
        "p75": float(np.percentile(terminal, 75)),
        "p95": float(np.percentile(terminal, 95)),
    }


def percentile_paths(paths: np.ndarray,
                      percentiles: tuple = (5, 25, 50, 75, 95)) -> dict:
    return {p: np.percentile(paths, p, axis=0) for p in percentiles}


def path_max_drawdowns(paths: np.ndarray) -> np.ndarray:
    """For each simulated path, compute max drawdown."""
    peaks = np.maximum.accumulate(paths, axis=1)
    dd = paths / peaks - 1
    return dd.min(axis=1)
