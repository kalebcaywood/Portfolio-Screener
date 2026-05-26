"""Portfolio optimization: efficient frontier, max Sharpe, min variance, risk parity."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_DAYS = 252
RISK_FREE_RATE = 0.04


def _stats(weights, mean_d, cov_d, rf):
    """Annualized (return, vol, sharpe) for daily mean & cov inputs."""
    ann_ret = float(weights @ mean_d * TRADING_DAYS)
    ann_vol = float(np.sqrt(weights @ cov_d @ weights * TRADING_DAYS))
    sr = (ann_ret - rf) / ann_vol if ann_vol > 0 else np.nan
    return ann_ret, ann_vol, sr


def max_sharpe(returns: pd.DataFrame, rf: float = RISK_FREE_RATE,
                allow_short: bool = False) -> dict:
    mean = returns.mean().values
    cov = returns.cov().values
    n = len(mean)
    bounds = tuple((-1.0, 1.0) if allow_short else (0.0, 1.0) for _ in range(n))
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    x0 = np.ones(n) / n

    def neg_sharpe(w):
        ret, vol, _ = _stats(w, mean, cov, rf)
        return 1e6 if vol == 0 else -(ret - rf) / vol

    res = minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    ret, vol, sr = _stats(res.x, mean, cov, rf)
    return {"weights": pd.Series(res.x, index=returns.columns),
            "return": ret, "vol": vol, "sharpe": sr, "success": bool(res.success)}


def min_variance(returns: pd.DataFrame, rf: float = RISK_FREE_RATE,
                  allow_short: bool = False) -> dict:
    cov = returns.cov().values
    mean = returns.mean().values
    n = len(returns.columns)
    bounds = tuple((-1.0, 1.0) if allow_short else (0.0, 1.0) for _ in range(n))
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    x0 = np.ones(n) / n

    res = minimize(lambda w: w @ cov @ w, x0, method="SLSQP",
                    bounds=bounds, constraints=constraints)
    ret, vol, sr = _stats(res.x, mean, cov, rf)
    return {"weights": pd.Series(res.x, index=returns.columns),
            "return": ret, "vol": vol, "sharpe": sr, "success": bool(res.success)}


def target_return(returns: pd.DataFrame, target: float,
                   rf: float = RISK_FREE_RATE, allow_short: bool = False) -> dict:
    """Minimum variance subject to target annualized return."""
    mean = returns.mean().values
    cov = returns.cov().values
    n = len(returns.columns)
    target_d = target / TRADING_DAYS

    bounds = tuple((-1.0, 1.0) if allow_short else (0.0, 1.0) for _ in range(n))
    constraints = [
        {"type": "eq", "fun": lambda w: w.sum() - 1},
        {"type": "eq", "fun": lambda w: w @ mean - target_d},
    ]
    x0 = np.ones(n) / n
    res = minimize(lambda w: w @ cov @ w, x0, method="SLSQP",
                    bounds=bounds, constraints=constraints)
    ret, vol, sr = _stats(res.x, mean, cov, rf)
    return {"weights": pd.Series(res.x, index=returns.columns),
            "return": ret, "vol": vol, "sharpe": sr, "success": bool(res.success)}


def efficient_frontier(returns: pd.DataFrame, n_points: int = 40,
                        rf: float = RISK_FREE_RATE, allow_short: bool = False) -> pd.DataFrame:
    mean = returns.mean().values
    n = len(mean)
    min_r = float(mean.min() * TRADING_DAYS)
    max_r = float(mean.max() * TRADING_DAYS)
    targets = np.linspace(min_r, max_r, n_points)

    rows = []
    for t in targets:
        out = target_return(returns, t, rf, allow_short)
        if out["success"]:
            rows.append({"target_return": t, "return": out["return"],
                          "vol": out["vol"], "sharpe": out["sharpe"]})
    return pd.DataFrame(rows)


def risk_parity(returns: pd.DataFrame, rf: float = RISK_FREE_RATE) -> dict:
    """Equal Risk Contribution (ERC) portfolio."""
    cov = returns.cov().values
    mean = returns.mean().values
    n = len(returns.columns)

    def objective(w):
        port_vol = np.sqrt(w @ cov @ w)
        if port_vol <= 0:
            return 1e6
        marginal = cov @ w / port_vol
        contrib = w * marginal
        target = port_vol / n
        return ((contrib - target) ** 2).sum() * 1e6

    bounds = tuple((1e-4, 1.0) for _ in range(n))
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    x0 = np.ones(n) / n
    res = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints,
                    options={"maxiter": 500})
    ret, vol, sr = _stats(res.x, mean, cov, rf)
    return {"weights": pd.Series(res.x, index=returns.columns),
            "return": ret, "vol": vol, "sharpe": sr, "success": bool(res.success)}


def inverse_vol(returns: pd.DataFrame, rf: float = RISK_FREE_RATE) -> dict:
    """Simple inverse-volatility weighting (risk-parity proxy)."""
    vol = returns.std()
    w = (1 / vol) / (1 / vol).sum()
    cov = returns.cov().values
    mean = returns.mean().values
    ret, port_vol, sr = _stats(w.values, mean, cov, rf)
    return {"weights": w, "return": ret, "vol": port_vol, "sharpe": sr, "success": True}


def equal_weight(returns: pd.DataFrame, rf: float = RISK_FREE_RATE) -> dict:
    n = len(returns.columns)
    w = pd.Series(1.0 / n, index=returns.columns)
    cov = returns.cov().values
    mean = returns.mean().values
    ret, vol, sr = _stats(w.values, mean, cov, rf)
    return {"weights": w, "return": ret, "vol": vol, "sharpe": sr, "success": True}


def all_strategies(returns: pd.DataFrame, rf: float = RISK_FREE_RATE,
                    allow_short: bool = False) -> dict:
    """Compute every standard allocation for comparison."""
    return {
        "Equal Weight": equal_weight(returns, rf),
        "Inverse Vol": inverse_vol(returns, rf),
        "Min Variance": min_variance(returns, rf, allow_short),
        "Max Sharpe": max_sharpe(returns, rf, allow_short),
        "Risk Parity (ERC)": risk_parity(returns, rf),
    }
