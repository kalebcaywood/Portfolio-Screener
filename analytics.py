"""Portfolio performance analytics: returns, ratios, drawdowns, attribution."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252
RISK_FREE_RATE = 0.04


# ─── Return aggregates ────────────────────────────────────────────────────────

def total_return(returns: pd.Series) -> float:
    return float((1 + returns).prod() - 1)


def cagr(returns: pd.Series) -> float:
    n_years = len(returns) / TRADING_DAYS
    if n_years <= 0:
        return np.nan
    total = (1 + returns).prod()
    if total <= 0:
        return -1.0
    return float(total ** (1 / n_years) - 1)


def annualized_return(returns: pd.Series) -> float:
    return float(returns.mean() * TRADING_DAYS)


def annualized_vol(returns: pd.Series) -> float:
    return float(returns.std() * np.sqrt(TRADING_DAYS))


def downside_deviation(returns: pd.Series, target: float = 0.0) -> float:
    diff = returns - target / TRADING_DAYS
    downside = diff[diff < 0]
    if len(downside) < 2:
        return np.nan
    return float(np.sqrt((downside ** 2).mean()) * np.sqrt(TRADING_DAYS))


# ─── Risk-adjusted ratios ─────────────────────────────────────────────────────

def sharpe(returns: pd.Series, rf: float = RISK_FREE_RATE) -> float:
    vol = annualized_vol(returns)
    if vol == 0 or np.isnan(vol):
        return np.nan
    return (annualized_return(returns) - rf) / vol


def sortino(returns: pd.Series, rf: float = RISK_FREE_RATE) -> float:
    dd = downside_deviation(returns, target=rf)
    if dd == 0 or np.isnan(dd):
        return np.nan
    return (annualized_return(returns) - rf) / dd


def calmar(returns: pd.Series) -> float:
    mdd = max_drawdown(returns)
    if mdd == 0 or np.isnan(mdd):
        return np.nan
    return annualized_return(returns) / abs(mdd)


def omega(returns: pd.Series, threshold: float = 0.0) -> float:
    daily_thresh = threshold / TRADING_DAYS
    gains = (returns[returns > daily_thresh] - daily_thresh).sum()
    losses = (daily_thresh - returns[returns < daily_thresh]).sum()
    if losses == 0:
        return np.nan
    return float(gains / losses)


def kappa_3(returns: pd.Series, threshold: float = 0.0) -> float:
    """Higher-order downside-risk-adjusted ratio."""
    daily_thresh = threshold / TRADING_DAYS
    excess = returns.mean() - daily_thresh
    downside = (daily_thresh - returns[returns < daily_thresh]) ** 3
    if len(downside) == 0:
        return np.nan
    lpm3 = downside.mean() ** (1 / 3)
    if lpm3 == 0:
        return np.nan
    return float(excess * TRADING_DAYS / (lpm3 * TRADING_DAYS ** (1 / 3)))


# ─── Drawdown analytics ───────────────────────────────────────────────────────

def drawdown_series(returns: pd.Series) -> pd.Series:
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    return cum / peak - 1


def max_drawdown(returns: pd.Series) -> float:
    return float(drawdown_series(returns).min())


def ulcer_index(returns: pd.Series) -> float:
    dd = drawdown_series(returns)
    return float(np.sqrt((dd ** 2).mean()))


def pain_index(returns: pd.Series) -> float:
    return float(abs(drawdown_series(returns).mean()))


def drawdown_episodes(returns: pd.Series, top_n: int = 10) -> pd.DataFrame:
    """Identify peak→trough→recovery episodes."""
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = cum / peak - 1
    if dd.empty:
        return pd.DataFrame()

    episodes = []
    in_dd = False
    peak_date = trough_date = None
    trough_val = 0.0

    for date, val in dd.items():
        if val < -1e-8 and not in_dd:
            in_dd = True
            peak_date = trough_date = date
            trough_val = val
        elif val < -1e-8 and in_dd:
            if val < trough_val:
                trough_val = val
                trough_date = date
        elif val >= -1e-8 and in_dd:
            episodes.append({
                "peak": peak_date,
                "trough": trough_date,
                "recovery": date,
                "depth": trough_val,
                "duration_days": (date - peak_date).days,
                "drawdown_days": (trough_date - peak_date).days,
                "recovery_days": (date - trough_date).days,
            })
            in_dd = False

    if in_dd:
        episodes.append({
            "peak": peak_date, "trough": trough_date, "recovery": None,
            "depth": trough_val,
            "duration_days": (dd.index[-1] - peak_date).days,
            "drawdown_days": (trough_date - peak_date).days,
            "recovery_days": None,
        })

    df = pd.DataFrame(episodes)
    if df.empty:
        return df
    return df.sort_values("depth").head(top_n).reset_index(drop=True)


# ─── Benchmark-relative ───────────────────────────────────────────────────────

def alpha_beta(returns: pd.Series, bench: pd.Series, rf: float = RISK_FREE_RATE) -> dict:
    df = pd.concat([returns, bench], axis=1, join="inner").dropna()
    df.columns = ["r", "m"]
    if len(df) < 30:
        return {k: np.nan for k in ["alpha", "beta", "r_squared", "treynor",
                                      "info_ratio", "tracking_error", "active_return",
                                      "up_capture", "down_capture"]}

    rf_d = rf / TRADING_DAYS
    excess_r = df["r"] - rf_d
    excess_m = df["m"] - rf_d

    slope, intercept, r_val, _, _ = stats.linregress(excess_m, excess_r)
    beta = slope
    alpha_annual = intercept * TRADING_DAYS
    r_squared = r_val ** 2

    treynor_val = (df["r"].mean() * TRADING_DAYS - rf) / beta if beta != 0 else np.nan
    active = df["r"] - df["m"]
    te = active.std() * np.sqrt(TRADING_DAYS)
    ir = (active.mean() * TRADING_DAYS) / te if te > 0 else np.nan

    up = df[df["m"] > 0]
    down = df[df["m"] < 0]
    up_cap = (up["r"].mean() / up["m"].mean()) if len(up) > 5 and up["m"].mean() != 0 else np.nan
    down_cap = (down["r"].mean() / down["m"].mean()) if len(down) > 5 and down["m"].mean() != 0 else np.nan

    return {
        "alpha": alpha_annual,
        "beta": beta,
        "r_squared": r_squared,
        "treynor": treynor_val,
        "info_ratio": ir,
        "tracking_error": te,
        "active_return": active.mean() * TRADING_DAYS,
        "up_capture": up_cap,
        "down_capture": down_cap,
    }


def m_squared(returns: pd.Series, bench: pd.Series, rf: float = RISK_FREE_RATE) -> float:
    sr = sharpe(returns, rf)
    bench_vol = annualized_vol(bench)
    if np.isnan(sr) or np.isnan(bench_vol):
        return np.nan
    return rf + sr * bench_vol


# ─── Rolling analytics ────────────────────────────────────────────────────────

def rolling_metrics(returns: pd.Series, window: int = 63,
                     rf: float = RISK_FREE_RATE) -> pd.DataFrame:
    ann_ret = returns.rolling(window).mean() * TRADING_DAYS
    ann_vol = returns.rolling(window).std() * np.sqrt(TRADING_DAYS)
    sr = (ann_ret - rf) / ann_vol
    cum = (1 + returns).cumprod()
    rolling_dd = (cum / cum.cummax() - 1).rolling(window).min()
    return pd.DataFrame({
        "rolling_return": ann_ret,
        "rolling_vol": ann_vol,
        "rolling_sharpe": sr,
        "rolling_max_dd": rolling_dd,
    })


def rolling_beta(returns: pd.Series, bench: pd.Series, window: int = 63) -> pd.Series:
    df = pd.concat([returns, bench], axis=1, join="inner").dropna()
    df.columns = ["r", "m"]
    cov = df["r"].rolling(window).cov(df["m"])
    var = df["m"].rolling(window).var()
    return cov / var


def rolling_correlation(returns: pd.Series, bench: pd.Series, window: int = 63) -> pd.Series:
    df = pd.concat([returns, bench], axis=1, join="inner").dropna()
    return df.iloc[:, 0].rolling(window).corr(df.iloc[:, 1])


# ─── Period returns ───────────────────────────────────────────────────────────

def monthly_returns(returns: pd.Series) -> pd.Series:
    return (1 + returns).resample("ME").prod() - 1


def annual_returns(returns: pd.Series) -> pd.Series:
    return (1 + returns).resample("YE").prod() - 1


def monthly_heatmap(returns: pd.Series) -> pd.DataFrame:
    """Months × years matrix of monthly returns."""
    m = monthly_returns(returns)
    df = pd.DataFrame({"ret": m.values, "year": m.index.year, "month": m.index.month})
    return df.pivot(index="year", columns="month", values="ret")


# ─── Summary aggregator ───────────────────────────────────────────────────────

def summary_stats(returns: pd.Series, bench: pd.Series | None = None,
                   rf: float = RISK_FREE_RATE) -> dict:
    out = {
        "total_return": total_return(returns),
        "cagr": cagr(returns),
        "ann_return": annualized_return(returns),
        "ann_vol": annualized_vol(returns),
        "downside_dev": downside_deviation(returns, target=rf),
        "sharpe": sharpe(returns, rf),
        "sortino": sortino(returns, rf),
        "calmar": calmar(returns),
        "omega": omega(returns),
        "max_drawdown": max_drawdown(returns),
        "ulcer_index": ulcer_index(returns),
        "pain_index": pain_index(returns),
        "skew": float(returns.skew()),
        "kurtosis": float(returns.kurtosis()),
        "best_day": float(returns.max()),
        "worst_day": float(returns.min()),
        "pct_positive_days": float((returns > 0).mean()),
        "avg_win": float(returns[returns > 0].mean()) if (returns > 0).any() else np.nan,
        "avg_loss": float(returns[returns < 0].mean()) if (returns < 0).any() else np.nan,
        "win_loss_ratio": (
            float(abs(returns[returns > 0].mean() / returns[returns < 0].mean()))
            if (returns < 0).any() and (returns > 0).any() else np.nan
        ),
        "n_observations": len(returns),
    }
    if bench is not None:
        out.update(alpha_beta(returns, bench, rf))
        out["m_squared"] = m_squared(returns, bench, rf)
    return out
