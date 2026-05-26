"""CAPM and multi-factor regression analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
from scipy import stats

TRADING_DAYS = 252
RISK_FREE_RATE = 0.04


def capm(returns: pd.Series, bench: pd.Series, rf: float = RISK_FREE_RATE) -> dict:
    """Single-factor (market) CAPM regression. Returns full regression output."""
    df = pd.concat([returns, bench], axis=1, join="inner").dropna()
    df.columns = ["r", "m"]
    if len(df) < 30:
        return {"error": "Need at least 30 overlapping observations"}

    rf_d = rf / TRADING_DAYS
    y = df["r"] - rf_d
    X = df["m"] - rf_d

    if HAS_STATSMODELS:
        Xc = sm.add_constant(X)
        model = sm.OLS(y, Xc).fit()
        return {
            "alpha_daily": float(model.params["const"]),
            "alpha_annual": float(model.params["const"] * TRADING_DAYS),
            "alpha_tstat": float(model.tvalues["const"]),
            "alpha_pvalue": float(model.pvalues["const"]),
            "beta": float(model.params["m"]),
            "beta_tstat": float(model.tvalues["m"]),
            "beta_pvalue": float(model.pvalues["m"]),
            "beta_ci_low": float(model.conf_int().loc["m", 0]),
            "beta_ci_high": float(model.conf_int().loc["m", 1]),
            "r_squared": float(model.rsquared),
            "r_squared_adj": float(model.rsquared_adj),
            "f_stat": float(model.fvalue),
            "f_pvalue": float(model.f_pvalue),
            "n_obs": int(model.nobs),
            "residual_std": float(model.resid.std()),
            "durbin_watson": float(sm.stats.stattools.durbin_watson(model.resid)),
            "_model": model,
            "_residuals": model.resid,
            "_fitted": model.fittedvalues,
        }
    else:
        slope, intercept, r_val, p_val, std_err = stats.linregress(X, y)
        return {
            "alpha_daily": float(intercept),
            "alpha_annual": float(intercept * TRADING_DAYS),
            "beta": float(slope),
            "beta_pvalue": float(p_val),
            "r_squared": float(r_val ** 2),
            "n_obs": len(df),
        }


def multi_factor_regression(returns: pd.Series, factor_returns: pd.DataFrame,
                             rf: float = RISK_FREE_RATE) -> dict:
    """OLS regression of excess returns on multiple factor returns."""
    if not HAS_STATSMODELS:
        return {"error": "statsmodels required for multi-factor regression"}

    df = pd.concat([returns.rename("r"), factor_returns], axis=1, join="inner").dropna()
    if len(df) < 30:
        return {"error": f"Need 30+ overlapping observations; have {len(df)}"}

    rf_d = rf / TRADING_DAYS
    y = df["r"] - rf_d
    X = df.drop(columns="r")
    Xc = sm.add_constant(X)
    model = sm.OLS(y, Xc).fit()

    rows = []
    for name in model.params.index:
        rows.append({
            "factor": name,
            "coefficient": float(model.params[name]),
            "annualized": float(model.params[name] * TRADING_DAYS) if name == "const" else float(model.params[name]),
            "std_error": float(model.bse[name]),
            "t_stat": float(model.tvalues[name]),
            "p_value": float(model.pvalues[name]),
            "ci_low": float(model.conf_int().loc[name, 0]),
            "ci_high": float(model.conf_int().loc[name, 1]),
            "significance": ("***" if model.pvalues[name] < 0.01 else
                              "**" if model.pvalues[name] < 0.05 else
                              "*" if model.pvalues[name] < 0.10 else ""),
        })

    return {
        "factor_table": pd.DataFrame(rows),
        "r_squared": float(model.rsquared),
        "r_squared_adj": float(model.rsquared_adj),
        "f_stat": float(model.fvalue),
        "f_pvalue": float(model.f_pvalue),
        "n_obs": int(model.nobs),
        "aic": float(model.aic),
        "bic": float(model.bic),
        "durbin_watson": float(sm.stats.stattools.durbin_watson(model.resid)),
        "residual_std": float(model.resid.std()),
        "_model": model,
        "_residuals": model.resid,
        "_fitted": model.fittedvalues,
    }


def rolling_capm(returns: pd.Series, bench: pd.Series, window: int = 63,
                  rf: float = RISK_FREE_RATE) -> pd.DataFrame:
    """Rolling alpha & beta over time."""
    df = pd.concat([returns, bench], axis=1, join="inner").dropna()
    df.columns = ["r", "m"]
    rf_d = rf / TRADING_DAYS
    ex_r = df["r"] - rf_d
    ex_m = df["m"] - rf_d
    cov = ex_r.rolling(window).cov(ex_m)
    var = ex_m.rolling(window).var()
    beta = cov / var
    alpha = (ex_r.rolling(window).mean() - beta * ex_m.rolling(window).mean()) * TRADING_DAYS
    return pd.DataFrame({"alpha": alpha, "beta": beta})


def factor_attribution(returns: pd.Series, factor_returns: pd.DataFrame,
                        rf: float = RISK_FREE_RATE) -> pd.DataFrame:
    """Decompose return into factor contributions: beta_i × mean(factor_i) × T."""
    if not HAS_STATSMODELS:
        return pd.DataFrame()
    res = multi_factor_regression(returns, factor_returns, rf)
    if "error" in res:
        return pd.DataFrame()
    ft = res["factor_table"]
    factor_means = factor_returns.mean() * TRADING_DAYS
    rows = []
    for _, r in ft.iterrows():
        name = r["factor"]
        if name == "const":
            rows.append({"factor": name, "exposure": r["coefficient"],
                          "factor_return": np.nan, "contribution": r["annualized"]})
        else:
            rows.append({"factor": name, "exposure": r["coefficient"],
                          "factor_return": float(factor_means.get(name, np.nan)),
                          "contribution": float(r["coefficient"] * factor_means.get(name, 0))})
    return pd.DataFrame(rows)
