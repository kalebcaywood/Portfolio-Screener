"""Statistical hypothesis tests for return series."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

try:
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
    from statsmodels.stats.stattools import jarque_bera, durbin_watson
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


def _conclude(p: float, h0_msg: str, h1_msg: str, level: float = 0.05) -> str:
    if pd.isna(p):
        return "—"
    return h1_msg if p < level else h0_msg


# ─── Normality ────────────────────────────────────────────────────────────────

def normality_tests(returns: pd.Series) -> pd.DataFrame:
    r = returns.dropna()
    out = []

    if 3 <= len(r) <= 5000:
        s, p = stats.shapiro(r)
        out.append(("Shapiro-Wilk", s, p, _conclude(p, "Cannot reject normality", "Reject normality")))

    s, p = stats.normaltest(r)
    out.append(("D'Agostino K²", s, p, _conclude(p, "Cannot reject normality", "Reject normality")))

    ad = stats.anderson(r, dist="norm")
    crit_5 = ad.critical_values[2]
    out.append(("Anderson-Darling", ad.statistic, np.nan,
                f"{'Reject' if ad.statistic > crit_5 else 'Cannot reject'} normality (5% crit = {crit_5:.3f})"))

    if HAS_STATSMODELS:
        jb_stat, jb_p, sk, kt = jarque_bera(r)
        out.append(("Jarque-Bera", jb_stat, jb_p,
                    _conclude(jb_p, "Cannot reject normality", "Reject normality")))
    else:
        jb_stat, jb_p = stats.jarque_bera(r)
        out.append(("Jarque-Bera", jb_stat, jb_p,
                    _conclude(jb_p, "Cannot reject normality", "Reject normality")))

    s, p = stats.kstest(r, "norm", args=(r.mean(), r.std()))
    out.append(("Kolmogorov-Smirnov", s, p,
                _conclude(p, "Cannot reject normality", "Reject normality")))

    return pd.DataFrame(out, columns=["test", "statistic", "p_value", "conclusion"])


# ─── Stationarity ─────────────────────────────────────────────────────────────

def stationarity_tests(series: pd.Series) -> pd.DataFrame:
    if not HAS_STATSMODELS:
        return pd.DataFrame()
    s = series.dropna()
    out = []

    try:
        adf_stat, adf_p, _, _, crit, _ = adfuller(s, autolag="AIC")
        out.append(("ADF (H₀: unit root)", adf_stat, adf_p,
                    _conclude(adf_p, "Non-stationary (unit root)", "Stationary")))
    except Exception:
        pass

    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kpss_stat, kpss_p, _, _ = kpss(s, regression="c", nlags="auto")
        out.append(("KPSS (H₀: stationary)", kpss_stat, kpss_p,
                    _conclude(kpss_p, "Stationary", "Non-stationary")))
    except Exception:
        pass

    return pd.DataFrame(out, columns=["test", "statistic", "p_value", "conclusion"])


# ─── Autocorrelation / serial correlation ─────────────────────────────────────

def autocorrelation_tests(returns: pd.Series, lags: int = 10) -> pd.DataFrame:
    if not HAS_STATSMODELS:
        return pd.DataFrame()
    r = returns.dropna()
    out = []

    lb = acorr_ljungbox(r, lags=[lags], return_df=True)
    s = float(lb["lb_stat"].iloc[0])
    p = float(lb["lb_pvalue"].iloc[0])
    out.append((f"Ljung-Box returns (lag={lags})", s, p,
                _conclude(p, "No serial autocorrelation", "Serial autocorrelation present")))

    lb_sq = acorr_ljungbox(r ** 2, lags=[lags], return_df=True)
    s = float(lb_sq["lb_stat"].iloc[0])
    p = float(lb_sq["lb_pvalue"].iloc[0])
    out.append((f"Ljung-Box squared (lag={lags})", s, p,
                _conclude(p, "No volatility clustering", "Volatility clustering present")))

    dw = float(durbin_watson(r))
    conc = "Positive autocorrelation" if dw < 1.5 else "Negative autocorrelation" if dw > 2.5 else "No autocorrelation"
    out.append(("Durbin-Watson", dw, np.nan, conc))

    return pd.DataFrame(out, columns=["test", "statistic", "p_value", "conclusion"])


# ─── Heteroscedasticity ───────────────────────────────────────────────────────

def heteroscedasticity_tests(returns: pd.Series, lags: int = 5) -> pd.DataFrame:
    if not HAS_STATSMODELS:
        return pd.DataFrame()
    r = returns.dropna()
    out = []

    try:
        arch_stat, arch_p, _, _ = het_arch(r, nlags=lags)
        out.append((f"Engle's ARCH-LM (lag={lags})", arch_stat, arch_p,
                    _conclude(arch_p, "Homoscedastic", "Heteroscedastic / ARCH effects")))
    except Exception:
        pass

    return pd.DataFrame(out, columns=["test", "statistic", "p_value", "conclusion"])


# ─── Mean / location tests ────────────────────────────────────────────────────

def t_test_mean(returns: pd.Series, mu0: float = 0.0) -> pd.DataFrame:
    r = returns.dropna()
    s, p = stats.ttest_1samp(r, mu0)
    return pd.DataFrame([{
        "test": f"One-sample t-test (H₀: μ = {mu0:.4f})",
        "statistic": s, "p_value": p,
        "conclusion": _conclude(p, f"Cannot reject μ = {mu0}", f"Reject μ = {mu0}"),
    }])


def sign_test(returns: pd.Series) -> pd.DataFrame:
    """Tests if returns are symmetric around 0."""
    r = returns.dropna()
    pos = (r > 0).sum()
    neg = (r < 0).sum()
    n = pos + neg
    if n == 0:
        return pd.DataFrame()
    # Binomial test against 0.5
    res = stats.binomtest(pos, n, p=0.5)
    return pd.DataFrame([{
        "test": "Sign test (H₀: median = 0)",
        "statistic": pos / n,
        "p_value": res.pvalue,
        "conclusion": _conclude(res.pvalue, "Median ≈ 0", "Median ≠ 0"),
    }])


# ─── Two-sample comparisons ───────────────────────────────────────────────────

def two_sample_tests(r1: pd.Series, r2: pd.Series,
                      label1: str = "A", label2: str = "B") -> pd.DataFrame:
    a, b = r1.dropna(), r2.dropna()
    out = []

    s, p = stats.ttest_ind(a, b, equal_var=False)
    out.append((f"Welch t-test ({label1} vs {label2})", s, p,
                _conclude(p, "Equal means", "Means differ")))

    s, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    out.append(("Mann-Whitney U", s, p,
                _conclude(p, "Same distribution", "Distributions differ")))

    s, p = stats.ks_2samp(a, b)
    out.append(("Kolmogorov-Smirnov 2-sample", s, p,
                _conclude(p, "Same distribution", "Distributions differ")))

    s, p = stats.levene(a, b)
    out.append(("Levene (equal variance)", s, p,
                _conclude(p, "Equal variance", "Variances differ")))

    return pd.DataFrame(out, columns=["test", "statistic", "p_value", "conclusion"])


# ─── Randomness / efficient-market tests ──────────────────────────────────────

def runs_test(returns: pd.Series) -> pd.DataFrame:
    """Wald-Wolfowitz runs test for randomness."""
    r = returns.dropna()
    signs = (r > r.median()).astype(int).values
    n1 = int(signs.sum())
    n2 = int(len(signs) - n1)
    if n1 == 0 or n2 == 0:
        return pd.DataFrame()
    runs = 1 + int((np.diff(signs) != 0).sum())
    expected = (2 * n1 * n2) / (n1 + n2) + 1
    var = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2) ** 2 * (n1 + n2 - 1))
    z = (runs - expected) / np.sqrt(var) if var > 0 else np.nan
    p = 2 * (1 - stats.norm.cdf(abs(z))) if not np.isnan(z) else np.nan
    return pd.DataFrame([{
        "test": "Runs test (Wald-Wolfowitz)",
        "statistic": z, "p_value": p,
        "conclusion": _conclude(p, "Cannot reject randomness", "Non-random sequence"),
    }])


def variance_ratio_test(returns: pd.Series, k: int = 5) -> pd.DataFrame:
    """Lo-MacKinlay variance ratio test (H₀: random walk)."""
    r = returns.dropna().values
    n = len(r)
    if n < 2 * k:
        return pd.DataFrame()
    mu = r.mean()
    var1 = ((r - mu) ** 2).sum() / (n - 1)
    rk = np.array([r[i:i + k].sum() for i in range(n - k + 1)])
    vark = ((rk - k * mu) ** 2).sum() / (k * (n - k + 1) * (1 - k / n))
    vr = vark / var1 if var1 > 0 else np.nan
    # Asymptotic z-stat
    phi = 2 * (2 * k - 1) * (k - 1) / (3 * k * n)
    z = (vr - 1) / np.sqrt(phi) if phi > 0 else np.nan
    p = 2 * (1 - stats.norm.cdf(abs(z))) if not np.isnan(z) else np.nan
    return pd.DataFrame([{
        "test": f"Variance Ratio (k={k})",
        "statistic": vr, "p_value": p,
        "conclusion": _conclude(p, "Random walk", "Mean reverting or trending"),
    }])


# ─── Distribution descriptors ────────────────────────────────────────────────

def describe_distribution(returns: pd.Series) -> dict:
    r = returns.dropna()
    return {
        "n": len(r),
        "mean": float(r.mean()),
        "median": float(r.median()),
        "std": float(r.std()),
        "min": float(r.min()),
        "max": float(r.max()),
        "skew": float(r.skew()),
        "excess_kurtosis": float(r.kurtosis()),
        "pct_5": float(np.percentile(r, 5)),
        "pct_25": float(np.percentile(r, 25)),
        "pct_75": float(np.percentile(r, 75)),
        "pct_95": float(np.percentile(r, 95)),
    }
