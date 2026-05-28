"""Return Stream Analyzer — CSV parser, frequency detection, frequency-aware stats.

Designed for pure return-time-series analysis (no holdings). Accepts:
  - Wide format:  Date, FundA, FundB, FundC, ...  (one column per stream)
  - Long format:  Date, Stream, Return            (one row per stream-period)

Auto-detects monthly / quarterly / weekly / daily and annualizes correctly.
"""
from __future__ import annotations

import io
import re

import numpy as np
import pandas as pd

# Periods-per-year by frequency code
PPY: dict[str, int] = {"D": 252, "W": 52, "M": 12, "Q": 4, "A": 1}

FREQ_LABEL = {"D": "Daily", "W": "Weekly", "M": "Monthly", "Q": "Quarterly", "A": "Annual"}


# ─── Column-alias matchers ───────────────────────────────────────────────────

_DATE_ALIASES = {"date", "period", "month", "quarter", "year", "asof", "asofdate",
                 "monthend", "monthenddate", "perioddate", "reportdate"}
_RETURN_ALIASES = {"return", "ret", "monthlyreturn", "netreturn", "grossreturn",
                   "performance", "perfreturn", "totalreturn", "tr", "pct",
                   "monthlypct", "rtn"}
_STREAM_ALIASES = {"stream", "fund", "manager", "strategy", "name", "ticker",
                   "account", "portfolio", "id", "label"}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _find_col(columns, aliases: set[str]) -> str | None:
    norm_map = {_norm(c): c for c in columns}
    for a in aliases:
        if _norm(a) in norm_map:
            return norm_map[_norm(a)]
    return None


# ─── Frequency detection ─────────────────────────────────────────────────────

def detect_frequency(index: pd.DatetimeIndex) -> str:
    """Return one of D/W/M/Q/A based on the median gap between observations."""
    if len(index) < 2:
        return "M"  # default
    gaps_days = pd.Series(index).diff().dt.days.dropna()
    if gaps_days.empty:
        return "M"
    median_gap = float(gaps_days.median())
    if median_gap <= 3:
        return "D"
    if median_gap <= 10:
        return "W"
    if median_gap <= 45:
        return "M"
    if median_gap <= 120:
        return "Q"
    return "A"


def periods_per_year(freq: str) -> int:
    return PPY.get(freq.upper(), 12)


# ─── Return value coercion ───────────────────────────────────────────────────

def _coerce_return(val) -> float:
    """Parse a single return cell: '1.42%' -> 0.0142, '(0.014)' -> -0.014, etc."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    if not s or s in {"-", "—", "n/a", "na", "nan"}:
        return np.nan
    neg = s.startswith("(") and s.endswith(")")
    s = s.replace("(", "").replace(")", "").strip()
    is_pct = s.endswith("%")
    if is_pct:
        s = s[:-1].strip()
    s = s.replace(",", "")
    try:
        v = float(s)
    except ValueError:
        return np.nan
    if is_pct:
        v = v / 100.0
    if neg:
        v = -v
    return v


def _clean_return_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Coerce every column to numeric return; if values look like percent (abs sum
    suggests percent scale) divide by 100."""
    notes: list[str] = []
    cleaned = df.copy()
    for col in cleaned.columns:
        cleaned[col] = cleaned[col].apply(_coerce_return)
    # Heuristic: if a column's median absolute value > 0.5, it's almost certainly
    # in percent (no fund returns 50%+ per period on median). Convert.
    for col in cleaned.columns:
        vals = cleaned[col].dropna()
        if len(vals) > 5 and vals.abs().median() > 0.5:
            cleaned[col] = cleaned[col] / 100.0
            notes.append(f"Column '{col}' looked like percent-scale; divided by 100.")
    return cleaned, notes


# ─── CSV parser ──────────────────────────────────────────────────────────────

def parse_return_csv(file_or_buffer) -> dict:
    """Parse a return-stream CSV and return a clean wide-format DataFrame.

    Returns dict with:
        df         — wide DataFrame: DatetimeIndex × stream columns (decimals)
        frequency  — 'D' / 'W' / 'M' / 'Q' / 'A'
        streams    — list of stream column names
        format     — 'wide' or 'long' (detected)
        errors     — list of fatal error messages
        warnings   — list of non-fatal messages
    """
    out: dict = {"df": None, "frequency": None, "streams": [],
                  "format": None, "errors": [], "warnings": []}
    try:
        raw = pd.read_csv(file_or_buffer)
    except UnicodeDecodeError:
        try:
            file_or_buffer.seek(0)
        except Exception:
            pass
        raw = pd.read_csv(file_or_buffer, encoding="latin-1")
    except Exception as e:
        out["errors"].append(f"Could not read CSV: {e}")
        return out

    if raw.empty:
        out["errors"].append("CSV is empty.")
        return out

    raw.columns = [str(c).strip().lstrip("﻿") for c in raw.columns]
    date_col = _find_col(raw.columns, _DATE_ALIASES)
    if date_col is None:
        # Try the first column if it parses as date
        first_col = raw.columns[0]
        parsed = pd.to_datetime(raw[first_col], errors="coerce")
        if parsed.notna().sum() / max(len(parsed), 1) > 0.8:
            date_col = first_col
        else:
            out["errors"].append(
                f"No date column found. Expected one of: {sorted(_DATE_ALIASES)}, "
                f"got: {list(raw.columns)}"
            )
            return out

    raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")
    n_bad_dates = int(raw[date_col].isna().sum())
    if n_bad_dates:
        out["warnings"].append(f"Dropped {n_bad_dates} rows with unparseable dates.")
        raw = raw.dropna(subset=[date_col])

    if raw.empty:
        out["errors"].append("No rows remain after date parsing.")
        return out

    # Detect format: long if there's a Return column AND a Stream-name column
    return_col = _find_col(raw.columns, _RETURN_ALIASES)
    stream_col = _find_col(raw.columns, _STREAM_ALIASES)

    if return_col is not None and stream_col is not None and stream_col != date_col:
        # ── Long format ────────────────────────────────────────────────────
        out["format"] = "long"
        long_df = raw[[date_col, stream_col, return_col]].copy()
        long_df.columns = ["__date", "__stream", "__return"]
        long_df["__return"] = long_df["__return"].apply(_coerce_return)
        wide = long_df.pivot_table(index="__date", columns="__stream",
                                     values="__return", aggfunc="mean")
        wide = wide.sort_index()
        wide.index.name = "Date"
        wide.columns.name = None
        # Pct-scale heuristic
        cleaned, notes = _clean_return_frame(wide)
        out["warnings"].extend(notes)
        wide = cleaned
    else:
        # ── Wide format ────────────────────────────────────────────────────
        out["format"] = "wide"
        wide = raw.set_index(date_col).sort_index()
        wide.index.name = "Date"
        # Drop any non-numeric-coercible columns
        wide, notes = _clean_return_frame(wide)
        # Drop columns that are entirely NaN
        all_nan = [c for c in wide.columns if wide[c].isna().all()]
        if all_nan:
            out["warnings"].append(f"Dropped non-numeric columns: {all_nan}")
            wide = wide.drop(columns=all_nan)
        out["warnings"].extend(notes)

    if wide.empty or wide.shape[1] == 0:
        out["errors"].append("No usable return columns after cleaning.")
        return out

    wide.index = pd.to_datetime(wide.index)
    freq = detect_frequency(wide.index)

    out["df"] = wide
    out["frequency"] = freq
    out["streams"] = list(wide.columns)
    return out


# ─── Frequency-aware analytics ──────────────────────────────────────────────

def cagr(returns: pd.Series, freq: str = "M") -> float:
    ppy = periods_per_year(freq)
    n_years = len(returns.dropna()) / ppy
    if n_years <= 0:
        return np.nan
    total = float((1 + returns.dropna()).prod())
    if total <= 0:
        return -1.0
    return float(total ** (1 / n_years) - 1)


def total_return(returns: pd.Series) -> float:
    return float((1 + returns.dropna()).prod() - 1)


def annualized_vol(returns: pd.Series, freq: str = "M") -> float:
    return float(returns.std() * np.sqrt(periods_per_year(freq)))


def sharpe_ratio(returns: pd.Series, freq: str = "M", rf: float = 0.04) -> float:
    vol = annualized_vol(returns, freq)
    if vol == 0 or np.isnan(vol):
        return np.nan
    ann_ret = returns.mean() * periods_per_year(freq)
    return float((ann_ret - rf) / vol)


def sortino_ratio(returns: pd.Series, freq: str = "M", rf: float = 0.04) -> float:
    downside = returns[returns < 0]
    if len(downside) < 2:
        return np.nan
    ppy = periods_per_year(freq)
    dd_std = downside.std() * np.sqrt(ppy)
    if dd_std == 0:
        return np.nan
    return float((returns.mean() * ppy - rf) / dd_std)


def drawdown_series(returns: pd.Series) -> pd.Series:
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    return cum / peak - 1


def max_drawdown(returns: pd.Series) -> float:
    return float(drawdown_series(returns).min())


def calmar_ratio(returns: pd.Series, freq: str = "M") -> float:
    mdd = max_drawdown(returns)
    if mdd == 0 or np.isnan(mdd):
        return np.nan
    return float(cagr(returns, freq) / abs(mdd))


def alpha_beta_against(returns: pd.Series, bench: pd.Series,
                        freq: str = "M", rf: float = 0.04) -> dict:
    """OLS regression of excess returns on excess benchmark returns."""
    df = pd.concat([returns, bench], axis=1, join="inner").dropna()
    df.columns = ["r", "m"]
    if len(df) < 6:
        return {k: np.nan for k in ["alpha", "beta", "r_squared", "info_ratio",
                                      "tracking_error", "up_capture", "down_capture"]}
    ppy = periods_per_year(freq)
    rf_period = rf / ppy
    ex_r = df["r"] - rf_period
    ex_m = df["m"] - rf_period
    cov = float(ex_r.cov(ex_m))
    var_m = float(ex_m.var())
    beta = cov / var_m if var_m > 0 else np.nan
    alpha_period = float(ex_r.mean() - beta * ex_m.mean()) if not np.isnan(beta) else np.nan
    alpha_ann = alpha_period * ppy if not np.isnan(alpha_period) else np.nan
    # R-squared
    corr = float(ex_r.corr(ex_m))
    r_sq = corr ** 2 if not np.isnan(corr) else np.nan
    # Active stats
    active = df["r"] - df["m"]
    te = float(active.std() * np.sqrt(ppy))
    ir = float(active.mean() * ppy / te) if te > 0 else np.nan
    # Capture
    up = df[df["m"] > 0]
    down = df[df["m"] < 0]
    up_cap = (up["r"].mean() / up["m"].mean()) if len(up) > 2 and up["m"].mean() != 0 else np.nan
    down_cap = (down["r"].mean() / down["m"].mean()) if len(down) > 2 and down["m"].mean() != 0 else np.nan
    return {
        "alpha": alpha_ann, "beta": beta, "r_squared": r_sq,
        "info_ratio": ir, "tracking_error": te,
        "up_capture": float(up_cap) if not np.isnan(up_cap) else np.nan,
        "down_capture": float(down_cap) if not np.isnan(down_cap) else np.nan,
        "active_return": float(active.mean() * ppy),
    }


def summary_stats(returns: pd.Series, freq: str = "M", rf: float = 0.04,
                   bench: pd.Series | None = None) -> dict:
    r = returns.dropna()
    out = {
        "n_obs": len(r),
        "total_return": total_return(r),
        "cagr": cagr(r, freq),
        "ann_return": float(r.mean() * periods_per_year(freq)),
        "ann_vol": annualized_vol(r, freq),
        "sharpe": sharpe_ratio(r, freq, rf),
        "sortino": sortino_ratio(r, freq, rf),
        "calmar": calmar_ratio(r, freq),
        "max_drawdown": max_drawdown(r),
        "skew": float(r.skew()),
        "kurtosis": float(r.kurtosis()),
        "best_period": float(r.max()),
        "worst_period": float(r.min()),
        "pct_positive": float((r > 0).mean()),
        "hit_rate": float((r > 0).mean()),
    }
    if bench is not None and len(bench) > 5:
        out.update(alpha_beta_against(r, bench, freq, rf))
    return out


def rolling_stats(returns: pd.Series, window: int, freq: str = "M",
                   rf: float = 0.04) -> pd.DataFrame:
    """Rolling annualized return, vol, Sharpe over `window` periods."""
    ppy = periods_per_year(freq)
    ann_ret = returns.rolling(window).mean() * ppy
    ann_vol = returns.rolling(window).std() * np.sqrt(ppy)
    sr = (ann_ret - rf) / ann_vol
    return pd.DataFrame({"return": ann_ret, "vol": ann_vol, "sharpe": sr})


# ─── Convenience: align returns to a benchmark (for freq matching) ───────────

def align_to_period(returns_df: pd.DataFrame, bench_returns: pd.Series,
                     freq: str) -> tuple[pd.DataFrame, pd.Series]:
    """Resample benchmark to the same period as the returns df.

    Benchmark is typically daily (Yahoo). If the streams are monthly, we need
    to compound the benchmark to month-end.
    """
    if freq == "D" or bench_returns.empty:
        common = returns_df.index.intersection(bench_returns.index)
        return returns_df.loc[common], bench_returns.loc[common]
    # Compound daily benchmark returns to the period frequency
    period_map = {"W": "W", "M": "ME", "Q": "QE", "A": "YE"}
    rule = period_map.get(freq, "ME")
    bench_period = (1 + bench_returns).resample(rule).prod() - 1
    # Align indices — pick the closest period-end date to each return-stream date
    bench_aligned = bench_period.reindex(returns_df.index, method="nearest",
                                            tolerance=pd.Timedelta(days=15))
    return returns_df, bench_aligned.dropna()
