"""Data fetching and quantitative metric computation for the equity screener."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

RISK_FREE_RATE = 0.04
TRADING_DAYS = 252
BENCHMARK = "^GSPC"


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_history(symbol: str, period: str = "2y") -> pd.DataFrame:
    try:
        hist = yf.Ticker(symbol).history(period=period, auto_adjust=True)
        return hist if hist is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_info(symbol: str) -> dict:
    try:
        return yf.Ticker(symbol).info or {}
    except Exception:
        return {}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_financials(symbol: str) -> dict:
    try:
        t = yf.Ticker(symbol)
        return {
            "income": t.income_stmt if t.income_stmt is not None else pd.DataFrame(),
            "balance": t.balance_sheet if t.balance_sheet is not None else pd.DataFrame(),
            "cashflow": t.cashflow if t.cashflow is not None else pd.DataFrame(),
        }
    except Exception:
        return {"income": pd.DataFrame(), "balance": pd.DataFrame(), "cashflow": pd.DataFrame()}


def returns_from_prices(prices: pd.Series) -> pd.Series:
    return prices.pct_change().dropna()


def total_return(prices: pd.Series, days: int) -> float:
    if len(prices) < days + 1:
        return np.nan
    return float(prices.iloc[-1] / prices.iloc[-days - 1] - 1)


def annualized_volatility(returns: pd.Series) -> float:
    if len(returns) < 2:
        return np.nan
    return float(returns.std() * np.sqrt(TRADING_DAYS))


def sharpe_ratio(returns: pd.Series, rf: float = RISK_FREE_RATE) -> float:
    vol = annualized_volatility(returns)
    if not vol or np.isnan(vol):
        return np.nan
    excess = returns.mean() * TRADING_DAYS - rf
    return float(excess / vol)


def sortino_ratio(returns: pd.Series, rf: float = RISK_FREE_RATE) -> float:
    if len(returns) < 2:
        return np.nan
    downside = returns[returns < 0]
    if len(downside) == 0:
        return np.nan
    dd_std = downside.std() * np.sqrt(TRADING_DAYS)
    if dd_std == 0:
        return np.nan
    excess = returns.mean() * TRADING_DAYS - rf
    return float(excess / dd_std)


def max_drawdown(prices: pd.Series) -> float:
    if len(prices) < 2:
        return np.nan
    cum = prices / prices.iloc[0]
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return float(dd.min())


def rsi(prices: pd.Series, period: int = 14) -> float:
    if len(prices) < period + 1:
        return np.nan
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    series = 100 - (100 / (1 + rs))
    val = series.iloc[-1]
    return float(val) if pd.notna(val) else np.nan


def beta_vs_benchmark(symbol_returns: pd.Series, bench_returns: pd.Series) -> float:
    df = pd.concat([symbol_returns, bench_returns], axis=1, join="inner").dropna()
    if len(df) < 30:
        return np.nan
    cov = df.cov().iloc[0, 1]
    var = df.iloc[:, 1].var()
    return float(cov / var) if var > 0 else np.nan


def _safe_loc(df: pd.DataFrame, key: str, col: int) -> float:
    if df.empty or key not in df.index or col >= df.shape[1]:
        return np.nan
    try:
        return float(df.loc[key].iloc[col])
    except (KeyError, IndexError, TypeError, ValueError):
        return np.nan


def piotroski_fscore(financials: dict) -> float:
    """Piotroski F-Score (0-9). Returns NaN if statements are unavailable."""
    inc, bs, cf = financials["income"], financials["balance"], financials["cashflow"]
    if inc.empty or bs.empty or cf.empty or bs.shape[1] < 2:
        return np.nan

    ni = _safe_loc(inc, "Net Income", 0)
    ta_curr = _safe_loc(bs, "Total Assets", 0)
    ta_prev = _safe_loc(bs, "Total Assets", 1)
    avg_assets = (ta_curr + ta_prev) / 2 if pd.notna(ta_curr) and pd.notna(ta_prev) else np.nan
    roa = ni / avg_assets if avg_assets and not np.isnan(avg_assets) else np.nan
    cfo = _safe_loc(cf, "Operating Cash Flow", 0)

    score = 0
    if pd.notna(ni) and ni > 0:
        score += 1
    if pd.notna(roa) and roa > 0:
        score += 1
    if pd.notna(cfo) and cfo > 0:
        score += 1
    if pd.notna(cfo) and pd.notna(ni) and cfo > ni:
        score += 1

    ltd_curr = _safe_loc(bs, "Long Term Debt", 0)
    ltd_prev = _safe_loc(bs, "Long Term Debt", 1)
    if pd.notna(ltd_curr) and pd.notna(ltd_prev) and ltd_curr < ltd_prev:
        score += 1

    ca_c, cl_c = _safe_loc(bs, "Current Assets", 0), _safe_loc(bs, "Current Liabilities", 0)
    ca_p, cl_p = _safe_loc(bs, "Current Assets", 1), _safe_loc(bs, "Current Liabilities", 1)
    cr_curr = ca_c / cl_c if cl_c else np.nan
    cr_prev = ca_p / cl_p if cl_p else np.nan
    if pd.notna(cr_curr) and pd.notna(cr_prev) and cr_curr > cr_prev:
        score += 1

    sh_c = _safe_loc(bs, "Ordinary Shares Number", 0)
    sh_p = _safe_loc(bs, "Ordinary Shares Number", 1)
    if pd.notna(sh_c) and pd.notna(sh_p) and sh_c <= sh_p:
        score += 1

    rev_c, rev_p = _safe_loc(inc, "Total Revenue", 0), _safe_loc(inc, "Total Revenue", 1)
    gp_c, gp_p = _safe_loc(inc, "Gross Profit", 0), _safe_loc(inc, "Gross Profit", 1)
    gm_c = gp_c / rev_c if rev_c else np.nan
    gm_p = gp_p / rev_p if rev_p else np.nan
    if pd.notna(gm_c) and pd.notna(gm_p) and gm_c > gm_p:
        score += 1

    ta_prev2 = _safe_loc(bs, "Total Assets", 2) if bs.shape[1] >= 3 else np.nan
    avg_prev = (ta_prev + ta_prev2) / 2 if pd.notna(ta_prev2) else np.nan
    at_curr = rev_c / avg_assets if avg_assets else np.nan
    at_prev = rev_p / avg_prev if avg_prev else np.nan
    if pd.notna(at_curr) and pd.notna(at_prev) and at_curr > at_prev:
        score += 1

    return float(score)


def _normalize_yf_pct(value):
    """yfinance sometimes returns ratios as percent (e.g. 2.5) and sometimes decimal (0.025).
    Heuristic: values >1 for yields/margins are assumed to be percent."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    return value / 100 if abs(value) > 1 else value


def compute_metrics(symbol: str, bench_returns: pd.Series | None = None) -> dict:
    info = fetch_info(symbol)
    hist = fetch_history(symbol, "2y")

    m: dict = {"ticker": symbol}
    m["name"] = info.get("shortName") or info.get("longName") or symbol
    m["sector"] = info.get("sector", "Unknown")
    m["industry"] = info.get("industry", "Unknown")
    m["country"] = info.get("country", "")
    m["currency"] = (info.get("currency") or "USD").upper()
    m["exchange"] = info.get("exchange") or info.get("fullExchangeName") or ""
    m["market_cap"] = info.get("marketCap", np.nan)
    m["price"] = info.get("currentPrice") or info.get("regularMarketPrice") or np.nan

    m["pe_trailing"] = info.get("trailingPE", np.nan)
    m["pe_forward"] = info.get("forwardPE", np.nan)
    m["peg"] = info.get("pegRatio", np.nan)
    m["pb"] = info.get("priceToBook", np.nan)
    m["ps"] = info.get("priceToSalesTrailing12Months", np.nan)
    m["ev_ebitda"] = info.get("enterpriseToEbitda", np.nan)
    m["ev_rev"] = info.get("enterpriseToRevenue", np.nan)

    m["roe"] = info.get("returnOnEquity", np.nan)
    m["roa"] = info.get("returnOnAssets", np.nan)
    m["gross_margin"] = info.get("grossMargins", np.nan)
    m["op_margin"] = info.get("operatingMargins", np.nan)
    m["net_margin"] = info.get("profitMargins", np.nan)

    m["rev_growth"] = info.get("revenueGrowth", np.nan)
    m["earnings_growth"] = info.get("earningsGrowth", np.nan)
    m["eps_growth_q"] = info.get("earningsQuarterlyGrowth", np.nan)

    de = info.get("debtToEquity", np.nan)
    m["debt_equity"] = de / 100 if pd.notna(de) and de > 5 else de
    m["current_ratio"] = info.get("currentRatio", np.nan)
    m["quick_ratio"] = info.get("quickRatio", np.nan)

    m["div_yield"] = _normalize_yf_pct(info.get("dividendYield", np.nan))
    m["payout_ratio"] = info.get("payoutRatio", np.nan)

    if not hist.empty and "Close" in hist.columns and len(hist) > 5:
        close = hist["Close"]
        rets = returns_from_prices(close)

        m["ret_1m"] = total_return(close, 21)
        m["ret_3m"] = total_return(close, 63)
        m["ret_6m"] = total_return(close, 126)
        m["ret_1y"] = total_return(close, 252)

        ytd_start = close[close.index.year == close.index[-1].year]
        m["ret_ytd"] = float(close.iloc[-1] / ytd_start.iloc[0] - 1) if len(ytd_start) > 0 else np.nan

        m["volatility"] = annualized_volatility(rets)
        m["sharpe"] = sharpe_ratio(rets)
        m["sortino"] = sortino_ratio(rets)
        m["max_dd"] = max_drawdown(close.iloc[-252:] if len(close) >= 252 else close)
        m["rsi_14"] = rsi(close)

        if len(close) >= 252:
            m["momentum_12_1"] = float(close.iloc[-21] / close.iloc[-252] - 1)
        else:
            m["momentum_12_1"] = np.nan

        high_52w = close.iloc[-252:].max() if len(close) >= 252 else close.max()
        m["pct_from_52w_high"] = float(close.iloc[-1] / high_52w - 1)

        if bench_returns is not None:
            m["beta_1y"] = beta_vs_benchmark(rets.iloc[-252:] if len(rets) >= 252 else rets, bench_returns)
        else:
            m["beta_1y"] = info.get("beta", np.nan)
    else:
        for k in ["ret_1m", "ret_3m", "ret_6m", "ret_1y", "ret_ytd", "volatility", "sharpe",
                  "sortino", "max_dd", "rsi_14", "momentum_12_1", "pct_from_52w_high", "beta_1y"]:
            m[k] = np.nan

    m["piotroski_f"] = piotroski_fscore(fetch_financials(symbol))

    return m


def compute_portfolio(symbols: list[str], progress_callback=None) -> pd.DataFrame:
    bench_hist = fetch_history(BENCHMARK, "2y")
    bench_returns = returns_from_prices(bench_hist["Close"]) if not bench_hist.empty else None

    rows = []
    for i, sym in enumerate(symbols):
        if progress_callback:
            progress_callback(i, len(symbols), sym)
        try:
            rows.append(compute_metrics(sym, bench_returns))
        except Exception as e:
            rows.append({"ticker": sym, "error": str(e)})
    return pd.DataFrame(rows)
