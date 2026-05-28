"""Shared portfolio data fetching utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

BENCHMARK = "^GSPC"
RISK_FREE_RATE = 0.04
TRADING_DAYS = 252


# ────────────────────────────────────────────────────────────────────────────
# Major benchmark indices — pulled live from Yahoo Finance.
# For MSCI indices (which aren't directly tradeable/quotable) we use the
# iShares ETF that tracks them — these track within a few basis points after
# fees and are what professional benchmarking platforms actually reference.
# ────────────────────────────────────────────────────────────────────────────

MAJOR_BENCHMARKS: dict[str, dict] = {
    # ── US ─────────────────────────────────────────────────────────────────
    "S&P 500": {
        "ticker": "^GSPC", "category": "US", "asset_class": "Equity",
        "description": "S&P 500 Index — US large-cap, 500 names (direct index quote)",
    },
    "S&P 500 (IVV ETF)": {
        "ticker": "IVV", "category": "US", "asset_class": "Equity",
        "description": "iShares Core S&P 500 ETF — investable proxy for the index",
    },
    "Nasdaq 100": {
        "ticker": "QQQ", "category": "US", "asset_class": "Equity",
        "description": "Invesco QQQ — top 100 non-financial Nasdaq names",
    },
    "Russell 2000": {
        "ticker": "IWM", "category": "US", "asset_class": "Equity",
        "description": "iShares Russell 2000 ETF — US small-cap",
    },
    "Dow Jones": {
        "ticker": "^DJI", "category": "US", "asset_class": "Equity",
        "description": "Dow Jones Industrial Average",
    },

    # ── Global (Developed + Emerging) ──────────────────────────────────────
    "MSCI ACWI": {
        "ticker": "ACWI", "category": "Global", "asset_class": "Equity",
        "description": "iShares MSCI ACWI ETF — All Country World (developed + emerging, ~3,000 names)",
    },
    "MSCI World": {
        "ticker": "URTH", "category": "Global", "asset_class": "Equity",
        "description": "iShares MSCI World ETF — developed markets only (~1,500 names)",
    },

    # ── International (ex-US) ──────────────────────────────────────────────
    "MSCI ACWI ex-US": {
        "ticker": "ACWX", "category": "International", "asset_class": "Equity",
        "description": "iShares MSCI ACWI ex-US ETF — global excluding the United States",
    },
    "MSCI EAFE": {
        "ticker": "EFA", "category": "International", "asset_class": "Equity",
        "description": "iShares MSCI EAFE ETF — developed markets ex-US & Canada",
    },
    "MSCI Europe": {
        "ticker": "VGK", "category": "International", "asset_class": "Equity",
        "description": "Vanguard FTSE Europe ETF — developed Europe",
    },
    "MSCI Japan": {
        "ticker": "EWJ", "category": "International", "asset_class": "Equity",
        "description": "iShares MSCI Japan ETF",
    },

    # ── Emerging ───────────────────────────────────────────────────────────
    "MSCI EM": {
        "ticker": "EEM", "category": "Emerging", "asset_class": "Equity",
        "description": "iShares MSCI Emerging Markets ETF",
    },
    "MSCI China": {
        "ticker": "MCHI", "category": "Emerging", "asset_class": "Equity",
        "description": "iShares MSCI China ETF",
    },
    "MSCI India": {
        "ticker": "INDA", "category": "Emerging", "asset_class": "Equity",
        "description": "iShares MSCI India ETF",
    },

    # ── Fixed income (for completeness — bond benchmarks) ──────────────────
    "US Aggregate Bond": {
        "ticker": "AGG", "category": "Fixed Income", "asset_class": "Bond",
        "description": "iShares Core US Aggregate Bond ETF",
    },
    "US 20+ Treasury": {
        "ticker": "TLT", "category": "Fixed Income", "asset_class": "Bond",
        "description": "iShares 20+ Year Treasury Bond ETF",
    },
    "EM Sovereign USD": {
        "ticker": "EMB", "category": "Fixed Income", "asset_class": "Bond",
        "description": "iShares JPMorgan USD Emerging Markets Bond ETF",
    },

    # ── Real assets ────────────────────────────────────────────────────────
    "Gold": {
        "ticker": "GLD", "category": "Real Assets", "asset_class": "Commodity",
        "description": "SPDR Gold Shares",
    },
    "Global REITs": {
        "ticker": "REET", "category": "Real Assets", "asset_class": "Real Estate",
        "description": "iShares Global REIT ETF",
    },
}


def benchmark_ticker(name: str) -> str:
    """Return the Yahoo ticker for a benchmark name (or pass-through if already a ticker)."""
    if name in MAJOR_BENCHMARKS:
        return MAJOR_BENCHMARKS[name]["ticker"]
    return name


def benchmark_picker_and_data(default: str = "S&P 500",
                                period: str | None = None,
                                location: str = "sidebar",
                                key: str = "selected_benchmark"
                                ) -> tuple[str, pd.Series, pd.Series]:
    """Render a benchmark selector and return (name, prices, returns).

    The selectbox is bound to session_state[key] so the user's choice
    persists across every analytics page automatically. Pass
    `location="inline"` to render in the main area instead of the sidebar.
    """
    if key not in st.session_state:
        st.session_state[key] = default

    target = st.sidebar if location == "sidebar" else st
    target.selectbox(
        "Comparison benchmark",
        options=list(MAJOR_BENCHMARKS.keys()),
        key=key,
        help="Used for alpha / beta / capture / info ratio / relative analysis on this page",
    )
    name = st.session_state[key]

    p = period or st.session_state.get("period", "5y")
    prices_df = fetch_benchmark_prices((name,), period=p)
    if prices_df.empty or name not in prices_df.columns:
        return name, pd.Series(dtype=float), pd.Series(dtype=float)
    series = prices_df[name].dropna()
    returns = series.pct_change().dropna()
    return name, series, returns


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_benchmark_prices(names: tuple, period: str = "5y") -> pd.DataFrame:
    """Fetch price series for one or more benchmarks by their friendly name.

    Returns a wide DataFrame: index = dates, columns = benchmark names (not tickers).
    Cached for 30 minutes — long enough to avoid hammering Yahoo, short enough
    that intra-day moves are reflected after a refresh.
    """
    if not names:
        return pd.DataFrame()
    ticker_to_name = {}
    for n in names:
        t = MAJOR_BENCHMARKS.get(n, {}).get("ticker", n)
        ticker_to_name[t] = n
    prices = fetch_prices(tuple(ticker_to_name.keys()), period=period)
    if prices.empty:
        return prices
    return prices.rename(columns=ticker_to_name)

# ────────────────────────────────────────────────────────────────────────────
# Currency, FX-pair, and interest-rate catalogs
# ────────────────────────────────────────────────────────────────────────────

# FX pairs available on Yahoo Finance. Convention: USD/XXX means "1 USD = N XXX",
# so a rising USD/JPY = JPY weakening vs USD = BAD for JPY holdings in USD terms.
FX_PAIRS = {
    "USD Index (DXY)": "DX-Y.NYB",
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "USDJPY=X",
    "USD/CHF": "USDCHF=X",
    "USD/CAD": "USDCAD=X",
    "AUD/USD": "AUDUSD=X",
    "NZD/USD": "NZDUSD=X",
    "USD/CNY": "USDCNY=X",
    "USD/HKD": "USDHKD=X",
    "USD/KRW": "USDKRW=X",
    "USD/INR": "USDINR=X",
    "USD/TWD": "USDTWD=X",
    "USD/SGD": "USDSGD=X",
    "USD/BRL": "USDBRL=X",
    "USD/MXN": "USDMXN=X",
    "USD/ZAR": "USDZAR=X",
    "USD/SEK": "USDSEK=X",
    "USD/NOK": "USDNOK=X",
    "USD/PLN": "USDPLN=X",
    "USD/TRY": "USDTRY=X",
}

# US Treasury yield indices (yfinance reports these as percent, e.g. 4.5 = 4.5%)
RATE_INDICATORS = {
    "US 13-Week (^IRX)": "^IRX",
    "US 5-Year (^FVX)": "^FVX",
    "US 10-Year (^TNX)": "^TNX",
    "US 30-Year (^TYX)": "^TYX",
}

# Bond / fixed-income ETF proxies for foreign rate exposure
BOND_ETFS = {
    "US 20+ Treasury (TLT)": "TLT",
    "US 7-10Y Treasury (IEF)": "IEF",
    "US 1-3Y Treasury (SHY)": "SHY",
    "US TIPS (TIP)": "TIP",
    "US Investment Grade (LQD)": "LQD",
    "US High Yield (HYG)": "HYG",
    "Emerging Market USD Bonds (EMB)": "EMB",
    "Intl Treasury USD-Hedged (BWX)": "BWX",
    "Total Intl Bond ex-US (BNDX)": "BNDX",
    "Japan Govt Bond ETF (1482.T)": "1482.T",
    "Euro Govt Bond ETF (IBGL.DE)": "IBGL.DE",
}

# Map ISO currency code → FX pair ticker (None = USD base, no conversion needed)
CURRENCY_TO_FX = {
    "USD": None,
    "EUR": "EURUSD=X",
    "GBP": "GBPUSD=X",
    "JPY": "USDJPY=X",
    "CHF": "USDCHF=X",
    "CAD": "USDCAD=X",
    "AUD": "AUDUSD=X",
    "NZD": "NZDUSD=X",
    "CNY": "USDCNY=X",
    "HKD": "USDHKD=X",
    "KRW": "USDKRW=X",
    "INR": "USDINR=X",
    "TWD": "USDTWD=X",
    "SGD": "USDSGD=X",
    "BRL": "USDBRL=X",
    "MXN": "USDMXN=X",
    "ZAR": "USDZAR=X",
    "SEK": "USDSEK=X",
    "NOK": "USDNOK=X",
    "PLN": "USDPLN=X",
    "TRY": "USDTRY=X",
}

# ────────────────────────────────────────────────────────────────────────────
# Approximate region weights for the major benchmarks
# ────────────────────────────────────────────────────────────────────────────
# Used by the Fund Holdings comparison tab to compare a fund's actual
# geographic exposure to a benchmark's known regional mix. These are
# approximations updated to roughly reflect ~2024-2025 index composition;
# they should be refreshed annually as benchmark weights drift.
#
# Region buckets match the labels in bloomberg.EXCHANGE_MAP so a fund's
# region totals can be diff'd against benchmark weights directly.
BENCHMARK_REGION_WEIGHTS: dict[str, dict[str, float]] = {
    "S&P 500": {
        "North America": 1.00,
    },
    "S&P 500 (IVV ETF)": {
        "North America": 1.00,
    },
    "Nasdaq 100": {
        "North America": 1.00,
    },
    "Russell 2000": {
        "North America": 1.00,
    },
    "Dow Jones": {
        "North America": 1.00,
    },
    "MSCI ACWI": {
        # ~ all-country world; weights approximate
        "North America": 0.665,   # US ~63% + Canada ~3%
        "Europe":        0.155,
        "Asia":          0.145,
        "Oceania":       0.018,
        "South America": 0.010,
        "Middle East":   0.004,
        "Africa":        0.003,
    },
    "MSCI World": {
        # developed markets only
        "North America": 0.72,
        "Europe":        0.17,
        "Asia":          0.08,
        "Oceania":       0.03,
    },
    "MSCI ACWI ex-US": {
        # ACWI minus US — Canada still in here
        "North America": 0.075,
        "Europe":        0.42,
        "Asia":          0.43,
        "Oceania":       0.05,
        "South America": 0.012,
        "Middle East":   0.008,
        "Africa":        0.005,
    },
    "MSCI EAFE": {
        # Europe, Australasia, Far East — DM ex-US/Canada
        "Europe":  0.65,
        "Asia":    0.27,
        "Oceania": 0.08,
    },
    "MSCI Europe": {
        "Europe": 1.00,
    },
    "MSCI Japan": {
        "Asia": 1.00,
    },
    "MSCI EM": {
        # emerging markets only
        "Asia":          0.78,
        "South America": 0.10,
        "Middle East":   0.04,
        "Africa":        0.04,
        "Europe":        0.04,
    },
    "MSCI China": {
        "Asia": 1.00,
    },
    "MSCI India": {
        "Asia": 1.00,
    },
}


# Currency symbols for display
CURRENCY_SYMBOLS = {
    "USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥", "CHF": "Fr.",
    "CAD": "C$", "AUD": "A$", "NZD": "NZ$", "CNY": "¥", "HKD": "HK$",
    "KRW": "₩", "INR": "₹", "TWD": "NT$", "SGD": "S$", "BRL": "R$",
    "MXN": "Mex$", "ZAR": "R", "SEK": "kr", "NOK": "kr", "PLN": "zł", "TRY": "₺",
}


# ────────────────────────────────────────────────────────────────────────────
# Factor proxy catalog (multi-factor regression on Factor Models page)
# ────────────────────────────────────────────────────────────────────────────

# Curated factor ETF proxies for multi-factor regression
FACTOR_PROXIES = {
    "Market (SPY)": "SPY",
    "Size (IWM small minus IWB)": "IWM",
    "Value (IWD large value)": "IWD",
    "Growth (IWF large growth)": "IWF",
    "Momentum (MTUM)": "MTUM",
    "Min-Vol / Low-Risk (USMV)": "USMV",
    "Quality (QUAL)": "QUAL",
    "Dividend (VYM)": "VYM",
    "International (EFA)": "EFA",
    "Emerging Markets (EEM)": "EEM",
    "Long-Term Treasury (TLT)": "TLT",
    "Investment Grade Credit (LQD)": "LQD",
    "Gold (GLD)": "GLD",
    "Commodities (DBC)": "DBC",
    "USD Index (UUP)": "UUP",
    "High Yield (HYG)": "HYG",
}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_prices(tickers: tuple, period: str = "5y") -> pd.DataFrame:
    """Fetch adjusted close prices for multiple tickers. Returns wide DataFrame."""
    if not tickers:
        return pd.DataFrame()
    tickers = list(tickers)
    try:
        raw = yf.download(tickers, period=period, auto_adjust=True,
                          progress=False, group_by="column")
    except Exception:
        return pd.DataFrame()
    if raw is None or raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            close = raw["Close"].copy()
        else:
            close = raw.xs("Close", axis=1, level=0, drop_level=True).copy()
    else:
        close = raw[["Close"]].copy()
        close.columns = [tickers[0]]

    close = close.dropna(axis=1, how="all")
    # Forward-fill small gaps (max 3 days) for non-trading days in mixed markets
    close = close.ffill(limit=3)
    return close


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns."""
    return prices.pct_change().dropna(how="all")


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna(how="all")


def portfolio_returns(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Portfolio return series given asset returns and (signed, leverage-respecting) weights.

    Does NOT re-normalize — the weights are expected to already represent the
    desired exposure structure (long-only 100%, 130/30 net 100% gross 160%,
    market-neutral, etc.). Callers should pass normalize_weights() output.
    """
    common = returns.columns.intersection(weights.index)
    if len(common) == 0:
        return pd.Series(dtype=float)
    w = weights.reindex(common).fillna(0)
    return returns[common].fillna(0).dot(w)


def portfolio_cumulative(returns: pd.DataFrame, weights: pd.Series,
                          initial_value: float = 100.0) -> pd.Series:
    pr = portfolio_returns(returns, weights)
    return initial_value * (1 + pr).cumprod()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_currency_info(ticker: str) -> dict:
    """Return {currency, country, exchange, long_name, quote_type} for a ticker.

    Defaults to USD if the fetch fails — the caller should treat that as a soft
    fallback, not as ground truth.
    """
    try:
        info = yf.Ticker(ticker).info or {}
        return {
            "currency": (info.get("currency") or "USD").upper(),
            "country": info.get("country") or "",
            "exchange": info.get("exchange") or info.get("fullExchangeName") or "",
            "long_name": info.get("longName") or info.get("shortName") or ticker,
            "quote_type": info.get("quoteType") or "",
            "sector": info.get("sector") or "",
            "industry": info.get("industry") or "",
        }
    except Exception:
        return {"currency": "USD", "country": "", "exchange": "",
                "long_name": ticker, "quote_type": "", "sector": "", "industry": ""}


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_sector(ticker: str) -> str:
    """Return GICS sector for a ticker, or '' if unavailable. Cached 24h."""
    try:
        info = yf.Ticker(ticker).info or {}
        return info.get("sector") or ""
    except Exception:
        return ""


def fetch_currency_map(tickers) -> dict:
    """Batch-fetch currency info for a list of tickers."""
    return {t: fetch_currency_info(t) for t in tickers}


def format_market_cap(value: float, currency: str = "USD") -> str:
    """Format a market cap with appropriate scale (T/B/M) and currency symbol."""
    if value is None or pd.isna(value) or value <= 0:
        return "—"
    sym = CURRENCY_SYMBOLS.get(currency, "")
    if value >= 1e12:
        return f"{sym}{value / 1e12:,.2f}T {currency}"
    if value >= 1e9:
        return f"{sym}{value / 1e9:,.1f}B {currency}"
    if value >= 1e6:
        return f"{sym}{value / 1e6:,.1f}M {currency}"
    return f"{sym}{value:,.0f} {currency}"


def require_portfolio() -> tuple:
    """Guard for analytics pages: confirm portfolio is loaded in session_state."""
    if "weights" not in st.session_state:
        st.warning("No portfolio loaded. Go to the **Home** page first to build a portfolio.")
        st.stop()
    return (
        st.session_state["tickers"],
        st.session_state["weights"],
        st.session_state["prices"],
        st.session_state["returns"],
        st.session_state["benchmark_prices"],
        st.session_state["benchmark_returns"],
        st.session_state.get("rf", RISK_FREE_RATE),
    )
