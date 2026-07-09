"""Data fetching and quantitative metric computation for the equity screener.

Resilience notes:
  Yahoo Finance routinely rate-limits or returns empty ``info`` payloads when
  called from cloud IP ranges (Streamlit Cloud, Heroku, etc.). To keep the
  screener from going blank under those conditions we:

    1. Use a ``curl_cffi`` session with browser impersonation when available
       (yfinance reads the ``session=`` kwarg and routes its HTTP calls through
       it). This sidesteps most Yahoo bot heuristics.
    2. Retry empty info responses once before giving up.
    3. Fall back to ``Ticker.fast_info`` and seed a minimal ``info`` dict so
       downstream code (tearsheet, screener) still gets headline fields like
       market cap, currency, and last price even when the rich endpoint is
       blocked.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from scipy import stats as sps

RISK_FREE_RATE = 0.04
TRADING_DAYS = 252
BENCHMARK = "^GSPC"

# Build a session that impersonates a real browser so Yahoo is less likely
# to rate-limit us. yfinance picks this up via the ``session=`` kwarg.
try:  # pragma: no cover - depends on optional dep
    from curl_cffi import requests as _curl_requests

    _YF_SESSION = _curl_requests.Session(impersonate="chrome")
except Exception:  # noqa: BLE001 - any failure → fall back to default requests
    _YF_SESSION = None


def _ticker(symbol: str) -> yf.Ticker:
    """Construct a yfinance Ticker, attaching our browser-impersonating session
    when available. Without the session, yfinance uses plain ``requests`` which
    Yahoo blocks from many cloud IPs."""
    if _YF_SESSION is not None:
        try:
            return yf.Ticker(symbol, session=_YF_SESSION)
        except TypeError:
            # Older yfinance versions don't accept session — fall through.
            pass
    return yf.Ticker(symbol)


def _fast_info_to_dict(t: yf.Ticker) -> dict:
    """Pull whatever we can from ``fast_info`` and shape it like ``info``.

    Used as a fallback when ``Ticker.info`` returns ``{}`` (cloud-IP block).
    """
    out: dict = {}
    try:
        fi = t.fast_info
    except Exception:
        return out
    mapping = {
        "last_price": "currentPrice",
        "market_cap": "marketCap",
        "currency": "currency",
        "shares": "sharesOutstanding",
        "year_high": "fiftyTwoWeekHigh",
        "year_low": "fiftyTwoWeekLow",
        "previous_close": "regularMarketPreviousClose",
        "fifty_day_average": "fiftyDayAverage",
        "two_hundred_day_average": "twoHundredDayAverage",
        "exchange": "exchange",
        "quote_type": "quoteType",
    }
    for src, dst in mapping.items():
        try:
            v = getattr(fi, src, None)
        except Exception:
            v = None
        if v is not None:
            out[dst] = v
    return out


# ── Fundamentals reconstruction (used when quoteSummary / .info is blocked) ──

def _stmt_row(df: pd.DataFrame, key: str, col: int = 0):
    """Safely pull a single statement value (latest col = 0). Returns None."""
    if df is None or df.empty or key not in df.index or col >= df.shape[1]:
        return None
    try:
        v = float(df.loc[key].iloc[col])
        return v if pd.notna(v) else None
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def _stmt_first(df: pd.DataFrame, keys: list[str], col: int = 0):
    """First non-None value among several candidate row labels."""
    for k in keys:
        v = _stmt_row(df, k, col)
        if v is not None:
            return v
    return None


def _has_fundamentals(info: dict) -> bool:
    """True if the rich fundamentals we care about are present.

    When Yahoo blocks quoteSummary these keys are all missing even though
    fast_info (price, market cap, shares) succeeds — that's our signal to
    rebuild from the statement endpoints.
    """
    return any(
        info.get(k) is not None
        for k in ("trailingPE", "priceToBook", "returnOnEquity",
                  "profitMargins", "totalRevenue")
    )


def _reconstruct_info_from_statements(symbol: str, t: yf.Ticker, base: dict) -> dict:
    """Rebuild the fundamental fields normally found in ``Ticker.info`` from
    the financial-statement, dividend, and fast_info endpoints.

    Yahoo's quoteSummary endpoint (which powers ``.info``) is frequently
    blocked for cloud IP ranges, returning an empty dict even though the
    statement and chart endpoints still work. This rebuilds P/E, P/B, P/S,
    ROE, ROA, margins, growth, leverage, liquidity, dividend yield, and EV
    multiples — keyed with yfinance's own ``.info`` names — so the screener
    and tearsheet stay populated on those hosts.
    """
    info = dict(base)
    fin = fetch_financials(symbol)
    inc, bs, cf = fin["income"], fin["balance"], fin["cashflow"]
    if (inc is None or inc.empty) and (bs is None or bs.empty):
        return info  # no statements to rebuild from

    price = info.get("currentPrice")
    shares = info.get("sharesOutstanding")
    mcap = info.get("marketCap")
    if mcap is None and price is not None and shares:
        mcap = price * shares
        info["marketCap"] = mcap

    # Income statement (latest = col 0; prior year = col 1 for growth)
    rev = _stmt_first(inc, ["Total Revenue", "Operating Revenue"])
    ni = _stmt_first(inc, ["Net Income Common Stockholders", "Net Income"])
    gp = _stmt_row(inc, "Gross Profit")
    opinc = _stmt_first(inc, ["Operating Income", "Total Operating Income As Reported"])
    ebitda = _stmt_first(inc, ["EBITDA", "Normalized EBITDA"])
    eps = _stmt_first(inc, ["Diluted EPS", "Basic EPS"])
    rev_p = _stmt_first(inc, ["Total Revenue", "Operating Revenue"], col=1)
    ni_p = _stmt_first(inc, ["Net Income Common Stockholders", "Net Income"], col=1)

    # Balance sheet
    equity = _stmt_first(bs, ["Stockholders Equity", "Common Stock Equity",
                              "Total Equity Gross Minority Interest"])
    debt = _stmt_row(bs, "Total Debt")
    cash = _stmt_first(bs, ["Cash And Cash Equivalents",
                            "Cash Cash Equivalents And Short Term Investments"])
    cur_a = _stmt_row(bs, "Current Assets")
    cur_l = _stmt_row(bs, "Current Liabilities")
    inv = _stmt_row(bs, "Inventory")
    assets = _stmt_row(bs, "Total Assets")

    # Cash flow
    ocf = _stmt_first(cf, ["Operating Cash Flow",
                           "Cash Flow From Continuing Operating Activities"])
    fcf = _stmt_row(cf, "Free Cash Flow")

    def _set(key, val):
        if val is not None and (not isinstance(val, float) or np.isfinite(val)):
            info[key] = val

    # Profitability / margins (decimals — yfinance convention)
    if rev:
        _set("totalRevenue", rev)
        if gp is not None:     _set("grossMargins", gp / rev)
        if opinc is not None:  _set("operatingMargins", opinc / rev)
        if ni is not None:     _set("profitMargins", ni / rev)
        if ebitda is not None: _set("ebitdaMargins", ebitda / rev)
    if ni is not None:
        _set("netIncomeToCommon", ni)
        if equity: _set("returnOnEquity", ni / equity)
        if assets: _set("returnOnAssets", ni / assets)
    if ebitda is not None: _set("ebitda", ebitda)
    if eps is not None:    _set("trailingEps", eps)

    # Growth (YoY)
    if rev and rev_p:               _set("revenueGrowth", rev / rev_p - 1)
    if ni and ni_p and ni_p != 0:   _set("earningsGrowth", ni / ni_p - 1)

    # Valuation multiples
    if price is not None and eps and eps != 0:
        _set("trailingPE", price / eps)
    if mcap and equity:
        _set("priceToBook", mcap / equity)
    if mcap and rev:
        _set("priceToSalesTrailing12Months", mcap / rev)
    if mcap is not None:
        ev = mcap + (debt or 0) - (cash or 0)
        _set("enterpriseValue", ev)
        if ebitda: _set("enterpriseToEbitda", ev / ebitda)
        if rev:    _set("enterpriseToRevenue", ev / rev)

    # Leverage / liquidity
    if debt is not None and equity:
        _set("debtToEquity", debt / equity * 100.0)  # yfinance reports as percent
    if cur_a and cur_l:
        _set("currentRatio", cur_a / cur_l)
        if inv is not None:
            _set("quickRatio", (cur_a - inv) / cur_l)
    if cash is not None: _set("totalCash", cash)
    if debt is not None: _set("totalDebt", debt)
    if ocf is not None:  _set("operatingCashflow", ocf)
    if fcf is not None:  _set("freeCashflow", fcf)

    # Dividend yield + rate from the dividends (chart) endpoint
    try:
        divs = t.dividends
    except Exception:
        divs = None
    if divs is not None and len(divs) and price:
        idx = pd.DatetimeIndex(divs.index)
        cutoff = idx.max() - pd.Timedelta(days=365)
        ttm = float(divs[idx >= cutoff].sum())
        if ttm > 0:
            _set("trailingAnnualDividendRate", ttm)
            _set("trailingAnnualDividendYield", ttm / price)  # decimal
            _set("dividendRate", ttm)

    info["_reconstructed"] = True
    return info


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_history(symbol: str, period: str = "2y") -> pd.DataFrame:
    """Fetch adjusted price history. Retries once on empty result."""
    for attempt in range(2):
        try:
            hist = _ticker(symbol).history(period=period, auto_adjust=True)
            if hist is not None and not hist.empty:
                return hist
        except Exception:
            pass
        if attempt == 0:
            time.sleep(0.3)
    return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_info(symbol: str) -> dict:
    """Fetch the rich ``info`` payload. When Yahoo blocks the request we
    fall back to ``fast_info`` so headline fields are still populated."""
    t = _ticker(symbol)
    info: dict | None = None
    for attempt in range(2):
        try:
            raw = t.info
            if raw:
                info = raw
                break
        except Exception:
            pass
        if attempt == 0:
            time.sleep(0.3)

    if not info:
        # Rich endpoint failed — synthesize from fast_info so downstream
        # consumers (tearsheet, screener) still get price + market cap.
        info = _fast_info_to_dict(t)
    else:
        # Backfill any holes from fast_info if it offers something info missed.
        fast = _fast_info_to_dict(t)
        for k, v in fast.items():
            info.setdefault(k, v)

    # If the rich fundamentals are missing — i.e. Yahoo blocked quoteSummary
    # for this host (common on cloud) — rebuild P/E, P/B, ROE, margins, etc.
    # from the statement and dividend endpoints, which aren't blocked.
    if not _has_fundamentals(info):
        info = _reconstruct_info_from_statements(symbol, t, info)

    return info or {}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_financials(symbol: str) -> dict:
    try:
        t = _ticker(symbol)
        return {
            "income": t.income_stmt if t.income_stmt is not None else pd.DataFrame(),
            "balance": t.balance_sheet if t.balance_sheet is not None else pd.DataFrame(),
            "cashflow": t.cashflow if t.cashflow is not None else pd.DataFrame(),
        }
    except Exception:
        return {"income": pd.DataFrame(), "balance": pd.DataFrame(), "cashflow": pd.DataFrame()}


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_prices_bulk(symbols: tuple, period: str = "2y") -> dict:
    """Download adjusted closes for many tickers in ONE request.

    Returns {symbol: close_series}. Dramatically faster than per-ticker
    ``history()`` for a universe (one HTTP round-trip vs N). Tickers that
    fail simply don't appear in the dict — callers fall back to per-ticker
    fetching for those.
    """
    syms = list(dict.fromkeys(s for s in symbols if s))  # dedupe, keep order
    if not syms:
        return {}
    kw = dict(period=period, auto_adjust=True, progress=False,
              group_by="column", threads=True)
    raw = None
    for attempt in range(2):
        try:
            if _YF_SESSION is not None:
                try:
                    raw = yf.download(syms, session=_YF_SESSION, **kw)
                except TypeError:
                    raw = yf.download(syms, **kw)  # older/newer signature
            else:
                raw = yf.download(syms, **kw)
            if raw is not None and not raw.empty:
                break
        except Exception:
            raw = None
        if attempt == 0:
            time.sleep(0.4)
    if raw is None or raw.empty:
        return {}

    out: dict = {}
    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = raw.columns.get_level_values(0)
        close = raw["Close"] if "Close" in lvl0 else None
        if close is None:
            return {}
        for s in syms:
            if s in close.columns:
                ser = close[s].dropna()
                if len(ser) > 1:
                    out[s] = ser
    else:
        # single-ticker download → flat columns
        if "Close" in raw.columns:
            ser = raw["Close"].dropna()
            if len(ser) > 1:
                out[syms[0]] = ser
    return out


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


def _align_returns(a: pd.Series, b: pd.Series) -> pd.DataFrame:
    """Inner-join two return series on calendar date, ignoring timezone.

    Foreign listings come back tz-localized to their home exchange
    (Asia/Tokyo, Asia/Taipei, Europe/Zurich, …) while a US benchmark like
    ^GSPC is America/New_York. A naive timestamp join then matches nothing
    (e.g. ``2024-01-15 00:00-05:00`` != ``2024-01-15 00:00+09:00``), which
    made beta NaN for *every* foreign equity. Normalizing both indices to
    tz-naive calendar dates restores cross-exchange alignment.
    """
    def _norm(s: pd.Series) -> pd.Series:
        idx = pd.DatetimeIndex(s.index)
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        out = pd.Series(np.asarray(s.values), index=idx.normalize())
        return out[~out.index.duplicated(keep="last")]

    return pd.concat([_norm(a), _norm(b)], axis=1, join="inner").dropna()


def beta_vs_benchmark(symbol_returns: pd.Series, bench_returns: pd.Series) -> float:
    df = _align_returns(symbol_returns, bench_returns)
    if len(df) < 30:
        return np.nan
    cov = df.cov().iloc[0, 1]
    var = df.iloc[:, 1].var()
    return float(cov / var) if var > 0 else np.nan


TTEST_LOOKBACKS = {"1y": 252, "2y": 504, "5y": None}  # None ⇒ full series
TTEST_FIELDS = ("n", "t_mean", "p_mean", "capm_alpha", "capm_t", "capm_p")
MIN_TTEST_OBS = 20


def _one_window_ttests(r: pd.Series, bench_returns: pd.Series | None) -> dict:
    """Drift + CAPM (Jensen's) alpha tests on one return-series window.

    drift — one-sample t-test, H0: mean daily return = 0.
    CAPM  — regress the stock's excess return on the market's excess return;
            the intercept is Jensen's alpha. Test H0: alpha = 0 via the
            intercept's own standard error (risk-adjusted, so a high-beta name
            doesn't get credit for merely riding the market up).
    """
    res = {"n": int(len(r)), "t_mean": np.nan, "p_mean": np.nan,
           "capm_alpha": np.nan, "capm_t": np.nan, "capm_p": np.nan}
    if len(r) >= MIN_TTEST_OBS and float(r.std()) > 0:
        t, p = sps.ttest_1samp(r, 0.0)
        res["t_mean"], res["p_mean"] = float(t), float(p)
    if bench_returns is not None:
        aligned = _align_returns(r, bench_returns)
        if len(aligned) >= MIN_TTEST_OBS:
            rf_d = RISK_FREE_RATE / TRADING_DAYS
            y = (aligned.iloc[:, 0] - rf_d).to_numpy()
            x = (aligned.iloc[:, 1] - rf_d).to_numpy()
            if np.std(x) > 0:
                lr = sps.linregress(x, y)
                a = float(lr.intercept)
                se = float(getattr(lr, "intercept_stderr", np.nan) or np.nan)
                if se > 0:
                    tt = a / se
                    dfree = max(len(x) - 2, 1)
                    res["capm_alpha"] = a * TRADING_DAYS
                    res["capm_t"] = float(tt)
                    res["capm_p"] = float(2 * sps.t.sf(abs(tt), dfree))
    return res


def ttest_stats(rets: pd.Series | None, bench_returns: pd.Series | None = None) -> dict:
    """Drift + CAPM-alpha significance at 1y / 2y / 5y lookbacks.

    Emits one set of fields per lookback, suffixed ``_1y`` / ``_2y`` / ``_5y``
    (e.g. ``t_mean_2y``, ``capm_p_5y``), so the UI can switch windows instantly
    without re-fetching.
    """
    out: dict = {}
    r_all = rets.dropna() if rets is not None else pd.Series(dtype=float)
    for label, w in TTEST_LOOKBACKS.items():
        r = r_all.iloc[-w:] if (w is not None and len(r_all) > w) else r_all
        for k, v in _one_window_ttests(r, bench_returns).items():
            out[f"{k}_{label}"] = v
    return out


def two_sample_ttest(rets_a: pd.Series, rets_b: pd.Series) -> dict:
    """Paired t-test comparing the mean daily returns of two stocks.

    The series are aligned on common trading dates and tested as a paired
    difference (H0: mean[A − B] = 0). Pairing removes the shared market move,
    so it's the right test for "does A out-return B by more than noise?".
    """
    out = {"n": 0, "t": np.nan, "p": np.nan, "diff_ann": np.nan,
           "mean_a_ann": np.nan, "mean_b_ann": np.nan}
    if rets_a is None or rets_b is None:
        return out
    df = _align_returns(rets_a.dropna(), rets_b.dropna())
    out["n"] = int(len(df))
    if len(df) < 20:
        return out
    a, b = df.iloc[:, 0], df.iloc[:, 1]
    diff = a - b
    if float(diff.std()) == 0:
        return out
    t, p = sps.ttest_1samp(diff, 0.0)
    out.update(t=float(t), p=float(p), diff_ann=float(diff.mean() * TRADING_DAYS),
               mean_a_ann=float(a.mean() * TRADING_DAYS),
               mean_b_ann=float(b.mean() * TRADING_DAYS))
    return out


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


def dividend_yield_decimal(info: dict) -> float:
    """Resolve a dividend yield to a decimal (0.025 == 2.5%).

    yfinance has repeatedly flip-flopped on units for ``dividendYield``:
    older builds returned a decimal (0.025) while current builds return a
    percent number (2.5 → meaning 2.5%, and 0.35 → meaning 0.35%). The naive
    ``value/100 if value > 1`` heuristic therefore reported absurd figures
    like 35% for Apple (raw 0.35, which is really 0.35%).

    We disambiguate using ``trailingAnnualDividendYield`` — which is always a
    decimal — as ground truth: pick whichever interpretation of
    ``dividendYield`` lands closest to it. When the trailing field is absent
    (some foreign listings), fall back to dividing by 100 (modern behavior).
    """
    dy = info.get("dividendYield")
    tay = info.get("trailingAnnualDividendYield")
    dy_ok = dy is not None and not (isinstance(dy, float) and np.isnan(dy))
    tay_ok = tay is not None and not (isinstance(tay, float) and np.isnan(tay))

    if dy_ok:
        dy = float(dy)
        if tay_ok and tay > 0:
            as_pct = dy / 100.0
            # Closer to the trailing decimal wins.
            return as_pct if abs(as_pct - tay) <= abs(dy - tay) else dy
        # No reliable reference — modern yfinance reports percent.
        return dy / 100.0
    if tay_ok:
        return float(tay)
    return np.nan


def compute_metrics(symbol: str, bench_returns: pd.Series | None = None,
                    close: pd.Series | None = None,
                    include_financials: bool = True) -> dict:
    """Compute one ticker's metric row.

    ``close``: a pre-fetched adjusted-close series (e.g. from
    ``fetch_prices_bulk``). When provided, the per-ticker history call is
    skipped. ``include_financials``: when False, the Piotroski F-Score (which
    needs 3 extra statement calls — the slowest part) is skipped (NaN).
    """
    info = fetch_info(symbol)
    if close is None or len(close) <= 5:
        hist = fetch_history(symbol, "2y")
        close = hist["Close"] if (not hist.empty and "Close" in hist.columns) else None

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

    m["div_yield"] = dividend_yield_decimal(info)
    m["payout_ratio"] = info.get("payoutRatio", np.nan)

    if close is not None and len(close) > 5:
        close_full = close                       # up to 5y — used for the t-tests
        close = close.iloc[-504:] if len(close) > 504 else close  # ~2y for the metrics below
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

        # Statistical-significance (t-test) evaluation across 1y/2y/5y windows,
        # using the FULL (up-to-5y) return series.
        m.update(ttest_stats(returns_from_prices(close_full), bench_returns))
    else:
        for k in ["ret_1m", "ret_3m", "ret_6m", "ret_1y", "ret_ytd", "volatility", "sharpe",
                  "sortino", "max_dd", "rsi_14", "momentum_12_1", "pct_from_52w_high", "beta_1y"]:
            m[k] = np.nan
        for lb in TTEST_LOOKBACKS:
            for f in TTEST_FIELDS:
                m[f"{f}_{lb}"] = np.nan

    m["piotroski_f"] = (piotroski_fscore(fetch_financials(symbol))
                        if include_financials else np.nan)

    return m


def compute_portfolio(symbols: list[str], progress_callback=None,
                       max_workers: int = 10,
                       include_financials: bool = False) -> pd.DataFrame:
    """Compute metrics for a list of tickers, optimized for larger universes.

    Speed model:
      • Prices for ALL tickers (+ benchmark) are pulled in ONE bulk
        ``yf.download`` instead of N per-ticker history calls.
      • Each ticker's ``.info`` runs in a worker thread.
      • Piotroski (3 extra statement calls — the slowest part) is OFF by
        default; pass ``include_financials=True`` to compute it.
    Order of the input list is preserved in the output DataFrame.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # One bulk request for every ticker's prices plus the benchmark. 5y of
    # history so the T-Test's 5y lookback has data; metrics still use ~2y.
    price_map = fetch_prices_bulk(tuple(symbols) + (BENCHMARK,), "5y")
    bench_close = price_map.get(BENCHMARK)
    if bench_close is None:
        bh = fetch_history(BENCHMARK, "5y")
        bench_close = bh["Close"] if (not bh.empty and "Close" in bh.columns) else None
    bench_returns = returns_from_prices(bench_close) if bench_close is not None else None

    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = {
            exe.submit(compute_metrics, s, bench_returns,
                       price_map.get(s), include_financials): s
            for s in symbols
        }
        done = 0
        for future in as_completed(futures):
            sym = futures[future]
            try:
                results[sym] = future.result()
            except Exception as e:
                results[sym] = {"ticker": sym, "error": str(e)}
            done += 1
            if progress_callback:
                progress_callback(done - 1, len(symbols), sym)
    return pd.DataFrame([results[s] for s in symbols])


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_fast_info(symbol: str) -> dict:
    """Lightweight quote (market cap, price, currency, 52w) via fast_info.

    Much cheaper than the full ``.info`` quoteSummary call — used for the
    fast universe-browse path where we don't need valuation/quality fields.
    """
    try:
        return _fast_info_to_dict(_ticker(symbol))
    except Exception:
        return {}


def compute_universe_lite(symbols: list[str], meta: dict | None = None,
                          progress_callback=None, max_workers: int = 12) -> pd.DataFrame:
    """Fast, price-only snapshot of a universe for *browsing* (not deep screening).

    One bulk price download drives all the return/vol/momentum/beta metrics;
    market cap comes from ``fast_info`` (parallel); name + sector come from the
    curated ``meta`` map ({symbol: {"name", "sector"}}). No ``.info`` and no
    financials, so a few-hundred-name universe loads in seconds. Use
    ``compute_portfolio`` afterwards for the full valuation/quality/composite
    pass on whatever the user narrows down to.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    meta = meta or {}

    price_map = fetch_prices_bulk(tuple(symbols) + (BENCHMARK,), "2y")
    bench_close = price_map.get(BENCHMARK)
    bench_returns = returns_from_prices(bench_close) if bench_close is not None else None

    fast: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = {exe.submit(fetch_fast_info, s): s for s in symbols}
        done = 0
        for future in as_completed(futures):
            s = futures[future]
            try:
                fast[s] = future.result()
            except Exception:
                fast[s] = {}
            done += 1
            if progress_callback:
                progress_callback(done - 1, len(symbols), s)

    rows = []
    for s in symbols:
        fi = fast.get(s, {})
        close = price_map.get(s)
        info_meta = meta.get(s, {})
        m: dict = {
            "ticker": s,
            "name": info_meta.get("name") or s,
            "sector": info_meta.get("sector", "Unknown"),
            "market_cap": fi.get("marketCap", np.nan),
            "price": fi.get("currentPrice", np.nan),
            "currency": (fi.get("currency") or "USD"),
        }
        if close is not None and len(close) > 5:
            rets = returns_from_prices(close)
            m["ret_1m"] = total_return(close, 21)
            m["ret_3m"] = total_return(close, 63)
            m["ret_6m"] = total_return(close, 126)
            m["ret_1y"] = total_return(close, 252)
            ytd = close[close.index.year == close.index[-1].year]
            m["ret_ytd"] = float(close.iloc[-1] / ytd.iloc[0] - 1) if len(ytd) else np.nan
            m["volatility"] = annualized_volatility(rets)
            m["max_dd"] = max_drawdown(close.iloc[-252:] if len(close) >= 252 else close)
            m["rsi_14"] = rsi(close)
            hi = close.iloc[-252:].max() if len(close) >= 252 else close.max()
            m["pct_from_52w_high"] = float(close.iloc[-1] / hi - 1)
            m["momentum_12_1"] = (float(close.iloc[-21] / close.iloc[-252] - 1)
                                  if len(close) >= 252 else np.nan)
            m["beta_1y"] = (beta_vs_benchmark(rets.iloc[-252:] if len(rets) >= 252 else rets,
                                              bench_returns)
                            if bench_returns is not None else np.nan)
        else:
            for k in ["ret_1m", "ret_3m", "ret_6m", "ret_1y", "ret_ytd", "volatility",
                      "max_dd", "rsi_14", "pct_from_52w_high", "momentum_12_1", "beta_1y"]:
                m[k] = np.nan
        rows.append(m)
    return pd.DataFrame(rows)
